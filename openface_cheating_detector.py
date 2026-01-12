"""
OpenFace Cheating Detection System
===================================
Uses OpenFace gaze and head pose data to differentiate between:
1. Looking at another screen (potential cheating)
2. Looking away to think (normal behavior)

Key behavioral differences:
- Cheating: Sustained horizontal gaze to fixed point, head turned to side, reading patterns
- Thinking: Variable gaze direction, upward/unfocused gaze, wandering eye movements

Requirements:
    - OpenFace 2.0+ installed (https://github.com/TadasBaltrusaitis/OpenFace)
    - Python 3.8+
    - numpy, pandas, scipy
    - opencv-python (for webcam mode)

Usage:
    # Process a video file
    python openface_cheating_detector.py --video interview.mp4

    # Process pre-extracted OpenFace CSV
    python openface_cheating_detector.py --csv openface_output.csv

    # Real-time webcam analysis
    python openface_cheating_detector.py --webcam

    # Webcam with specific device
    python openface_cheating_detector.py --webcam --camera-id 1
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
from collections import deque
import subprocess
import tempfile
import os
import argparse
import json
import time
import threading
import signal
import sys
from pathlib import Path

# Optional imports for webcam mode
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class GazeBehavior(Enum):
    """Classification of gaze behavior."""
    NORMAL = "normal"                    # Looking at camera/screen
    THINKING = "thinking"                # Looking away to think
    SUSPICIOUS = "suspicious"            # Possible cheating indicator
    CHEATING = "cheating"               # Strong cheating indicator


@dataclass
class GazeFrame:
    """Single frame of gaze data from OpenFace."""
    frame_num: int
    timestamp: float

    # Gaze direction vectors (3D unit vectors for each eye)
    gaze_0_x: float  # Left eye gaze x
    gaze_0_y: float  # Left eye gaze y
    gaze_0_z: float  # Left eye gaze z
    gaze_1_x: float  # Right eye gaze x
    gaze_1_y: float  # Right eye gaze y
    gaze_1_z: float  # Right eye gaze z

    # Gaze angles (radians)
    gaze_angle_x: float  # Horizontal gaze angle
    gaze_angle_y: float  # Vertical gaze angle

    # Head pose
    pose_Rx: float  # Head pitch (radians)
    pose_Ry: float  # Head yaw (radians)
    pose_Rz: float  # Head roll (radians)

    # Facial Action Units relevant to thinking/concentration
    AU04_r: float = 0.0  # Brow lowerer (concentration)
    AU07_r: float = 0.0  # Lid tightener (focus)
    AU14_r: float = 0.0  # Dimpler (thinking)
    AU23_r: float = 0.0  # Lip tightener (concentration)

    # Confidence
    confidence: float = 1.0
    success: bool = True


@dataclass
class BehaviorWindow:
    """Analysis of gaze behavior over a time window."""
    start_time: float
    end_time: float
    behavior: GazeBehavior
    confidence: float

    # Metrics that led to classification
    avg_horizontal_gaze: float
    gaze_variability: float
    avg_head_yaw: float
    head_stability: float
    vertical_gaze_tendency: float
    thinking_au_score: float

    details: Dict = field(default_factory=dict)


@dataclass
class CheatingReport:
    """Full analysis report for a video/session."""
    total_duration: float
    total_frames: int

    # Time spent in each behavior
    normal_time: float
    thinking_time: float
    suspicious_time: float
    cheating_time: float

    # Suspicious events
    suspicious_events: List[BehaviorWindow]
    cheating_events: List[BehaviorWindow]

    # Overall assessment
    overall_risk_score: float  # 0-1, higher = more suspicious
    assessment: str


class OpenFaceCheatingDetector:
    """
    Detects cheating behavior by analyzing OpenFace gaze and pose data.

    The detector uses multiple heuristics to differentiate between:

    1. THINKING behavior (normal):
       - Gaze tends upward (looking up to recall information)
       - Variable gaze direction (eyes wandering while thinking)
       - May look slightly down (internal reflection)
       - Head may tilt but doesn't fix on a specific direction
       - Often accompanied by concentration AUs (brow furrow)

    2. CHEATING behavior (looking at another screen):
       - Sustained horizontal gaze to one side
       - Head turned and maintained at fixed angle (reading position)
       - Low vertical gaze variance (reading horizontal text)
       - Repetitive return to same gaze position
       - May show small horizontal saccades (reading patterns)
    """

    # Thresholds (in radians unless noted)
    HORIZONTAL_GAZE_THRESHOLD = 0.25  # ~14 degrees - looking significantly to side
    SUSTAINED_GAZE_DURATION = 2.0     # Seconds of sustained side gaze for suspicion
    CHEATING_DURATION = 4.0           # Seconds for high-confidence cheating
    HEAD_YAW_THRESHOLD = 0.20         # ~11 degrees head turn
    THINKING_VERTICAL_THRESHOLD = 0.15  # Upward gaze angle for thinking
    GAZE_VARIABILITY_THRESHOLD = 0.08   # Low variability suggests reading

    # Window settings
    ANALYSIS_WINDOW_SEC = 2.0  # Analyze behavior in 2-second windows
    WINDOW_OVERLAP = 0.5       # 50% overlap between windows

    def __init__(
        self,
        horizontal_threshold: float = None,
        sustained_duration: float = None,
        openface_path: str = None
    ):
        """
        Initialize the detector with optional custom thresholds.

        Args:
            horizontal_threshold: Radians for horizontal gaze threshold
            sustained_duration: Seconds of sustained gaze for suspicion
            openface_path: Path to OpenFace FeatureExtraction binary
        """
        self.horizontal_threshold = horizontal_threshold or self.HORIZONTAL_GAZE_THRESHOLD
        self.sustained_duration = sustained_duration or self.SUSTAINED_GAZE_DURATION

        # Find OpenFace binary
        self.openface_path = openface_path or self._find_openface()

        # State for real-time processing
        self.frame_buffer: deque = deque(maxlen=300)  # ~10 sec at 30fps
        self.current_behavior = GazeBehavior.NORMAL
        self.behavior_history: List[BehaviorWindow] = []

    def _find_openface(self) -> Optional[str]:
        """Locate OpenFace FeatureExtraction binary."""
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()

        possible_paths = [
            # Local project paths (check these first)
            os.path.join(cwd, "OpenFace", "build", "bin", "FeatureExtraction"),
            os.path.join(cwd, "OpenFace", "FeatureExtraction"),
            os.path.join(cwd, "OpenFace", "x64", "Release", "FeatureExtraction.exe"),
            os.path.join(script_dir, "OpenFace", "build", "bin", "FeatureExtraction"),
            os.path.join(script_dir, "OpenFace", "FeatureExtraction"),
            os.path.join(cwd, "..", "OpenFace", "build", "bin", "FeatureExtraction"),
            # System paths
            "/usr/local/bin/FeatureExtraction",
            "/opt/OpenFace/build/bin/FeatureExtraction",
            "~/OpenFace/build/bin/FeatureExtraction",
            # Windows paths
            "C:/OpenFace/FeatureExtraction.exe",
            "C:/OpenFace/build/bin/Release/FeatureExtraction.exe",
            # In PATH
            "FeatureExtraction",
        ]

        for path in possible_paths:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                return expanded

        # Search recursively in OpenFace directory if it exists
        openface_dirs = [
            os.path.join(cwd, "OpenFace"),
            os.path.join(script_dir, "OpenFace"),
        ]
        for of_dir in openface_dirs:
            if os.path.isdir(of_dir):
                for binary_name in ["FeatureExtraction", "FeatureExtraction.exe"]:
                    for root, dirs, files in os.walk(of_dir):
                        if binary_name in files:
                            return os.path.join(root, binary_name)

        return None

    def extract_features_from_video(self, video_path: str) -> str:
        """
        Run OpenFace on a video file to extract features.

        Args:
            video_path: Path to input video

        Returns:
            Path to generated CSV file
        """
        if not self.openface_path:
            raise RuntimeError(
                "OpenFace not found. Install from: "
                "https://github.com/TadasBaltrusaitis/OpenFace"
            )

        output_dir = tempfile.mkdtemp(prefix="openface_")

        cmd = [
            self.openface_path,
            "-f", video_path,
            "-out_dir", output_dir,
            "-pose",      # Extract head pose
            "-gaze",      # Extract gaze
            "-aus",       # Extract action units
            "-2Dfp",      # 2D facial landmarks
        ]

        print(f"Running OpenFace: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)

        # Find the output CSV
        video_name = Path(video_path).stem
        csv_path = os.path.join(output_dir, f"{video_name}.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"OpenFace output not found: {csv_path}")

        return csv_path

    def load_openface_csv(self, csv_path: str) -> List[GazeFrame]:
        """
        Load and parse OpenFace CSV output.

        Args:
            csv_path: Path to OpenFace CSV output

        Returns:
            List of GazeFrame objects
        """
        df = pd.read_csv(csv_path)

        # Clean column names (OpenFace adds spaces)
        df.columns = df.columns.str.strip()

        frames = []
        for _, row in df.iterrows():
            try:
                frame = GazeFrame(
                    frame_num=int(row['frame']),
                    timestamp=float(row['timestamp']),

                    # Gaze vectors
                    gaze_0_x=float(row['gaze_0_x']),
                    gaze_0_y=float(row['gaze_0_y']),
                    gaze_0_z=float(row['gaze_0_z']),
                    gaze_1_x=float(row['gaze_1_x']),
                    gaze_1_y=float(row['gaze_1_y']),
                    gaze_1_z=float(row['gaze_1_z']),

                    # Gaze angles
                    gaze_angle_x=float(row['gaze_angle_x']),
                    gaze_angle_y=float(row['gaze_angle_y']),

                    # Head pose
                    pose_Rx=float(row['pose_Rx']),
                    pose_Ry=float(row['pose_Ry']),
                    pose_Rz=float(row['pose_Rz']),

                    # Action units (may not all be present)
                    AU04_r=float(row.get('AU04_r', 0)),
                    AU07_r=float(row.get('AU07_r', 0)),
                    AU14_r=float(row.get('AU14_r', 0)),
                    AU23_r=float(row.get('AU23_r', 0)),

                    confidence=float(row.get('confidence', 1.0)),
                    success=bool(row.get('success', 1))
                )
                frames.append(frame)
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping frame due to error: {e}")
                continue

        return frames

    def _analyze_window(self, frames: List[GazeFrame]) -> BehaviorWindow:
        """
        Analyze a window of frames to classify behavior.

        Args:
            frames: List of GazeFrame objects for this window

        Returns:
            BehaviorWindow with classification
        """
        if not frames:
            return None

        # Extract metrics
        horizontal_gazes = [f.gaze_angle_x for f in frames if f.success]
        vertical_gazes = [f.gaze_angle_y for f in frames if f.success]
        head_yaws = [f.pose_Ry for f in frames if f.success]

        if not horizontal_gazes:
            return None

        # Calculate statistics
        avg_horizontal = np.mean(horizontal_gazes)
        horizontal_std = np.std(horizontal_gazes)
        avg_vertical = np.mean(vertical_gazes)
        vertical_std = np.std(vertical_gazes)
        avg_head_yaw = np.mean(head_yaws)
        head_yaw_std = np.std(head_yaws)

        # Gaze variability (combined)
        gaze_variability = np.sqrt(horizontal_std**2 + vertical_std**2)

        # Thinking AU score (concentration indicators)
        thinking_aus = [
            np.mean([f.AU04_r for f in frames if f.success]),  # Brow lowerer
            np.mean([f.AU07_r for f in frames if f.success]),  # Lid tightener
        ]
        thinking_au_score = np.mean(thinking_aus)

        # Determine behavior
        behavior, confidence, details = self._classify_behavior(
            avg_horizontal=avg_horizontal,
            horizontal_std=horizontal_std,
            avg_vertical=avg_vertical,
            vertical_std=vertical_std,
            avg_head_yaw=avg_head_yaw,
            head_yaw_std=head_yaw_std,
            gaze_variability=gaze_variability,
            thinking_au_score=thinking_au_score
        )

        return BehaviorWindow(
            start_time=frames[0].timestamp,
            end_time=frames[-1].timestamp,
            behavior=behavior,
            confidence=confidence,
            avg_horizontal_gaze=avg_horizontal,
            gaze_variability=gaze_variability,
            avg_head_yaw=avg_head_yaw,
            head_stability=1.0 - min(head_yaw_std * 5, 1.0),
            vertical_gaze_tendency=avg_vertical,
            thinking_au_score=thinking_au_score,
            details=details
        )

    def _classify_behavior(
        self,
        avg_horizontal: float,
        horizontal_std: float,
        avg_vertical: float,
        vertical_std: float,
        avg_head_yaw: float,
        head_yaw_std: float,
        gaze_variability: float,
        thinking_au_score: float
    ) -> Tuple[GazeBehavior, float, Dict]:
        """
        Classify behavior based on extracted metrics.

        Returns:
            Tuple of (behavior, confidence, details_dict)
        """
        details = {
            'horizontal_deviation': abs(avg_horizontal),
            'vertical_tendency': avg_vertical,
            'gaze_stability': 1.0 - min(gaze_variability * 5, 1.0),
            'head_alignment': abs(avg_head_yaw),
        }

        # THINKING indicators:
        # - Gaze tends upward (positive vertical angle)
        # - OR gaze wanders (high variability)
        # - Head doesn't fix on specific angle
        thinking_score = 0.0

        if avg_vertical > self.THINKING_VERTICAL_THRESHOLD:
            # Looking up - classic thinking pose
            thinking_score += 0.4
            details['thinking_indicator'] = 'upward_gaze'

        if gaze_variability > self.GAZE_VARIABILITY_THRESHOLD:
            # Wandering gaze - thinking/processing
            thinking_score += 0.3
            details['gaze_wandering'] = True

        if thinking_au_score > 0.3:
            # Concentration facial expressions
            thinking_score += 0.2
            details['concentration_aus'] = True

        if vertical_std > 0.1:
            # Variable vertical gaze - looking around while thinking
            thinking_score += 0.1

        # CHEATING indicators:
        # - Sustained horizontal gaze to one side
        # - Head turned to match gaze (reading position)
        # - Low gaze variability (reading)
        # - Gaze and head aligned (both pointing same direction)
        cheating_score = 0.0

        horizontal_deviation = abs(avg_horizontal)
        head_deviation = abs(avg_head_yaw)

        if horizontal_deviation > self.horizontal_threshold:
            # Looking significantly to the side
            cheating_score += 0.35
            details['side_gaze'] = 'left' if avg_horizontal < 0 else 'right'

        if head_deviation > self.HEAD_YAW_THRESHOLD:
            # Head turned to side
            cheating_score += 0.25
            details['head_turned'] = True

        # Check if gaze and head are aligned (both pointing same direction)
        # This is characteristic of reading from a side screen
        gaze_head_alignment = 1.0 - abs(avg_horizontal - avg_head_yaw) / (horizontal_deviation + 0.01)
        if gaze_head_alignment > 0.7 and horizontal_deviation > 0.15:
            cheating_score += 0.2
            details['gaze_head_aligned'] = True

        if gaze_variability < self.GAZE_VARIABILITY_THRESHOLD * 0.5:
            # Very stable gaze - likely reading
            cheating_score += 0.15
            details['stable_reading_gaze'] = True

        if head_yaw_std < 0.05 and horizontal_deviation > 0.15:
            # Head very stable while looking to side
            cheating_score += 0.1
            details['head_fixed'] = True

        # Determine final classification
        # Thinking takes precedence if both scores are similar (benefit of doubt)
        if thinking_score > 0.5 and thinking_score > cheating_score - 0.1:
            return GazeBehavior.THINKING, thinking_score, details

        if cheating_score > 0.7:
            return GazeBehavior.CHEATING, cheating_score, details

        if cheating_score > 0.4:
            return GazeBehavior.SUSPICIOUS, cheating_score, details

        if thinking_score > 0.3:
            return GazeBehavior.THINKING, thinking_score, details

        return GazeBehavior.NORMAL, 1.0 - max(thinking_score, cheating_score), details

    def analyze_frames(self, frames: List[GazeFrame]) -> CheatingReport:
        """
        Perform full analysis on a list of frames.

        Args:
            frames: List of GazeFrame objects

        Returns:
            CheatingReport with full analysis
        """
        if not frames:
            return CheatingReport(
                total_duration=0,
                total_frames=0,
                normal_time=0,
                thinking_time=0,
                suspicious_time=0,
                cheating_time=0,
                suspicious_events=[],
                cheating_events=[],
                overall_risk_score=0,
                assessment="No data to analyze"
            )

        # Get frame rate from timestamps
        if len(frames) > 1:
            avg_frame_time = (frames[-1].timestamp - frames[0].timestamp) / len(frames)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 30.0
        else:
            fps = 30.0

        window_frames = int(self.ANALYSIS_WINDOW_SEC * fps)
        step_frames = int(window_frames * (1 - self.WINDOW_OVERLAP))

        windows: List[BehaviorWindow] = []

        # Analyze in sliding windows
        for start_idx in range(0, len(frames) - window_frames + 1, step_frames):
            window = frames[start_idx:start_idx + window_frames]
            result = self._analyze_window(window)
            if result:
                windows.append(result)

        # Aggregate results
        total_duration = frames[-1].timestamp - frames[0].timestamp

        time_by_behavior = {
            GazeBehavior.NORMAL: 0.0,
            GazeBehavior.THINKING: 0.0,
            GazeBehavior.SUSPICIOUS: 0.0,
            GazeBehavior.CHEATING: 0.0,
        }

        suspicious_events = []
        cheating_events = []

        for window in windows:
            window_duration = window.end_time - window.start_time
            time_by_behavior[window.behavior] += window_duration

            if window.behavior == GazeBehavior.SUSPICIOUS:
                suspicious_events.append(window)
            elif window.behavior == GazeBehavior.CHEATING:
                cheating_events.append(window)

        # Merge consecutive events
        suspicious_events = self._merge_consecutive_events(suspicious_events)
        cheating_events = self._merge_consecutive_events(cheating_events)

        # Calculate risk score
        cheating_ratio = time_by_behavior[GazeBehavior.CHEATING] / max(total_duration, 1)
        suspicious_ratio = time_by_behavior[GazeBehavior.SUSPICIOUS] / max(total_duration, 1)

        risk_score = min(1.0, cheating_ratio * 2 + suspicious_ratio * 0.5)

        # Generate assessment
        assessment = self._generate_assessment(
            risk_score=risk_score,
            cheating_events=cheating_events,
            suspicious_events=suspicious_events,
            thinking_time=time_by_behavior[GazeBehavior.THINKING],
            total_duration=total_duration
        )

        return CheatingReport(
            total_duration=total_duration,
            total_frames=len(frames),
            normal_time=time_by_behavior[GazeBehavior.NORMAL],
            thinking_time=time_by_behavior[GazeBehavior.THINKING],
            suspicious_time=time_by_behavior[GazeBehavior.SUSPICIOUS],
            cheating_time=time_by_behavior[GazeBehavior.CHEATING],
            suspicious_events=suspicious_events,
            cheating_events=cheating_events,
            overall_risk_score=risk_score,
            assessment=assessment
        )

    def _merge_consecutive_events(
        self,
        events: List[BehaviorWindow],
        max_gap: float = 1.0
    ) -> List[BehaviorWindow]:
        """Merge events that are close together in time."""
        if not events:
            return []

        merged = [events[0]]

        for event in events[1:]:
            last = merged[-1]
            if event.start_time - last.end_time <= max_gap:
                # Merge by extending end time and averaging metrics
                merged[-1] = BehaviorWindow(
                    start_time=last.start_time,
                    end_time=event.end_time,
                    behavior=last.behavior,
                    confidence=(last.confidence + event.confidence) / 2,
                    avg_horizontal_gaze=(last.avg_horizontal_gaze + event.avg_horizontal_gaze) / 2,
                    gaze_variability=(last.gaze_variability + event.gaze_variability) / 2,
                    avg_head_yaw=(last.avg_head_yaw + event.avg_head_yaw) / 2,
                    head_stability=(last.head_stability + event.head_stability) / 2,
                    vertical_gaze_tendency=(last.vertical_gaze_tendency + event.vertical_gaze_tendency) / 2,
                    thinking_au_score=(last.thinking_au_score + event.thinking_au_score) / 2,
                    details={**last.details, **event.details}
                )
            else:
                merged.append(event)

        return merged

    def _generate_assessment(
        self,
        risk_score: float,
        cheating_events: List[BehaviorWindow],
        suspicious_events: List[BehaviorWindow],
        thinking_time: float,
        total_duration: float
    ) -> str:
        """Generate human-readable assessment."""
        lines = []

        if risk_score < 0.1:
            lines.append("LOW RISK: No significant cheating indicators detected.")
            lines.append("The subject maintained appropriate gaze behavior throughout.")

        elif risk_score < 0.3:
            lines.append("MINIMAL RISK: Some suspicious moments detected, likely benign.")
            lines.append(f"Found {len(suspicious_events)} brief instances of side-gazing.")
            lines.append("These patterns are consistent with normal distraction or thinking.")

        elif risk_score < 0.5:
            lines.append("MODERATE RISK: Notable suspicious behavior detected.")
            lines.append(f"Found {len(cheating_events)} potential cheating events "
                        f"and {len(suspicious_events)} suspicious moments.")
            lines.append("Recommend manual review of flagged timestamps.")

        elif risk_score < 0.7:
            lines.append("HIGH RISK: Multiple cheating indicators detected.")
            lines.append(f"Found {len(cheating_events)} likely cheating events.")
            if cheating_events:
                longest = max(cheating_events, key=lambda e: e.end_time - e.start_time)
                duration = longest.end_time - longest.start_time
                lines.append(f"Longest suspicious period: {duration:.1f}s "
                           f"at {longest.start_time:.1f}s-{longest.end_time:.1f}s")
            lines.append("Strong recommendation for manual review.")

        else:
            lines.append("CRITICAL RISK: Strong cheating behavior detected.")
            lines.append(f"Found {len(cheating_events)} cheating events with high confidence.")
            lines.append("Subject spent significant time looking at what appears to be")
            lines.append("an external screen or reference material.")

        # Add thinking context
        if thinking_time > 0 and total_duration > 0:
            thinking_pct = (thinking_time / total_duration) * 100
            if thinking_pct > 20:
                lines.append(f"\nNote: {thinking_pct:.0f}% of time classified as 'thinking' behavior,")
                lines.append("which is normal and expected during interviews/exams.")

        return "\n".join(lines)

    def analyze_video(self, video_path: str) -> CheatingReport:
        """
        Analyze a video file for cheating behavior.

        Args:
            video_path: Path to video file

        Returns:
            CheatingReport with analysis
        """
        print(f"Extracting features from: {video_path}")
        csv_path = self.extract_features_from_video(video_path)

        print(f"Loading OpenFace output: {csv_path}")
        frames = self.load_openface_csv(csv_path)

        print(f"Analyzing {len(frames)} frames...")
        return self.analyze_frames(frames)

    def analyze_csv(self, csv_path: str) -> CheatingReport:
        """
        Analyze pre-extracted OpenFace CSV.

        Args:
            csv_path: Path to OpenFace CSV output

        Returns:
            CheatingReport with analysis
        """
        frames = self.load_openface_csv(csv_path)
        return self.analyze_frames(frames)

    def process_realtime_frame(self, frame_data: dict) -> Optional[GazeBehavior]:
        """
        Process a single frame in real-time mode.

        Args:
            frame_data: Dictionary with OpenFace output for single frame

        Returns:
            Current behavior classification, or None if buffer not full
        """
        try:
            frame = GazeFrame(
                frame_num=frame_data.get('frame', 0),
                timestamp=frame_data.get('timestamp', 0),
                gaze_0_x=frame_data['gaze_0_x'],
                gaze_0_y=frame_data['gaze_0_y'],
                gaze_0_z=frame_data['gaze_0_z'],
                gaze_1_x=frame_data['gaze_1_x'],
                gaze_1_y=frame_data['gaze_1_y'],
                gaze_1_z=frame_data['gaze_1_z'],
                gaze_angle_x=frame_data['gaze_angle_x'],
                gaze_angle_y=frame_data['gaze_angle_y'],
                pose_Rx=frame_data['pose_Rx'],
                pose_Ry=frame_data['pose_Ry'],
                pose_Rz=frame_data['pose_Rz'],
                AU04_r=frame_data.get('AU04_r', 0),
                AU07_r=frame_data.get('AU07_r', 0),
                confidence=frame_data.get('confidence', 1.0),
                success=frame_data.get('success', True)
            )
        except KeyError as e:
            print(f"Missing required field: {e}")
            return None

        self.frame_buffer.append(frame)

        # Need at least 1 second of data
        if len(self.frame_buffer) < 30:
            return self.current_behavior

        # Analyze recent frames
        recent_frames = list(self.frame_buffer)[-60:]  # Last 2 seconds
        window = self._analyze_window(recent_frames)

        if window:
            self.current_behavior = window.behavior

        return self.current_behavior

    def run_webcam_demo(self, camera_id: int = 0, show_visualization: bool = True):
        """
        Run live webcam cheating detection demo.

        Args:
            camera_id: Camera device ID (default 0)
            show_visualization: Whether to show OpenCV visualization
        """
        if not CV2_AVAILABLE:
            raise RuntimeError(
                "OpenCV not installed. Install with: pip install opencv-python"
            )

        processor = WebcamProcessor(self, camera_id, show_visualization)
        processor.run()


class WebcamProcessor:
    """
    Real-time webcam processor for cheating detection.

    Supports two modes:
    1. With OpenFace binary: Full gaze tracking via OpenFace subprocess
    2. Simulation mode: Uses face + eye detection for demo when OpenFace unavailable

    DETECTION LOGIC EXPLAINED:
    ==========================

    The system analyzes gaze direction and head pose to classify behavior:

    1. NORMAL: Looking at screen/camera
       - Gaze angle near center (within ~10 degrees horizontal)
       - Head facing forward
       - This is the baseline state

    2. THINKING: Looking away to recall/process information
       - Gaze tends UPWARD (looking up to recall memories - proven cognitive behavior)
       - OR gaze WANDERS in multiple directions (processing information)
       - High gaze variability over time window
       - May show concentration facial expressions (brow furrow)

    3. SUSPICIOUS: Possible cheating indicator
       - Moderate horizontal gaze to one side (10-20 degrees)
       - Head slightly turned
       - Brief duration (2-4 seconds)

    4. CHEATING: Strong cheating indicator
       - SUSTAINED horizontal gaze to one side (>20 degrees)
       - Head turned AND fixed in reading position
       - LOW gaze variability (stable reading pattern)
       - Gaze and head aligned (both pointing same direction)
       - Duration > 4 seconds

    Key Metrics:
    - gaze_angle_x: Horizontal gaze direction (negative=left, positive=right)
    - gaze_angle_y: Vertical gaze direction (negative=down, positive=up)
    - pose_Ry: Head yaw (rotation left/right)
    - gaze_variability: How much gaze moves over time window
    """

    # Colors for visualization (BGR format)
    COLORS = {
        GazeBehavior.NORMAL: (0, 255, 0),      # Green
        GazeBehavior.THINKING: (255, 191, 0),  # Deep sky blue
        GazeBehavior.SUSPICIOUS: (0, 165, 255),  # Orange
        GazeBehavior.CHEATING: (0, 0, 255),    # Red
    }

    STATUS_TEXT = {
        GazeBehavior.NORMAL: "NORMAL - Looking at screen",
        GazeBehavior.THINKING: "THINKING - Processing/recalling",
        GazeBehavior.SUSPICIOUS: "SUSPICIOUS - Extended side gaze",
        GazeBehavior.CHEATING: "CHEATING - Looking at other screen",
    }

    def __init__(
        self,
        detector: OpenFaceCheatingDetector,
        camera_id: int = 0,
        show_visualization: bool = True,
        glasses_mode: bool = False
    ):
        self.detector = detector
        self.camera_id = camera_id
        self.show_visualization = show_visualization
        self.glasses_mode = glasses_mode

        self.running = False
        self.current_behavior = GazeBehavior.NORMAL
        self.behavior_history = deque(maxlen=150)  # 5 seconds at 30fps
        self.frame_count = 0
        self.start_time = None

        # Statistics tracking
        self.behavior_times = {
            GazeBehavior.NORMAL: 0.0,
            GazeBehavior.THINKING: 0.0,
            GazeBehavior.SUSPICIOUS: 0.0,
            GazeBehavior.CHEATING: 0.0,
        }
        self.last_behavior_time = None

        # OpenFace process
        self.openface_process = None
        self.openface_thread = None
        self.latest_openface_data = None
        self.data_lock = threading.Lock()

        # Face detection fallback (when OpenFace unavailable)
        self.face_cascade = None
        self.eye_cascade = None
        self.use_simulation = False

        # Calibration - store baseline face/eye position
        self.calibration_frames = []
        self.baseline_face_center = None
        self.baseline_eye_positions = None
        self.is_calibrated = False

        # Tracking history for better detection
        self.gaze_history = deque(maxlen=30)  # 1 second of gaze data
        self.head_pose_history = deque(maxlen=30)

        # Debug info for display
        self.debug_info = {
            'gaze_x': 0.0,
            'gaze_y': 0.0,
            'head_yaw': 0.0,
            'variability': 0.0,
            'confidence': 0.0,
            'eyes_detected': False,
        }

    def _find_openface_live(self) -> Optional[str]:
        """Find OpenFace binary for live video processing."""
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()

        possible_paths = [
            # Local project paths (most common for users)
            os.path.join(cwd, "OpenFace", "build", "bin", "FaceLandmarkVid"),
            os.path.join(cwd, "OpenFace", "build", "bin", "FeatureExtraction"),
            os.path.join(cwd, "OpenFace", "FaceLandmarkVid"),
            os.path.join(cwd, "OpenFace", "FeatureExtraction"),
            os.path.join(cwd, "OpenFace", "x64", "Release", "FaceLandmarkVid.exe"),
            os.path.join(cwd, "OpenFace", "x64", "Release", "FeatureExtraction.exe"),
            # Script directory paths
            os.path.join(script_dir, "OpenFace", "build", "bin", "FaceLandmarkVid"),
            os.path.join(script_dir, "OpenFace", "build", "bin", "FeatureExtraction"),
            os.path.join(script_dir, "OpenFace", "FaceLandmarkVid"),
            os.path.join(script_dir, "OpenFace", "FeatureExtraction"),
            # Parent directory
            os.path.join(cwd, "..", "OpenFace", "build", "bin", "FaceLandmarkVid"),
            os.path.join(cwd, "..", "OpenFace", "build", "bin", "FeatureExtraction"),
            # System paths
            "/usr/local/bin/FaceLandmarkVid",
            "/usr/local/bin/FeatureExtraction",
            "/opt/OpenFace/build/bin/FaceLandmarkVid",
            "/opt/OpenFace/build/bin/FeatureExtraction",
            "~/OpenFace/build/bin/FaceLandmarkVid",
            "~/OpenFace/build/bin/FeatureExtraction",
            # Windows paths
            "C:/OpenFace/FaceLandmarkVid.exe",
            "C:/OpenFace/FeatureExtraction.exe",
            "C:/OpenFace/build/bin/Release/FaceLandmarkVid.exe",
            "C:/OpenFace/build/bin/Release/FeatureExtraction.exe",
            # In PATH
            "FaceLandmarkVid",
            "FeatureExtraction",
        ]

        for path in possible_paths:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                print(f"Found OpenFace at: {expanded}")
                return expanded

        # Also search recursively in OpenFace directory if it exists
        openface_dirs = [
            os.path.join(cwd, "OpenFace"),
            os.path.join(script_dir, "OpenFace"),
        ]
        for of_dir in openface_dirs:
            if os.path.isdir(of_dir):
                for binary_name in ["FaceLandmarkVid", "FeatureExtraction",
                                   "FaceLandmarkVid.exe", "FeatureExtraction.exe"]:
                    for root, dirs, files in os.walk(of_dir):
                        if binary_name in files:
                            found_path = os.path.join(root, binary_name)
                            print(f"Found OpenFace at: {found_path}")
                            return found_path

        return None

    def _start_openface(self):
        """Start OpenFace subprocess for real-time processing."""
        openface_path = self._find_openface_live()

        if not openface_path:
            print("OpenFace not found - running in SIMULATION mode")
            print("(Uses face/eye detection as proxy for gaze)")
            print("\nFor full accuracy, install OpenFace from:")
            print("https://github.com/TadasBaltrusaitis/OpenFace")
            print("\n" + "=" * 50)
            print("SIMULATION MODE INSTRUCTIONS:")
            print("- Move your HEAD to simulate gaze direction")
            print("- Look LEFT/RIGHT: Turn head sideways")
            print("- Look UP: Tilt head up (triggers THINKING)")
            print("- Stay centered: Face camera directly (NORMAL)")
            print("=" * 50 + "\n")
            self.use_simulation = True
            self._init_cascades()
            return

        output_dir = tempfile.mkdtemp(prefix="openface_live_")

        # Use device camera
        cmd = [
            openface_path,
            "-device", str(self.camera_id),
            "-out_dir", output_dir,
            "-pose",
            "-gaze",
            "-aus",
        ]

        print(f"Starting OpenFace: {' '.join(cmd)}")

        try:
            self.openface_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Start thread to read OpenFace output
            self.openface_thread = threading.Thread(
                target=self._read_openface_output,
                args=(output_dir,),
                daemon=True
            )
            self.openface_thread.start()

        except Exception as e:
            print(f"Failed to start OpenFace: {e}")
            print("Running in simulation mode")
            self.use_simulation = True
            self._init_cascades()

    def _init_cascades(self):
        """Initialize OpenCV cascades for simulation mode."""
        # Face cascade
        face_cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
        ]
        for path in face_cascade_paths:
            if os.path.exists(path):
                self.face_cascade = cv2.CascadeClassifier(path)
                break

        # Eye cascade for better tracking
        eye_cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_eye.xml',
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_eye.xml',
        ]
        for path in eye_cascade_paths:
            if os.path.exists(path):
                self.eye_cascade = cv2.CascadeClassifier(path)
                break

    def _read_openface_output(self, output_dir: str):
        """Read OpenFace CSV output in real-time."""
        csv_pattern = os.path.join(output_dir, "*.csv")
        csv_file = None

        # Wait for CSV file to be created
        for _ in range(50):  # 5 seconds timeout
            import glob
            files = glob.glob(csv_pattern)
            if files:
                csv_file = files[0]
                break
            time.sleep(0.1)

        if not csv_file:
            print("Warning: OpenFace CSV output not found")
            return

        # Tail the CSV file
        last_pos = 0
        header = None

        while self.running:
            try:
                with open(csv_file, 'r') as f:
                    f.seek(last_pos)
                    lines = f.readlines()
                    last_pos = f.tell()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if header is None:
                        header = [h.strip() for h in line.split(',')]
                        continue

                    values = line.split(',')
                    if len(values) != len(header):
                        continue

                    data = dict(zip(header, values))

                    try:
                        frame_data = {
                            'frame': int(float(data.get('frame', 0))),
                            'timestamp': float(data.get('timestamp', 0)),
                            'gaze_0_x': float(data.get('gaze_0_x', 0)),
                            'gaze_0_y': float(data.get('gaze_0_y', 0)),
                            'gaze_0_z': float(data.get('gaze_0_z', 1)),
                            'gaze_1_x': float(data.get('gaze_1_x', 0)),
                            'gaze_1_y': float(data.get('gaze_1_y', 0)),
                            'gaze_1_z': float(data.get('gaze_1_z', 1)),
                            'gaze_angle_x': float(data.get('gaze_angle_x', 0)),
                            'gaze_angle_y': float(data.get('gaze_angle_y', 0)),
                            'pose_Rx': float(data.get('pose_Rx', 0)),
                            'pose_Ry': float(data.get('pose_Ry', 0)),
                            'pose_Rz': float(data.get('pose_Rz', 0)),
                            'AU04_r': float(data.get('AU04_r', 0)),
                            'AU07_r': float(data.get('AU07_r', 0)),
                            'confidence': float(data.get('confidence', 1)),
                            'success': int(float(data.get('success', 1))),
                        }

                        with self.data_lock:
                            self.latest_openface_data = frame_data

                    except (ValueError, KeyError):
                        continue

                time.sleep(0.01)  # Small delay

            except Exception:
                time.sleep(0.1)

    def _detect_eyes_in_face(self, gray, face_roi, face_x, face_y):
        """Detect eyes within face region and return their positions."""
        if self.eye_cascade is None:
            return None, None

        eyes = self.eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
            maxSize=(80, 80)
        )

        if len(eyes) < 2:
            return None, None

        # Sort by x position to get left and right eye
        eyes = sorted(eyes, key=lambda e: e[0])
        left_eye = eyes[0]
        right_eye = eyes[-1]

        # Calculate eye centers relative to full frame
        left_center = (
            face_x + left_eye[0] + left_eye[2] // 2,
            face_y + left_eye[1] + left_eye[3] // 2
        )
        right_center = (
            face_x + right_eye[0] + right_eye[2] // 2,
            face_y + right_eye[1] + right_eye[3] // 2
        )

        return left_center, right_center

    def _calibrate(self, frame, face, gray):
        """Calibrate baseline position from first few frames."""
        x, y, w, h = face
        face_center = (x + w // 2, y + h // 2)

        # Get eye positions
        face_roi = gray[y:y+h, x:x+w]
        left_eye, right_eye = self._detect_eyes_in_face(gray, face_roi, x, y)

        self.calibration_frames.append({
            'face_center': face_center,
            'left_eye': left_eye,
            'right_eye': right_eye
        })

        # Need 15 frames to calibrate
        if len(self.calibration_frames) >= 15:
            # Average the positions
            valid_frames = [f for f in self.calibration_frames
                          if f['left_eye'] is not None]

            if valid_frames:
                self.baseline_face_center = (
                    np.mean([f['face_center'][0] for f in self.calibration_frames]),
                    np.mean([f['face_center'][1] for f in self.calibration_frames])
                )
                self.baseline_eye_positions = {
                    'left': (
                        np.mean([f['left_eye'][0] for f in valid_frames]),
                        np.mean([f['left_eye'][1] for f in valid_frames])
                    ),
                    'right': (
                        np.mean([f['right_eye'][0] for f in valid_frames]),
                        np.mean([f['right_eye'][1] for f in valid_frames])
                    )
                }
            else:
                self.baseline_face_center = (
                    np.mean([f['face_center'][0] for f in self.calibration_frames]),
                    np.mean([f['face_center'][1] for f in self.calibration_frames])
                )

            self.is_calibrated = True
            print("Calibration complete! Baseline position recorded.")

    def _simulate_gaze_from_face(self, frame, faces, gray) -> Optional[dict]:
        """
        Simulate gaze data from face/eye detection for demo mode.

        This improved version:
        1. Detects eyes within face for better accuracy
        2. Uses calibrated baseline for relative movement
        3. Tracks both face position AND eye positions
        4. More sensitive thresholds tuned for head movement
        """
        if len(faces) == 0:
            self.debug_info['eyes_detected'] = False
            return None

        h, w = frame.shape[:2]

        # Use largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = face
        face_center_x = fx + fw / 2
        face_center_y = fy + fh / 2

        # Get eye positions
        face_roi = gray[fy:fy+fh, fx:fx+fw]
        left_eye, right_eye = self._detect_eyes_in_face(gray, face_roi, fx, fy)
        self.debug_info['eyes_detected'] = left_eye is not None

        # Calibrate if needed
        if not self.is_calibrated:
            self._calibrate(frame, face, gray)
            # Return neutral during calibration
            return {
                'frame': self.frame_count,
                'timestamp': time.time() - self.start_time if self.start_time else 0,
                'gaze_0_x': 0, 'gaze_0_y': 0, 'gaze_0_z': 1,
                'gaze_1_x': 0, 'gaze_1_y': 0, 'gaze_1_z': 1,
                'gaze_angle_x': 0, 'gaze_angle_y': 0,
                'pose_Rx': 0, 'pose_Ry': 0, 'pose_Rz': 0,
                'AU04_r': 0, 'AU07_r': 0,
                'confidence': 0.5, 'success': 1,
            }

        # Calculate offset from calibrated baseline
        # Normalized to frame dimensions for consistent sensitivity
        if self.baseline_face_center:
            # Head position offset (primary signal in simulation mode)
            h_offset = (face_center_x - self.baseline_face_center[0]) / (w * 0.15)
            v_offset = (self.baseline_face_center[1] - face_center_y) / (h * 0.15)
        else:
            # Fallback to frame center
            h_offset = (face_center_x - w/2) / (w * 0.2)
            v_offset = (h/2 - face_center_y) / (h * 0.2)

        # Clamp to reasonable range
        h_offset = np.clip(h_offset, -1.5, 1.5)
        v_offset = np.clip(v_offset, -1.0, 1.0)

        # Eye-based refinement if eyes detected
        eye_gaze_x = 0
        eye_gaze_y = 0
        if left_eye and right_eye and self.baseline_eye_positions:
            # Calculate eye movement from baseline
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            eye_center_y = (left_eye[1] + right_eye[1]) / 2

            baseline_eye_center_x = (
                self.baseline_eye_positions['left'][0] +
                self.baseline_eye_positions['right'][0]
            ) / 2
            baseline_eye_center_y = (
                self.baseline_eye_positions['left'][1] +
                self.baseline_eye_positions['right'][1]
            ) / 2

            eye_gaze_x = (eye_center_x - baseline_eye_center_x) / (fw * 0.3)
            eye_gaze_y = (baseline_eye_center_y - eye_center_y) / (fh * 0.3)

        # Combine head pose and eye position
        # In simulation mode, head pose is primary, eyes are secondary
        combined_gaze_x = h_offset * 0.7 + eye_gaze_x * 0.3
        combined_gaze_y = v_offset * 0.7 + eye_gaze_y * 0.3

        # Convert to approximate radians (scaled for sensitivity)
        # These values are tuned to trigger the detection thresholds
        gaze_x = combined_gaze_x * 0.35  # ~20 degrees at full offset
        gaze_y = combined_gaze_y * 0.25  # ~14 degrees at full offset

        # Head pose estimation from face position
        head_yaw = h_offset * 0.3   # Head turn angle
        head_pitch = -v_offset * 0.2  # Head tilt angle

        # Add small noise for realism
        gaze_x += np.random.normal(0, 0.01)
        gaze_y += np.random.normal(0, 0.01)

        # Update debug info
        self.debug_info['gaze_x'] = gaze_x
        self.debug_info['gaze_y'] = gaze_y
        self.debug_info['head_yaw'] = head_yaw
        self.debug_info['confidence'] = 0.8 if left_eye else 0.5

        return {
            'frame': self.frame_count,
            'timestamp': time.time() - self.start_time if self.start_time else 0,
            'gaze_0_x': gaze_x,
            'gaze_0_y': gaze_y,
            'gaze_0_z': 0.95,
            'gaze_1_x': gaze_x,
            'gaze_1_y': gaze_y,
            'gaze_1_z': 0.95,
            'gaze_angle_x': gaze_x,
            'gaze_angle_y': gaze_y,
            'pose_Rx': head_pitch,
            'pose_Ry': head_yaw,
            'pose_Rz': 0,
            'AU04_r': 0,
            'AU07_r': 0,
            'confidence': 0.8 if left_eye else 0.5,
            'success': 1,
        }

    def _classify_realtime(self, frame_data: dict) -> GazeBehavior:
        """
        Classify behavior in real-time with adjusted thresholds for responsiveness.

        This is a simplified version optimized for real-time feedback.
        """
        gaze_x = frame_data['gaze_angle_x']
        gaze_y = frame_data['gaze_angle_y']
        head_yaw = frame_data['pose_Ry']

        # Store in history for variability calculation
        self.gaze_history.append((gaze_x, gaze_y))
        self.head_pose_history.append(head_yaw)

        # Calculate gaze variability over recent history
        variability = 0.0
        if len(self.gaze_history) >= 10:
            recent_x = [g[0] for g in list(self.gaze_history)[-15:]]
            recent_y = [g[1] for g in list(self.gaze_history)[-15:]]
            variability = np.sqrt(np.std(recent_x)**2 + np.std(recent_y)**2)

        self.debug_info['variability'] = variability

        # Thresholds (adjusted for better sensitivity)
        # These are in radians: 0.15 rad ≈ 8.6°, 0.25 rad ≈ 14.3°
        SIDE_GAZE_THRESHOLD = 0.12      # Looking to side
        STRONG_SIDE_THRESHOLD = 0.20    # Strong side gaze
        UP_GAZE_THRESHOLD = 0.10        # Looking up (thinking)
        HEAD_TURN_THRESHOLD = 0.10      # Head turned
        HIGH_VARIABILITY = 0.06         # Wandering gaze

        abs_gaze_x = abs(gaze_x)
        abs_head_yaw = abs(head_yaw)

        # THINKING detection:
        # - Looking UP (positive gaze_y) - classic recall behavior
        # - OR high gaze variability (eyes wandering while processing)
        if gaze_y > UP_GAZE_THRESHOLD:
            return GazeBehavior.THINKING

        if variability > HIGH_VARIABILITY and abs_gaze_x < SIDE_GAZE_THRESHOLD:
            return GazeBehavior.THINKING

        # CHEATING detection:
        # - Strong sustained side gaze
        # - Head turned in same direction as gaze
        # - Low variability (stable reading position)
        gaze_head_aligned = (gaze_x * head_yaw > 0)  # Same direction

        if abs_gaze_x > STRONG_SIDE_THRESHOLD:
            if gaze_head_aligned and abs_head_yaw > HEAD_TURN_THRESHOLD:
                if variability < HIGH_VARIABILITY:
                    return GazeBehavior.CHEATING
            return GazeBehavior.SUSPICIOUS

        # SUSPICIOUS detection:
        # - Moderate side gaze
        if abs_gaze_x > SIDE_GAZE_THRESHOLD:
            if abs_head_yaw > HEAD_TURN_THRESHOLD * 0.5:
                return GazeBehavior.SUSPICIOUS

        # Default: NORMAL
        return GazeBehavior.NORMAL

    def _process_frame(self, frame) -> GazeBehavior:
        """Process a single frame and return behavior classification."""
        frame_data = None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.use_simulation and self.face_cascade is not None:
            # Simulation mode using face/eye detection
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )
            frame_data = self._simulate_gaze_from_face(frame, faces, gray)

            if frame_data and self.is_calibrated:
                # Use our own real-time classifier for simulation mode
                self.current_behavior = self._classify_realtime(frame_data)
        else:
            # Real OpenFace mode
            with self.data_lock:
                if self.latest_openface_data:
                    frame_data = self.latest_openface_data.copy()

            if frame_data:
                # Use the detector's processing for OpenFace data
                behavior = self.detector.process_realtime_frame(frame_data)
                if behavior:
                    self.current_behavior = behavior

                # Update debug info
                self.debug_info['gaze_x'] = frame_data.get('gaze_angle_x', 0)
                self.debug_info['gaze_y'] = frame_data.get('gaze_angle_y', 0)
                self.debug_info['head_yaw'] = frame_data.get('pose_Ry', 0)
                self.debug_info['confidence'] = frame_data.get('confidence', 0)

        return self.current_behavior

    def _draw_overlay(self, frame):
        """Draw status overlay on frame."""
        h, w = frame.shape[:2]
        behavior = self.current_behavior
        color = self.COLORS[behavior]
        status_text = self.STATUS_TEXT[behavior]

        # Draw colored border based on status
        border_thickness = 8
        cv2.rectangle(frame, (0, 0), (w, h), color, border_thickness)

        # Draw status bar at top
        bar_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw status indicator circle
        indicator_radius = 20
        cv2.circle(frame, (40, bar_height // 2), indicator_radius, color, -1)
        cv2.circle(frame, (40, bar_height // 2), indicator_radius, (255, 255, 255), 2)

        # Draw status text
        cv2.putText(
            frame, status_text,
            (75, bar_height // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        # Draw GAZE DEBUG panel on the LEFT
        debug_panel_width = 200
        debug_panel_x = 10
        debug_panel_y = bar_height + 10

        # Semi-transparent debug panel
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (debug_panel_x, debug_panel_y),
            (debug_panel_x + debug_panel_width, debug_panel_y + 160),
            (30, 30, 30), -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Debug panel title
        cv2.putText(
            frame, "Gaze Debug",
            (debug_panel_x + 10, debug_panel_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )

        # Gaze values with visual indicators
        gaze_x = self.debug_info.get('gaze_x', 0)
        gaze_y = self.debug_info.get('gaze_y', 0)
        head_yaw = self.debug_info.get('head_yaw', 0)
        variability = self.debug_info.get('variability', 0)
        confidence = self.debug_info.get('confidence', 0)
        eyes_detected = self.debug_info.get('eyes_detected', False)

        # Convert to degrees for display
        gaze_x_deg = np.degrees(gaze_x)
        gaze_y_deg = np.degrees(gaze_y)
        head_yaw_deg = np.degrees(head_yaw)

        debug_lines = [
            (f"Gaze X: {gaze_x_deg:+.1f} deg", self._get_gaze_color(abs(gaze_x), 0.12, 0.20)),
            (f"Gaze Y: {gaze_y_deg:+.1f} deg", self._get_gaze_color(gaze_y, 0.10, 0.20) if gaze_y > 0 else (200, 200, 200)),
            (f"Head Yaw: {head_yaw_deg:+.1f} deg", self._get_gaze_color(abs(head_yaw), 0.10, 0.15)),
            (f"Variability: {variability:.3f}", (0, 255, 255) if variability > 0.06 else (200, 200, 200)),
            (f"Confidence: {confidence:.1%}", (0, 255, 0) if confidence > 0.6 else (0, 165, 255)),
            (f"Eyes: {'YES' if eyes_detected else 'NO'}", (0, 255, 0) if eyes_detected else (100, 100, 100)),
        ]

        for i, (text, text_color) in enumerate(debug_lines):
            y_pos = debug_panel_y + 45 + i * 18
            cv2.putText(
                frame, text,
                (debug_panel_x + 10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1
            )

        # Draw gaze direction indicator (visual crosshair)
        indicator_center_x = debug_panel_x + debug_panel_width // 2
        indicator_center_y = debug_panel_y + 160 + 50
        indicator_radius = 40

        # Draw indicator background circle
        cv2.circle(frame, (indicator_center_x, indicator_center_y), indicator_radius, (60, 60, 60), -1)
        cv2.circle(frame, (indicator_center_x, indicator_center_y), indicator_radius, (100, 100, 100), 2)

        # Draw crosshair
        cv2.line(frame, (indicator_center_x - indicator_radius, indicator_center_y),
                (indicator_center_x + indicator_radius, indicator_center_y), (80, 80, 80), 1)
        cv2.line(frame, (indicator_center_x, indicator_center_y - indicator_radius),
                (indicator_center_x, indicator_center_y + indicator_radius), (80, 80, 80), 1)

        # Draw gaze point (clamped to circle)
        gaze_point_x = int(indicator_center_x + np.clip(gaze_x * 150, -indicator_radius + 5, indicator_radius - 5))
        gaze_point_y = int(indicator_center_y - np.clip(gaze_y * 150, -indicator_radius + 5, indicator_radius - 5))
        cv2.circle(frame, (gaze_point_x, gaze_point_y), 8, color, -1)
        cv2.circle(frame, (gaze_point_x, gaze_point_y), 8, (255, 255, 255), 2)

        # Label for indicator
        cv2.putText(
            frame, "Gaze Direction",
            (indicator_center_x - 45, indicator_center_y + indicator_radius + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1
        )

        # Calibration status
        if not self.is_calibrated:
            cal_text = f"CALIBRATING... {len(self.calibration_frames)}/15"
            cv2.putText(frame, cal_text, (w // 2 - 100, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Look at the camera", (w // 2 - 80, h // 2 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw statistics panel on the right
        panel_width = 250
        panel_x = w - panel_width - 10
        panel_y = bar_height + 10

        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (w - 10, panel_y + 180),
            (30, 30, 30), -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Panel title
        cv2.putText(
            frame, "Session Stats",
            (panel_x + 10, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        # Calculate elapsed time
        elapsed = time.time() - self.start_time if self.start_time else 0

        # Stats text
        stats = [
            f"Time: {elapsed:.0f}s",
            f"Frames: {self.frame_count}",
            "",
            f"Normal: {self.behavior_times[GazeBehavior.NORMAL]:.1f}s",
            f"Thinking: {self.behavior_times[GazeBehavior.THINKING]:.1f}s",
            f"Suspicious: {self.behavior_times[GazeBehavior.SUSPICIOUS]:.1f}s",
            f"Cheating: {self.behavior_times[GazeBehavior.CHEATING]:.1f}s",
        ]

        for i, text in enumerate(stats):
            y_pos = panel_y + 50 + i * 18
            text_color = (200, 200, 200)

            # Color-code the behavior lines
            if "Normal:" in text:
                text_color = self.COLORS[GazeBehavior.NORMAL]
            elif "Thinking:" in text:
                text_color = self.COLORS[GazeBehavior.THINKING]
            elif "Suspicious:" in text:
                text_color = self.COLORS[GazeBehavior.SUSPICIOUS]
            elif "Cheating:" in text:
                text_color = self.COLORS[GazeBehavior.CHEATING]

            cv2.putText(
                frame, text,
                (panel_x + 15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1
            )

        # Draw behavior timeline at bottom
        timeline_height = 30
        timeline_y = h - timeline_height - 10

        # Timeline background
        cv2.rectangle(
            frame,
            (10, timeline_y),
            (w - 10, timeline_y + timeline_height),
            (40, 40, 40), -1
        )

        # Draw recent behavior history as colored segments
        if self.behavior_history:
            segment_width = (w - 20) / len(self.behavior_history)
            for i, b in enumerate(self.behavior_history):
                x1 = int(10 + i * segment_width)
                x2 = int(10 + (i + 1) * segment_width)
                cv2.rectangle(
                    frame,
                    (x1, timeline_y + 2),
                    (x2, timeline_y + timeline_height - 2),
                    self.COLORS[b], -1
                )

        # Instructions
        cv2.putText(
            frame, "Press 'q' to quit | 'r' to reset stats | 'c' to recalibrate",
            (10, h - timeline_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1
        )

        # Mode indicator
        mode_text = "SIMULATION MODE" if self.use_simulation else "OPENFACE MODE"
        mode_color = (0, 165, 255) if self.use_simulation else (0, 255, 0)
        cv2.putText(
            frame, mode_text,
            (w - 180, bar_height // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1
        )

        return frame

    def _get_gaze_color(self, value, threshold1, threshold2):
        """Return color based on gaze value thresholds."""
        if abs(value) > threshold2:
            return (0, 0, 255)  # Red - high
        elif abs(value) > threshold1:
            return (0, 165, 255)  # Orange - medium
        else:
            return (0, 255, 0)  # Green - normal

    def _update_stats(self):
        """Update behavior time statistics."""
        current_time = time.time()

        if self.last_behavior_time is not None:
            elapsed = current_time - self.last_behavior_time
            self.behavior_times[self.current_behavior] += elapsed

        self.last_behavior_time = current_time
        self.behavior_history.append(self.current_behavior)

    def run(self):
        """Run the webcam processing loop."""
        print(f"\nStarting webcam cheating detection (camera {self.camera_id})...")
        print("=" * 50)

        # Open camera
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        self.start_time = time.time()
        self.last_behavior_time = self.start_time

        # Start OpenFace (or simulation mode)
        self._start_openface()

        print("\nWebcam opened successfully!")
        print("Press 'q' to quit, 'r' to reset statistics")
        print("=" * 50 + "\n")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                self.frame_count += 1

                # Process frame
                self._process_frame(frame)
                self._update_stats()

                # Draw visualization
                if self.show_visualization:
                    frame = self._draw_overlay(frame)
                    cv2.imshow('Cheating Detection - Live', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r'):
                    # Reset statistics
                    self.behavior_times = {b: 0.0 for b in GazeBehavior}
                    self.behavior_history.clear()
                    self.start_time = time.time()
                    self.last_behavior_time = self.start_time
                    print("Statistics reset")
                elif key == ord('c'):
                    # Recalibrate
                    self.is_calibrated = False
                    self.calibration_frames = []
                    self.baseline_face_center = None
                    self.baseline_eye_positions = None
                    self.gaze_history.clear()
                    print("Recalibrating... Look at the camera")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.running = False

            # Cleanup
            if self.openface_process:
                self.openface_process.terminate()
                self.openface_process.wait()

            cap.release()
            cv2.destroyAllWindows()

            # Print final statistics
            self._print_final_stats()

    def _print_final_stats(self):
        """Print final session statistics."""
        total_time = sum(self.behavior_times.values())

        print("\n" + "=" * 50)
        print("SESSION COMPLETE")
        print("=" * 50)
        print(f"\nTotal Duration: {total_time:.1f} seconds")
        print(f"Total Frames: {self.frame_count}")
        print("\nBehavior Breakdown:")

        for behavior, time_spent in self.behavior_times.items():
            pct = (time_spent / total_time * 100) if total_time > 0 else 0
            print(f"  {behavior.value:12s}: {time_spent:6.1f}s ({pct:5.1f}%)")

        # Calculate risk score
        cheating_pct = self.behavior_times[GazeBehavior.CHEATING] / max(total_time, 1)
        suspicious_pct = self.behavior_times[GazeBehavior.SUSPICIOUS] / max(total_time, 1)
        risk_score = min(1.0, cheating_pct * 2 + suspicious_pct * 0.5)

        print(f"\nOverall Risk Score: {risk_score:.2f} / 1.00")

        if risk_score < 0.2:
            print("Assessment: LOW RISK - Normal behavior observed")
        elif risk_score < 0.4:
            print("Assessment: MINIMAL RISK - Some suspicious moments")
        elif risk_score < 0.6:
            print("Assessment: MODERATE RISK - Review recommended")
        else:
            print("Assessment: HIGH RISK - Significant cheating indicators")

        print("=" * 50)


def format_report(report: CheatingReport) -> str:
    """Format a CheatingReport as a readable string."""
    lines = [
        "=" * 70,
        "CHEATING DETECTION ANALYSIS REPORT",
        "=" * 70,
        "",
        f"Total Duration: {report.total_duration:.1f} seconds",
        f"Total Frames: {report.total_frames}",
        "",
        "TIME BREAKDOWN:",
        f"  Normal:     {report.normal_time:6.1f}s ({report.normal_time/max(report.total_duration,1)*100:5.1f}%)",
        f"  Thinking:   {report.thinking_time:6.1f}s ({report.thinking_time/max(report.total_duration,1)*100:5.1f}%)",
        f"  Suspicious: {report.suspicious_time:6.1f}s ({report.suspicious_time/max(report.total_duration,1)*100:5.1f}%)",
        f"  Cheating:   {report.cheating_time:6.1f}s ({report.cheating_time/max(report.total_duration,1)*100:5.1f}%)",
        "",
        f"OVERALL RISK SCORE: {report.overall_risk_score:.2f} / 1.00",
        "",
    ]

    if report.cheating_events:
        lines.append("CHEATING EVENTS DETECTED:")
        for i, event in enumerate(report.cheating_events, 1):
            duration = event.end_time - event.start_time
            lines.append(f"  {i}. {event.start_time:.1f}s - {event.end_time:.1f}s "
                        f"(duration: {duration:.1f}s, confidence: {event.confidence:.0%})")
            if event.details.get('side_gaze'):
                lines.append(f"      Looking {event.details['side_gaze']}, "
                           f"horizontal angle: {np.degrees(event.avg_horizontal_gaze):.1f}°")
        lines.append("")

    if report.suspicious_events:
        lines.append("SUSPICIOUS EVENTS:")
        for i, event in enumerate(report.suspicious_events, 1):
            duration = event.end_time - event.start_time
            lines.append(f"  {i}. {event.start_time:.1f}s - {event.end_time:.1f}s "
                        f"(duration: {duration:.1f}s)")
        lines.append("")

    lines.append("ASSESSMENT:")
    for line in report.assessment.split('\n'):
        lines.append(f"  {line}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Command-line interface for cheating detection."""
    parser = argparse.ArgumentParser(
        description="OpenFace-based cheating detection for interviews/exams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Live webcam analysis:
    python openface_cheating_detector.py --webcam

  Webcam with specific camera:
    python openface_cheating_detector.py --webcam --camera-id 1

  Analyze a video file:
    python openface_cheating_detector.py --video interview.mp4

  Analyze pre-extracted OpenFace CSV:
    python openface_cheating_detector.py --csv openface_output.csv

  Adjust sensitivity:
    python openface_cheating_detector.py --video interview.mp4 --threshold 0.3

  Output as JSON:
    python openface_cheating_detector.py --csv output.csv --json
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--webcam', action='store_true',
                            help='Run live webcam cheating detection')
    input_group.add_argument('--video', type=str, help='Path to video file')
    input_group.add_argument('--csv', type=str, help='Path to OpenFace CSV output')
    input_group.add_argument('--demo', action='store_true',
                            help='Run demo with synthetic data')

    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID for webcam mode (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.25,
                       help='Horizontal gaze threshold in radians (default: 0.25)')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Sustained gaze duration for suspicion (default: 2.0s)')
    parser.add_argument('--openface-path', type=str,
                       help='Path to OpenFace FeatureExtraction binary')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    parser.add_argument('--output', type=str,
                       help='Save report to file')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable visualization window (webcam mode)')

    args = parser.parse_args()

    detector = OpenFaceCheatingDetector(
        horizontal_threshold=args.threshold,
        sustained_duration=args.duration,
        openface_path=args.openface_path
    )

    # Webcam mode - live analysis
    if args.webcam:
        if not CV2_AVAILABLE:
            print("Error: OpenCV not installed. Install with: pip install opencv-python")
            sys.exit(1)

        detector.run_webcam_demo(
            camera_id=args.camera_id,
            show_visualization=not args.no_display
        )
        return  # Webcam mode handles its own output

    if args.demo:
        # Generate synthetic demo data
        print("Running demo with synthetic data...")
        frames = generate_demo_data()
        report = detector.analyze_frames(frames)

    elif args.video:
        report = detector.analyze_video(args.video)

    else:  # args.csv
        report = detector.analyze_csv(args.csv)

    # Output results
    if args.json:
        result = {
            'total_duration': report.total_duration,
            'total_frames': report.total_frames,
            'time_breakdown': {
                'normal': report.normal_time,
                'thinking': report.thinking_time,
                'suspicious': report.suspicious_time,
                'cheating': report.cheating_time
            },
            'risk_score': report.overall_risk_score,
            'cheating_events': [
                {
                    'start': e.start_time,
                    'end': e.end_time,
                    'confidence': e.confidence,
                    'details': e.details
                }
                for e in report.cheating_events
            ],
            'suspicious_events': [
                {
                    'start': e.start_time,
                    'end': e.end_time,
                    'confidence': e.confidence
                }
                for e in report.suspicious_events
            ],
            'assessment': report.assessment
        }
        output = json.dumps(result, indent=2)
    else:
        output = format_report(report)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Report saved to: {args.output}")
    else:
        print(output)


def generate_demo_data() -> List[GazeFrame]:
    """Generate synthetic demo data showing different behaviors."""
    frames = []
    fps = 30

    # Scenario 1: Normal behavior (0-10s)
    for i in range(fps * 10):
        frames.append(GazeFrame(
            frame_num=len(frames),
            timestamp=len(frames) / fps,
            gaze_0_x=np.random.normal(0, 0.02),
            gaze_0_y=np.random.normal(0, 0.02),
            gaze_0_z=0.99,
            gaze_1_x=np.random.normal(0, 0.02),
            gaze_1_y=np.random.normal(0, 0.02),
            gaze_1_z=0.99,
            gaze_angle_x=np.random.normal(0, 0.05),
            gaze_angle_y=np.random.normal(0, 0.05),
            pose_Rx=np.random.normal(0, 0.02),
            pose_Ry=np.random.normal(0, 0.02),
            pose_Rz=np.random.normal(0, 0.01),
        ))

    # Scenario 2: Thinking (10-20s) - looking up, variable gaze
    for i in range(fps * 10):
        t = i / fps
        frames.append(GazeFrame(
            frame_num=len(frames),
            timestamp=len(frames) / fps,
            gaze_0_x=np.random.normal(0, 0.1) + 0.05 * np.sin(t),
            gaze_0_y=np.random.normal(0.2, 0.08),  # Looking up
            gaze_0_z=0.97,
            gaze_1_x=np.random.normal(0, 0.1) + 0.05 * np.sin(t),
            gaze_1_y=np.random.normal(0.2, 0.08),
            gaze_1_z=0.97,
            gaze_angle_x=np.random.normal(0, 0.1) + 0.05 * np.sin(t),
            gaze_angle_y=np.random.normal(0.2, 0.08),  # Upward gaze
            pose_Rx=np.random.normal(-0.1, 0.03),  # Slight head tilt up
            pose_Ry=np.random.normal(0, 0.05),
            pose_Rz=np.random.normal(0, 0.02),
            AU04_r=0.5,  # Brow lowerer - concentration
        ))

    # Scenario 3: Cheating (20-35s) - sustained side gaze
    for i in range(fps * 15):
        frames.append(GazeFrame(
            frame_num=len(frames),
            timestamp=len(frames) / fps,
            gaze_0_x=-0.35 + np.random.normal(0, 0.02),  # Looking left
            gaze_0_y=np.random.normal(0, 0.02),
            gaze_0_z=0.93,
            gaze_1_x=-0.35 + np.random.normal(0, 0.02),
            gaze_1_y=np.random.normal(0, 0.02),
            gaze_1_z=0.93,
            gaze_angle_x=-0.35 + np.random.normal(0, 0.02),  # Strong left gaze
            gaze_angle_y=np.random.normal(0, 0.02),  # Low vertical variance (reading)
            pose_Rx=np.random.normal(0, 0.01),
            pose_Ry=-0.25 + np.random.normal(0, 0.02),  # Head turned left
            pose_Rz=np.random.normal(0, 0.01),
        ))

    # Scenario 4: Back to normal with brief suspicious moment (35-45s)
    for i in range(fps * 10):
        t = i / fps
        is_suspicious = 3 < t < 5  # Brief suspicious moment

        if is_suspicious:
            h_gaze = 0.28 + np.random.normal(0, 0.03)
            h_yaw = 0.15 + np.random.normal(0, 0.02)
        else:
            h_gaze = np.random.normal(0, 0.05)
            h_yaw = np.random.normal(0, 0.02)

        frames.append(GazeFrame(
            frame_num=len(frames),
            timestamp=len(frames) / fps,
            gaze_0_x=h_gaze,
            gaze_0_y=np.random.normal(0, 0.03),
            gaze_0_z=0.98,
            gaze_1_x=h_gaze,
            gaze_1_y=np.random.normal(0, 0.03),
            gaze_1_z=0.98,
            gaze_angle_x=h_gaze,
            gaze_angle_y=np.random.normal(0, 0.03),
            pose_Rx=np.random.normal(0, 0.02),
            pose_Ry=h_yaw,
            pose_Rz=np.random.normal(0, 0.01),
        ))

    return frames


if __name__ == "__main__":
    main()
