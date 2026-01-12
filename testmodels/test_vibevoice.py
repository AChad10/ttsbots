"""
VibeVoice Realtime TTS Test Script
===================================
Tests Microsoft's VibeVoice Realtime 0.5B model for text-to-speech synthesis.

Installation:
    git clone https://github.com/vibevoice-community/VibeVoice.git
    cd VibeVoice
    pip install -e .
    huggingface-cli download microsoft/VibeVoice-Realtime-0.5B --local-dir ./checkpoints/0.5B

Tuning Parameters:
    - cfg_scale: 1.0-2.0 (classifier-free guidance scale, higher = more faithful to text)
    - ddpm_steps: 3-10 (diffusion steps, higher = better quality but slower)

Special Features:
    - Streaming-optimized (~300ms latency)
    - Single speaker: Carter (male voice)
    - Based on Qwen2.5-0.5B
    - Fast inference with DDPM-based diffusion

Known Issues:
    - Single speaker only (Carter) - no multi-speaker support
    - Requires voice preset file (en-Carter.pt)
    - Developers explicitly warn against commercial use
    - Flash Attention 2 recommended for CUDA devices
"""

import os
import sys
import time
import copy
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

# Test phrases for hiring agent
TEST_PHRASES = [
    "Hello, thank you for applying to this position. Can you tell me about your experience?",
    "That's great! Your background in software development sounds very relevant.",
    "Let me ask you a few technical questions to better understand your skills.",
    "Could you walk me through a challenging project you've worked on recently?",
]

# Output directory
OUTPUT_DIR = "outputs/vibevoice"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample rate for VibeVoice Realtime
SAMPLE_RATE = 24000


def get_device():
    """
    Get the best available device with cross-platform support.

    Priority: CUDA > DirectML (AMD GPU) > MPS (Apple Silicon) > CPU

    Platform Support:
        - NVIDIA GPU: CUDA (recommended)
        - AMD GPU: DirectML (Windows only, install torch-directml)
        - Apple Silicon: MPS (native support)
        - CPU: Fallback for all platforms
    """
    # NVIDIA CUDA
    if torch.cuda.is_available():
        return "cuda"

    # AMD GPU via DirectML (Windows ThinkPad)
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print(f"DirectML device detected: {dml_device}")
        return dml_device
    except ImportError:
        pass
    except Exception as e:
        print(f"DirectML initialization failed: {e}")
        pass

    # Apple Silicon MPS
    if torch.backends.mps.is_available():
        return "mps"

    # CPU fallback
    return "cpu"


def check_vibevoice_installed():
    """Check if VibeVoice streaming is installed."""
    try:
        from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
        return True
    except ImportError:
        return False


def _find_voice_preset_path(speaker_name: str = "Carter") -> str:
    """
    Find the voice preset file for the given speaker.

    Searches multiple possible locations:
    1. VibeVoice directory in project root
    2. demo/voices/streaming_model/ in sys.path
    3. Relative to current directory
    4. Common installation paths

    Returns:
        Path to the voice preset .pt file

    Raises:
        FileNotFoundError: If preset file not found
    """
    preset_filename = f"en-{speaker_name}_man.pt"

    # Search paths
    search_paths = []

    # 1. Search in project's VibeVoice directory first (most likely location)
    project_root = Path(__file__).parent.parent  # Go up from testmodels to ttsbots
    vibevoice_path = project_root / "VibeVoice" / "demo" / "voices" / "streaming_model" / preset_filename
    search_paths.append(vibevoice_path)

    # Also check in checkpoint directory
    checkpoint_path = project_root / "VibeVoice" / "checkpoints" / "0.5B" / "voices" / preset_filename
    search_paths.append(checkpoint_path)

    # 2. Search in sys.path for demo/voices/streaming_model
    for path in sys.path:
        candidate = Path(path) / "demo" / "voices" / "streaming_model" / preset_filename
        search_paths.append(candidate)

    # 3. Relative to current directory
    search_paths.extend([
        Path.cwd() / "VibeVoice" / "demo" / "voices" / "streaming_model" / preset_filename,
        Path.cwd() / "demo" / "voices" / "streaming_model" / preset_filename,
        Path.cwd() / "voices" / "streaming_model" / preset_filename,
        Path.cwd() / preset_filename,
    ])

    # 3. Common installation paths (cross-platform: Windows + Unix)
    import tempfile
    import platform

    # Universal paths
    search_paths.append(Path.home() / ".cache" / "vibevoice" / "voices" / preset_filename)
    search_paths.append(Path(tempfile.gettempdir()) / "vibevoice" / "voices" / preset_filename)

    # Platform-specific paths
    if platform.system() == "Windows":
        search_paths.append(Path.home() / "AppData" / "Local" / "vibevoice" / "voices" / preset_filename)
        search_paths.append(Path.home() / "AppData" / "Roaming" / "vibevoice" / "voices" / preset_filename)
    else:
        search_paths.append(Path("/usr/local/share/vibevoice/voices") / preset_filename)
        search_paths.append(Path("/opt/vibevoice/voices") / preset_filename)

    # Try each path
    for path in search_paths:
        if path.exists():
            print(f"Found voice preset: {path}")
            return str(path)

    # Not found - raise helpful error
    raise FileNotFoundError(
        f"Voice preset file '{preset_filename}' not found.\n"
        f"Searched in:\n" + "\n".join(f"  - {p}" for p in search_paths[:5]) +
        f"\n\nPlease ensure the VibeVoice repository demo/voices/streaming_model/ directory is accessible."
    )


class VibeVoiceRealtimeTTS:
    """
    Wrapper for VibeVoice Realtime 0.5B streaming model.

    Single-speaker (Carter) streaming TTS optimized for low latency.
    """

    def __init__(self, model_path: str = "microsoft/VibeVoice-Realtime-0.5B"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = None
        self.voice_preset = None

    def load(self, speaker_name: str = "Carter"):
        """Load the streaming model and voice preset."""
        print(f"Loading VibeVoice Realtime 0.5B model from {self.model_path}...")
        self.device = get_device()
        print(f"Using device: {self.device}")

        try:
            from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
            from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

            # Load processor
            print("Loading processor...")
            self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

            # Determine dtype and attention based on device
            if str(self.device) == "cuda":
                dtype = torch.bfloat16
                attn_implementation = "flash_attention_2"
                print("Using bfloat16 with Flash Attention 2")
            elif str(self.device) == "mps":
                dtype = torch.float32
                attn_implementation = "sdpa"
                print("Using float32 with SDPA (MPS mode)")
            else:
                dtype = torch.float32
                attn_implementation = "sdpa"
                print("Using float32 with SDPA (CPU mode)")

            # Try to load model with flash attention, fallback to sdpa
            try:
                print(f"Loading model with {attn_implementation}...")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    attn_implementation=attn_implementation,
                    torch_dtype=dtype,
                    device_map=self.device
                )
            except Exception as e:
                if attn_implementation == "flash_attention_2":
                    print(f"Flash Attention 2 not available ({e}), falling back to SDPA...")
                    self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        attn_implementation="sdpa",
                        torch_dtype=dtype,
                        device_map=self.device
                    )
                else:
                    raise

            # Set DDPM inference steps
            print("Configuring DDPM with 5 steps...")
            self.model.set_ddpm_inference_steps(num_steps=5)

            # Load voice preset
            print(f"Loading voice preset for {speaker_name}...")
            preset_path = _find_voice_preset_path(speaker_name)
            self.voice_preset = torch.load(preset_path, map_location=self.device)

            print("Model loaded successfully!")

        except ImportError as e:
            print(f"\nError: VibeVoice streaming modules not installed.")
            print("Please install with:")
            print("  git clone https://github.com/vibevoice-community/VibeVoice.git")
            print("  cd VibeVoice && pip install -e .")
            print("  huggingface-cli download microsoft/VibeVoice-Realtime-0.5B --local-dir ./checkpoints/0.5B")
            raise e
        except Exception as e:
            print(f"\nError loading model: {e}")
            raise e

    def generate(
        self,
        text: str,
        # Tuning parameters
        cfg_scale: float = 1.5,
        ddpm_steps: int = 5,
        speaker_name: str = "Carter",
    ) -> tuple:
        """
        Generate speech from text using streaming inference.

        Args:
            text: Input text to synthesize
            cfg_scale: Classifier-free guidance scale (1.0-2.0, higher = more faithful)
            ddpm_steps: Number of diffusion steps (3-10, higher = better quality)
            speaker_name: Voice preset name (currently only "Carter")

        Returns:
            Tuple of (audio_array, sample_rate, generation_time_ms)
        """
        start_time = time.time()

        # Update DDPM steps if different
        self.model.set_ddpm_inference_steps(num_steps=ddpm_steps)

        # Prepare inputs with cached voice prompt
        inputs = self.processor.process_input_with_cached_prompt(
            text,
            cached_prompt=self.voice_preset
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                cfg_scale=cfg_scale,
                all_prefilled_outputs=copy.deepcopy(self.voice_preset)
            )

        # Extract audio
        audio = outputs.speech_outputs[0]

        generation_time = (time.time() - start_time) * 1000

        # Convert to numpy
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()

        return audio, SAMPLE_RATE, generation_time

    def save_audio(self, audio, sample_rate: int, filepath: str):
        """Save audio to WAV file."""
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()

        # Ensure 1D audio
        audio = np.squeeze(audio)

        sf.write(filepath, audio, sample_rate)
        print(f"Saved: {filepath}")


def run_basic_test(tts: VibeVoiceRealtimeTTS):
    """Run basic test with default settings."""
    print("\n" + "="*60)
    print("BASIC TEST (Default Settings - Carter voice)")
    print("="*60)

    for i, phrase in enumerate(TEST_PHRASES, 1):
        print(f"\nPhrase {i}: {phrase[:50]}...")
        audio, sr, gen_time = tts.generate(phrase)

        filepath = os.path.join(OUTPUT_DIR, f"basic_phrase_{i}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = len(audio) / sr
        rtf = gen_time / 1000 / audio_duration
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s | RTF: {rtf:.2f}")


def run_tuning_test(tts: VibeVoiceRealtimeTTS):
    """Test different tuning parameters."""
    print("\n" + "="*60)
    print("TUNING TESTS")
    print("="*60)

    test_phrase = TEST_PHRASES[0]

    # Test different cfg_scale values
    print("\n--- Testing CFG Scale ---")
    for cfg in [1.0, 1.3, 1.5, 1.7, 2.0]:
        print(f"\nCFG Scale: {cfg}")
        audio, sr, gen_time = tts.generate(
            test_phrase,
            cfg_scale=cfg
        )

        filepath = os.path.join(OUTPUT_DIR, f"cfg_{cfg}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = len(audio) / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")

    # Test different ddpm_steps
    print("\n--- Testing DDPM Steps ---")
    for steps in [3, 5, 7, 10]:
        print(f"\nDDPM Steps: {steps}")
        audio, sr, gen_time = tts.generate(
            test_phrase,
            ddpm_steps=steps
        )

        filepath = os.path.join(OUTPUT_DIR, f"steps_{steps}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = len(audio) / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")


def run_recommended_settings(tts: VibeVoiceRealtimeTTS):
    """Run with recommended settings for balanced quality."""
    print("\n" + "="*60)
    print("RECOMMENDED SETTINGS (Balanced Quality)")
    print("="*60)
    print("Settings: Carter voice, cfg_scale=1.5, ddpm_steps=5")

    for i, phrase in enumerate(TEST_PHRASES, 1):
        print(f"\nPhrase {i}: {phrase[:50]}...")
        audio, sr, gen_time = tts.generate(
            phrase,
            cfg_scale=1.5,
            ddpm_steps=5,
            speaker_name="Carter"
        )

        filepath = os.path.join(OUTPUT_DIR, f"recommended_phrase_{i}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = len(audio) / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")


def main():
    print("="*60)
    print("VIBEVOICE REALTIME 0.5B TEST")
    print("="*60)

    # Check if VibeVoice is installed
    if not check_vibevoice_installed():
        print("\n" + "!"*60)
        print("VibeVoice Realtime is NOT installed!")
        print("!"*60)
        print("\nTo install VibeVoice Realtime:")
        print("  1. git clone https://github.com/vibevoice-community/VibeVoice.git")
        print("  2. cd VibeVoice")
        print("  3. pip install -e .")
        print("  4. huggingface-cli download microsoft/VibeVoice-Realtime-0.5B --local-dir ./checkpoints/0.5B")
        print("\nNote: This is the 0.5B streaming model (single-speaker: Carter)")
        print("\n" + "!"*60)
        print("WARNING: Developers advise against commercial use!")
        print("!"*60)
        return

    # Initialize and load model
    tts = VibeVoiceRealtimeTTS()
    tts.load(speaker_name="Carter")

    # Run tests
    run_basic_test(tts)
    run_tuning_test(tts)
    run_recommended_settings(tts)

    print("\n" + "="*60)
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("="*60)
    print("\nFiles generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.wav'):
            print(f"  - {f}")

    print("\n" + "="*60)
    print("RECOMMENDATION FOR HIRING AGENT")
    print("="*60)
    print("For balanced quality and speed, use:")
    print("  speaker_name='Carter'  (only available speaker)")
    print("  cfg_scale=1.5          (balanced text faithfulness)")
    print("  ddpm_steps=5           (good quality/speed tradeoff)")
    print("\nListen to 'recommended_phrase_*.wav' files for examples.")


if __name__ == "__main__":
    main()
