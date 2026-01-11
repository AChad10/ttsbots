"""
MeloTTS Test Script
===================
Tests MyShell AI's MeloTTS model for text-to-speech synthesis.

Installation:
    git clone https://github.com/myshell-ai/MeloTTS.git
    cd MeloTTS
    pip install -e .
    python -m unidic download

Tuning Parameters:
    - speed: 0.5-2.0 (speaking rate, 1.0 = normal)
    - noise_scale: 0.1-1.0 (voice variation/expressiveness)
    - noise_scale_w: 0.1-1.0 (duration variation)
    - sdp_ratio: 0.0-1.0 (stochastic duration predictor ratio)

Special Features:
    - Multi-accent English: EN-US, EN-BR, EN-INDIA, EN-AU, EN-Default
    - Fast CPU inference (real-time capable)
    - MIT License (enterprise-friendly)
    - Good pronunciation quality

Known Issues:
    - Requires unidic dictionary download (one-time setup)
    - English-only model (separate models for other languages)
    - Can sound slightly robotic at default settings
"""

import os
import time
import torch
import soundfile as sf

# Test phrases for hiring agent
TEST_PHRASES = [
    "Hello, thank you for applying to this position. Can you tell me about your experience?",
    "That's great! Your background in software development sounds very relevant.",
    "Let me ask you a few technical questions to better understand your skills.",
    "Could you walk me through a challenging project you've worked on recently?",
]

# Output directory
OUTPUT_DIR = "outputs/melotts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def check_melotts_installed():
    """Check if MeloTTS is installed."""
    try:
        from melo.api import TTS
        return True
    except ImportError:
        return False


class MeloTTSWrapper:
    """
    Wrapper for MeloTTS model.

    Multi-accent English TTS with fast CPU inference.
    """

    def __init__(self, language='EN'):
        self.model = None
        self.device = None
        self.language = language
        self.sample_rate = None
        self.speaker_ids = None

    def load(self):
        """Load the model."""
        print("Loading MeloTTS model...")
        self.device = get_device()
        print(f"Using device: {self.device}")

        try:
            from melo.api import TTS

            # Initialize TTS model
            self.model = TTS(language=self.language, device='auto')

            # Get sample rate and speaker IDs
            self.sample_rate = self.model.hps.data.sampling_rate
            self.speaker_ids = self.model.hps.data.spk2id

            print(f"Model loaded successfully!")
            print(f"Sample rate: {self.sample_rate}")
            print(f"Available speakers: {list(self.speaker_ids.keys())}")

        except ImportError as e:
            print(f"\nError: MeloTTS not installed.")
            print("Please install with:")
            print("  git clone https://github.com/myshell-ai/MeloTTS.git")
            print("  cd MeloTTS")
            print("  pip install -e .")
            print("  python -m unidic download")
            raise e

    def generate(
        self,
        text: str,
        # Tuning parameters
        speaker_id: str = "EN-US",
        speed: float = 1.0,
        noise_scale: float = 0.6,
        noise_scale_w: float = 0.8,
        sdp_ratio: float = 0.2,
    ) -> tuple:
        """
        Generate speech from text.

        Args:
            text: Input text to synthesize
            speaker_id: Accent/speaker (EN-US, EN-BR, EN-INDIA, EN-AU, EN-Default)
            speed: Speaking rate (0.5-2.0, 1.0 = normal)
            noise_scale: Voice variation/expressiveness (0.1-1.0)
            noise_scale_w: Duration variation (0.1-1.0)
            sdp_ratio: Stochastic duration predictor ratio (0.0-1.0)

        Returns:
            Tuple of (audio_array, sample_rate, generation_time_ms)
        """
        start_time = time.time()

        # Get speaker index from speaker_id string
        if speaker_id in self.speaker_ids:
            speaker_idx = self.speaker_ids[speaker_id]
        else:
            print(f"Warning: Speaker '{speaker_id}' not found. Using EN-US.")
            speaker_idx = self.speaker_ids.get("EN-US", 0)

        # Generate audio
        audio = self.model.tts_to_file(
            text,
            speaker_idx,
            None,  # reference_audio (not used)
            speed=speed,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            quiet=True
        )

        generation_time = (time.time() - start_time) * 1000

        return audio, self.sample_rate, generation_time

    def save_audio(self, audio, sample_rate: int, filepath: str):
        """Save audio to WAV file."""
        sf.write(filepath, audio, sample_rate)
        print(f"Saved: {filepath}")


def run_basic_test(tts: MeloTTSWrapper):
    """Run basic test with default settings."""
    print("\n" + "="*60)
    print("BASIC TEST (Default Settings - EN-US)")
    print("="*60)

    for i, phrase in enumerate(TEST_PHRASES, 1):
        print(f"\nPhrase {i}: {phrase[:50]}...")
        audio, sr, gen_time = tts.generate(phrase)

        filepath = os.path.join(OUTPUT_DIR, f"basic_phrase_{i}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = len(audio) / sr
        rtf = gen_time / 1000 / audio_duration
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s | RTF: {rtf:.2f}")


def run_accent_test(tts: MeloTTSWrapper):
    """Test different English accents."""
    print("\n" + "="*60)
    print("ACCENT TESTS")
    print("="*60)

    test_phrase = TEST_PHRASES[0]
    accents = ["EN-US", "EN-BR", "EN-INDIA", "EN-AU"]

    for accent in accents:
        if accent not in tts.speaker_ids:
            print(f"\nSkipping {accent} (not available)")
            continue

        print(f"\nAccent: {accent}")
        audio, sr, gen_time = tts.generate(
            test_phrase,
            speaker_id=accent
        )

        filepath = os.path.join(OUTPUT_DIR, f"accent_{accent}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = len(audio) / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")


def run_speed_test(tts: MeloTTSWrapper):
    """Test different speaking speeds."""
    print("\n" + "="*60)
    print("SPEED TESTS")
    print("="*60)

    test_phrase = TEST_PHRASES[0]

    for speed in [0.8, 0.9, 1.0, 1.1, 1.2]:
        print(f"\nSpeed: {speed}")
        audio, sr, gen_time = tts.generate(
            test_phrase,
            speed=speed
        )

        filepath = os.path.join(OUTPUT_DIR, f"speed_{speed}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = len(audio) / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")


def run_expressiveness_test(tts: MeloTTSWrapper):
    """Test different expressiveness levels (noise_scale)."""
    print("\n" + "="*60)
    print("EXPRESSIVENESS TESTS (noise_scale)")
    print("="*60)

    test_phrase = TEST_PHRASES[0]

    for noise in [0.4, 0.6, 0.8]:
        print(f"\nNoise scale: {noise}")
        audio, sr, gen_time = tts.generate(
            test_phrase,
            noise_scale=noise
        )

        filepath = os.path.join(OUTPUT_DIR, f"noise_{noise}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = len(audio) / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")


def run_recommended_settings(tts: MeloTTSWrapper):
    """Run with recommended settings for professional hiring agent."""
    print("\n" + "="*60)
    print("RECOMMENDED SETTINGS (Professional Hiring Agent)")
    print("="*60)
    print("Settings: EN-US, speed=1.0, noise_scale=0.6")
    print("(Balanced quality and naturalness)")

    for i, phrase in enumerate(TEST_PHRASES, 1):
        print(f"\nPhrase {i}: {phrase[:50]}...")
        audio, sr, gen_time = tts.generate(
            phrase,
            speaker_id="EN-US",
            speed=1.0,
            noise_scale=0.6,
            noise_scale_w=0.8,
            sdp_ratio=0.2
        )

        filepath = os.path.join(OUTPUT_DIR, f"recommended_phrase_{i}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = len(audio) / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")


def main():
    print("="*60)
    print("MELOTTS TEST")
    print("="*60)

    # Check if MeloTTS is installed
    if not check_melotts_installed():
        print("\n" + "!"*60)
        print("MeloTTS is NOT installed!")
        print("!"*60)
        print("\nTo install MeloTTS:")
        print("  git clone https://github.com/myshell-ai/MeloTTS.git")
        print("  cd MeloTTS")
        print("  pip install -e .")
        print("  python -m unidic download")
        print("\nNote: Good open-source option!")
        print("  - MIT license (enterprise-friendly)")
        print("  - Fast CPU inference")
        print("  - Multi-accent English support")
        return

    # Initialize and load model
    tts = MeloTTSWrapper(language='EN')
    tts.load()

    # Run tests
    run_basic_test(tts)
    run_accent_test(tts)
    run_speed_test(tts)
    run_expressiveness_test(tts)
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
    print("For a professional hiring agent voice, use:")
    print("  speaker_id='EN-US'  (American English)")
    print("  speed=1.0           (normal speed)")
    print("  noise_scale=0.6     (moderate expressiveness)")
    print("\nListen to 'recommended_phrase_*.wav' files for examples.")


if __name__ == "__main__":
    main()
