"""
Chatterbox TTS Test Script
==========================
Tests Resemble AI's Chatterbox model for text-to-speech synthesis.

Installation:
    pip install chatterbox-tts

Tuning Parameters:
    - exaggeration: 0.0-1.0 (expressiveness level)
    - cfg_weight: 0.0-1.0 (classifier-free guidance weight)
    - temperature: 0.1-1.5 (sampling randomness)
    - top_p: 0.5-1.0 (nucleus sampling threshold)

Special Features:
    - Paralinguistic tags: [laugh], [chuckle], [sigh], [gasp], [cough]
    - Native MPS support (best for Apple Silicon)
    - Excellent streaming support

Known Issues:
    - Watermarking: All output includes imperceptible watermark
    - English only (Turbo model) - Multilingual requires separate model
    - Needs ~8GB memory for comfortable inference
"""

import os
import time
import torch
import numpy as np

# Test phrases for hiring agent
TEST_PHRASES = [
    "Hello, thank you for applying to this position. Can you tell me about your experience?",
    "That's great! Your background in software development sounds very relevant.",
    "Let me ask you a few technical questions to better understand your skills.",
    "Could you walk me through a challenging project you've worked on recently?",
]

# Test phrases with paralinguistic tags
EXPRESSIVE_PHRASES = [
    "[chuckle] That's a great answer! I really appreciate your enthusiasm.",
    "Interesting approach. [pause] Let me think about that for a moment.",
    "[laugh] That's exactly the kind of creative thinking we're looking for!",
]

# Output directory
OUTPUT_DIR = "outputs/chatterbox"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_device():
    """
    Get the best available device with cross-platform support.

    Priority: CUDA > DirectML (AMD GPU) > MPS (Apple Silicon) > CPU

    Platform Support:
        - NVIDIA GPU: CUDA (recommended)
        - AMD GPU: DirectML (Windows only, install torch-directml)
        - Apple Silicon: MPS (native support, Chatterbox works great!)
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

    # Apple Silicon MPS - Chatterbox works great on MPS!
    if torch.backends.mps.is_available():
        return "mps"

    # CPU fallback
    return "cpu"


def check_chatterbox_installed():
    """Check if Chatterbox is installed."""
    try:
        from chatterbox import ChatterboxTTS
        return True
    except ImportError:
        return False


class ChatterboxTTSWrapper:
    """
    Wrapper for Chatterbox TTS model.

    This is the recommended model for real-time streaming applications.
    """

    def __init__(self):
        self.model = None
        self.device = None

    def load(self):
        """Load the model."""
        print("Loading Chatterbox model...")
        self.device = get_device()
        print(f"Using device: {self.device}")

        try:
            from chatterbox import ChatterboxTTS

            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            print(f"Model loaded successfully! Sample rate: {self.model.sr}")

        except ImportError as e:
            print(f"\nError: Chatterbox not installed.")
            print("Please install with:")
            print("  pip install chatterbox-tts")
            raise e

    def generate(
        self,
        text: str,
        # Tuning parameters
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> tuple:
        """
        Generate speech from text.

        Args:
            text: Input text to synthesize (supports paralinguistic tags)
            exaggeration: Expressiveness level (0.0-1.0, higher = more expressive)
            cfg_weight: Classifier-free guidance weight (0.0-1.0)
            temperature: Sampling randomness (0.1-1.5)
            top_p: Nucleus sampling threshold (0.5-1.0)

        Returns:
            Tuple of (audio_tensor, sample_rate, generation_time_ms)
        """
        start_time = time.time()

        # Generate with tuning parameters
        try:
            wav = self.model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                top_p=top_p,
            )
        except TypeError:
            # Fallback if some parameters aren't supported in this version
            wav = self.model.generate(text)

        generation_time = (time.time() - start_time) * 1000

        return wav, self.model.sr, generation_time

    def save_audio(self, audio, sample_rate: int, filepath: str):
        """Save audio to WAV file."""
        import torchaudio as ta

        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        ta.save(filepath, audio.cpu(), sample_rate)
        print(f"Saved: {filepath}")


def run_basic_test(tts: ChatterboxTTSWrapper):
    """Run basic test with default settings."""
    print("\n" + "="*60)
    print("BASIC TEST (Default Settings)")
    print("="*60)

    for i, phrase in enumerate(TEST_PHRASES, 1):
        print(f"\nPhrase {i}: {phrase[:50]}...")
        audio, sr, gen_time = tts.generate(phrase)

        filepath = os.path.join(OUTPUT_DIR, f"basic_phrase_{i}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = audio.shape[-1] / sr
        rtf = gen_time / 1000 / audio_duration
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s | RTF: {rtf:.2f}")


def run_exaggeration_test(tts: ChatterboxTTSWrapper):
    """Test different expressiveness levels."""
    print("\n" + "="*60)
    print("EXAGGERATION TESTS (Expressiveness)")
    print("="*60)

    test_phrase = TEST_PHRASES[0]

    for exag in [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]:
        print(f"\nExaggeration: {exag}")
        audio, sr, gen_time = tts.generate(
            test_phrase,
            exaggeration=exag
        )

        filepath = os.path.join(OUTPUT_DIR, f"exag_{exag}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = audio.shape[-1] / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")


def run_temperature_test(tts: ChatterboxTTSWrapper):
    """Test different temperature values."""
    print("\n" + "="*60)
    print("TEMPERATURE TESTS")
    print("="*60)

    test_phrase = TEST_PHRASES[0]

    for temp in [0.5, 0.7, 0.8, 1.0, 1.2]:
        print(f"\nTemperature: {temp}")
        audio, sr, gen_time = tts.generate(
            test_phrase,
            temperature=temp
        )

        filepath = os.path.join(OUTPUT_DIR, f"temp_{temp}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = audio.shape[-1] / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")


def run_cfg_test(tts: ChatterboxTTSWrapper):
    """Test different CFG weight values."""
    print("\n" + "="*60)
    print("CFG WEIGHT TESTS")
    print("="*60)

    test_phrase = TEST_PHRASES[0]

    for cfg in [0.0, 0.3, 0.5, 0.7, 1.0]:
        print(f"\nCFG Weight: {cfg}")
        audio, sr, gen_time = tts.generate(
            test_phrase,
            cfg_weight=cfg
        )

        filepath = os.path.join(OUTPUT_DIR, f"cfg_{cfg}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = audio.shape[-1] / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")


def run_paralinguistic_test(tts: ChatterboxTTSWrapper):
    """Test paralinguistic tags (unique Chatterbox feature)."""
    print("\n" + "="*60)
    print("PARALINGUISTIC TAGS TEST")
    print("="*60)
    print("Chatterbox supports: [laugh], [chuckle], [sigh], [gasp], [cough]")

    for i, phrase in enumerate(EXPRESSIVE_PHRASES, 1):
        print(f"\nPhrase {i}: {phrase[:60]}...")
        audio, sr, gen_time = tts.generate(phrase)

        filepath = os.path.join(OUTPUT_DIR, f"expressive_phrase_{i}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = audio.shape[-1] / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")


def run_recommended_settings(tts: ChatterboxTTSWrapper):
    """Run with recommended settings for professional hiring agent."""
    print("\n" + "="*60)
    print("RECOMMENDED SETTINGS (Professional Hiring Agent)")
    print("="*60)
    print("Settings: exaggeration=0.3, cfg_weight=0.5, temperature=0.7")
    print("(Moderate expressiveness - professional but warm)")

    for i, phrase in enumerate(TEST_PHRASES, 1):
        print(f"\nPhrase {i}: {phrase[:50]}...")
        audio, sr, gen_time = tts.generate(
            phrase,
            exaggeration=0.3,
            cfg_weight=0.5,
            temperature=0.7
        )

        filepath = os.path.join(OUTPUT_DIR, f"recommended_phrase_{i}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = audio.shape[-1] / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")


def run_expressive_settings(tts: ChatterboxTTSWrapper):
    """Run with more expressive settings."""
    print("\n" + "="*60)
    print("EXPRESSIVE SETTINGS (More Personality)")
    print("="*60)
    print("Settings: exaggeration=0.7, temperature=0.9")

    for i, phrase in enumerate(TEST_PHRASES, 1):
        print(f"\nPhrase {i}: {phrase[:50]}...")
        audio, sr, gen_time = tts.generate(
            phrase,
            exaggeration=0.7,
            temperature=0.9
        )

        filepath = os.path.join(OUTPUT_DIR, f"expressive_setting_phrase_{i}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = audio.shape[-1] / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")


def main():
    print("="*60)
    print("CHATTERBOX TTS TEST")
    print("="*60)

    # Check if Chatterbox is installed
    if not check_chatterbox_installed():
        print("\n" + "!"*60)
        print("Chatterbox is NOT installed!")
        print("!"*60)
        print("\nTo install Chatterbox:")
        print("  pip install chatterbox-tts")
        print("\nNote: This is the recommended model for your use case!")
        print("  - MIT license (enterprise-friendly)")
        print("  - Native MPS support (great for Apple Silicon)")
        print("  - Excellent streaming support")
        print("  - Good tuning options for professional tone")
        return

    # Initialize and load model
    tts = ChatterboxTTSWrapper()
    tts.load()

    # Run tests
    run_basic_test(tts)
    run_exaggeration_test(tts)
    run_temperature_test(tts)
    run_cfg_test(tts)
    run_paralinguistic_test(tts)
    run_recommended_settings(tts)
    run_expressive_settings(tts)

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
    print("  exaggeration=0.3  (moderate expressiveness)")
    print("  cfg_weight=0.5    (balanced guidance)")
    print("  temperature=0.7   (slight variation)")
    print("\nListen to 'recommended_phrase_*.wav' files for examples.")


if __name__ == "__main__":
    main()
