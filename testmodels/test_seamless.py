"""
Seamless M4T v2 TTS Test Script
================================
Tests Facebook's Seamless M4T v2 model for text-to-speech synthesis.

Installation:
    pip install transformers torch torchaudio sentencepiece

Tuning Parameters:
    - num_beams: 1-10 (higher = more stable but slower)
    - do_sample: True/False (enable sampling for variety)
    - temperature: 0.1-2.0 (higher = more expressive, requires do_sample=True)
    - speaker_id: 0-7 (different voice embeddings)

Known Issues:
    - Robotic sound is inherent to the model architecture
    - Use CPU on Apple Silicon (MPS has compatibility issues)
    - License: CC-BY-NC (non-commercial) - check enterprise compatibility
"""

import os
import time
import torch
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, SeamlessM4Tv2Model

# Test phrases for hiring agent
TEST_PHRASES = [
    "Hello, thank you for applying to this position. Can you tell me about your experience?",
    "That's great! Your background in software development sounds very relevant.",
    "Let me ask you a few technical questions to better understand your skills.",
    "Could you walk me through a challenging project you've worked on recently?",
]

# Output directory
OUTPUT_DIR = "outputs/seamless"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_device():
    """Get the best available device. Seamless works best on CPU for Apple Silicon."""
    # Seamless has MPS compatibility issues, so we default to CPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SeamlessTTS:
    def __init__(self, model_name: str = "facebook/seamless-m4t-v2-large"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = None

    def load(self):
        """Load the model and processor."""
        print(f"Loading Seamless M4T v2 model...")
        self.device = get_device()
        print(f"Using device: {self.device}")

        # Use slow tokenizer to avoid compatibility issues with fast tokenizer
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            use_fast=False
        )
        self.model = SeamlessM4Tv2Model.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def generate(
        self,
        text: str,
        src_lang: str = "eng",
        tgt_lang: str = "eng",
        # Tuning parameters
        num_beams: int = 5,
        do_sample: bool = False,
        temperature: float = 1.0,
        speaker_id: int = 0,
    ) -> tuple:
        """
        Generate speech from text.

        Args:
            text: Input text to synthesize
            src_lang: Source language code (e.g., "eng", "fra", "deu")
            tgt_lang: Target language code for output speech
            num_beams: Beam search width (1 = greedy, higher = more stable)
            do_sample: Enable sampling for more variety
            temperature: Sampling temperature (requires do_sample=True)
            speaker_id: Speaker embedding index (0-7 typically)

        Returns:
            Tuple of (audio_array, sample_rate, generation_time_ms)
        """
        inputs = self.processor(
            text=text,
            src_lang=src_lang,
            return_tensors="pt"
        ).to(self.device)

        start_time = time.time()

        with torch.no_grad():
            # Build generation kwargs
            gen_kwargs = {
                "tgt_lang": tgt_lang,
                "num_beams": num_beams,
            }

            # Add sampling parameters if enabled
            if do_sample:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature

            # Add speaker ID if supported
            try:
                gen_kwargs["speaker_id"] = speaker_id
            except Exception:
                pass  # Some model versions may not support this

            output = self.model.generate(**inputs, **gen_kwargs)

        generation_time = (time.time() - start_time) * 1000

        # Extract audio from output
        audio = output[0].cpu().numpy().squeeze()
        sample_rate = 16000  # Seamless outputs at 16kHz

        return audio, sample_rate, generation_time

    def save_audio(self, audio, sample_rate: int, filepath: str):
        """Save audio to WAV file."""
        # Convert to numpy array if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Ensure correct shape for soundfile (samples,) or (samples, channels)
        if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
            audio = audio.T  # Transpose from (channels, samples) to (samples, channels)

        sf.write(filepath, audio, sample_rate)
        print(f"Saved: {filepath}")


def run_basic_test(tts: SeamlessTTS):
    """Run basic test with default settings."""
    print("\n" + "="*60)
    print("BASIC TEST (Default Settings)")
    print("="*60)

    for i, phrase in enumerate(TEST_PHRASES, 1):
        print(f"\nPhrase {i}: {phrase[:50]}...")
        audio, sr, gen_time = tts.generate(phrase)

        filepath = os.path.join(OUTPUT_DIR, f"basic_phrase_{i}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = len(audio) / sr
        rtf = gen_time / 1000 / audio_duration
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s | RTF: {rtf:.2f}")


def run_tuning_test(tts: SeamlessTTS):
    """Run tests with different tuning parameters."""
    print("\n" + "="*60)
    print("TUNING TESTS")
    print("="*60)

    test_phrase = TEST_PHRASES[0]  # Use first phrase for tuning tests

    # Test different speaker IDs
    print("\n--- Testing Speaker IDs ---")
    for speaker_id in range(4):  # Test speakers 0-3
        try:
            audio, sr, gen_time = tts.generate(
                test_phrase,
                speaker_id=speaker_id
            )
            filepath = os.path.join(OUTPUT_DIR, f"speaker_{speaker_id}.wav")
            tts.save_audio(audio, sr, filepath)
            print(f"Speaker {speaker_id}: {gen_time:.0f}ms")
        except Exception as e:
            print(f"Speaker {speaker_id}: Not supported ({e})")

    # Test sampling with temperature
    print("\n--- Testing Temperature (with sampling) ---")
    for temp in [0.5, 0.7, 1.0, 1.3]:
        audio, sr, gen_time = tts.generate(
            test_phrase,
            do_sample=True,
            temperature=temp,
            num_beams=1  # Use greedy with sampling
        )
        filepath = os.path.join(OUTPUT_DIR, f"temp_{temp}.wav")
        tts.save_audio(audio, sr, filepath)
        print(f"Temperature {temp}: {gen_time:.0f}ms")

    # Test beam sizes
    print("\n--- Testing Beam Sizes ---")
    for beams in [1, 3, 5]:
        audio, sr, gen_time = tts.generate(
            test_phrase,
            num_beams=beams
        )
        filepath = os.path.join(OUTPUT_DIR, f"beams_{beams}.wav")
        tts.save_audio(audio, sr, filepath)
        print(f"Beams {beams}: {gen_time:.0f}ms")


def run_recommended_settings(tts: SeamlessTTS):
    """Run with recommended settings for less robotic output."""
    print("\n" + "="*60)
    print("RECOMMENDED SETTINGS (Less Robotic)")
    print("="*60)
    print("Settings: speaker_id=1, do_sample=True, temperature=0.7")

    for i, phrase in enumerate(TEST_PHRASES, 1):
        print(f"\nPhrase {i}: {phrase[:50]}...")
        audio, sr, gen_time = tts.generate(
            phrase,
            speaker_id=1,
            do_sample=True,
            temperature=0.7,
            num_beams=1
        )

        filepath = os.path.join(OUTPUT_DIR, f"recommended_phrase_{i}.wav")
        tts.save_audio(audio, sr, filepath)

        audio_duration = len(audio) / sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {audio_duration:.2f}s")


def main():
    print("="*60)
    print("SEAMLESS M4T V2 TTS TEST")
    print("="*60)

    # Initialize and load model
    tts = SeamlessTTS()
    tts.load()

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


if __name__ == "__main__":
    main()
