"""
TTS Model Comparison Script
===========================
Runs all TTS models with the same test phrase and generates a comparison report.

Usage:
    python compare_outputs.py

This script will:
1. Check which models are available
2. Run each available model with ONE test phrase
3. Generate a comparison table with timing metrics
4. Save one output file per model to outputs/ directory
"""

import os
import sys
import time
import gc
import copy
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import torch

# Single test phrase for comparison
TEST_PHRASE = "Hello, thank you for applying to this position. Can you tell me about your experience? That's great! Your background in software development sounds very relevant."

# Recommended tuning parameters for each model
RECOMMENDED_SETTINGS = {
    "seamless": {
        "speaker_id": 1,
        "do_sample": True,
        "temperature": 0.7,
        "num_beams": 1,
    },
    "vibevoice": {
        "speaker_name": "Carter",
        "cfg_scale": 1.5,
        "ddpm_steps": 5,
    },
    "chatterbox": {
        "exaggeration": 0.3,
        "cfg_weight": 0.5,
        "temperature": 0.7,
    },
    "melotts": {
        "speaker_id": "EN-US",
        "speed": 1.0,
        "noise_scale": 0.6,
        "sdp_ratio": 0.2,
        "noise_scale_w": 0.8,
    },
}


@dataclass
class ModelResult:
    """Results from running a model."""
    model_name: str
    available: bool
    error_message: Optional[str] = None
    generation_time: Optional[float] = None
    audio_duration: Optional[float] = None
    sample_rate: Optional[int] = None


def cleanup_memory():
    """Clean up GPU/MPS memory between model runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def check_model_available(model_name: str) -> tuple:
    """Check if a model is available for testing."""
    if model_name == "seamless":
        try:
            from transformers import AutoProcessor, SeamlessM4Tv2Model
            return True, None
        except ImportError as e:
            return False, f"transformers not installed: {e}"

    elif model_name == "vibevoice":
        try:
            from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
            from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
            return True, None
        except ImportError:
            return False, "VibeVoice Realtime 0.5B not installed. See test_vibevoice.py for instructions."

    elif model_name == "chatterbox":
        try:
            from chatterbox import ChatterboxTTS
            return True, None
        except ImportError:
            return False, "Chatterbox not installed. Run: pip install chatterbox-tts"

    elif model_name == "melotts":
        try:
            from melo.api import TTS
            return True, None
        except ImportError:
            return False, "MeloTTS not installed. See test_melotts.py for instructions."

    return False, f"Unknown model: {model_name}"


def _find_vibevoice_voice_preset(speaker_name: str = "Carter") -> str:
    """Find the VibeVoice voice preset file (cross-platform: Windows + Unix)."""
    import tempfile
    import platform

    preset_filename = f"en-{speaker_name}.pt"

    search_paths = []

    # Search in sys.path for demo/voices/streaming_model
    for path in sys.path:
        candidate = Path(path) / "demo" / "voices" / "streaming_model" / preset_filename
        search_paths.append(candidate)

    # Relative to current directory
    search_paths.extend([
        Path.cwd() / "demo" / "voices" / "streaming_model" / preset_filename,
        Path.cwd() / "voices" / "streaming_model" / preset_filename,
        Path.cwd() / preset_filename,
    ])

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
            return str(path)

    raise FileNotFoundError(
        f"Voice preset file '{preset_filename}' not found. "
        f"Please ensure the VibeVoice repository demo/voices/streaming_model/ directory is accessible."
    )


def run_seamless() -> ModelResult:
    """Run Seamless M4T v2 model."""
    print("\n" + "="*60)
    print("Running Seamless M4T v2...")
    print("="*60)

    try:
        from transformers import AutoProcessor, SeamlessM4Tv2Model
        import torchaudio

        processor = AutoProcessor.from_pretrained(
            "facebook/seamless-m4t-v2-large",
            use_fast=False
        )
        model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
        model.to("cpu")  # CPU for stability on Apple Silicon
        model.eval()

        settings = RECOMMENDED_SETTINGS["seamless"]

        inputs = processor(text=TEST_PHRASE, src_lang="eng", return_tensors="pt")

        start = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                tgt_lang="eng",
                **settings
            )
        gen_time = (time.time() - start) * 1000

        audio = output[0].cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        os.makedirs("outputs/seamless", exist_ok=True)
        filepath = "outputs/seamless/compare_output.wav"
        torchaudio.save(filepath, audio, 16000)

        duration = audio.shape[-1] / 16000
        print(f"  Generation: {gen_time:.0f}ms | Duration: {duration:.2f}s")

        del model, processor
        cleanup_memory()

        return ModelResult(
            model_name="Seamless M4T v2",
            available=True,
            generation_time=gen_time,
            audio_duration=duration,
            sample_rate=16000
        )

    except Exception as e:
        return ModelResult(
            model_name="Seamless M4T v2",
            available=False,
            error_message=str(e)
        )


def run_vibevoice() -> ModelResult:
    """Run VibeVoice Realtime 0.5B model."""
    print("\n" + "="*60)
    print("Running VibeVoice Realtime 0.5B...")
    print("="*60)

    try:
        from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
        import soundfile as sf

        model_path = "microsoft/VibeVoice-Realtime-0.5B"

        # Detect device
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
            attn = "flash_attention_2"
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float32
            attn = "sdpa"
        else:
            device = "cpu"
            dtype = torch.float32
            attn = "sdpa"

        print(f"Using device: {device}")

        # Load processor
        processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

        # Load model with fallback
        try:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                attn_implementation=attn,
                torch_dtype=dtype,
                device_map=device
            )
        except Exception:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                attn_implementation="sdpa",
                torch_dtype=dtype,
                device_map=device
            )

        # Set DDPM steps
        model.set_ddpm_inference_steps(num_steps=5)

        # Load voice preset
        preset_path = _find_vibevoice_voice_preset("Carter")
        voice_preset = torch.load(preset_path, map_location=device)

        settings = RECOMMENDED_SETTINGS["vibevoice"]

        # Prepare inputs
        inputs = processor.process_input_with_cached_prompt(
            TEST_PHRASE,
            cached_prompt=voice_preset
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Generate
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                cfg_scale=settings["cfg_scale"],
                all_prefilled_outputs=copy.deepcopy(voice_preset)
            )
        gen_time = (time.time() - start) * 1000

        audio = outputs.speech_outputs[0]
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()

        os.makedirs("outputs/vibevoice", exist_ok=True)
        filepath = "outputs/vibevoice/compare_output.wav"
        sf.write(filepath, audio, 24000)

        duration = len(audio) / 24000
        print(f"  Generation: {gen_time:.0f}ms | Duration: {duration:.2f}s")

        del model, processor, voice_preset
        cleanup_memory()

        return ModelResult(
            model_name="VibeVoice 0.5B",
            available=True,
            generation_time=gen_time,
            audio_duration=duration,
            sample_rate=24000
        )

    except Exception as e:
        return ModelResult(
            model_name="VibeVoice 0.5B",
            available=False,
            error_message=str(e)
        )


def run_chatterbox() -> ModelResult:
    """Run Chatterbox model."""
    print("\n" + "="*60)
    print("Running Chatterbox...")
    print("="*60)

    try:
        from chatterbox import ChatterboxTTS
        import torchaudio as ta

        # Device detection with DirectML support
        if torch.cuda.is_available():
            device = "cuda"
        else:
            try:
                import torch_directml
                device = torch_directml.device()
                print(f"DirectML device detected: {device}")
            except:
                device = "mps" if torch.backends.mps.is_available() else "cpu"

        print(f"Using device: {device}")

        model = ChatterboxTTS.from_pretrained(device=device)
        settings = RECOMMENDED_SETTINGS["chatterbox"]

        start = time.time()
        wav = model.generate(TEST_PHRASE, **settings)
        gen_time = (time.time() - start) * 1000

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        os.makedirs("outputs/chatterbox", exist_ok=True)
        filepath = "outputs/chatterbox/compare_output.wav"
        ta.save(filepath, wav.cpu(), model.sr)

        duration = wav.shape[-1] / model.sr
        print(f"  Generation: {gen_time:.0f}ms | Duration: {duration:.2f}s")

        sample_rate = model.sr
        del model
        cleanup_memory()

        return ModelResult(
            model_name="Chatterbox",
            available=True,
            generation_time=gen_time,
            audio_duration=duration,
            sample_rate=sample_rate
        )

    except Exception as e:
        return ModelResult(
            model_name="Chatterbox",
            available=False,
            error_message=str(e)
        )


def run_melotts() -> ModelResult:
    """Run MeloTTS model."""
    print("\n" + "="*60)
    print("Running MeloTTS...")
    print("="*60)

    try:
        from melo.api import TTS
        import soundfile as sf

        model = TTS(language='EN', device='auto')
        sample_rate = model.hps.data.sampling_rate
        speaker_ids = model.hps.data.spk2id

        settings = RECOMMENDED_SETTINGS["melotts"]
        speaker_idx = speaker_ids.get(settings["speaker_id"], 0)

        start = time.time()
        audio = model.tts_to_file(
            TEST_PHRASE,
            speaker_idx,
            None,
            speed=settings["speed"],
            sdp_ratio=settings["sdp_ratio"],
            noise_scale=settings["noise_scale"],
            noise_scale_w=settings["noise_scale_w"],
            quiet=True
        )
        gen_time = (time.time() - start) * 1000

        os.makedirs("outputs/melotts", exist_ok=True)
        filepath = "outputs/melotts/compare_output.wav"
        sf.write(filepath, audio, sample_rate)

        duration = len(audio) / sample_rate
        print(f"  Generation: {gen_time:.0f}ms | Duration: {duration:.2f}s")

        del model
        cleanup_memory()

        return ModelResult(
            model_name="MeloTTS",
            available=True,
            generation_time=gen_time,
            audio_duration=duration,
            sample_rate=sample_rate
        )

    except Exception as e:
        return ModelResult(
            model_name="MeloTTS",
            available=False,
            error_message=str(e)
        )


def print_comparison_table(results: List[ModelResult]):
    """Print a comparison table of all results."""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    # Header
    print(f"\n{'Model':<20} {'Status':<12} {'Time':<12} {'Duration':<12} {'RTF':<10} {'Sample Rate'}")
    print("-" * 80)

    for result in results:
        if result.available and result.generation_time:
            rtf = (result.generation_time / 1000) / result.audio_duration

            print(f"{result.model_name:<20} {'OK':<12} {result.generation_time:>8.0f}ms  {result.audio_duration:>8.2f}s    {rtf:>8.2f}   {result.sample_rate}")
        else:
            status = "SKIPPED"
            print(f"{result.model_name:<20} {status:<12} {'N/A':<12} {'N/A':<12} {'N/A':<10} N/A")
            if result.error_message:
                print(f"  Error: {result.error_message[:60]}...")

    print("-" * 80)
    print("\nRTF = Real-Time Factor (< 1.0 means faster than real-time)")


def print_recommendations(results: List[ModelResult]):
    """Print recommendations based on results."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    available = [r for r in results if r.available and r.generation_time]

    if not available:
        print("\nNo models were successfully tested. Please install the required dependencies.")
        return

    # Sort by RTF (lower is better)
    sorted_by_rtf = sorted(
        available,
        key=lambda r: (r.generation_time / 1000) / r.audio_duration
    )

    print("\nFor REAL-TIME streaming (lowest latency):")
    for i, result in enumerate(sorted_by_rtf[:3], 1):
        rtf = (result.generation_time / 1000) / result.audio_duration
        print(f"  {i}. {result.model_name} (RTF: {rtf:.2f})")

    print("\nLicense considerations:")
    print("  - Chatterbox: MIT (enterprise-friendly)")
    print("  - MeloTTS: MIT (enterprise-friendly)")
    print("  - VibeVoice: MIT (but devs warn against commercial use)")
    print("  - Seamless: CC-BY-NC (non-commercial only!)")

    print("\nFor your HIRING AGENT use case, recommended order:")
    print("  1. Chatterbox - Best streaming, MIT license, native MPS")
    print("  2. MeloTTS - Good quality, MIT license, fast CPU inference")
    print("  3. VibeVoice 0.5B - If commercial warning is acceptable")
    print("  4. Seamless - Only if non-commercial or license obtained")


def main():
    print("="*80)
    print("TTS MODEL COMPARISON")
    print("="*80)

    # Device info
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = "CUDA"
    elif torch.backends.mps.is_available():
        device_name = "MPS"
    else:
        try:
            import torch_directml
            device_name = "DirectML (AMD)"
        except:
            pass

    print(f"\nDevice: {device_name}")
    print(f"Test phrase length: {len(TEST_PHRASE)} chars")

    # Check which models are available
    print("\n" + "-"*40)
    print("Checking model availability...")
    print("-"*40)

    models = ["seamless", "vibevoice", "chatterbox", "melotts"]
    model_runners = {
        "seamless": run_seamless,
        "vibevoice": run_vibevoice,
        "chatterbox": run_chatterbox,
        "melotts": run_melotts,
    }

    available_models = []
    for model in models:
        is_available, error = check_model_available(model)
        status = "OK" if is_available else "MISSING"
        print(f"  {model:<12}: {status}")
        if not is_available and error:
            print(f"    -> {error[:60]}...")
        if is_available:
            available_models.append(model)

    if not available_models:
        print("\nNo models available for testing!")
        print("Please install at least one model. See individual test scripts for instructions.")
        return

    print(f"\nWill test {len(available_models)} model(s): {', '.join(available_models)}")

    # Run each available model
    results = []
    for model in models:
        if model in available_models:
            result = model_runners[model]()
        else:
            result = ModelResult(
                model_name=model.title(),
                available=False,
                error_message="Not installed"
            )
        results.append(result)

    # Print comparison
    print_comparison_table(results)
    print_recommendations(results)

    print("\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80)
    print("\nGenerated audio files:")
    for model in models:
        output_file = f"outputs/{model}/compare_output.wav"
        if os.path.exists(output_file):
            print(f"  {output_file}")

    print("\n" + "="*80)
    print("Listen to the 'compare_output.wav' files to subjectively compare quality!")
    print("="*80)


if __name__ == "__main__":
    main()
