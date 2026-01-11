# TTS Comparison Project - Implementation Summary
# made by arnav c
## Overview



### 1. **testmodels/test_melotts.py** (NEW FILE)
- Complete MeloTTS wrapper following established patterns
- Supports multi-accent English (EN-US, EN-BR, EN-INDIA, EN-AU)
- Parameter tuning: speed, noise_scale, noise_scale_w, sdp_ratio
- DirectML support built-in for AMD GPUs
- Recommended settings: EN-US, speed=1.0, noise_scale=0.6

### 2. **testmodels/test_vibevoice.py** (MAJOR REWRITE)
- Migrated from VibeVoice 1.5B to 0.5B Realtime streaming model
- Single-speaker only: Carter (male voice)
- New streaming API: `VibeVoiceStreamingForConditionalGenerationInference`
- Parameters: cfg_scale (1.0-2.0), ddpm_steps (3-10)
- Voice preset loading with multi-path search
- DirectML support for AMD GPUs
- Device-specific optimizations (Flash Attention 2 for CUDA, SDPA for MPS/CPU)

### 3. **compare_outputs.py** (SIMPLIFIED)
- Generates ONE audio file per model (not multiple)
- Single test phrase: "Hello, thank you for applying to this position..."
- Added MeloTTS integration
- Updated VibeVoice to use 0.5B Realtime API
- DirectML support in Chatterbox runner
- Outputs: `outputs/{model}/compare_output.wav`
- Simple comparison table with RTF metrics

### 4. **testmodels/test_chatterbox.py** (MINOR UPDATE)
- Updated `get_device()` function with DirectML support
- Cross-platform device detection: CUDA > DirectML > MPS > CPU

## Installation Instructions

### MeloTTS
```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
python -m unidic download
```

### VibeVoice Realtime 0.5B
```bash
git clone https://github.com/vibevoice-community/VibeVoice.git
cd VibeVoice
pip install -e .
huggingface-cli download microsoft/VibeVoice-Realtime-0.5B --local-dir ./checkpoints/0.5B
```

### Chatterbox
```bash
pip install chatterbox-tts
```

### Seamless M4T v2
```bash
pip install transformers sentencepiece
# Model downloads automatically on first run (9.24 GB)
```

### DirectML (Optional - AMD GPU on Windows)
```bash
pip install torch-directml
```

## Usage

### Test Individual Models

```bash
# Test MeloTTS
python3 testmodels/test_melotts.py

# Test VibeVoice 0.5B Realtime
python3 testmodels/test_vibevoice.py

# Test Chatterbox
python3 testmodels/test_chatterbox.py

# Test Seamless
python3 testmodels/test_seamless.py
```

### Run Complete Comparison

```bash
python3 compare_outputs.py
```

This will:
- Check which models are installed
- Generate ONE audio file per model
- Display comparison table with timing metrics
- Save outputs to `outputs/{model}/compare_output.wav`

## Platform Support

### Mac (Apple Silicon)
- **Chatterbox**: MPS acceleration (native support)
- **MeloTTS**: MPS acceleration
- **VibeVoice 0.5B**: MPS with float32 + SDPA
- **Seamless**: CPU only (MPS has known issues)

### Windows ThinkPad (AMD Radeon 790M)
- **All models**: DirectML support (install `torch-directml`)
- **Fallback**: CPU mode if DirectML not available
- **Detection**: Automatic via enhanced `get_device()` function

### NVIDIA GPU
- **All models**: CUDA support
- **VibeVoice**: Flash Attention 2 optimization (bfloat16)

## Output Files

After running `compare_outputs.py`, you'll have:

```
outputs/
├── chatterbox/
│   └── compare_output.wav
├── melotts/
│   └── compare_output.wav
├── vibevoice/
│   └── compare_output.wav
└── seamless/
    └── compare_output.wav
```

Listen to these files to subjectively compare audio quality!

## Recommended Settings (Per Model)

### Chatterbox
- `exaggeration=0.3` (moderate expressiveness)
- `cfg_weight=0.5` (balanced guidance)
- `temperature=0.7` (slight variation)

### MeloTTS
- `speaker_id='EN-US'` (American English)
- `speed=1.0` (normal speed)
- `noise_scale=0.6` (moderate expressiveness)

### VibeVoice 0.5B Realtime
- `speaker_name='Carter'` (only available speaker)
- `cfg_scale=1.5` (balanced text faithfulness)
- `ddpm_steps=5` (good quality/speed tradeoff)

### Seamless M4T v2
- `speaker_id=1`
- `temperature=0.7`
- `do_sample=True`

## License Summary

- **Chatterbox**: MIT ✅ (enterprise-friendly)
- **MeloTTS**: MIT ✅ (enterprise-friendly)
- **VibeVoice**: MIT ⚠️ (but developers warn against commercial use)
- **Seamless**: CC-BY-NC ❌ (non-commercial only)

## For Mucognitron Integration

**Recommended prioritization:**

1. **Chatterbox** - Best for real-time streaming, MIT license, native MPS
2. **MeloTTS** - Good quality, MIT license, fast CPU inference
3. **VibeVoice 0.5B** - Only if commercial warning is acceptable
4. **Seamless** - Only if non-commercial or license obtained

## Performance Metrics

Run `compare_outputs.py` to see actual metrics on your hardware. Expected RTF (Real-Time Factor):

- **RTF < 1.0** = Faster than real-time (good for streaming)
- **RTF = 1.0** = Exactly real-time
- **RTF > 1.0** = Slower than real-time (not suitable for streaming)

## Troubleshooting

### VibeVoice voice preset not found
Ensure the VibeVoice repository's `demo/voices/streaming_model/en-Carter.pt` file is accessible. The code searches multiple paths automatically.

### DirectML not working
- DirectML only works on Windows
- Install: `pip install torch-directml`
- Falls back to CPU automatically if unavailable

### Seamless downloading slowly
The model is 9.24 GB and downloads on first run. This is a one-time operation.

### MeloTTS unidic error
Run: `python -m unidic download`

## Next Steps for Testing

1. **Mac (Current System)**:
   ```bash
   python3 compare_outputs.py
   ```
   Expected: MPS acceleration for Chatterbox, MeloTTS, VibeVoice

2. **ThinkPad (When Available)**:
   ```bash
   pip install torch-directml
   python3 compare_outputs.py
   ```
   Expected: DirectML detection for AMD Radeon 790M

3. **Listen to outputs**:
   Compare `outputs/*/compare_output.wav` files for subjective quality

4. **Parameter tuning** (optional):
   Run individual `test_*.py` scripts to experiment with different settings

## Implementation Complete! 🎉

All 4 enhancements have been successfully implemented:
- ✅ MeloTTS integration
- ✅ VibeVoice 0.5B Realtime migration
- ✅ DirectML cross-platform support
- ✅ Simplified comparison workflow

The project is now ready for comprehensive TTS model comparison on both Mac and Windows ThinkPad!
