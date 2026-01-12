# Windows Installation Guide for MeloTTS

When installing MeloTTS on Windows, you may encounter build errors for `fugashi` and `tokenizers`. These packages have native components (C/Rust) that require compilers.

## Quick Fix (Recommended)

### Step 1: Upgrade pip first
```bash
pip install --upgrade pip setuptools wheel
```

### Step 2: Install tokenizers from pre-built wheel
```bash
pip install tokenizers
```
This usually works if pip is up-to-date, as pre-built wheels are available on PyPI.

### Step 3: Install fugashi with pre-built wheel
```bash
pip install fugashi[unidic-lite]
```
Pre-built wheels are available for most Python versions (3.8-3.12).

### Step 4: Install MeloTTS
```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
python -m unidic download
```

---

## If Pre-built Wheels Don't Work

### For `tokenizers` (Rust required)

**Option A: Install Rust**
1. Download and install Rust from https://rustup.rs
2. Restart your terminal
3. Retry the installation

**Option B: Use conda**
```bash
conda install -c huggingface tokenizers
```

### For `fugashi` (C++ compiler required)

**Option A: Install Visual C++ Build Tools**
1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer
3. Select "Desktop development with C++"
4. Install and restart your terminal
5. Retry the installation

**Option B: Use conda**
```bash
conda install -c conda-forge fugashi
```

---

## Alternative: Use Conda Environment (Easiest)

Using conda/mamba handles all compiler dependencies automatically:

```bash
# Create conda environment
conda create -n melotts python=3.10
conda activate melotts

# Install build dependencies
conda install -c conda-forge fugashi unidic-lite
conda install -c huggingface tokenizers

# Install MeloTTS
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
python -m unidic download
```

---

## Troubleshooting

### Error: "Microsoft Visual C++ 14.0 or greater is required"
- Install Visual C++ Build Tools (link above)
- Or use conda to bypass compilation

### Error: "can't find Rust compiler"
- Install Rust from https://rustup.rs
- Or upgrade pip and let it find pre-built wheels: `pip install --upgrade pip`

### Error: "Failed building wheel"
- Try installing the package separately first before MeloTTS
- Use `--no-build-isolation` flag: `pip install --no-build-isolation fugashi`

### Still having issues?
- Use Python 3.10 or 3.11 (best wheel availability)
- Consider using WSL (Windows Subsystem for Linux) for easier installation
