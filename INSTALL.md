# CLIFS Installation Guide

This guide provides multiple installation methods for CLIFS.

## Quick Start (Recommended)

### Using pip with pyproject.toml

#### 1. GPU Installation (CUDA 11.8)

```bash
# Clone the repository
git clone https://github.com/DevinW-sudo/CLIFS.git
cd CLIFS

# Download the fine-tuned model
DIR="models/best_mbert_model/best_mbert_model/modern_BERT_fusion_augmented_data_finegrain"
mkdir -p "$DIR"

curl -s https://api.github.com/repos/DevinW-sudo/CLIFS/releases/tags/v0.1.0 \
| jq -r '.assets[].browser_download_url' \
| while read -r url; do
    echo "Downloading $(basename "$url") ..."
    curl -L -o "$DIR/$(basename "$url")" "$url"
  done

# Install with GPU support
pip install -e ".[gpu,ensemble]"

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

#### 2. CPU-Only Installation

```bash
# Clone and download models (same as above)
git clone https://github.com/DevinW-sudo/CLIFS.git
cd CLIFS

# Download models (same commands as GPU version)

# Install with CPU support
pip install -e ".[cpu,ensemble]"

# Download language models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

#### 3. Development Installation

```bash
# Install with all development tools
pip install -e ".[gpu,ensemble,dev]"

# Or for CPU
pip install -e ".[cpu,ensemble,dev]"
```

#### 4. SageMaker Deployment Installation

```bash
# Install with SageMaker dependencies
pip install -e ".[gpu,ensemble,sagemaker]"
```

## Installation Options Reference

The `pyproject.toml` defines several optional dependency groups:

- **`gpu`**: PyTorch with CUDA support, faiss-gpu, cuML
- **`cpu`**: PyTorch CPU-only, faiss-cpu
- **`ensemble`**: OpenAI API client for ensemble mode
- **`dev`**: Development tools (pytest, black, mypy, jupyter)
- **`sagemaker`**: AWS SageMaker deployment dependencies
- **`all`**: Everything (GPU + ensemble + SageMaker)

### Installation Combinations

```bash
# Minimal (no PyTorch - you must install separately)
pip install -e .

# Classification only (GPU)
pip install -e ".[gpu]"

# Classification + Ensemble (GPU)
pip install -e ".[gpu,ensemble]"

# Full installation (GPU + everything)
pip install -e ".[all]"

# Development with CPU
pip install -e ".[cpu,ensemble,dev]"
```

## Legacy Installation (Conda)

If you prefer the original conda-based installation:

### GPU Version

```bash
conda env create -f environment.yml
conda activate clifs
pip install git+https://github.com/huggingface/transformers.git@31ab7168ff7e07f61c90134e5238c4d97606aa70
pip install -e libs/clifs --use-pep517
```

### CPU Version

```bash
conda env create -f environment_cpu.yml
conda activate clifs_cpu
pip install git+https://github.com/huggingface/transformers.git@31ab7168ff7e07f61c90134e5238c4d97606aa70
pip install -e libs/clifs --use-pep517
```

## Verify Installation

```bash
# Check CLIFS can be imported
python -c "from clifs import clifs; print('CLIFS imported successfully')"

# Check CLI is available
clifs --version

# Check models are loaded
python -c "
import torch
from clifs import clifs
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

## Model Files

The fine-tuned ModernBERT model (~500MB) must be downloaded separately:

```bash
# Automated download
DIR="models/best_mbert_model/best_mbert_model/modern_BERT_fusion_augmented_data_finegrain"
mkdir -p "$DIR"

curl -s https://api.github.com/repos/DevinW-sudo/CLIFS/releases/tags/v0.1.0 \
| jq -r '.assets[].browser_download_url' \
| while read -r url; do
    curl -L -o "$DIR/$(basename "$url")" "$url"
  done
```

## Troubleshooting

### CUDA Version Mismatch

If you have a different CUDA version:

```bash
# Check your CUDA version
nvcc --version

# Install PyTorch for your CUDA version
# See: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Transformers Installation Issues

The project requires a specific transformers commit:

```bash
pip install git+https://github.com/huggingface/transformers.git@31ab7168ff7e07f61c90134e5238c4d97606aa70
```

### SpaCy Model Not Found

```bash
python -m spacy download en_core_web_sm
```

### NLTK Data Missing

```bash
python -c "import nltk; nltk.download('punkt')"
```

### FAISS Import Errors

Ensure you installed the correct FAISS variant:

```bash
# For GPU
pip install faiss-gpu

# For CPU
pip install faiss-cpu
```

### Import Errors

Make sure you're in the correct directory when installing:

```bash
# Install from project root
cd /path/to/CLIFS
pip install -e ".[gpu,ensemble]"
```

## System Requirements

### Minimum Requirements

- Python 3.10 or 3.11
- 16GB RAM
- 10GB disk space

### Recommended for GPU

- NVIDIA GPU with 14GB+ VRAM (e.g., RTX 4090, A100)
- CUDA 11.8 or compatible
- 32GB RAM
- 20GB disk space

### CPU-Only Performance

- Expect 10-20x slower inference
- 32GB+ RAM recommended
- Consider using smaller batch sizes

## Next Steps

After installation, see the [README.md](README.md) for usage examples.

### Quick Test

```python
import pandas as pd
from clifs import clifs

# Create test data
df = pd.DataFrame({
    'text': [
        "I love my country and would do anything for it.",
        "I went to that university but didn't enjoy it."
    ]
})

# Run CLIFS
results = clifs.clifs(
    df=df,
    known_groups=["country", "university"],
    save_path='test_results.csv'
)

print(results[['text', 'clifs', 'clifs_fusion_numeric']])
```

### Using the CLI

```bash
# Create a test CSV
echo "text" > test_input.csv
echo "I am proud to be an American. I love my country." >> test_input.csv

# Run classification
clifs --input test_input.csv --output test_output.csv --groups country usa america

# Check results
cat test_output.csv
```