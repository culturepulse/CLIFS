# CLIFS Quick Start Guide

## Installation

### Option 1: pip (Recommended)

```bash
# GPU version
pip install -e ".[gpu,ensemble]"

# CPU version
pip install -e ".[cpu,ensemble]"
```

### Option 2: conda (Original)

```bash
conda env create -f environment.yml
conda activate clifs
pip install git+https://github.com/huggingface/transformers.git@31ab7168ff7e07f61c90134e5238c4d97606aa70
pip install -e libs/clifs --use-pep517
```

## Download Model Files

```bash
DIR="models/best_mbert_model/best_mbert_model/modern_BERT_fusion_augmented_data_finegrain"
mkdir -p "$DIR"

curl -s https://api.github.com/repos/DevinW-sudo/CLIFS/releases/tags/v0.1.0 \
| jq -r '.assets[].browser_download_url' \
| while read -r url; do
    curl -L -o "$DIR/$(basename "$url")" "$url"
  done
```

## Usage

### Python API

#### 1. Basic Classification (Fast)

```python
import pandas as pd
from clifs import clifs

# Load your data
df = pd.DataFrame({
    'text': [
        "I am proud to be an American. I love my country deeply.",
        "I attended that university but had a terrible experience."
    ]
})

# Run CLIFS
results = clifs.clifs(
    df=df,
    known_groups=["country", "america", "university"],
    save_path='results.csv'
)

# Check predictions
print(results[['text', 'clifs', 'clifs_fusion_numeric']])
# clifs column: 'low', 'medium', or 'high'
# clifs_fusion_numeric: 0=high, 1=low, 2=medium
```

#### 2. Regression (Continuous Scores)

```python
results = clifs.clifs(
    df=df,
    known_groups=["country", "america"],
    regression=True,
    save_path='regression_results.csv'
)

# Check continuous fusion scores
print(results[['text', 'clifs_fusion_r']])
# clifs_fusion_r: continuous score (typically 1-7)
```

#### 3. Ensemble Mode (Best Accuracy, Slower)

```python
# Requires OpenAI and DeepSeek API keys
# You'll be prompted to enter them

results = clifs.clifs(
    df=df,
    known_groups=["country", "america"],
    ensemble=True,
    save_path='ensemble_results.csv'
)

# Multiple predictions available
print(results[['clifs', 'sbert_rf', 'rag_4o', 'rag_r1', 'clifs_fusion']])
# clifs_fusion: hard voting result from all models
```

### Command Line Interface

#### Basic Usage

```bash
# Create input CSV
cat > input.csv << EOF
text
"I love my country and would die for it."
"I went to that school but hated it."
EOF

# Run classification
clifs --input input.csv --output results.csv --groups country school

# Check results
cat results.csv
```

#### Advanced Options

```bash
# Regression mode
clifs -i input.csv -o output.csv -r --groups religion church

# Ensemble mode
clifs -i input.csv -o output.csv -e --groups political party

# Custom text column
clifs -i data.csv -o results.csv --text-column "essay" --groups college

# Get help
clifs --help
```

## Output Columns

### Classification Mode

- `clifs`: Prediction label ('low', 'medium', 'high')
- `clifs_fusion_numeric`: Numeric encoding (0=high, 1=low, 2=medium)

### Regression Mode

- `clifs_fusion_r`: Continuous fusion score (1.0 - 7.0 scale)

### Ensemble Mode

- `sbert_rf`: SBERT Random Forest prediction
- `clifs`: CLIFS Random Forest prediction
- `rag_4o`: RAG with GPT-4o prediction
- `rag_r1`: RAG with DeepSeek R1 prediction
- `clifs_fusion`: Final ensemble prediction (hard voting)
- `clifs_fusion_numeric`: Numeric encoding

## Known Groups Parameter

The `known_groups` parameter is optional but **highly recommended**:

```python
# Generic groups (less accurate)
results = clifs.clifs(df=df)

# Specific groups (better accuracy)
results = clifs.clifs(
    df=df,
    known_groups=["religion", "church", "god", "faith"]
)
```

**Tips:**
- Include synonyms and related terms
- System automatically expands these using word embeddings
- More specific groups = better accuracy

## Performance Notes

### Speed (per sample, GPU)
- **Classification**: 2-5 seconds
- **Regression**: 2-5 seconds
- **Ensemble**: 15-30 seconds (API calls)

### Speed (per sample, CPU)
- **Classification**: 20-50 seconds
- **Regression**: 20-50 seconds
- **Ensemble**: 20-40 seconds (mostly API wait time)

### Memory Requirements
- **GPU**: 14GB+ VRAM
- **CPU**: 16GB+ RAM
- **Disk**: 2GB for models

## Common Issues

### "Module not found"
```bash
pip install -e ".[gpu,ensemble]"
python -m spacy download en_core_web_sm
```

### "CUDA out of memory"
- Use CPU version
- Process smaller batches
- Close other GPU applications

### "API key required"
Ensemble mode requires:
- OpenAI API key ([get here](https://platform.openai.com/api-keys))
- DeepSeek API key ([get here](https://www.deepseek.com/))

### "Model files not found"
```bash
# Re-download model files
DIR="models/best_mbert_model/best_mbert_model/modern_BERT_fusion_augmented_data_finegrain"
mkdir -p "$DIR"
# ... (see installation section)
```

## Example: Real Data Analysis

```python
import pandas as pd
from clifs import clifs

# Load real data
df = pd.read_csv('survey_responses.csv')
# Must have 'text' column with essay responses

# Define context-specific groups
GROUPS = [
    "america", "usa", "country", "nation",  # National identity
    "religion", "church", "god", "faith",   # Religious identity
    "university", "college", "school"       # Educational identity
]

# Run analysis
print(f"Analyzing {len(df)} texts...")
results = clifs.clifs(
    df=df,
    known_groups=GROUPS,
    save_path='fusion_analysis.csv'
)

# Summary statistics
print("\nFusion Distribution:")
print(results['clifs'].value_counts())

# High fusion examples
high_fusion = results[results['clifs'] == 'high']
print(f"\nFound {len(high_fusion)} high fusion texts")
print(high_fusion[['text', 'clifs']].head())
```

## Next Steps

- Read full documentation: [README.md](README.md)
- See installation details: [INSTALL.md](INSTALL.md)
- Check API reference: [libs/clifs/clifs/clifs.py](libs/clifs/clifs/clifs.py)
- Review paper: EMNLP 2025 (citation in README)

## Citation

If you use CLIFS in your research:

```bibtex
@inproceedings{wright2025clifs,
  title={Cognitive Linguistic Identity Fusion Score (CLIFS): A Scalable Cognition-Informed Approach to Quantifying Identity Fusion from Text},
  author={Wright, Devin R. and An, Jisun and Ahn, Yong-Yeol},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2025},
  note={In press}
}
```