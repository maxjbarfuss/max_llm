# TinyStories Data Preparation

Automated pipeline for the **karpathy/tinystories-gpt4-clean** dataset. Stories are separated by blank lines and processed with boundary-aware extraction.

## Quick Start

Prepare a tokenized subset:

```bash
python scripts/data/run_data_prep.py \
  --config scripts/data/tinystories/tinystories_utf8_small.yaml
```

This creates `data/fast/tinystories_100k_tokens__utf8.npy` ready for training.

## Available Configs

| Config | Token Count | Use Case |
|--------|-------------|----------|
| `tinystories_utf8_small.yaml` | 100K | Fast iteration and testing |
| `tinystories_utf8_full.yaml` | 5M | Full training runs |

Both configs use `article` cutoff mode to respect story boundaries (complete stories only).

## Dataset Format

Stories are separated by blank lines:

```
Once upon a time, there was a little girl...
She loved to play in the sun.

One day, a boy found a big red ball...
He kicked it high into the sky.

...
```

The `BlankLineBoundary` detector (registered in `src/data/datasets/boundary.py`) ensures extracts contain complete stories only.

## Training Integration

Once prepared, update your experiment config to use the TinyStories cache:

```toml
[data]
dataset_path = "data/fast/tinystories_100k_tokens__utf8.npy"
```

Then train:

```bash
python -m src.training.train --config config/milestones/p2_tinystories_baseline.toml
```

## Dataset Source

Download from Hugging Face:

```bash
# Via datasets library (recommended)
python -c "
from datasets import load_dataset
ds = load_dataset('karpathy/tinystories-gpt4-clean', split='train')
with open('data/raw/tinystories_train.txt', 'w') as f:
    for story in ds:
        f.write(story['story'] + '\n\n')
"
```

Or via git-lfs:

```bash
git clone https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean
```

**Dataset stats:**
- ~2M short stories (3-5 sentences each)
- Cleaned and filtered by GPT-4
- Simple vocabulary suitable for small models
- No normalization required (already clean)

## Manual Pipeline Steps

The YAML config automates these steps. For manual control:

```bash
# 1. Extract text subset (respects story boundaries)
python -m src.data.pipeline.extract_text \
  --input data/raw/tinystories_train.txt \
  --output data/fast/tinystories_subset.txt \
  --size 2M \
  --dataset tinystories

# 2. Tokenize (UTF-8 character-level)
python -m src.data.pipeline.tokenize \
  --input data/fast/tinystories_subset.txt \
  --output data/fast/tinystories_tokens.npy \
  --mode utf8 \
  --vocab-size 256

# 3. Extract token subset
python -m src.data.pipeline.extract_tokens \
  --input data/fast/tinystories_tokens.npy \
  --output data/fast/tinystories_100k_tokens__utf8.npy \
  --size 100K
```

See `src/data/README.md` for pipeline architecture details.
