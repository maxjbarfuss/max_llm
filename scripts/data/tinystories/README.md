# TinyStories Pipeline

Setup for the TinyStories dataset, following the same pattern as WikiText-103.

## Quick Start

After downloading TinyStories, prepare a small subset for testing:

```bash
# Using YAML config (recommended)
python scripts/data/run_data_prep.py --config scripts/data/tinystories/tinystories_utf8_small.yaml

# Or via shell script (convenience wrapper)
scripts/data/tinystories/prepare_tinystories_tokens.sh \
  --source /path/to/tinystories/train.txt \
  --output-dir data/fast \
  --size 100K \
  --mode utf8
```

## Configs

Two standard YAML configurations are provided:

- **`tinystories_utf8_small.yaml`**: Extract 100K tokens for quick iteration
  - Cutoff mode: `article` (respects `<|endoftext|>` boundaries)
  - Suitable for testing pipeline without waiting for large tokenization

- **`tinystories_utf8_full.yaml`**: Extract 5M tokens for realistic training
  - Cutoff mode: `none` (extract at byte limit)
  - Use for Phase 2 validation training runs

## Dataset Details

**TinyStories** uses a simple story-per-line format with `<|endoftext|>` document separators:

```
Once upon a time...
<|endoftext|>
Another story starts here...
<|endoftext|>
...
```

The pipeline automatically detects story boundaries via the **TinyStoriesBoundary** detector (registered in `src/data/datasets/boundary.py`).

### Normalization

No normalization needed for TinyStories (unlike WikiText). The data is already clean.

### Tokenization

Standard character-level tokenization (UTF-8 mode, vocab_size=256) matches Phase 2 requirements.

## Workflow

1. **Extract text subset** (respects story boundaries)
   ```bash
   python -m src.data.pipeline.extract_text \
     --input /path/to/tinystories/train.txt \
     --output data/fast/tinystories_subset.txt \
     --size 2M \
     --dataset tinystories
   ```

2. **Tokenize** (caches as .npy)
   ```bash
   python -m src.data.pipeline.tokenize \
     --input data/fast/tinystories_subset.txt \
     --output data/fast/tinystories_100k_tokens__utf8.npy \
     --vocab-size 256 \
     --mode utf8
   ```

3. **Extract token subset** (if needed)
   ```bash
   python -m src.data.pipeline.extract_tokens \
     --input data/fast/tinystories_100k_tokens__utf8.npy \
     --output data/fast/tinystories_100k_tokens__utf8.npy \
     --size 100K
   ```

All steps are coordinated by `run_data_prep.py` when using a YAML config.

## Phase 2 Integration

To use TinyStories in training:

1. Prepare via YAML:
   ```bash
   python scripts/data/run_data_prep.py --config scripts/data/tinystories/tinystories_utf8_small.yaml
   ```

2. Update `config/experiment.toml` to reference the TinyStories cache:
   ```toml
   [data]
   train_tokens_path = "data/fast/tinystories_100k_tokens__utf8.npy"
   ```

3. Train:
   ```bash
   python -m src.training.train --config config/experiment.toml
   ```

## Download Instructions

TinyStories is publicly available from Hugging Face:

```bash
# Via git-lfs (requires setup)
git clone https://huggingface.co/datasets/roneneldan/TinyStories

# Or programmatically
python -c "
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories')
with open('/path/to/tinystories/train.txt', 'w') as f:
    for example in ds['train']:
        f.write(example['text'] + '\n')
"
```

See `src/data/README.md` for more detail on the full pipeline.
