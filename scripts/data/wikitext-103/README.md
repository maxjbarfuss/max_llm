# WikiText-103 Data Preparation

Automated pipeline for the **WikiText-103** dataset. Articles are identified by top-level headers and processed with normalization to fix formatting artifacts.

## Quick Start

Prepare a tokenized subset:

```bash
python scripts/data/run_data_prep.py \
  --config scripts/data/wikitext-103/wikitext-103_utf8_small.yaml
```

This creates `data/fast/wikitext_100k_tokens__utf8.npy` ready for training.

## Available Configs

| Config | Token Count | Tokenizer | Use Case |
|--------|-------------|-----------|----------|
| `wikitext-103_utf8_small.yaml` | 100K | UTF-8 (256 vocab) | Fast iteration and testing |
| `wikitext-103_utf8_full.yaml` | 5M | UTF-8 (256 vocab) | Full training runs |
| `wikitext-103_default_ascii_small.yaml` | 100K | ASCII (128 vocab) | ASCII-only experiments |
| `wikitext-103_default_ascii.yaml` | 5M | ASCII (128 vocab) | ASCII baseline training |

Small configs use `article` cutoff mode to extract complete articles. Full configs use `none` mode for maximum token coverage.

## Dataset Format

Articles are marked by top-level headers (single `=` on each side):

```
 = Valkyria Chronicles III =

Valkyria Chronicles III is a tactical role-playing game...
The game was released in 2011.

 == Gameplay ==

The game features turn-based combat...

 = Another Article =

Another article starts here...
```

The `WikiTextBoundary` detector (registered in `src/data/datasets/boundary.py`) recognizes the pattern `^\s*=\s+[^=]+\s+=\s*$` to identify article boundaries.

## Normalization

WikiText-103 requires normalization to fix formatting artifacts:

- **@ tokens**: `@-@` → `-`, `@.@` → `.`
- **Header spacing**: `= = Title = =` → `== Title ==`
- **Punctuation spacing**: Removes spaces before `,`, `.`, `!`, `?`, etc.
- **Quote spacing**: Fixes spacing around quotation marks
- **Extra newlines**: Collapses 3+ consecutive newlines to 2

Run normalization manually:

```bash
python -m src.data.datasets.wikitext.normalize \
  data/raw/wikitext_train.txt \
  data/raw/wikitext_train_normalized.txt
```

Or enable in YAML config with `workflow.normalize: true`.

## Training Integration

Once prepared, update your experiment config to use the WikiText cache:

```toml
[data]
dataset_path = "data/fast/wikitext_100k_tokens__utf8.npy"
```

Then train:

```bash
python -m src.training.train --config config/experiment.toml
```

## Dataset Source

Download from Hugging Face:

```bash
# Via datasets library (recommended)
python -c "
from datasets import load_dataset
ds = load_dataset('cchoi1022/wikitext-103-v1', split='train')
with open('data/raw/wikitext_train.txt', 'w') as f:
    f.write(ds[0]['text'])
"
```

Or via git-lfs:

```bash
git clone https://huggingface.co/datasets/cchoi1022/wikitext-103-v1
```

**Dataset stats:**
- ~1.7M articles from Wikipedia (2010 snapshot)
- ~103M tokens (word-level)
- Pre-split train/validation/test sets
- Requires normalization before tokenization

## Manual Pipeline Steps

The YAML config automates these steps. For manual control:

```bash
# 1. Normalize (fix WikiText artifacts)
python -m src.data.datasets.wikitext.normalize \
  data/raw/wikitext_train.txt \
  data/raw/wikitext_train_normalized.txt

# 2. Extract text subset (respects article boundaries)
python -m src.data.pipeline.extract_text \
  --input data/raw/wikitext_train_normalized.txt \
  --output data/fast/wikitext_subset.txt \
  --size 2M \
  --dataset wikitext

# 3. Tokenize (UTF-8 character-level)
python -m src.data.pipeline.tokenize \
  --input data/fast/wikitext_subset.txt \
  --output data/fast/wikitext_tokens.npy \
  --mode utf8 \
  --vocab-size 256

# 4. Extract token subset
python -m src.data.pipeline.extract_tokens \
  --input data/fast/wikitext_tokens.npy \
  --output data/fast/wikitext_100k_tokens__utf8.npy \
  --size 100K
```

## Normalization Examples

Before normalization:
```
= = Career = =

He won the championship in 2003 @-@ 2004 season .
The score was 3 @.@ 14 seconds .
```

After normalization:
```
== Career ==

He won the championship in 2003-2004 season.
The score was 3.14 seconds.
```

See `src/data/datasets/wikitext/normalize.py` for full normalization logic.
