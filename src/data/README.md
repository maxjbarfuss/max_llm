# src/data — Data Pipeline

Production data infrastructure for preprocessing, tokenizing, and serving training datasets.

## Structure

```
src/data/
├── preparation/      First-class dataset preparation framework
│   ├── config.py     Config types + JSON/TOML loading + defaults
│   ├── strategies.py Readers/mixers/curriculum/split strategies
│   ├── pipeline.py   Orchestration + outputs/stats/manifest
│   └── __main__.py   CLI entrypoint
├── pipeline/          Common, dataset-agnostic pipeline steps
│   ├── tokenize.py    Text → cached .npy token array
│   ├── extract_tokens.py  Slice a token subset from a .npy cache
│   └── extract_text.py    Slice a text subset respecting document boundaries
└── datasets/          Dataset-specific processing
    ├── wikitext/      WikiText-103 normalization (see README.md)
    └── tinystories/   TinyStories (Phase 2 placeholder)
```

For end-to-end dataset preparation, run:

```bash
python -m src.data.preparation --config <config.json|config.toml>
```

The `pipeline/` modules below remain useful as focused building blocks for tokenization and extraction workflows.

## Pipeline Tools

Each tool runs standalone.

### tokenize.py

Convert a normalized text file to a cached `.npy` token array (run once, reuse forever).

```bash
python -m src.data.pipeline.tokenize \
    --input  /slow/wikitext/train_normalized.txt \
    --output /slow/wikitext/train_tokens__utf8.npy \
    --tokenizer char --mode utf8
```

### extract_tokens.py

Slice the first N tokens from a `.npy` cache into a fast-disk subset for training.

```bash
python -m src.data.pipeline.extract_tokens \
    --input  /slow/wikitext/train_tokens__utf8.npy \
    --output data/fast/wikitext_1m__utf8.npy \
    --size 1M
```

Size formats: `100K`, `500K`, `1M`, `5M`, `10M` (plain integers also accepted).

### extract_text.py

Slice a text subset that respects document boundaries (complete articles, stories, etc.).

```bash
# WikiText: extract complete articles
python -m src.data.pipeline.extract_text \
    --input  /slow/wikitext/train_normalized.txt \
    --output data/fast/wikitext_10mb.txt \
    --size 10M --dataset wikitext

# TinyStories: extract complete stories
python -m src.data.pipeline.extract_text \
    --input  /slow/tinystories/train.txt \
    --output data/fast/tinystories_10mb.txt \
    --size 10M --dataset tinystories

# Custom boundary pattern
python -m src.data.pipeline.extract_text \
    --input  data.txt --output subset.txt --size 5M \
    --boundary-pattern "^<\|endoftext\|>"

# No boundary detection (cut at byte limit)
python -m src.data.pipeline.extract_text \
    --input data.txt --output subset.txt --size 5M
```

## Datasets

Each dataset directory contains at minimum:
- A normalization or preprocessing script (`normalize.py` or similar)
- A `README.md` documenting the dataset-specific tool and any quirks

### Adding a new dataset

1. Create `src/data/datasets/<name>/`
2. Add `__init__.py` — document the dataset's boundary pattern for `extract_text.py`
3. Add `normalize.py` (or equivalent) if the dataset needs preprocessing
4. Add `README.md` — usage, what it fixes, before/after example
5. Add YAML configs in `scripts/data/<name>/` for the YAML-driven workflow

### Dataset boundary patterns

`extract_text.py` resolves document boundaries via the `--dataset` flag (preferred) or `--boundary-pattern` (custom regex). Detectors are registered in `src/data/datasets/boundary.py`.

| Dataset | Flag | Boundary |
|---|---|---|
| WikiText-103 | `--dataset wikitext` | Top-level article headers (`= Title =`) |
| TinyStories | `--dataset tinystories` | Story separator (`<\|endoftext\|>`) |
| Custom | `--boundary-pattern REGEX` | Arbitrary regex, takes precedence over `--dataset` |
| None | _(omit both)_ | No detection; cut at byte limit |

To add a new dataset: implement a `BoundaryDetector` subclass and register it in `_REGISTRY` in `boundary.py`.
