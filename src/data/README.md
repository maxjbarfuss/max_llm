# src/data — Data Pipeline

Production data infrastructure for preprocessing, tokenizing, and serving training datasets.

## Structure

```
src/data/
├── preparation/           First-class dataset preparation framework
│   ├── config.py          Config types + JSON/TOML loading + defaults
│   ├── strategies.py      Readers/mixers/curriculum/split strategies
│   ├── pipeline/          Orchestration (split into focused modules)
│   │   ├── __init__.py    PreparationPipeline orchestrator
│   │   ├── _spill.py      _SpilledDocs + read_and_spill()
│   │   ├── _corpus.py     Tokenizer construction + corpus helpers
│   │   └── _save.py       Output writers, sharding, stats/manifest
│   └── __main__.py        CLI entrypoint
└── datasets/              Dataset-specific processing
    ├── wikitext/          WikiText-103 normalization (see README.md)
    └── tinystories/       TinyStories (Phase 2 placeholder)
```

## Preparation Framework

The first-class path for building training datasets from raw sources.

```bash
python -m src.data.preparation --config <config.json|config.toml>
```

Progress is printed to stdout as four numbered steps; per-source tqdm bars show docs/s and cumulative token count in real time.

### Configuration

Configs are JSON or TOML files.  Top-level sections:

| Section | Description |
|---|---|
| `[tokenizer]` | Tokenizer type, vocab size, character coverage, optional pre-built model path |
| `[[datasets]]` | One entry per source (name, path, format, weight, …) |
| `[mixing]` | How sources are combined (`interleave` or `concatenate`), optional per-source ratios, and optional global doc/token budgets |
| `[splits]` | Train/val/test ratios, shuffle, stratified flag |
| `[output]` | Output directory, file prefix, EOS token, shard size |
| `[curriculum]` | Optional curriculum learning stages (length-based, domain, or custom) |

### Supported source formats

| `format` | What it reads |
|---|---|
| `text` | Plain `.txt` file(s); split on `delimiter` (default `\n\n`) |
| `jsonl` | One JSON object per line; field selected by `text_field` (default `"text"`) |
| `npy` | Pre-tokenized `.npy` uint16/uint32 token array; chunked by `chunk_size` |
| `utf8_tokens` | `.npy` of raw UTF-8 byte values (uint8); decoded then re-tokenized |

### Text cleaning

Every document goes through two cleaning steps before tokenization, applied consistently to both the tokenizer training corpus and the main data path:

1. **NFKC normalization** — collapses fullwidth, ligature, and other compatibility Unicode forms into their canonical equivalents (common in scraped web text and OpenWebText).
2. **Control-character stripping** — removes ASCII and Latin-1 control bytes while preserving `\n`, `\r`, and `\t`.

After tokenization, a **two-stage unk filter** is applied to every sequence (default `max_unk_rate = 2 %`):

| Unk fraction in sequence | Action |
|---|---|
| > `max_unk_rate` | Drop the entire document — too noisy for useful training signal |
| ≤ `max_unk_rate` | Strip individual unk tokens in place |

This prevents the model from ever generating SentencePiece's `⁇` artifact (the "double-?" characters seen in inference output from models trained on noisy web data).  Dropping high-unk documents avoids word-fragment corruption (e.g. `"Gupta" → "upta"`) that arises when individual tokens are stripped from proper nouns; stripping low-unk documents handles isolated stray characters without discarding otherwise clean text.

> **Note — vocab size matters:** With the 1 024-token SentencePiece model used in Phase 3, roughly 63 % of OpenWebText paragraphs exceed the 2 % threshold and are dropped because uncommon capital letters (`G`, `R`, `U`, `V`, `X`, `Z`) have no standalone token.  Upgrading to a ≥ 8 192-token vocabulary covers these characters and reduces the drop rate dramatically.

Pre-tokenized `npy` sources have the same unk filter applied at the array level (always a hard drop of token 0, since there is no text to normalize).

### Token budgets and source caps

`[mixing]` supports either `target_total_docs` or `target_total_tokens`.

- Use `target_total_docs` when your ratios are meant in documents.
- Use `target_total_tokens` with `weight_by = "tokens"` when your ratios are meant in tokens.
- For token-weighted multi-source mixes, set `target_total_tokens` explicitly unless every source has its own `max_tokens` cap. Otherwise the pipeline will size itself to the full readable source inventory.

Each `[[datasets]]` entry may also set `max_docs` or `max_tokens` to stop reading a large source early. This is especially useful for very large corpora such as FineWeb where reading the full source just to downsample later would waste hours.

### OWT hygiene thresholds (text sources)

`[[datasets]]` entries for `format = "text"` can enable extra quality gates used in the Wave 0 OWT hygiene pass:

- `min_punctuation_ended_line_ratio` (0.0-1.0): minimum fraction of non-empty lines that must end with sentence punctuation (`. ! ? ; :`).
- `max_duplicate_line_ratio` (0.0-1.0): maximum allowed duplicate-line share across non-empty lines.
- `max_symbol_to_word_ratio` (>= 0.0): maximum symbol-density allowed, computed as symbol characters divided by alphanumeric word count.

If any configured threshold fails, the document is skipped before tokenization.

### Language filtering

Set top-level `lang_model_path` to a fastText language-ID model such as `lid.176.bin`, then opt individual sources into filtering with `allowed_languages`.

- `lang_model_path`: path to the fastText supervised language-ID model loaded once at pipeline start.
- `allowed_languages`: per-source ISO language allowlist such as `["en"]`.

Language filtering currently applies to text, JSONL, and parquet readers. Very short texts under 50 characters are allowed through to avoid unstable predictions on tiny samples.

### MinHash near-dedup

Use the top-level `[dedup]` table to enable cross-source MinHash LSH deduplication after reading/spill and before mixing:

- `enabled`: turn near-dedup on or off.
- `jaccard_threshold`: approximate Jaccard similarity threshold for duplicate detection.
- `num_perm`: number of MinHash permutations.
- `shingle_size`: contiguous token n-gram size used to build document shingles.

The first document in a near-duplicate cluster is kept and later matches are dropped. Dedup diagnostics record the global drop rate and per-source drop counts.

### Memory model

Sources are streamed to temporary binary spill files under `<output_dir>/.prep_spill/` and backed by read-only memory maps.  Peak RAM is bounded to a single ~10 MB write buffer per source, regardless of corpus size.  Previously-read sources are memory-mapped so the OS can page them out while the next source is being read.  Spill files are removed unconditionally on exit (success or failure).

Output `.npy` files are written via `np.lib.format.open_memmap` — a pre-allocated memory map is filled one document at a time, flushing every 5 M tokens, so the output side also never requires a full in-memory concatenation.

### Output files

| File | Description |
|---|---|
| `<prefix>_train.npy` | Training split (or shards `_train_00000.npy`, …) |
| `<prefix>_val.npy` | Validation split |
| `<prefix>_test.npy` | Test split (if `test > 0`) |
| `<prefix>_stats.json` | Per-split / per-source document and token counts |
| `<prefix>_manifest.json` | Shard paths, shard size, tokenizer path |
| `<prefix>_tokenizer.model` | Trained SentencePiece model (unigram type only) |

Set `shard_size_tokens > 0` to write multiple fixed-size shards instead of a single file per split.

## Datasets

Each dataset directory contains at minimum:
- A normalization or preprocessing script (`normalize.py` or similar)
- A `README.md` documenting the dataset-specific tool and any quirks

### Adding a new dataset

1. Create `src/data/datasets/<name>/`
2. Add `__init__.py`
3. Add `normalize.py` (or equivalent) if the dataset needs preprocessing
4. Add `README.md` — usage, what it fixes, before/after example
