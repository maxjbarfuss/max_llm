# Data Tooling Plan: Production Pipeline for Quality Datasets

**Status**: 🔄 In Progress (Phase 2)
**Purpose**: Comprehensive data pipeline for download, preprocessing, tokenization, and efficient training delivery
**Primary Document**: [PLAN.md](PLAN.md) — Phase 2, Step 3

---

## Overview

Build a production-ready data pipeline for wikitext-103 with pre-tokenization, metadata caching, and intelligent chunked staging from slow→fast storage with background prefetching. Architecture designed for immediate wikitext-103 delivery, then rapid extension to tinystories and future datasets.

---

## Key Design Choices

- **Pre-tokenize once, cache forever**: Tokenize on slow disk, eliminating runtime overhead
- **Parquet metadata**: Columnar format enables efficient queries for chunk ranges, splits, statistics
- **Chunked staging with prefetching**: Copy token chunks (e.g., 50M tokens) to fast drive while training on current chunk. Background thread prefetches next chunk asynchronously
- **Extensible abstractions**: `DatasetDownloader`, `DatasetProcessor`, `ChunkedCache` designed for reuse across wikitext-103 → tinystories → arbitrary HuggingFace datasets

---

## Implementation Steps

### 1. Create data module structure

- ✅ Create `src/data/` package directory
- ✅ Create `src/data/processors/` for dataset-specific preprocessing
  - ✅ Migrate `normalize_wikitext.py` → `src/data/processors/wikitext.py`
- ✅ Create `src/data/cli/` for command-line tools
- Add `src/data/downloader.py`: `HuggingFaceDownloader` class
- Add `src/data/processor.py`: `DatasetProcessor` ABC base class
- Add `src/data/metadata.py`: Parquet schema definitions (`DatasetMetadata`, `ChunkMetadata`, `TokenStatistics`)
- Add `src/data/cache.py`: `ChunkedTokenCache` for slow→fast staging with prefetch
- Add `src/data/dataset.py`: `CachedTokenDataset` (torch Dataset wrapper)

### 2. Implement download & discovery

- `HuggingFaceDownloader.download(dataset_name, cache_dir)`: Use `datasets.load_dataset()` to download to slow disk
- `HuggingFaceDownloader.discover_schema()`: Inspect dataset structure (columns, splits, row counts, sample data)
- Generate discovery report as Parquet: dataset name, splits, column types, row counts, total chars/tokens estimate
- Add `--discover` mode to `src/data/cli/download.py` CLI

### 3. Build pre-processing pipeline

- `DatasetProcessor` ABC with `process(raw_data) → processed_text`
- `WikiTextProcessor` implementation: Integrate existing wikitext normalization logic
- Add quality filters: min/max length, empty removal, encoding validation
- Emit processing statistics: docs processed, filtered, character histograms
- Write to intermediate Parquet on slow disk (row-level: doc_id, text, split, char_count)

### 4. Implement tokenization pipeline

- `Tokenizer.tokenize_dataset()`: Batch-process preprocessed Parquet
- Use existing tokenizer infrastructure (`CharTokenizer` for Phase 2)
- Generate chunked token arrays (numpy int32) on slow disk: `tokens_chunk_000.npy`, `tokens_chunk_001.npy`, ...
- Chunk size configurable (default 50M tokens per chunk)
- Write comprehensive metadata to `metadata.parquet`: chunk_id, token_count, byte_offset, split, sha256 hash

### 5. Create chunked cache manager

- `ChunkedTokenCache(slow_dir, fast_dir, chunk_size_mb)`
- `stage_chunk(chunk_id)`: Copy specific chunk from slow→fast with progress tracking
- `prefetch_next(chunk_id, background=True)`: Async prefetch using `threading.Thread`
- `get_chunk_path(chunk_id) → fast_path`: Returns path to staged chunk (stages if missing)
- LRU eviction on fast drive when space limited (configurable `max_fast_gb`)
- Track staging state in `staging_status.parquet`: chunk_id, staged, last_access, size_mb

### 6. Build PyTorch dataset integration

- `CachedTokenDataset(metadata_path, cache, split, sequence_length)`
- Lazy-load chunks via `ChunkedTokenCache` on `__getitem__`
- Non-overlapping sequence creation (current [train.py](../train.py#L40) logic)
- Prefetch next chunk when 80% through current chunk
- Returns `(input_ids, targets)` for next-token prediction

### 7. Update training loader factory

- Refactor [train.py](../train.py) `create_simple_loaders()` → `create_cached_loaders()`
- Accept `DataConfig` with slow/fast paths, chunk settings
- Initialize `ChunkedTokenCache` and `CachedTokenDataset`
- Start background prefetch thread before returning DataLoader
- Maintain existing DataLoader config: 6 workers, prefetch 2, pin_memory, persistent_workers

### 8. Create CLI tooling

- `python -m src.data.cli.download --dataset wikitext-103-raw-v1 --output /mnt/d/dev/data --discover`
- `python -m src.data.cli.preprocess --input /slow/wikitext --processor wikitext --output /slow/wikitext_processed`
- `python -m src.data.cli.tokenize --input /slow/wikitext_processed --tokenizer char --chunk-size 50M`
- `python -m src.data.cli.inspect --metadata /slow/wikitext_tokens/metadata.parquet`

### 9. Update configurations

- Extend [src/config/data.py](../src/config/data.py) `DataConfig`: Add `slow_storage_dir`, `fast_storage_dir`, `chunk_size_tokens`, `max_fast_storage_gb`, `prefetch_chunks`
- Update [config/data.toml](../config/data.toml) with Phase 2 wikitext-103 paths
- Document slow/fast storage expectations in config comments

### 10. Validate with wikitext-103

- Download wikitext-103-raw-v1 to slow disk
- Preprocess with `WikiTextProcessor`
- Tokenize with `CharTokenizer` into 50M token chunks
- Inspect metadata: verify train/val/test splits, token counts match expectations
- Stage first chunk to fast drive, run short training loop
- Verify prefetch triggers before chunk exhaustion
- Measure: download time, tokenization time, staging time, training throughput

### 11. Extend to tinystories (validation)

- Create `TinyStoriesProcessor` (minimal processing needed)
- Download + tokenize tinystories using same pipeline
- Verify metadata schema accommodates different dataset shape
- Compare tokenization stats: tokens/doc, vocab coverage
- Document any processor-specific quirks

---

## Verification Checklist

- [ ] Run full pipeline: download wikitext-103 → preprocess → tokenize → stage → train for 100 steps
- [ ] Check Parquet metadata with `inspect_metadata.py`: splits present, token counts align with wikitext-103 docs (103M tokens train)
- [ ] Monitor fast disk during training: verify chunk prefetch triggers, old chunks evicted
- [ ] Measure throughput: target >100K tokens/sec on training loop (bound by model, not I/O)
- [ ] Repeat for tinystories: verify same tools work with different dataset structure
- [ ] Unit tests: `ChunkedTokenCache` (staging, prefetch, LRU), `CachedTokenDataset` (sequence creation), processors (filtering logic)

---

## Design Decisions

**Pre-tokenization over on-the-fly**: Trades disk space for training speed and reproducibility. Enables offline preprocessing, chunk-level caching.

**Parquet over JSON/SQLite**: Efficient columnar queries, native pandas/polars integration, compression, schema evolution.

**Chunked staging over full copy**: Reduces fast disk requirements (50M tokens ≈ 200MB vs multi-GB full datasets). Enables prefetching for seamless training.

**Background prefetch thread over blocking**: Training continues while next chunk stages. Simple threading sufficient (no multiprocessing overhead for I/O-bound copy).

**Wikitext-103 MVP with clean interfaces**: Delivers immediate Phase 2 value while establishing patterns for tinystories and OpenWebText (Phase 3).
