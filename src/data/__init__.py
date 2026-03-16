"""Data pipeline for downloading, preprocessing, tokenizing, and serving datasets.

Structure:
- pipeline/: Dataset-agnostic pipeline steps (tokenize, extract_tokens, extract_text)
- datasets/: Dataset-specific processing (wikitext/, tinystories/, ...)
- preparation/: First-class dataset preparation framework for mixed-source
        tokenization, splitting, and export.

Planned components (Phase 3+):
- downloader: HuggingFace dataset downloading and schema discovery
- metadata: Parquet-based metadata schemas
- cache: Chunked token cache with LRU eviction and prefetching (Phase 4)
- dataset: PyTorch Dataset wrapper for cached tokens (Phase 4)
"""

__all__: list[str] = []
