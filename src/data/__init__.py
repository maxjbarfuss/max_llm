"""Data pipeline for downloading, preprocessing, tokenizing, and serving datasets.

This module provides a production-ready data infrastructure for:
- Downloading datasets from HuggingFace and other sources
- Preprocessing and normalizing text (wikitext, tinystories, etc.)
- Pre-tokenizing and caching tokens on slow storage
- Chunked staging to fast storage with background prefetching
- PyTorch Dataset integration for efficient training

Key Components:
- downloader: Dataset downloading and discovery
- processors: Dataset-specific preprocessing (wikitext, tinystories)
- metadata: Parquet-based metadata schemas
- cache: Chunked token cache with LRU eviction and prefetching
- dataset: PyTorch Dataset wrapper for cached tokens
- cli: Command-line tools for pipeline operations
"""

__all__ = [
    # Populated as modules are implemented
]
