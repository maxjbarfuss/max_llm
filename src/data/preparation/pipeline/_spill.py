"""Spill tokenized source documents to disk for memory-efficient pipeline resumption."""

import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import psutil
from tqdm import tqdm

from src.data.preparation.config import DataSource
from src.data.preparation.strategies import FormatReader


class _SpilledDocs:
    """Memory-mapped tokenized documents stored on disk.

    Wraps a flat token mmap and a document-offset array so callers can
    index individual documents without loading the full corpus into RAM.
    """

    def __init__(self, mmap: np.ndarray, offsets: np.ndarray) -> None:
        self.mmap = mmap
        self.offsets = offsets

    def __len__(self) -> int:
        return max(0, len(self.offsets) - 1)

    def __getitem__(self, idx: int) -> np.ndarray:
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        return self.mmap[start:end]

    @property
    def total_tokens(self) -> int:
        return int(self.offsets[-1]) if len(self.offsets) > 0 else 0

    @property
    def dtype(self) -> "np.dtype[Any]":
        return self.mmap.dtype

    @property
    def doc_lengths(self) -> np.ndarray:
        return np.diff(self.offsets).astype(np.int64)

    @property
    def offsets_bytes(self) -> int:
        return int(self.offsets.nbytes)

    @property
    def token_bytes(self) -> int:
        return int(self.mmap.nbytes)


def _cache_paths(bin_path: Path) -> tuple[Path, Path]:
    """Return (offsets_path, meta_path) for a given spill bin path."""
    return (
        bin_path.with_suffix(bin_path.suffix + ".offsets.npy"),
        bin_path.with_suffix(bin_path.suffix + ".meta.json"),
    )


def _load_cached_spill(bin_path: Path) -> _SpilledDocs | None:
    """Load a previously spilled source from disk, or return None on cache miss."""
    offsets_path, meta_path = _cache_paths(bin_path)
    if not (bin_path.exists() and offsets_path.exists() and meta_path.exists()):
        return None
    if bin_path.stat().st_size == 0:
        return None
    try:
        meta = json.loads(meta_path.read_text())
        dtype = np.dtype(meta["dtype"])
        offsets = np.load(str(offsets_path))
        total_tokens = int(offsets[-1])
        if total_tokens == 0:
            return _SpilledDocs(np.array([], dtype=dtype), offsets)
        mmap = np.memmap(str(bin_path), dtype=dtype, mode="r", shape=(total_tokens,))
        return _SpilledDocs(mmap, offsets)
    except Exception:
        return None


def _source_limit_reached(source: DataSource, n_docs: int, n_tokens: int, doc_len: int) -> bool:
    """Return True if the source has hit its configured document or token limit."""
    max_docs: int | None = getattr(source, "max_docs", None)
    if max_docs is not None and n_docs >= max_docs:
        return True
    max_tokens: int | None = getattr(source, "max_tokens", None)
    if max_tokens is not None and n_tokens + doc_len > max_tokens:
        return True
    return False


def _save_spill_cache(bin_path: Path, offsets: list[int], dtype: "np.dtype[Any]") -> None:
    offsets_path, meta_path = _cache_paths(bin_path)
    np.save(str(offsets_path), np.array(offsets, dtype=np.int64))
    meta_path.write_text(json.dumps({"dtype": str(dtype)}))


def read_and_spill(
    source: DataSource,
    reader: FormatReader,
    tokenizer: Any,
    spill_dir: Path,
    diagnostic_output_dir: Path | None = None,
    diagnostic_prefix: str | None = None,
) -> _SpilledDocs:
    """Tokenize all documents from a source into a binary spill file.

    On subsequent calls with the same source name, loads from the on-disk
    cache (bin + offsets + meta) rather than re-tokenizing.
    """
    bin_path = spill_dir / f"{source.name}.bin"
    cached = _load_cached_spill(bin_path)
    if cached is not None:
        print(
            f"  {source.name}: cache hit — {len(cached):,} docs  {cached.total_tokens:,} tokens",
            flush=True,
        )
        return cached

    dtype: np.dtype[Any] = np.dtype(np.uint16)
    offsets: list[int] = [0]
    buffer: list[np.ndarray] = []
    buffered_tokens = 0
    process = psutil.Process()

    import time

    start_time = last_status_time = time.time()

    with tqdm(
        desc=f"  {source.name}",
        unit="doc",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    ) as pbar:
        with open(bin_path, "wb") as f:
            for _, doc in reader.iter_documents(source, tokenizer):
                if _source_limit_reached(source, len(offsets) - 1, offsets[-1], len(doc)):
                    break
                if doc.dtype == np.uint32:
                    dtype = np.dtype(np.uint32)
                arr = doc.astype(dtype, copy=False)
                buffer.append(arr)
                buffered_tokens += len(arr)
                offsets.append(offsets[-1] + len(arr))
                if buffered_tokens >= 5_000_000:
                    f.write(np.concatenate(buffer).tobytes())
                    f.flush()
                    buffer = []
                    buffered_tokens = 0
                pbar.update(1)
                now = time.time()
                if now - last_status_time > 10:
                    tokens = offsets[-1]
                    docs = len(offsets) - 1
                    speed = tokens / max(now - start_time, 1e-6)
                    print(
                        f"[STATUS] {source.name}: {docs:,} docs, {tokens:,} tokens,"
                        f" {speed / 1e3:.2f}k tok/s,"
                        f" {process.memory_info().rss / 1024**2:.1f}MB RAM",
                        flush=True,
                    )
                    last_status_time = now
            if buffer:
                f.write(np.concatenate(buffer).tobytes())
                f.flush()
        cast(Any, pbar).set_postfix(tokens=f"{offsets[-1] / 1e6:.1f}M", done=True)
        pbar.refresh()

    total_tokens = offsets[-1]
    _save_spill_cache(bin_path, offsets, dtype)
    if total_tokens == 0:
        return _SpilledDocs(np.array([], dtype=dtype), np.array([0], dtype=np.int64))
    mmap = np.memmap(str(bin_path), dtype=dtype, mode="r", shape=(total_tokens,))
    return _SpilledDocs(mmap, np.array(offsets, dtype=np.int64))
