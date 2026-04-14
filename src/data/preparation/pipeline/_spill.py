"""Memory-efficient document spilling to disk via mmap."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.data.preparation.config import DataSource
from src.data.preparation.strategies import FormatReader

logger = logging.getLogger(__name__)


class _SpilledDocs:
    """Memory-mapped sequence of tokenized documents backed by a temp binary file."""

    def __init__(self, mmap: np.ndarray, offsets: np.ndarray) -> None:
        self._mmap = mmap  # shape: (total_tokens,)
        self._offsets = offsets  # shape: (n_docs + 1,) int64

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def __getitem__(self, idx: int) -> np.ndarray:
        start = int(self._offsets[idx])
        end = int(self._offsets[idx + 1])
        return self._mmap[start:end]

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self[i]

    @property
    def total_tokens(self) -> int:
        return int(self._offsets[-1])


def _source_limit_reached(
    source: DataSource, doc_count: int, current_tokens: int, doc_len: int
) -> bool:
    """Return True if any configured source limit has been hit."""
    if source.max_docs is not None and doc_count >= source.max_docs:
        return True
    if source.max_tokens is not None:
        if current_tokens >= source.max_tokens:
            return True
        if current_tokens > 0 and current_tokens + doc_len > source.max_tokens:
            return True
    return False


def _cache_paths(bin_path: Path) -> tuple[Path, Path]:
    """Return (offsets_path, meta_path) for a given spill .bin file."""
    return bin_path.with_suffix(".bin.offsets.npy"), bin_path.with_suffix(".bin.meta.json")


def _load_cached_spill(bin_path: Path) -> _SpilledDocs | None:
    """Return a _SpilledDocs loaded from disk cache, or None if cache is absent/incomplete."""
    offsets_path, meta_path = _cache_paths(bin_path)
    if not (bin_path.exists() and offsets_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        dtype = np.dtype(meta["dtype"])
        offsets = np.load(str(offsets_path))
        n_docs = len(offsets) - 1
        total_tokens = int(offsets[-1])
        if n_docs == 0:
            # Empty cache from a failed/interrupted run — treat as missing so we re-tokenize.
            logger.warning("Spill cache for %s has 0 docs — discarding and re-tokenizing", bin_path)
            return None
        mmap = np.memmap(str(bin_path), dtype=dtype, mode="r", shape=(total_tokens,))
        return _SpilledDocs(mmap, offsets)
    except Exception as exc:
        logger.warning("Failed to load spill cache for %s: %s — will re-tokenize", bin_path, exc)
        return None


def _save_spill_cache(bin_path: Path, offsets: list[int], dtype: np.dtype[Any]) -> None:
    """Persist offsets and dtype alongside the .bin so future runs can skip tokenization."""
    offsets_path, meta_path = _cache_paths(bin_path)
    np.save(str(offsets_path), np.array(offsets, dtype=np.int64))
    meta_path.write_text(json.dumps({"dtype": str(dtype)}))


def read_and_spill(
    source: DataSource,
    reader: FormatReader,
    tokenizer: Any,
    spill_dir: Path,
) -> _SpilledDocs:
    """Stream source docs to a temp binary file; peak RAM is bounded to one flush buffer (~10 MB).

    If a complete cache (*.bin + *.bin.offsets.npy + *.bin.meta.json) already exists in
    *spill_dir*, tokenization is skipped and the cached data is returned immediately.
    """
    bin_path = spill_dir / f"{source.name}.bin"

    cached = _load_cached_spill(bin_path)
    if cached is not None:
        logger.info(
            "Spill cache hit: name=%s docs=%d tokens=%d",
            source.name,
            len(cached),
            cached.total_tokens,
        )
        print(
            f"  {source.name}: cache hit — {len(cached):,} docs  {cached.total_tokens:,} tokens",
            flush=True,
        )
        return cached

    import time

    import psutil

    dtype: np.dtype[np.unsignedinteger] = np.dtype(np.uint16)  # sufficient for vocab ≤ 65535
    offsets: list[int] = [0]
    buffer: list[np.ndarray] = []
    buffered_tokens = 0
    last_status_time = time.time()
    last_tokens = 0
    last_docs = 0
    last_flush_time = time.time()
    stall_warned = False
    N_TOKENS_STATUS = 1_000_000
    process = psutil.Process()
    start_time = time.time()

    with tqdm(
        desc=f"  {source.name}",
        unit="doc",
        unit_scale=True,
        dynamic_ncols=True,
    ) as pbar:
        with open(bin_path, "wb") as f:
            for _, doc in reader.iter_documents(source, tokenizer):
                if _source_limit_reached(source, len(offsets) - 1, offsets[-1], len(doc)):
                    break

                if doc.dtype == np.uint32:
                    dtype = np.dtype(np.uint32)
                coerced = doc.astype(dtype, copy=False)
                buffer.append(coerced)
                buffered_tokens += len(coerced)
                offsets.append(offsets[-1] + len(coerced))

                if buffered_tokens >= 5_000_000:
                    f.write(np.concatenate(buffer).tobytes())
                    f.flush()
                    buffer = []
                    buffered_tokens = 0
                    last_flush_time = time.time()

                pbar.update(1)
                now = time.time()
                # Status every N tokens or 10s
                if (offsets[-1] // N_TOKENS_STATUS) > (last_tokens // N_TOKENS_STATUS) or (
                    now - last_status_time
                ) > 10:
                    elapsed = now - start_time
                    tokens = offsets[-1]
                    docs = len(offsets) - 1
                    speed = (tokens - last_tokens) / max(now - last_status_time, 1e-6)
                    docs_speed = (docs - last_docs) / max(now - last_status_time, 1e-6)
                    mem = process.memory_info().rss / 1024**2
                    io_counters = process.io_counters()
                    eta = (tokens and (tokens / max(tokens / elapsed, 1e-6))) or 0
                    pbar.set_postfix(
                        tokens=f"{tokens / 1e6:.1f}M",
                        docs=f"{docs:,}",
                        speed=f"{speed/1e3:.2f}k/s",
                        docs_speed=f"{docs_speed:.1f}/s",
                        mem=f"{mem:.1f}MB",
                        io_write=f"{io_counters.write_bytes/1024**2:.1f}MB",
                        eta=f"{eta/60:.1f}m",
                    )
                    print(
                        f"[STATUS] {source.name}: {docs:,} docs, {tokens:,} tokens, {speed/1e3:.2f}k tok/s, {mem:.1f}MB RAM, {io_counters.write_bytes/1024**2:.1f}MB written, ETA {eta/60:.1f}m",
                        flush=True,
                    )
                    last_status_time = now
                    last_tokens = tokens
                    last_docs = docs
                    stall_warned = False
                # Stall warning if no progress for 60s
                if not stall_warned and (now - last_status_time) > 60:
                    print(
                        f"[WARNING] No progress for 60s in {source.name} (docs={len(offsets)-1:,}, tokens={offsets[-1]:,})",
                        flush=True,
                    )
                    stall_warned = True
                # Always flush tqdm
                pbar.refresh()

            if buffer:
                f.write(np.concatenate(buffer).tobytes())
                f.flush()
            pbar.set_postfix(tokens=f"{offsets[-1] / 1e6:.1f}M", done=True)
            pbar.refresh()

    total_tokens = offsets[-1]
    _save_spill_cache(bin_path, offsets, dtype)

    if total_tokens == 0:
        return _SpilledDocs(
            np.array([], dtype=dtype),
            np.array([0], dtype=np.int64),
        )

    mmap = np.memmap(str(bin_path), dtype=dtype, mode="r", shape=(total_tokens,))
    return _SpilledDocs(mmap, np.array(offsets, dtype=np.int64))
