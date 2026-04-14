# --- Imports ---
import sys
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from tqdm import tqdm

# If DataSource and FormatReader are project-specific, import from their modules
try:
    from src.data.inference.reader import DataSource, FormatReader
except ImportError:
    DataSource = Any
    FormatReader = Any


# Minimal _SpilledDocs implementation
class _SpilledDocs:
    def __init__(self, mmap: np.ndarray, offsets: np.ndarray):
        self.mmap = mmap
        self.offsets = offsets

    def __len__(self):
        return len(self.offsets) - 1

    @property
    def total_tokens(self):
        return int(self.offsets[-1]) if len(self.offsets) > 0 else 0


# Minimal cache loader (no-op, always miss)
def _load_cached_spill(bin_path: Path):
    return None


# Minimal source limit checker (never limit)
def _source_limit_reached(source, n_docs, n_tokens, doc_len):
    return False


def _save_spill_cache(bin_path: Path, offsets: list[int], dtype: np.dtype[Any]) -> None:
    offsets_path = bin_path.with_suffix(bin_path.suffix + ".offsets.npy")
    meta_path = bin_path.with_suffix(bin_path.suffix + ".meta.json")
    np.save(str(offsets_path), np.array(offsets, dtype=np.int64))
    import json

    meta_path.write_text(json.dumps({"dtype": str(dtype)}))


# Minimal _SpilledDocs implementation
class _SpilledDocs:
    def __init__(self, mmap: np.ndarray, offsets: np.ndarray):
        self.mmap = mmap
        self.offsets = offsets

    def __len__(self):
        return len(self.offsets) - 1

    @property
    def total_tokens(self):
        return int(self.offsets[-1]) if len(self.offsets) > 0 else 0


# Minimal cache loader (no-op, always miss)
def _load_cached_spill(bin_path: Path):
    return None


# Minimal source limit checker (never limit)
def _source_limit_reached(source, n_docs, n_tokens, doc_len):
    return False


def read_and_spill(
    source: DataSource,
    reader: FormatReader,
    tokenizer: Any,
    spill_dir: Path,
    diagnostic_output_dir: Path | None = None,
    diagnostic_prefix: str | None = None,
) -> _SpilledDocs:
    bin_path = spill_dir / f"{source.name}.bin"
    cached = _load_cached_spill(bin_path)
    if cached is not None:
        print(
            f"  {source.name}: cache hit — {len(cached):,} docs  {cached.total_tokens:,} tokens",
            flush=True,
        )
        return cached
    dtype = np.dtype(np.uint16)
    offsets = [0]
    buffer = []
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

                    meta_path.write_text(json.dumps({"dtype": str(dtype)}))

                def read_and_spill(
                    source: DataSource,
                    reader: FormatReader,
                    tokenizer: Any,
                    spill_dir: Path,
                    diagnostic_output_dir: Path | None = None,
                    diagnostic_prefix: str | None = None,
                ) -> _SpilledDocs:
                    bin_path = spill_dir / f"{source.name}.bin"
                    cached = _load_cached_spill(bin_path)
                    if cached is not None:
                        print(
                            f"  {source.name}: cache hit — {len(cached):,} docs  {cached.total_tokens:,} tokens",
                            flush=True,
                        )
                        return cached
                    dtype = np.dtype(np.uint16)
                    offsets = [0]
                    buffer = []
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
                                if _source_limit_reached(
                                    source, len(offsets) - 1, offsets[-1], len(doc)
                                ):
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
                                        f"[STATUS] {source.name}: {docs:,} docs, {tokens:,} tokens, {speed/1e3:.2f}k tok/s, {process.memory_info().rss/1024**2:.1f}MB RAM",
                                        flush=True,
                                    )
                                    last_status_time = now
                        if buffer:
                            f.write(np.concatenate(buffer).tobytes())
                            f.flush()
                        pbar.set_postfix(tokens=f"{offsets[-1] / 1e6:.1f}M", done=True)
                        pbar.refresh()
                    total_tokens = offsets[-1]
                    _save_spill_cache(bin_path, offsets, dtype)
                    if total_tokens == 0:
                        return _SpilledDocs(
                            np.array([], dtype=dtype), np.array([0], dtype=np.int64)
                        )
                    mmap = np.memmap(str(bin_path), dtype=dtype, mode="r", shape=(total_tokens,))
                    return _SpilledDocs(mmap, np.array(offsets, dtype=np.int64))
