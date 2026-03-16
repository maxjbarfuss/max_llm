"""Memory-efficient document spilling to disk via mmap."""

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

    def __init__(self, mmap: np.memmap, offsets: np.ndarray) -> None:
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


def read_and_spill(
    source: DataSource,
    reader: FormatReader,
    tokenizer: Any,
    spill_dir: Path,
) -> _SpilledDocs:
    """Stream source docs to a temp binary file; peak RAM is bounded to one flush buffer (~10 MB)."""
    bin_path = spill_dir / f"{source.name}.bin"
    dtype: np.dtype[np.unsignedinteger] = np.dtype(np.uint16)  # sufficient for vocab ≤ 65535
    offsets: list[int] = [0]
    buffer: list[np.ndarray] = []
    buffered_tokens = 0

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

                pbar.update(1)
                if pbar.n % 5_000 == 0:
                    pbar.set_postfix(tokens=f"{offsets[-1] / 1e6:.1f}M")

            if buffer:
                f.write(np.concatenate(buffer).tobytes())
                f.flush()
            pbar.set_postfix(tokens=f"{offsets[-1] / 1e6:.1f}M", done=True)

    total_tokens = offsets[-1]
    if total_tokens == 0:
        return _SpilledDocs(
            np.memmap(bin_path, dtype=dtype, mode="r", shape=(0,)),
            np.array([0], dtype=np.int64),
        )

    mmap = np.memmap(bin_path, dtype=dtype, mode="r", shape=(total_tokens,))
    return _SpilledDocs(mmap, np.array(offsets, dtype=np.int64))
