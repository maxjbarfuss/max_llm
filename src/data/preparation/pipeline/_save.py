"""Output writing: single files, shards, stats, and manifests."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.data.preparation.config import DataPreparationConfig

logger = logging.getLogger(__name__)


def save_outputs(
    splits: dict[str, list[tuple[str, np.ndarray]]],
    tokenizer_path: str | None,
    config: DataPreparationConfig,
) -> dict[str, Any]:
    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, str] = {}
    output_shards: dict[str, list[str]] = {}

    for split_name, docs in splits.items():
        if not docs:
            continue

        logger.info(
            "Saving split: %s docs=%d shard_size_tokens=%d",
            split_name,
            len(docs),
            config.output.shard_size_tokens,
        )
        shard_paths, total_tokens = _save_split_outputs(split_name, docs, config)
        output_shards[split_name] = shard_paths
        output_paths[split_name] = shard_paths[0]
        logger.info(
            "Saved split: %s shards=%d first_path=%s tokens=%d",
            split_name,
            len(shard_paths),
            shard_paths[0],
            total_tokens,
        )

    if config.output.save_stats:
        stats_path = output_dir / f"{config.output.prefix}_stats.json"
        stats_path.write_text(
            json.dumps(_compute_stats(splits), indent=2), encoding="utf-8"
        )

    manifest: dict[str, Any] = {
        "tokenizer_path": tokenizer_path,
        "output_paths": output_paths,
        "output_shards": output_shards,
        "shard_size_tokens": config.output.shard_size_tokens,
    }

    if config.output.save_manifest:
        manifest_path = output_dir / f"{config.output.prefix}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def _save_split_outputs(
    split_name: str,
    docs: list[tuple[str, np.ndarray]],
    config: DataPreparationConfig,
) -> tuple[list[str], int]:
    output_dir = Path(config.output.dir)
    prefix = config.output.prefix
    dtype = (
        np.dtype(np.uint32)
        if any(doc.dtype == np.uint32 for _, doc in docs)
        else np.dtype(np.uint16)
    )
    has_eos = config.output.eos_token_id >= 0
    eos_val = config.output.eos_token_id
    shard_size_tokens = config.output.shard_size_tokens

    if shard_size_tokens <= 0:
        return _save_single_file(split_name, docs, output_dir, prefix, dtype, has_eos, eos_val)

    return _save_sharded(
        split_name, docs, output_dir, prefix, dtype, has_eos, eos_val, shard_size_tokens
    )


def _save_single_file(
    split_name: str,
    docs: list[tuple[str, np.ndarray]],
    output_dir: Path,
    prefix: str,
    dtype: np.dtype[Any],
    has_eos: bool,
    eos_val: int,
) -> tuple[list[str], int]:
    """Write all docs into a pre-allocated memmap, flushing every 5 M tokens."""
    total_tokens = sum(len(doc) + (1 if has_eos else 0) for _, doc in docs)
    file_path = output_dir / f"{prefix}_{split_name}.npy"
    out = np.lib.format.open_memmap(
        str(file_path), mode="w+", dtype=dtype, shape=(total_tokens,)
    )
    cursor = 0
    next_flush_at = 5_000_000

    with tqdm(
        total=len(docs),
        desc=f"  {split_name}",
        unit="doc",
        unit_scale=True,
        dynamic_ncols=True,
    ) as pbar:
        for _, doc in docs:
            chunk = doc.astype(dtype, copy=False)
            n = len(chunk)
            out[cursor : cursor + n] = chunk
            cursor += n
            if has_eos:
                out[cursor] = eos_val
                cursor += 1
            if cursor >= next_flush_at:
                out.flush()
                next_flush_at = cursor + 5_000_000
            pbar.update(1)

    out.flush()
    del out
    return [str(file_path)], total_tokens


def _save_sharded(
    split_name: str,
    docs: list[tuple[str, np.ndarray]],
    output_dir: Path,
    prefix: str,
    dtype: np.dtype[Any],
    has_eos: bool,
    eos_val: int,
    shard_size_tokens: int,
) -> tuple[list[str], int]:
    """Fill fixed-size shards sequentially, flushing each to disk when full."""
    eos = None if not has_eos else np.array([eos_val], dtype=dtype)
    shard_paths: list[str] = []
    parts: list[np.ndarray] = []
    buffered_tokens = 0
    total_tokens = 0
    shard_index = 0

    def flush() -> None:
        nonlocal buffered_tokens, parts, shard_index
        if not parts:
            return
        shard_path = output_dir / f"{prefix}_{split_name}_{shard_index:05d}.npy"
        np.save(shard_path, np.concatenate(parts))
        shard_paths.append(str(shard_path))
        parts = []
        buffered_tokens = 0
        shard_index += 1

    with tqdm(
        total=len(docs),
        desc=f"  {split_name}",
        unit="doc",
        unit_scale=True,
        dynamic_ncols=True,
    ) as pbar:
        for _, doc in docs:
            chunk = doc.astype(dtype, copy=False)
            doc_tokens = int(len(chunk)) + (0 if eos is None else 1)

            if buffered_tokens > 0 and buffered_tokens + doc_tokens > shard_size_tokens:
                flush()

            parts.append(chunk)
            if eos is not None:
                parts.append(eos)
            buffered_tokens += doc_tokens
            total_tokens += doc_tokens

            if buffered_tokens >= shard_size_tokens:
                flush()

            pbar.update(1)
            if shard_paths:
                pbar.set_postfix(shards=len(shard_paths))

        flush()

    return shard_paths, total_tokens


def _compute_stats(splits: dict[str, list[tuple[str, np.ndarray]]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}

    for split_name, docs in splits.items():
        by_source: dict[str, list[int]] = defaultdict(list)
        for source_name, doc in docs:
            by_source[source_name].append(len(doc))

        source_stats: dict[str, Any] = {}
        for source_name, lengths in by_source.items():
            source_stats[source_name] = {
                "num_docs": len(lengths),
                "total_tokens": int(sum(lengths)),
                "mean_length": float(np.mean(lengths)),
                "median_length": float(np.median(lengths)),
                "min_length": int(min(lengths)),
                "max_length": int(max(lengths)),
            }

        stats[split_name] = {
            "total_docs": len(docs),
            "total_tokens": int(sum(len(doc) for _, doc in docs)),
            "by_source": source_stats,
        }

    return stats
