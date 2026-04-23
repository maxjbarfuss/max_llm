"""Output writing: single files, shards, stats, and manifests."""

import json
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.data.preparation.config import DataPreparationConfig
from src.data.preparation.diagnostics import emit_prep_diagnostic

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
    packing_summaries: dict[str, dict[str, Any]] = {}
    packing_metadata_paths: dict[str, str] = {}

    for split_name, docs in splits.items():
        if not docs:
            continue

        logger.info(
            "Saving split: %s docs=%d shard_size_tokens=%d",
            split_name,
            len(docs),
            config.output.shard_size_tokens,
        )
        shard_paths, total_tokens, packing_info = _save_split_outputs(split_name, docs, config)
        output_shards[split_name] = shard_paths
        output_paths[split_name] = shard_paths[0]
        if packing_info is not None:
            packing_summaries[split_name] = packing_info["summary"]
            if "metadata_path" in packing_info:
                packing_metadata_paths[split_name] = packing_info["metadata_path"]
        logger.info(
            "Saved split: %s shards=%d first_path=%s tokens=%d",
            split_name,
            len(shard_paths),
            shard_paths[0],
            total_tokens,
        )

    if config.output.save_stats:
        stats_path = output_dir / f"{config.output.prefix}_stats.json"
        stats = _compute_stats(splits)
        for split_name, summary in packing_summaries.items():
            if split_name in stats:
                stats[split_name]["packing"] = summary
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    manifest: dict[str, Any] = {
        "tokenizer_path": tokenizer_path,
        "output_paths": output_paths,
        "output_shards": output_shards,
        "shard_size_tokens": config.output.shard_size_tokens,
    }
    if config.packing.enabled:
        manifest["packing"] = {
            "enabled": True,
            "sequence_length": config.packing.sequence_length,
        }
        manifest["packing_metadata_paths"] = packing_metadata_paths

    if config.output.save_manifest:
        manifest_path = output_dir / f"{config.output.prefix}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def save_outputs_from_source_indices(
    split_indices: dict[str, dict[str, np.ndarray]],
    source_documents: dict[str, Any],
    tokenizer_path: str | None,
    config: DataPreparationConfig,
    shuffle_seed: int,
) -> dict[str, Any]:
    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, str] = {}
    output_shards: dict[str, list[str]] = {}
    stats = _compute_indexed_stats(split_indices, source_documents)
    packing_summaries: dict[str, dict[str, Any]] = {}
    packing_metadata_paths: dict[str, str] = {}

    for split_name, by_source in split_indices.items():
        total_docs = sum(len(indices) for indices in by_source.values())
        if total_docs == 0:
            continue

        logger.info(
            "Saving split from source indices: %s docs=%d shard_size_tokens=%d",
            split_name,
            total_docs,
            config.output.shard_size_tokens,
        )
        shard_paths, total_tokens, packing_info = _save_split_outputs_from_indices(
            split_name,
            by_source,
            source_documents,
            config,
            shuffle_seed,
        )
        output_shards[split_name] = shard_paths
        output_paths[split_name] = shard_paths[0]
        if packing_info is not None:
            packing_summaries[split_name] = packing_info["summary"]
            if "metadata_path" in packing_info:
                packing_metadata_paths[split_name] = packing_info["metadata_path"]
        logger.info(
            "Saved indexed split: %s shards=%d first_path=%s tokens=%d",
            split_name,
            len(shard_paths),
            shard_paths[0],
            total_tokens,
        )

    if config.output.save_stats:
        for split_name, summary in packing_summaries.items():
            if split_name in stats:
                stats[split_name]["packing"] = summary
        stats_path = output_dir / f"{config.output.prefix}_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    manifest: dict[str, Any] = {
        "tokenizer_path": tokenizer_path,
        "output_paths": output_paths,
        "output_shards": output_shards,
        "shard_size_tokens": config.output.shard_size_tokens,
    }
    if config.packing.enabled:
        manifest["packing"] = {
            "enabled": True,
            "sequence_length": config.packing.sequence_length,
        }
        manifest["packing_metadata_paths"] = packing_metadata_paths

    if config.output.save_manifest:
        manifest_path = output_dir / f"{config.output.prefix}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def _save_split_outputs(
    split_name: str,
    docs: list[tuple[str, np.ndarray]],
    config: DataPreparationConfig,
) -> tuple[list[str], int, dict[str, Any] | None]:
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

    if config.packing.enabled:
        return _save_packed_from_docs(split_name, docs, config, dtype)

    if shard_size_tokens <= 0:
        shard_paths, total_tokens = _save_single_file(
            split_name, docs, output_dir, prefix, dtype, has_eos, eos_val
        )
        return shard_paths, total_tokens, None

    shard_paths, total_tokens = _save_sharded(
        split_name, docs, output_dir, prefix, dtype, has_eos, eos_val, shard_size_tokens
    )
    return shard_paths, total_tokens, None


def _source_dtype(
    source_documents: dict[str, Any], split_indices: dict[str, np.ndarray]
) -> np.dtype[Any]:
    return (
        np.dtype(np.uint32)
        if any(
            source_documents[name].dtype == np.uint32 and len(indices) > 0
            for name, indices in split_indices.items()
        )
        else np.dtype(np.uint16)
    )


def _source_schedule(
    split_indices: dict[str, np.ndarray],
    shuffle_seed: int,
    split_name: str,
    shuffle: bool,
) -> tuple[list[str], np.ndarray]:
    source_names = [name for name, indices in split_indices.items() if len(indices) > 0]
    total_docs = sum(len(split_indices[name]) for name in source_names)
    dtype = np.uint8 if len(source_names) <= np.iinfo(np.uint8).max else np.uint16
    schedule = np.empty(total_docs, dtype=dtype)
    cursor = 0
    for source_id, name in enumerate(source_names):
        count = len(split_indices[name])
        schedule[cursor : cursor + count] = source_id
        cursor += count
    if shuffle and total_docs > 1:
        split_offsets = {"train": 0, "val": 1, "test": 2}
        rng = np.random.RandomState(shuffle_seed + split_offsets.get(split_name, 3))
        rng.shuffle(schedule)
    return source_names, schedule


def _save_split_outputs_from_indices(  # noqa: C901
    split_name: str,
    split_indices: dict[str, np.ndarray],
    source_documents: dict[str, Any],
    config: DataPreparationConfig,
    shuffle_seed: int,
) -> tuple[list[str], int, dict[str, Any] | None]:
    output_dir = Path(config.output.dir)
    prefix = config.output.prefix
    dtype = _source_dtype(source_documents, split_indices)
    has_eos = config.output.eos_token_id >= 0
    eos_val = config.output.eos_token_id
    shard_size_tokens = config.output.shard_size_tokens
    source_names, schedule = _source_schedule(
        split_indices,
        shuffle_seed,
        split_name,
        shuffle=config.splits.shuffle,
    )
    emit_prep_diagnostic(
        output_dir,
        prefix,
        "save_indexed_split_schedule",
        split=split_name,
        schedule_docs=int(len(schedule)),
        schedule_bytes=int(schedule.nbytes),
        source_doc_counts={name: int(len(split_indices[name])) for name in source_names},
    )
    pointers = dict.fromkeys(source_names, 0)
    lengths = {
        name: np.asarray(source_documents[name].doc_lengths)[indices]
        for name, indices in split_indices.items()
        if len(indices) > 0
    }
    emit_prep_diagnostic(
        output_dir,
        prefix,
        "save_indexed_split_lengths",
        split=split_name,
        length_bytes={name: int(values.nbytes) for name, values in lengths.items()},
        total_length_bytes=int(sum(values.nbytes for values in lengths.values())),
    )
    total_tokens = int(
        sum(int(values.sum()) for values in lengths.values()) + (len(schedule) if has_eos else 0)
    )
    emit_prep_diagnostic(
        output_dir,
        prefix,
        "save_indexed_split_total_tokens",
        split=split_name,
        total_tokens=total_tokens,
        has_eos=has_eos,
        output_dtype=str(dtype),
    )

    if config.packing.enabled:
        return _save_packed_from_indices(
            split_name,
            split_indices,
            source_documents,
            source_names,
            schedule,
            config,
            dtype,
            total_doc_tokens=int(sum(int(values.sum()) for values in lengths.values())),
        )

    if shard_size_tokens <= 0:
        file_path = output_dir / f"{prefix}_{split_name}.npy"
        out = np.lib.format.open_memmap(
            str(file_path), mode="w+", dtype=dtype, shape=(total_tokens,)
        )
        cursor = 0
        next_flush_at = 5_000_000
        with tqdm(
            total=len(schedule),
            desc=f"  {split_name}",
            unit="doc",
            unit_scale=True,
            dynamic_ncols=True,
        ) as pbar:
            for source_id in schedule:
                name = source_names[int(source_id)]
                index = int(split_indices[name][pointers[name]])
                pointers[name] += 1
                chunk = source_documents[name][index].astype(dtype, copy=False)
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
        return [str(file_path)], total_tokens, None

    eos = None if not has_eos else np.array([eos_val], dtype=dtype)
    shard_paths: list[str] = []
    parts: list[np.ndarray] = []
    buffered_tokens = 0
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
        total=len(schedule),
        desc=f"  {split_name}",
        unit="doc",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    ) as pbar:
        for source_id in schedule:
            name = source_names[int(source_id)]
            index = int(split_indices[name][pointers[name]])
            pointers[name] += 1
            chunk = source_documents[name][index].astype(dtype, copy=False)
            doc_tokens = int(len(chunk)) + (0 if eos is None else 1)

            if buffered_tokens > 0 and buffered_tokens + doc_tokens > shard_size_tokens:
                flush()

            parts.append(chunk)
            if eos is not None:
                parts.append(eos)
            buffered_tokens += doc_tokens

            if buffered_tokens >= shard_size_tokens:
                flush()

            pbar.update(1)

        flush()

    return shard_paths, total_tokens, None


def _save_packing_metadata(
    output_dir: Path,
    prefix: str,
    split_name: str,
    sequence_offsets: list[int],
    boundaries: list[int],
) -> str:
    path = output_dir / f"{prefix}_{split_name}_packing_meta.npz"
    np.savez(
        path,
        sequence_offsets=np.asarray(sequence_offsets, dtype=np.int64),
        boundaries=np.asarray(boundaries, dtype=np.int32),
    )
    return str(path)


def _save_packed_from_docs(
    split_name: str,
    docs: list[tuple[str, np.ndarray]],
    config: DataPreparationConfig,
    dtype: np.dtype[Any],
) -> tuple[list[str], int, dict[str, Any]]:
    output_dir = Path(config.output.dir)
    prefix = config.output.prefix
    seq_len = config.packing.sequence_length
    pad_val = config.output.eos_token_id if config.output.eos_token_id >= 0 else 0
    shard_size_tokens = config.output.shard_size_tokens
    total_doc_tokens = int(sum(len(doc) for _, doc in docs))
    doc_iter = (doc for _, doc in docs)
    return _save_packed_sequences(
        split_name,
        doc_iter,
        doc_count=len(docs),
        total_doc_tokens=total_doc_tokens,
        output_dir=output_dir,
        prefix=prefix,
        dtype=dtype,
        sequence_length=seq_len,
        pad_token=pad_val,
        shard_size_tokens=shard_size_tokens,
        save_metadata=config.packing.save_metadata,
    )


def _save_packed_from_indices(
    split_name: str,
    split_indices: dict[str, np.ndarray],
    source_documents: dict[str, Any],
    source_names: list[str],
    schedule: np.ndarray,
    config: DataPreparationConfig,
    dtype: np.dtype[Any],
    total_doc_tokens: int,
) -> tuple[list[str], int, dict[str, Any]]:
    output_dir = Path(config.output.dir)
    prefix = config.output.prefix
    seq_len = config.packing.sequence_length
    pad_val = config.output.eos_token_id if config.output.eos_token_id >= 0 else 0
    shard_size_tokens = config.output.shard_size_tokens
    pointers = dict.fromkeys(source_names, 0)

    def iter_docs() -> Iterable[np.ndarray]:
        for source_id in schedule:
            name = source_names[int(source_id)]
            index = int(split_indices[name][pointers[name]])
            pointers[name] += 1
            yield source_documents[name][index]

    return _save_packed_sequences(
        split_name,
        iter_docs(),
        doc_count=int(len(schedule)),
        total_doc_tokens=total_doc_tokens,
        output_dir=output_dir,
        prefix=prefix,
        dtype=dtype,
        sequence_length=seq_len,
        pad_token=pad_val,
        shard_size_tokens=shard_size_tokens,
        save_metadata=config.packing.save_metadata,
    )


def _save_packed_sequences(  # noqa: C901
    split_name: str,
    doc_iter: Iterable[np.ndarray],
    doc_count: int,
    total_doc_tokens: int,
    output_dir: Path,
    prefix: str,
    dtype: np.dtype[Any],
    sequence_length: int,
    pad_token: int,
    shard_size_tokens: int,
    save_metadata: bool,
) -> tuple[list[str], int, dict[str, Any]]:
    if total_doc_tokens <= 0:
        return (
            [],
            0,
            {
                "summary": {
                    "enabled": True,
                    "sequence_length": sequence_length,
                    "packed_sequences": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "fill_ratio": 0.0,
                }
            },
        )

    num_sequences = (total_doc_tokens + sequence_length - 1) // sequence_length
    output_tokens = num_sequences * sequence_length
    seq_offsets: list[int] = [0]
    boundaries_flat: list[int] = []

    summary: dict[str, Any] = {
        "enabled": True,
        "sequence_length": sequence_length,
        "packed_sequences": int(num_sequences),
        "input_tokens": int(total_doc_tokens),
        "output_tokens": int(output_tokens),
        "fill_ratio": float(total_doc_tokens / max(1, output_tokens)),
    }

    def append_metadata(boundaries: list[int]) -> None:
        boundaries_flat.extend(boundaries)
        seq_offsets.append(len(boundaries_flat))

    if shard_size_tokens <= 0:
        file_path = output_dir / f"{prefix}_{split_name}.npy"
        out = np.lib.format.open_memmap(
            str(file_path), mode="w+", dtype=dtype, shape=(output_tokens,)
        )
        seq_buffer = np.full((sequence_length,), pad_token, dtype=dtype)
        seq_cursor = 0
        seq_idx = 0
        boundaries_cur = []

        with tqdm(
            total=doc_count,
            desc=f"  {split_name}",
            unit="doc",
            unit_scale=True,
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        ) as pbar:
            for doc in doc_iter:
                chunk = doc.astype(dtype, copy=False)
                pos = 0
                while pos < len(chunk):
                    space = sequence_length - seq_cursor
                    take = min(space, len(chunk) - pos)
                    seq_buffer[seq_cursor : seq_cursor + take] = chunk[pos : pos + take]
                    seq_cursor += take
                    pos += take
                    if pos == len(chunk):
                        boundaries_cur.append(seq_cursor)
                    if seq_cursor == sequence_length:
                        start = seq_idx * sequence_length
                        out[start : start + sequence_length] = seq_buffer
                        append_metadata(boundaries_cur)
                        seq_idx += 1
                        seq_buffer.fill(pad_token)
                        seq_cursor = 0
                        boundaries_cur = []
                pbar.update(1)

        if seq_cursor > 0:
            start = seq_idx * sequence_length
            out[start : start + sequence_length] = seq_buffer
            append_metadata(boundaries_cur)

        out.flush()
        del out
        shard_paths = [str(file_path)]
    else:
        shard_paths = []
        parts: list[np.ndarray] = []
        buffered_tokens = 0
        shard_index = 0
        seq_buffer = np.full((sequence_length,), pad_token, dtype=dtype)
        seq_cursor = 0
        boundaries_cur = []

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
            total=doc_count,
            desc=f"  {split_name}",
            unit="doc",
            unit_scale=True,
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        ) as pbar:
            for doc in doc_iter:
                chunk = doc.astype(dtype, copy=False)
                pos = 0
                while pos < len(chunk):
                    space = sequence_length - seq_cursor
                    take = min(space, len(chunk) - pos)
                    seq_buffer[seq_cursor : seq_cursor + take] = chunk[pos : pos + take]
                    seq_cursor += take
                    pos += take
                    if pos == len(chunk):
                        boundaries_cur.append(seq_cursor)
                    if seq_cursor == sequence_length:
                        if (
                            buffered_tokens > 0
                            and buffered_tokens + sequence_length > shard_size_tokens
                        ):
                            flush()
                        parts.append(seq_buffer.copy())
                        buffered_tokens += sequence_length
                        append_metadata(boundaries_cur)
                        if buffered_tokens >= shard_size_tokens:
                            flush()
                        seq_buffer.fill(pad_token)
                        seq_cursor = 0
                        boundaries_cur = []
                pbar.update(1)

        if seq_cursor > 0:
            if buffered_tokens > 0 and buffered_tokens + sequence_length > shard_size_tokens:
                flush()
            parts.append(seq_buffer.copy())
            buffered_tokens += sequence_length
            append_metadata(boundaries_cur)

        flush()

    result: dict[str, Any] = {"summary": summary}
    if save_metadata:
        result["metadata_path"] = _save_packing_metadata(
            output_dir,
            prefix,
            split_name,
            seq_offsets,
            boundaries_flat,
        )

    return shard_paths, output_tokens, result


def _compute_indexed_stats(
    splits: dict[str, dict[str, np.ndarray]],
    source_documents: dict[str, Any],
) -> dict[str, Any]:
    stats: dict[str, Any] = {}

    for split_name, by_source in splits.items():
        source_stats: dict[str, Any] = {}
        total_docs = 0
        total_tokens = 0
        for source_name, indices in by_source.items():
            if len(indices) == 0:
                continue
            lengths = np.asarray(source_documents[source_name].doc_lengths)[indices]
            source_stats[source_name] = {
                "num_docs": int(len(indices)),
                "total_tokens": int(lengths.sum()),
                "mean_length": float(np.mean(lengths)),
                "median_length": float(np.median(lengths)),
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
            }
            total_docs += int(len(indices))
            total_tokens += int(lengths.sum())

        stats[split_name] = {
            "total_docs": total_docs,
            "total_tokens": total_tokens,
            "by_source": source_stats,
        }

    return stats


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
    out = np.lib.format.open_memmap(str(file_path), mode="w+", dtype=dtype, shape=(total_tokens,))
    cursor = 0
    next_flush_at = 5_000_000

    with tqdm(
        total=len(docs),
        desc=f"  {split_name}",
        unit="doc",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
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
        disable=not sys.stderr.isatty(),
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
