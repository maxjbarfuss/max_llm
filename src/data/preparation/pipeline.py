"""Orchestration pipeline for dataset preparation."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.data.preparation.config import DataPreparationConfig, DataSource, load_config
from src.data.preparation.strategies import (
    FormatReader,
    _normalize_text,
    resolve_curriculum_strategy,
    resolve_format_reader,
    resolve_mixing_strategy,
    resolve_split_strategy,
)
from src.tokenizer import TokenizerFactory, UnigramTokenizer

logger = logging.getLogger(__name__)


class _SpilledDocs:
    """Memory-efficient sequence of documents backed by a raw binary temp file.

    Token data is memory-mapped from disk; only view metadata lives in RAM.
    """

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


class _UTF8Tokenizer:
    def __init__(self) -> None:
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="ignore")


class PreparationPipeline:
    """Main preparation pipeline."""

    def run(self, config: DataPreparationConfig) -> dict[str, Any]:
        config.validate()
        logger.info("Preparation started: output_dir=%s", config.output.dir)

        output_dir = Path(config.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        spill_dir = output_dir / ".prep_spill"
        spill_dir.mkdir(exist_ok=True)
        try:
            return self._run_with_spill_dir(config, output_dir, spill_dir)
        finally:
            shutil.rmtree(spill_dir, ignore_errors=True)

    def _run_with_spill_dir(
        self, config: DataPreparationConfig, output_dir: Path, spill_dir: Path
    ) -> dict[str, Any]:
        n_active = sum(1 for s in config.datasets if s.weight > 0)

        # ── Step 1: tokenizer ────────────────────────────────────────────────
        print(f"\n[1/4] Tokenizer  ({config.tokenizer.type})", flush=True)
        logger.info("Building tokenizer: type=%s", config.tokenizer.type)
        tokenizer_prefix = str(output_dir / f"{config.output.prefix}_tokenizer")
        tokenizer, tokenizer_path = self._build_tokenizer(config, tokenizer_prefix)
        print(f"      ready → {tokenizer_path or 'in-memory'}", flush=True)
        logger.info("Tokenizer ready: path=%s", tokenizer_path)

        # ── Step 2: read sources ─────────────────────────────────────────────
        print(f"\n[2/4] Reading {n_active} source(s)", flush=True)
        all_documents: dict[str, _SpilledDocs] = {}
        for source in config.datasets:
            if source.weight == 0:
                continue
            logger.info(
                "Reading source: name=%s format=%s path=%s", source.name, source.format, source.path
            )
            reader = resolve_format_reader(source.format)
            spilled = self._read_and_spill(source, reader, tokenizer, spill_dir)
            all_documents[source.name] = spilled
            total_tok = spilled.total_tokens
            print(f"      {source.name}: {len(spilled):,} docs  {total_tok:,} tokens", flush=True)
            logger.info(
                "Source loaded: name=%s docs=%d tokens=%d", source.name, len(spilled), total_tok
            )

        # ── Step 3: mix ──────────────────────────────────────────────────────
        print(f"\n[3/4] Mixing  ({config.mixing.strategy})", flush=True)
        logger.info("Mixing documents: strategy=%s", config.mixing.strategy)
        mixed_documents = resolve_mixing_strategy(config.mixing).mix(
            all_documents,  # type: ignore[arg-type]
            config.mixing,
        )
        print(f"      {len(mixed_documents):,} documents", flush=True)
        logger.info("Mix complete: mixed_docs=%d", len(mixed_documents))

        # ── Step 4: split + save ─────────────────────────────────────────────
        print("\n[4/4] Splitting and saving", flush=True)
        if config.curriculum.enabled:
            logger.info("Applying curriculum: type=%s", config.curriculum.type)
            stages = resolve_curriculum_strategy(config.curriculum).build(
                mixed_documents,
                config.curriculum,
            )
            splits = self._split_curriculum(stages, config)
        else:
            logger.info(
                "Applying standard split: train=%.3f val=%.3f test=%.3f",
                config.splits.train,
                config.splits.val,
                config.splits.test,
            )
            splits = resolve_split_strategy(config.splits).split(mixed_documents, config.splits)

        manifest = self._save_outputs(splits, tokenizer_path, config)

        shard_counts = {
            name: len(paths) for name, paths in manifest.get("output_shards", {}).items()
        }
        print(f"\nDone. Outputs in {output_dir}", flush=True)
        print(f"      splits: {shard_counts}", flush=True)
        logger.info(
            "Preparation complete: outputs=%s shard_counts=%s",
            manifest.get("output_paths", {}),
            shard_counts,
        )
        return manifest

    def run_from_config_path(self, config_path: str | Path) -> dict[str, Any]:
        config = load_config(config_path)
        return self.run(config)

    def _read_and_spill(
        self,
        source: DataSource,
        reader: FormatReader,
        tokenizer: Any,
        spill_dir: Path,
    ) -> _SpilledDocs:
        """Stream a source's documents to a temp binary file and return a mmap-backed view.

        This keeps peak RAM bounded to one source's flush buffer (~10 MB) regardless
        of how many tokens the source contains.  Previously-read sources are backed by
        mmap so the OS can page them out while the next source is being read.
        """
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
        offsets_arr = np.array(offsets, dtype=np.int64)
        return _SpilledDocs(mmap, offsets_arr)

    def _build_tokenizer(
        self,
        config: DataPreparationConfig,
        tokenizer_prefix: str,
    ) -> tuple[Any, str | None]:
        tok_cfg = config.tokenizer

        if tok_cfg.type == "unigram":
            if tok_cfg.model_path:
                return UnigramTokenizer(tok_cfg.model_path), str(tok_cfg.model_path)

            corpus_path = self._create_training_corpus(config.datasets, tok_cfg.max_corpus_mb)
            model_path = UnigramTokenizer.train_model(
                input_path=corpus_path,
                model_prefix=tokenizer_prefix,
                vocab_size=tok_cfg.vocab_size,
                character_coverage=tok_cfg.character_coverage,
            )
            Path(corpus_path).unlink(missing_ok=True)
            return UnigramTokenizer(model_path), str(model_path)

        if tok_cfg.type == "char":
            tokenizer = TokenizerFactory.create("char", vocab_size=tok_cfg.vocab_size)
            return tokenizer, None

        if tok_cfg.type == "bpe":
            tokenizer = TokenizerFactory.create("bpe")
            return tokenizer, None

        if tok_cfg.type == "hf_bpe":
            tokenizer = TokenizerFactory.create("hf_bpe")
            return tokenizer, None

        if tok_cfg.type == "utf8":
            return _UTF8Tokenizer(), None

        raise ValueError(f"Unknown tokenizer type: {tok_cfg.type}")

    def _create_training_corpus(
        self,
        sources: list[DataSource],
        max_corpus_mb: float | None,
    ) -> str:
        corpus_file = tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".txt",
            encoding="utf-8",
        )

        total_bytes = 0
        max_bytes = int(max_corpus_mb * 1024 * 1024) if max_corpus_mb else None

        try:
            for source in sources:
                if max_bytes and total_bytes >= max_bytes:
                    break

                written = self._write_source_to_corpus(
                    source=source,
                    handle=corpus_file,
                    current_bytes=total_bytes,
                    max_bytes=max_bytes,
                )
                total_bytes += written
                logger.info(
                    "Tokenizer corpus streaming: source=%s wrote=%d bytes total=%d",
                    source.name,
                    written,
                    total_bytes,
                )
        finally:
            corpus_file.close()

        return corpus_file.name

    def _write_with_limit(
        self,
        handle: Any,
        text: str,
        current_bytes: int,
        max_bytes: int | None,
    ) -> int:
        if not text:
            return 0

        remaining = None if max_bytes is None else max_bytes - current_bytes
        if remaining is not None and remaining <= 0:
            return 0

        payload = text if remaining is None else text[:remaining]
        handle.write(payload)
        return len(payload)

    def _write_source_to_corpus(  # noqa: C901
        self,
        source: DataSource,
        handle: Any,
        current_bytes: int,
        max_bytes: int | None,
    ) -> int:
        path = Path(source.path)
        written = 0

        if source.format == "text":
            files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
            for file_path in files:
                with file_path.open(encoding="utf-8", errors="ignore") as src:
                    for chunk in iter(lambda: src.read(1024 * 1024), ""):
                        delta = self._write_with_limit(
                            handle,
                            _normalize_text(chunk),
                            current_bytes + written,
                            max_bytes,
                        )
                        written += delta
                        if delta < len(chunk):
                            return written

                sep_delta = self._write_with_limit(
                    handle,
                    "\n\n",
                    current_bytes + written,
                    max_bytes,
                )
                written += sep_delta
                if sep_delta < 2:
                    return written
            return written

        if source.format == "utf8_tokens":
            tokens = np.load(path, mmap_mode="r")
            byte_tokens = np.asarray(tokens)
            if byte_tokens.size and (int(byte_tokens.min()) < 0 or int(byte_tokens.max()) > 255):
                raise ValueError(
                    f"utf8_tokens source contains out-of-byte-range values: {source.path}"
                )
            decoded = _normalize_text(
                byte_tokens.astype(np.uint8, copy=False).tobytes().decode("utf-8", errors="ignore")
            )
            return self._write_with_limit(handle, decoded, current_bytes, max_bytes)

        if source.format == "jsonl":
            with path.open(encoding="utf-8") as src:
                for line in src:
                    obj = json.loads(line)
                    text = obj.get(source.text_field, "")
                    if not text:
                        continue
                    delta = self._write_with_limit(
                        handle,
                        _normalize_text(text),
                        current_bytes + written,
                        max_bytes,
                    )
                    written += delta
                    if delta < len(text):
                        return written

                    sep_delta = self._write_with_limit(
                        handle,
                        "\n\n",
                        current_bytes + written,
                        max_bytes,
                    )
                    written += sep_delta
                    if sep_delta < 2:
                        return written
            return written

        raise ValueError(f"Cannot convert format '{source.format}' to text corpus")

    def _load_source_as_text(self, source: DataSource) -> str:
        path = Path(source.path)

        if source.format == "text":
            if path.is_dir():
                parts = [
                    p.read_text(encoding="utf-8", errors="ignore")
                    for p in sorted(path.glob("*.txt"))
                ]
                return "\n\n".join(parts)
            return path.read_text(encoding="utf-8", errors="ignore")

        if source.format == "utf8_tokens":
            tokens = np.load(path)
            byte_tokens = np.asarray(tokens)
            if byte_tokens.size and (int(byte_tokens.min()) < 0 or int(byte_tokens.max()) > 255):
                raise ValueError(
                    f"utf8_tokens source contains out-of-byte-range values: {source.path}"
                )
            return (
                byte_tokens.astype(np.uint8, copy=False).tobytes().decode("utf-8", errors="ignore")
            )

        if source.format == "jsonl":
            lines: list[str] = []
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    obj = json.loads(line)
                    lines.append(obj.get(source.text_field, ""))
            return "\n\n".join(lines)

        raise ValueError(f"Cannot convert format '{source.format}' to text corpus")

    def _split_curriculum(
        self,
        stages: dict[str, list[tuple[str, np.ndarray]]],
        config: DataPreparationConfig,
    ) -> dict[str, list[tuple[str, np.ndarray]]]:
        splitter = resolve_split_strategy(config.splits)

        if config.curriculum.output_mode == "separate":
            separate_splits: dict[str, list[tuple[str, np.ndarray]]] = {}
            for stage_name, stage_docs in stages.items():
                split_result = splitter.split(stage_docs, config.splits)
                for split_name, docs in split_result.items():
                    separate_splits[f"{stage_name}_{split_name}"] = docs
            return separate_splits

        if config.curriculum.output_mode == "both":
            both_splits: dict[str, list[tuple[str, np.ndarray]]] = {}
            merged_docs_both: list[tuple[str, np.ndarray]] = []
            for stage_name, stage_docs in stages.items():
                split_result = splitter.split(stage_docs, config.splits)
                for split_name, docs in split_result.items():
                    both_splits[f"{stage_name}_{split_name}"] = docs
                merged_docs_both.extend(stage_docs)
            merged_splits = splitter.split(merged_docs_both, config.splits)
            for split_name, docs in merged_splits.items():
                both_splits[f"merged_{split_name}"] = docs
            return both_splits

        merged_docs: list[tuple[str, np.ndarray]] = []
        for stage_docs in stages.values():
            merged_docs.extend(stage_docs)
        return splitter.split(merged_docs, config.splits)

    def _save_outputs(
        self,
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
            shard_paths, total_tokens = self._save_split_outputs(split_name, docs, config)
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
                json.dumps(self._compute_stats(splits), indent=2), encoding="utf-8"
            )

        manifest = {
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
        self,
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
            return self._save_single_file(
                split_name, docs, output_dir, prefix, dtype, has_eos, eos_val
            )

        return self._save_sharded(
            split_name, docs, output_dir, prefix, dtype, has_eos, eos_val, shard_size_tokens
        )

    def _save_single_file(
        self,
        split_name: str,
        docs: list[tuple[str, np.ndarray]],
        output_dir: Path,
        prefix: str,
        dtype: np.dtype[Any],
        has_eos: bool,
        eos_val: int,
    ) -> tuple[list[str], int]:
        """Stream all docs into a single .npy via a pre-allocated memmap.

        Pre-computing the token count lets us write each doc directly into its
        final position without holding more than one doc in RAM at a time.
        Flushes every 5 M tokens so the OS page cache doesn't grow unbounded.
        """
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
        self,
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
                if len(shard_paths) > 0:
                    pbar.set_postfix(shards=len(shard_paths))

            flush()

        return shard_paths, total_tokens

    def _compute_stats(self, splits: dict[str, list[tuple[str, np.ndarray]]]) -> dict[str, Any]:
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
