"""Dataset preparation pipeline."""

import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.data.preparation.config import DataPreparationConfig, DataSource, load_config
from src.data.preparation.diagnostics import emit_prep_diagnostic
from src.data.preparation.strategies import (
    load_lang_model,
    resolve_curriculum_strategy,
    resolve_format_reader,
    resolve_mixing_strategy,
    resolve_split_strategy,
    select_source_indices,
)

from ._corpus import build_tokenizer, load_source_as_text
from ._dedup import _FilteredSpilledDocs, run_minhash_dedup
from ._save import save_outputs, save_outputs_from_source_indices
from ._spill import _SpilledDocs, read_and_spill

logger = logging.getLogger(__name__)


def _derive_source_ratios(config: DataPreparationConfig) -> dict[str, float] | None:
    """Return effective source ratios derived from config or dataset weights."""
    if config.mixing.source_ratios is not None:
        return dict(config.mixing.source_ratios)

    active_weights = {s.name: s.weight for s in config.datasets if s.weight > 0}
    unique_weights = set(active_weights.values())
    if config.mixing.weight_by == "tokens" or len(unique_weights) > 1:
        return active_weights
    return None


def _summarize_mixing_plan(config: DataPreparationConfig) -> list[str]:
    """Return human-readable summary lines for the configured mixing budget."""
    ratios = _derive_source_ratios(config)
    if not ratios:
        return []

    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        return []

    normalized = {name: weight / total_ratio for name, weight in ratios.items()}
    lines: list[str] = []
    if config.mixing.weight_by == "tokens" and config.mixing.target_total_tokens is not None:
        total_tokens = config.mixing.target_total_tokens
        lines.append(f"      target budget: {total_tokens:,} tokens")
        for name, ratio in normalized.items():
            lines.append(
                f"      target {name}: {ratio * 100:5.1f}% -> {int(total_tokens * ratio):,} tokens"
            )
        return lines

    if config.mixing.weight_by == "docs" and config.mixing.target_total_docs is not None:
        total_docs = config.mixing.target_total_docs
        lines.append(f"      target budget: {total_docs:,} docs")
        for name, ratio in normalized.items():
            lines.append(
                f"      target {name}: {ratio * 100:5.1f}% -> {int(total_docs * ratio):,} docs"
            )
        return lines

    return []


def _cleanup_spill_dir(spill_dir: Path) -> None:
    """Remove the spill dir but preserve per-source cache files for future resume."""
    if not spill_dir.exists():
        return
    for path in spill_dir.iterdir():
        name = path.name
        # Keep: raw bin, offsets, and meta — these form the resume cache
        if (
            name.endswith(".bin")
            or name.endswith(".bin.offsets.npy")
            or name.endswith(".bin.meta.json")
        ):
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _split_curriculum(
    stages: dict[str, list[tuple[str, np.ndarray]]],
    config: DataPreparationConfig,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    splitter = resolve_split_strategy(config.splits)

    if config.curriculum.output_mode == "separate":
        result: dict[str, list[tuple[str, np.ndarray]]] = {}
        for stage_name, stage_docs in stages.items():
            for split_name, docs in splitter.split(stage_docs, config.splits).items():
                result[f"{stage_name}_{split_name}"] = docs
        return result

    if config.curriculum.output_mode == "both":
        result = {}
        merged: list[tuple[str, np.ndarray]] = []
        for stage_name, stage_docs in stages.items():
            for split_name, docs in splitter.split(stage_docs, config.splits).items():
                result[f"{stage_name}_{split_name}"] = docs
            merged.extend(stage_docs)
        for split_name, docs in splitter.split(merged, config.splits).items():
            result[f"merged_{split_name}"] = docs
        return result

    merged = []
    for stage_docs in stages.values():
        merged.extend(stage_docs)
    return splitter.split(merged, config.splits)


def _split_source_indices(
    selected_indices: dict[str, np.ndarray],
    config: DataPreparationConfig,
) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.RandomState(config.splits.seed)
    splits: dict[str, dict[str, np.ndarray]] = {"train": {}, "val": {}, "test": {}}

    for name, indices in selected_indices.items():
        if len(indices) == 0:
            continue
        source_indices = np.array(indices, copy=True)
        if config.splits.shuffle and len(source_indices) > 1:
            rng.shuffle(source_indices)

        count = len(source_indices)
        train_end = int(count * config.splits.train)
        val_end = int(count * (config.splits.train + config.splits.val))

        splits["train"][name] = source_indices[:train_end]
        splits["val"][name] = source_indices[train_end:val_end]
        splits["test"][name] = source_indices[val_end:]

    return splits


class PreparationPipeline:
    """Orchestrates dataset preparation: tokenize → read → mix → split+save."""

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
            _cleanup_spill_dir(spill_dir)

    def run_from_config_path(self, config_path: str | Path) -> dict[str, Any]:
        return self.run(load_config(config_path))

    def _load_source_as_text(self, source: DataSource) -> str:
        return load_source_as_text(source)

    def _run_with_spill_dir(
        self, config: DataPreparationConfig, output_dir: Path, spill_dir: Path
    ) -> dict[str, Any]:
        n_active = sum(1 for s in config.datasets if s.weight > 0)
        derived_ratios = _derive_source_ratios(config)
        if config.mixing.source_ratios is None and derived_ratios is not None:
            config.mixing.source_ratios = derived_ratios
        emit_prep_diagnostic(
            output_dir,
            config.output.prefix,
            "pipeline_start",
            active_sources=n_active,
            strategy=config.mixing.strategy,
            weight_by=config.mixing.weight_by,
            stratified=config.splits.stratified,
            curriculum_enabled=config.curriculum.enabled,
        )

        # ── Step 1: tokenizer ────────────────────────────────────────────────
        print(f"\n[1/4] Tokenizer  ({config.tokenizer.type})", flush=True)
        logger.info("Building tokenizer: type=%s", config.tokenizer.type)
        tokenizer_prefix = str(output_dir / f"{config.output.prefix}_tokenizer")
        tokenizer, tokenizer_path = build_tokenizer(config, tokenizer_prefix)
        print(f"      ready → {tokenizer_path or 'in-memory'}", flush=True)
        logger.info("Tokenizer ready: path=%s", tokenizer_path)

        # ── Step 2: read sources ─────────────────────────────────────────────
        print(f"\n[2/4] Reading {n_active} source(s)", flush=True)
        for line in _summarize_mixing_plan(config):
            print(line, flush=True)
            logger.info(line.strip())

        if config.lang_model_path:
            print(f"      Loading language model: {config.lang_model_path}", flush=True)
            logger.info("Loading lang model: path=%s", config.lang_model_path)
            load_lang_model(config.lang_model_path)

        all_documents: dict[str, Any] = {}
        for source in config.datasets:
            if source.weight == 0:
                continue
            logger.info(
                "Reading source: name=%s format=%s path=%s",
                source.name,
                source.format,
                source.path,
            )
            spilled = read_and_spill(
                source,
                resolve_format_reader(source.format),
                tokenizer,
                spill_dir,
                diagnostic_output_dir=output_dir,
                diagnostic_prefix=config.output.prefix,
            )
            all_documents[source.name] = spilled
            total_tok = spilled.total_tokens
            print(f"      {source.name}: {len(spilled):,} docs  {total_tok:,} tokens", flush=True)
            logger.info(
                "Source loaded: name=%s docs=%d tokens=%d", source.name, len(spilled), total_tok
            )
            emit_prep_diagnostic(
                output_dir,
                config.output.prefix,
                "source_loaded",
                source=source.name,
                docs=int(len(spilled)),
                tokens=int(total_tok),
                offsets_bytes=spilled.offsets_bytes,
                token_mmap_bytes=spilled.token_bytes,
            )

        # ── Step 2.5: MinHash near-dedup ─────────────────────────────────────
        if config.dedup.enabled:
            print("\n[2.5/4] Near-dedup (MinHash LSH)", flush=True)
            logger.info(
                "Running MinHash dedup: threshold=%s num_perm=%s shingle_size=%s",
                config.dedup.jaccard_threshold,
                config.dedup.num_perm,
                config.dedup.shingle_size,
            )
            kept_indices, dedup_stats = run_minhash_dedup(
                all_documents,
                jaccard_threshold=config.dedup.jaccard_threshold,
                num_perm=config.dedup.num_perm,
                shingle_size=config.dedup.shingle_size,
            )
            emit_prep_diagnostic(
                output_dir,
                config.output.prefix,
                "minhash_dedup_complete",
                **{k: v for k, v in dedup_stats.items() if k != "drop_counts_per_source"},
                drop_counts=dedup_stats["drop_counts_per_source"],
            )
            all_documents = {
                name: _FilteredSpilledDocs(spilled, kept_indices[name])
                for name, spilled in all_documents.items()
            }

        if not config.curriculum.enabled and config.splits.stratified:
            print("\n[3/4] Stratified split  (streaming optimized)", flush=True)
            logger.info("Using streaming stratified path for source-index-based splitting")
            selected_indices = select_source_indices(
                cast(dict[str, Sequence[np.ndarray]], all_documents),
                config.mixing,
                diagnostic_context={
                    "output_dir": str(output_dir),
                    "prefix": config.output.prefix,
                },
            )
            split_indices = _split_source_indices(selected_indices, config)
            emit_prep_diagnostic(
                output_dir,
                config.output.prefix,
                "split_source_indices_done",
                split_doc_counts={
                    split_name: {name: int(len(indices)) for name, indices in by_source.items()}
                    for split_name, by_source in split_indices.items()
                },
                split_index_bytes={
                    split_name: int(sum(indices.nbytes for indices in by_source.values()))
                    for split_name, by_source in split_indices.items()
                },
            )
            manifest = save_outputs_from_source_indices(
                split_indices,
                all_documents,
                tokenizer_path,
                config,
                shuffle_seed=config.splits.seed,
            )
            shard_counts = {
                name: len(paths) for name, paths in manifest.get("output_shards", {}).items()
            }
            print(f"\nDone. Outputs in {output_dir}", flush=True)
            print(f"      splits: {shard_counts}", flush=True)
            logger.info(
                "Preparation complete (streaming stratified): outputs=%s shard_counts=%s",
                manifest.get("output_paths", {}),
                shard_counts,
            )
            return manifest

        # ── Step 3: mix ──────────────────────────────────────────────────────
        print(f"\n[3/4] Mixing  ({config.mixing.strategy})", flush=True)
        logger.info("Mixing documents: strategy=%s", config.mixing.strategy)
        mixed_documents = resolve_mixing_strategy(config.mixing).mix(
            cast(dict[str, Sequence[np.ndarray]], all_documents),
            config.mixing,
        )
        print(f"      {len(mixed_documents):,} documents", flush=True)
        logger.info("Mix complete: mixed_docs=%d", len(mixed_documents))

        # ── Step 4: split + save ─────────────────────────────────────────────
        print("\n[4/4] Splitting and saving", flush=True)
        if config.curriculum.enabled:
            logger.info("Applying curriculum: type=%s", config.curriculum.type)
            stages = resolve_curriculum_strategy(config.curriculum).build(
                mixed_documents, config.curriculum
            )
            splits = _split_curriculum(stages, config)
        else:
            logger.info(
                "Applying standard split: train=%.3f val=%.3f test=%.3f",
                config.splits.train,
                config.splits.val,
                config.splits.test,
            )
            splits = resolve_split_strategy(config.splits).split(mixed_documents, config.splits)

        manifest = save_outputs(splits, tokenizer_path, config)

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
