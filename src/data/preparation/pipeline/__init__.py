"""Dataset preparation pipeline."""

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.data.preparation.config import DataPreparationConfig, DataSource, load_config
from src.data.preparation.strategies import (
    resolve_curriculum_strategy,
    resolve_format_reader,
    resolve_mixing_strategy,
    resolve_split_strategy,
)

from ._corpus import build_tokenizer, load_source_as_text
from ._save import save_outputs
from ._spill import _SpilledDocs, read_and_spill

logger = logging.getLogger(__name__)


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
            shutil.rmtree(spill_dir, ignore_errors=True)

    def run_from_config_path(self, config_path: str | Path) -> dict[str, Any]:
        return self.run(load_config(config_path))

    def _load_source_as_text(self, source: DataSource) -> str:
        return load_source_as_text(source)

    def _run_with_spill_dir(
        self, config: DataPreparationConfig, output_dir: Path, spill_dir: Path
    ) -> dict[str, Any]:
        n_active = sum(1 for s in config.datasets if s.weight > 0)

        # ── Step 1: tokenizer ────────────────────────────────────────────────
        print(f"\n[1/4] Tokenizer  ({config.tokenizer.type})", flush=True)
        logger.info("Building tokenizer: type=%s", config.tokenizer.type)
        tokenizer_prefix = str(output_dir / f"{config.output.prefix}_tokenizer")
        tokenizer, tokenizer_path = build_tokenizer(config, tokenizer_prefix)
        print(f"      ready → {tokenizer_path or 'in-memory'}", flush=True)
        logger.info("Tokenizer ready: path=%s", tokenizer_path)

        # ── Step 2: read sources ─────────────────────────────────────────────
        print(f"\n[2/4] Reading {n_active} source(s)", flush=True)
        all_documents: dict[str, _SpilledDocs] = {}
        for source in config.datasets:
            if source.weight == 0:
                continue
            logger.info(
                "Reading source: name=%s format=%s path=%s",
                source.name,
                source.format,
                source.path,
            )
            spilled = read_and_spill(source, resolve_format_reader(source.format), tokenizer, spill_dir)
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
