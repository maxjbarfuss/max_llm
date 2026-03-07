"""Orchestration pipeline for dataset preparation."""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
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
from src.tokenizer import TokenizerFactory, UnigramTokenizer


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

        output_dir = Path(config.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer_prefix = str(output_dir / f"{config.output.prefix}_tokenizer")
        tokenizer, tokenizer_path = self._build_tokenizer(config, tokenizer_prefix)

        all_documents: dict[str, list[np.ndarray]] = {}
        for source in config.datasets:
            if source.weight == 0:
                continue
            reader = resolve_format_reader(source.format)
            docs = reader.read_documents(source, tokenizer)
            all_documents[source.name] = [doc for _, doc in docs]

        mixed_documents = resolve_mixing_strategy(config.mixing).mix(all_documents, config.mixing)

        if config.curriculum.enabled:
            stages = resolve_curriculum_strategy(config.curriculum).build(
                mixed_documents,
                config.curriculum,
            )
            splits = self._split_curriculum(stages, config)
        else:
            splits = resolve_split_strategy(config.splits).split(mixed_documents, config.splits)

        manifest = self._save_outputs(splits, tokenizer_path, config)
        return manifest

    def run_from_config_path(self, config_path: str | Path) -> dict[str, Any]:
        config = load_config(config_path)
        return self.run(config)

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

                text = self._load_source_as_text(source)
                if max_bytes and total_bytes + len(text) > max_bytes:
                    text = text[: max_bytes - total_bytes]

                corpus_file.write(text)
                corpus_file.write("\n\n")
                total_bytes += len(text)
        finally:
            corpus_file.close()

        return corpus_file.name

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
            return "".join(chr(int(t)) for t in tokens)

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

        for split_name, docs in splits.items():
            if not docs:
                continue

            tokens = self._flatten_docs_with_optional_eos(docs, config.output.eos_token_id)
            file_path = output_dir / f"{config.output.prefix}_{split_name}.npy"
            np.save(file_path, tokens)
            output_paths[split_name] = str(file_path)

        if config.output.save_stats:
            stats_path = output_dir / f"{config.output.prefix}_stats.json"
            stats_path.write_text(
                json.dumps(self._compute_stats(splits), indent=2), encoding="utf-8"
            )

        manifest = {
            "tokenizer_path": tokenizer_path,
            "output_paths": output_paths,
        }

        if config.output.save_manifest:
            manifest_path = output_dir / f"{config.output.prefix}_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return manifest

    def _flatten_docs_with_optional_eos(
        self,
        docs: list[tuple[str, np.ndarray]],
        eos_token_id: int,
    ) -> np.ndarray:
        if eos_token_id < 0:
            return np.concatenate([doc for _, doc in docs])

        dtype = np.uint32 if any(doc.dtype == np.uint32 for _, doc in docs) else np.uint16
        eos = np.array([eos_token_id], dtype=dtype)
        parts: list[np.ndarray] = []
        for _, doc in docs:
            parts.append(doc.astype(dtype, copy=False))
            parts.append(eos)
        return np.concatenate(parts)

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
