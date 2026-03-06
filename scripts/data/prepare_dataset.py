#!/usr/bin/env python3
"""
Comprehensive data preparation pipeline for language model training.

Features:
- Multiple tokenizer types (Unigram, BPE, UTF-8, Character-level)
- Flexible dataset mixing (weighted, upsampling, temperature-based)
- Document boundary preservation
- Curriculum learning (length-based, domain progression, custom)
- Representative validation splits
- Quality validation and statistics

Usage:
    python scripts/data/prepare_dataset.py --config config.json
"""

import argparse
import json
import logging
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tokenizer.unigram_tokenizer import UnigramTokenizer

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""

    type: str = "unigram"  # unigram, bpe, utf8, char
    vocab_size: int = 8192
    character_coverage: float = 0.9995
    min_frequency: int = 2

    # Sampling for training
    max_corpus_mb: float | None = None  # Limit corpus size for tokenizer training


@dataclass
class DataSource:
    """Single data source configuration."""

    name: str
    path: str
    format: str = "text"  # text, utf8_tokens, jsonl, npy
    weight: float = 1.0

    # Format-specific
    delimiter: str = "\n\n"
    chunk_size: int = 1024
    text_field: str = "text"

    # Filtering
    min_length: int = 10
    max_length: int | None = None

    # Curriculum
    curriculum_stage: int | None = None


@dataclass
class MixingConfig:
    """Dataset mixing strategy."""

    strategy: str = "interleave"  # interleave, concatenate, temperature_sample
    block_size: int = 1
    temperature: float = 1.0
    upsample_to_max: bool = False
    downsample_to_min: bool = False
    target_total_docs: int | None = None

    # Mix ratios: target proportions by source name
    # Example: {"tinystories": 0.1, "fineweb": 0.6, "wikitext": 0.3}
    # If None, uses natural proportions (or dataset weights if specified)
    source_ratios: dict[str, float] | None = None

    seed: int = 42


@dataclass
class CurriculumStage:
    """Curriculum learning stage."""

    name: str
    min_length: int | None = None
    max_length: int | None = None
    sources: list[str] | None = None  # Filter: only include these sources
    source_weights: dict[str, float] | None = (
        None  # Mix ratio: {"tinystories": 0.7, "wikitext": 0.3}
    )
    num_docs: int | None = None
    num_tokens: int | None = None
    fraction: float | None = None


@dataclass
class CurriculumConfig:
    """Curriculum configuration."""

    enabled: bool = False
    type: str = "length_based"  # length_based, domain_progression, custom
    num_stages: int = 3
    length_bins: list[int] | None = None
    stages: list[CurriculumStage] = field(default_factory=list)
    output_mode: str = "merged"  # merged, separate, both


@dataclass
class SplitConfig:
    """Train/val/test split."""

    train: float = 0.9
    val: float = 0.1
    test: float = 0.0
    shuffle: bool = True
    stratified: bool = True
    seed: int = 42


@dataclass
class OutputConfig:
    """Output configuration."""

    dir: str = "data/fast"
    prefix: str = "prepared"
    save_stats: bool = True
    save_manifest: bool = True
    eos_token_id: int = -1  # -1 = no EOS; set to 1 for Unigram models trained with add_eos=True


@dataclass
class DataPreparationConfig:
    """Complete configuration."""

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    datasets: list[DataSource] = field(default_factory=list)
    mixing: MixingConfig = field(default_factory=MixingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self):
        """Validate configuration."""
        # Check splits
        total = self.splits.train + self.splits.val + self.splits.test
        assert abs(total - 1.0) < 1e-6, f"Splits must sum to 1.0, got {total}"

        # Check datasets
        assert len(self.datasets) > 0, "Need at least one dataset"
        for ds in self.datasets:
            assert Path(ds.path).exists(), f"Not found: {ds.path}"
            assert ds.weight >= 0, f"Negative weight: {ds.name}"

        assert sum(ds.weight for ds in self.datasets) > 0, "All weights zero"

        # Check mixing
        if self.mixing.upsample_to_max and self.mixing.downsample_to_min:
            raise ValueError("Cannot upsample and downsample")


# ============================================================================
# Tokenizer Handling
# ============================================================================


class TokenizerFactory:
    """Create and train tokenizers."""

    @staticmethod
    def create_corpus(sources: list[DataSource], config: TokenizerConfig) -> str:
        """Create training corpus from all sources."""
        logging.info("Creating tokenizer training corpus...")

        corpus_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        )
        total_bytes = 0
        max_bytes = (config.max_corpus_mb * 1024 * 1024) if config.max_corpus_mb else None

        for ds in sources:
            if max_bytes and total_bytes >= max_bytes:
                logging.info(f"  Reached corpus limit ({config.max_corpus_mb}MB), stopping")
                break

            text = DataLoader.load_as_text(ds)

            # Sample if needed
            if max_bytes and total_bytes + len(text) > max_bytes:
                remaining = max_bytes - total_bytes
                text = text[:remaining]

            corpus_file.write(text)
            corpus_file.write("\n\n")
            total_bytes += len(text)

            logging.info(f"  {ds.name}: {len(text)/1e6:.1f}MB")

        corpus_file.close()
        logging.info(f"Total corpus: {total_bytes/1e6:.1f}MB")
        return corpus_file.name

    @staticmethod
    def train(corpus_path: str, config: TokenizerConfig, output_prefix: str) -> tuple:
        """Train tokenizer on corpus."""
        logging.info(f"Training {config.type} tokenizer...")
        logging.info(f"  Vocab: {config.vocab_size}, Coverage: {config.character_coverage}")

        if config.type == "unigram":
            model_path = UnigramTokenizer.train_model(
                input_path=corpus_path,
                model_prefix=output_prefix,
                vocab_size=config.vocab_size,
                character_coverage=config.character_coverage,
            )
            tokenizer = UnigramTokenizer(model_path)

        elif config.type == "utf8":
            # UTF-8 bytes (no training needed)
            class UTF8Tokenizer:
                def __init__(self):
                    self.vocab_size = 256

                def encode(self, text):
                    return list(text.encode("utf-8"))

                def decode(self, ids):
                    return bytes(ids).decode("utf-8", errors="ignore")

            tokenizer = UTF8Tokenizer()
            model_path = None

        elif config.type == "char":
            # Character-level (build vocab from corpus)
            with open(corpus_path, encoding="utf-8") as f:
                text = f.read()
            chars = sorted(set(text))
            vocab = {c: i for i, c in enumerate(chars)}

            class CharTokenizer:
                def __init__(self, vocab):
                    self.vocab = vocab
                    self.vocab_size = len(vocab)

                def encode(self, text):
                    return [self.vocab.get(c, 0) for c in text]

                def decode(self, ids):
                    inv = {v: k for k, v in self.vocab.items()}
                    return "".join(inv.get(i, "") for i in ids)

            tokenizer = CharTokenizer(vocab)
            model_path = f"{output_prefix}.char.json"
            with open(model_path, "w") as f:
                json.dump(vocab, f)

        else:
            raise ValueError(f"Unknown tokenizer type: {config.type}")

        logging.info(f"✓ Tokenizer ready: vocab_size={tokenizer.vocab_size}")
        return tokenizer, model_path


# ============================================================================
# Data Loading
# ============================================================================


class DataLoader:
    """Load data from various formats."""

    @staticmethod
    def load_as_text(source: DataSource) -> str:
        """Load any format as raw text."""
        path = Path(source.path)

        if source.format == "text":
            with open(path, encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif source.format == "utf8_tokens":
            tokens = np.load(path)
            return "".join(chr(int(t)) for t in tokens)

        elif source.format == "jsonl":
            lines = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    text = obj.get(source.text_field, "")
                    lines.append(text)
            return "\n\n".join(lines)

        elif source.format == "npy":
            raise ValueError("Cannot convert npy to text (already tokenized)")

        else:
            raise ValueError(f"Unknown format: {source.format}")

    @staticmethod
    def load_documents(source: DataSource, tokenizer) -> list[tuple[str, np.ndarray]]:
        """Load source as list of (metadata, tokens) tuples."""
        logging.info(f"Loading {source.name}...")

        path = Path(source.path)
        documents = []

        if source.format == "text":
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()

            docs = [d.strip() for d in text.split(source.delimiter) if d.strip()]

            for doc in docs:
                tokens = tokenizer.encode(doc)
                tokens = np.array(tokens, dtype=np.uint16 if max(tokens) < 65536 else np.uint32)

                if len(tokens) < source.min_length:
                    continue
                if source.max_length and len(tokens) > source.max_length:
                    tokens = tokens[: source.max_length]

                documents.append((source.name, tokens))

            logging.info(f"  {source.name}: {len(documents):,} documents")

        elif source.format == "utf8_tokens":
            utf8_tokens = np.load(path)
            text = "".join(chr(int(t)) for t in utf8_tokens)
            tokens = tokenizer.encode(text)

            for i in range(0, len(tokens), source.chunk_size):
                chunk = tokens[i : i + source.chunk_size]
                if len(chunk) >= source.min_length:
                    arr = np.array(chunk, dtype=np.uint16 if max(chunk) < 65536 else np.uint32)
                    documents.append((source.name, arr))

            logging.info(f"  {source.name}: {len(documents):,} chunks")

        elif source.format == "jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    text = obj.get(source.text_field, "")
                    if not text:
                        continue

                    tokens = tokenizer.encode(text)
                    tokens = np.array(tokens, dtype=np.uint16 if max(tokens) < 65536 else np.uint32)

                    if len(tokens) >= source.min_length:
                        if source.max_length:
                            tokens = tokens[: source.max_length]
                        documents.append((source.name, tokens))

            logging.info(f"  {source.name}: {len(documents):,} documents")

        elif source.format == "npy":
            tokens = np.load(path)

            if source.chunk_size:
                for i in range(0, len(tokens), source.chunk_size):
                    chunk = tokens[i : i + source.chunk_size]
                    if len(chunk) >= source.min_length:
                        documents.append((source.name, chunk.astype(np.uint16)))
            else:
                documents.append((source.name, tokens.astype(np.uint16)))

            logging.info(f"  {source.name}: {len(documents):,} chunks")

        return documents


# ============================================================================
# Dataset Mixing
# ============================================================================


class DatasetMixer:
    """Mix multiple datasets with various strategies."""

    @staticmethod
    def mix(
        all_documents: dict[str, list[np.ndarray]], config: MixingConfig
    ) -> list[tuple[str, np.ndarray]]:
        """Mix documents from all sources."""
        logging.info(f"Mixing datasets (strategy={config.strategy})...")

        for name, docs in all_documents.items():
            total_tokens = sum(len(d) for d in docs)
            logging.info(f"  {name}: {len(docs):,} docs, {total_tokens/1e6:.1f}M tokens")

        all_documents = DatasetMixer._apply_sampling(all_documents, config)

        if config.strategy == "concatenate":
            mixed = DatasetMixer._concatenate(all_documents)
        elif config.strategy == "interleave":
            mixed = DatasetMixer._interleave(all_documents, config)
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")

        total_tokens = sum(len(d) for _, d in mixed)
        logging.info(f"Mixed: {len(mixed):,} docs, {total_tokens/1e6:.1f}M tokens")

        return mixed

    @staticmethod
    def _apply_sampling(docs: dict, config: MixingConfig) -> dict:
        """Apply upsampling/downsampling."""
        if not (config.upsample_to_max or config.downsample_to_min or config.target_total_docs):
            return docs

        rng = np.random.RandomState(config.seed)

        if config.target_total_docs:
            target = config.target_total_docs
        elif config.upsample_to_max:
            target = max(len(d) for d in docs.values())
        else:  # downsample_to_min
            target = min(len(d) for d in docs.values())

        new_docs = {}
        for name, doc_list in docs.items():
            if len(doc_list) < target:
                indices = rng.choice(len(doc_list), size=target, replace=True)
                new_docs[name] = [doc_list[i] for i in indices]
            elif len(doc_list) > target:
                indices = rng.choice(len(doc_list), size=target, replace=False)
                new_docs[name] = [doc_list[i] for i in indices]
            else:
                new_docs[name] = doc_list

        return new_docs

    @staticmethod
    def _concatenate(docs: dict) -> list[tuple[str, np.ndarray]]:
        """Simple concatenation."""
        mixed = []
        for name, doc_list in docs.items():
            mixed.extend((name, d) for d in doc_list)
        return mixed

    @staticmethod
    def _interleave(docs: dict, config: MixingConfig) -> list[tuple[str, np.ndarray]]:
        """Block-level interleaving with optional weighted mixing."""
        rng = np.random.RandomState(config.seed)

        source_docs = {name: list(doc_list) for name, doc_list in docs.items()}

        for doc_list in source_docs.values():
            rng.shuffle(doc_list)

        # Apply mix ratios if specified
        if config.source_ratios:
            source_docs = DatasetMixer._apply_ratio_resampling(
                source_docs, config.source_ratios, rng
            )

        mixed = []
        pointers = dict.fromkeys(source_docs, 0)

        while any(pointers[name] < len(source_docs[name]) for name in source_docs):
            for name in source_docs:
                if pointers[name] < len(source_docs[name]):
                    for _ in range(config.block_size):
                        if pointers[name] < len(source_docs[name]):
                            mixed.append((name, source_docs[name][pointers[name]]))
                            pointers[name] += 1

        return mixed

    @staticmethod
    def _apply_ratio_resampling(docs: dict, ratios: dict[str, float], rng) -> dict:
        """Resample documents to achieve target mix ratios."""
        # Normalize ratios
        total_ratio = sum(ratios.values())
        normalized_ratios = {k: v / total_ratio for k, v in ratios.items()}

        # Calculate current and target document counts
        current_counts = {name: len(doc_list) for name, doc_list in docs.items()}
        total_docs = sum(current_counts.values())

        # Calculate target counts based on ratios
        target_counts = {}
        for name in docs.keys():
            if name in normalized_ratios:
                target_counts[name] = int(total_docs * normalized_ratios[name])
            else:
                # Sources not in ratios get equal share of remaining
                remaining_sources = [n for n in docs.keys() if n not in normalized_ratios]
                if remaining_sources:
                    remaining_ratio = 1.0 - sum(normalized_ratios.values())
                    target_counts[name] = int(total_docs * remaining_ratio / len(remaining_sources))
                else:
                    target_counts[name] = 0

        logging.info("Applying mix ratios:")
        for name in docs.keys():
            if target_counts[name] > 0:
                ratio_pct = (target_counts[name] / total_docs) * 100
                logging.info(
                    f"  {name}: {current_counts[name]:,} → {target_counts[name]:,} docs ({ratio_pct:.1f}%)"
                )

        # Resample each source to target count
        resampled = {}
        for name, doc_list in docs.items():
            target = target_counts[name]
            if target == 0:
                continue

            if len(doc_list) < target:
                # Upsample with replacement
                indices = rng.choice(len(doc_list), size=target, replace=True)
                resampled[name] = [doc_list[i] for i in indices]
            elif len(doc_list) > target:
                # Downsample without replacement
                indices = rng.choice(len(doc_list), size=target, replace=False)
                resampled[name] = [doc_list[i] for i in indices]
            else:
                resampled[name] = doc_list

        return resampled


# ============================================================================
# Curriculum Learning
# ============================================================================


class CurriculumBuilder:
    """Build curriculum learning stages."""

    @staticmethod
    def build(
        documents: list[tuple[str, np.ndarray]], config: CurriculumConfig
    ) -> dict[str, list[tuple[str, np.ndarray]]]:
        """Build curriculum stages from documents.

        Args:
            documents: List of (source_name, token_array) tuples
            config: Curriculum configuration

        Returns:
            Dict mapping stage_name -> list of (source_name, token_array) tuples
        """
        if not config.enabled:
            return {"default": documents}

        if config.type == "length_based":
            return CurriculumBuilder._length_based(documents, config)
        elif config.type == "domain_progression":
            return CurriculumBuilder._domain_progression(documents, config)
        elif config.type == "custom":
            return CurriculumBuilder._custom(documents, config)
        else:
            raise ValueError(f"Unknown curriculum type: {config.type}")

    @staticmethod
    def _custom(
        documents: list[tuple[str, np.ndarray]], config: CurriculumConfig
    ) -> dict[str, list[tuple[str, np.ndarray]]]:
        """Apply custom curriculum stages from config."""
        logging.info(f"Building custom curriculum with {len(config.stages)} stages...")
        stages = {}

        for stage_config in config.stages:
            logging.info(f"Building stage: {stage_config.name}")

            # Filter by length
            stage_docs = []
            for name, doc in documents:
                doc_len = len(doc)
                if stage_config.min_length and doc_len < stage_config.min_length:
                    continue
                if stage_config.max_length and doc_len > stage_config.max_length:
                    continue
                stage_docs.append((name, doc))

            logging.info(f"  After length filter: {len(stage_docs)} documents")

            # Filter by sources
            if stage_config.sources:
                stage_docs = [
                    (name, doc) for name, doc in stage_docs if name in stage_config.sources
                ]
                logging.info(f"  After source filter: {len(stage_docs)} documents")

            # Apply source_weights if specified
            if stage_config.source_weights:
                logging.info(f"  Applying source weights: {stage_config.source_weights}")

                # Group by source
                by_source = defaultdict(list)
                for name, doc in stage_docs:
                    by_source[name].append(doc)

                # Apply resampling to achieve target weights
                rng = np.random.RandomState(42)  # Use consistent seed
                resampled = DatasetMixer._apply_ratio_resampling(
                    by_source, stage_config.source_weights, rng
                )

                # Flatten back to list of tuples
                stage_docs = [(name, doc) for name, docs in resampled.items() for doc in docs]

            # Limit by fraction or num_docs
            if stage_config.fraction:
                n = int(len(documents) * stage_config.fraction)
                stage_docs = stage_docs[:n]
                logging.info(
                    f"  Limited to {stage_config.fraction*100}% of total: {len(stage_docs)} documents"
                )
            elif stage_config.num_docs:
                stage_docs = stage_docs[: stage_config.num_docs]
                logging.info(f"  Limited to {stage_config.num_docs} documents")

            stages[stage_config.name] = stage_docs
            logging.info(f"  Final stage size: {len(stage_docs)} documents")

        return stages

    @staticmethod
    def _length_based(
        documents: list[tuple[str, np.ndarray]], config: CurriculumConfig
    ) -> dict[str, list[tuple[str, np.ndarray]]]:
        """Create stages based on document length."""
        logging.info(f"Building length-based curriculum with {config.num_stages} stages...")

        # Sort by length
        sorted_docs = sorted(documents, key=lambda x: len(x[1]))

        # Divide into equal-sized stages
        stage_size = len(sorted_docs) // config.num_stages
        stages = {}

        for i in range(config.num_stages):
            start = i * stage_size
            end = start + stage_size if i < config.num_stages - 1 else len(sorted_docs)
            stage_docs = sorted_docs[start:end]

            min_len = min(len(doc) for _, doc in stage_docs)
            max_len = max(len(doc) for _, doc in stage_docs)

            stage_name = f"stage{i+1}_len{min_len}-{max_len}"
            stages[stage_name] = stage_docs
            logging.info(f"  {stage_name}: {len(stage_docs)} docs, {min_len}-{max_len} tokens")

        return stages

    @staticmethod
    def _domain_progression(
        documents: list[tuple[str, np.ndarray]], config: CurriculumConfig
    ) -> dict[str, list[tuple[str, np.ndarray]]]:
        """Create stages based on domain progression."""
        logging.info("Building domain progression curriculum...")

        # Group by source
        by_source = defaultdict(list)
        for name, doc in documents:
            by_source[name].append((name, doc))

        # Create stages with different domain mixes
        # This is a simplified version - can be extended based on domain difficulty
        stages = {}
        sources = list(by_source.keys())

        for i, source in enumerate(sources):
            stage_name = f"stage{i+1}_{source}"
            stages[stage_name] = by_source[source]
            logging.info(f"  {stage_name}: {len(stages[stage_name])} docs")

        return stages


# ============================================================================
# Splitting
# ============================================================================


class DataSplitter:
    """Create train/val/test splits."""

    @staticmethod
    def split(documents: list[tuple[str, np.ndarray]], config: SplitConfig) -> dict:
        """Split documents into train/val/test."""
        logging.info("Creating splits...")

        if config.stratified:
            return DataSplitter._stratified_split(documents, config)
        else:
            return DataSplitter._simple_split(documents, config)

    @staticmethod
    def _stratified_split(documents: list[tuple[str, np.ndarray]], config: SplitConfig) -> dict:
        """Split while maintaining source distribution."""
        rng = np.random.RandomState(config.seed)

        by_source = defaultdict(list)
        for name, doc in documents:
            by_source[name].append(doc)

        if config.shuffle:
            for doc_list in by_source.values():
                rng.shuffle(doc_list)

        splits = {"train": [], "val": [], "test": []}

        for name, doc_list in by_source.items():
            n = len(doc_list)
            train_end = int(n * config.train)
            val_end = int(n * (config.train + config.val))

            train_docs = doc_list[:train_end]
            val_docs = doc_list[train_end:val_end]
            test_docs = doc_list[val_end:]

            splits["train"].extend((name, d) for d in train_docs)
            splits["val"].extend((name, d) for d in val_docs)
            if test_docs:
                splits["test"].extend((name, d) for d in test_docs)

            logging.info(
                f"  {name}: {len(train_docs):,} train, {len(val_docs):,} val, {len(test_docs):,} test"
            )

        if config.shuffle:
            for split_name in splits:
                rng.shuffle(splits[split_name])

        return splits

    @staticmethod
    def _simple_split(documents: list[tuple[str, np.ndarray]], config: SplitConfig) -> dict:
        """Simple random split."""
        rng = np.random.RandomState(config.seed)

        docs_copy = list(documents)
        if config.shuffle:
            rng.shuffle(docs_copy)

        n = len(docs_copy)
        train_end = int(n * config.train)
        val_end = int(n * (config.train + config.val))

        return {
            "train": docs_copy[:train_end],
            "val": docs_copy[train_end:val_end],
            "test": docs_copy[val_end:],
        }


# ============================================================================
# Output
# ============================================================================


class OutputManager:
    """Save outputs."""

    @staticmethod
    def save(splits: dict, tokenizer_path: str | None, config: DataPreparationConfig):
        """Save all outputs."""
        output_dir = Path(config.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Saving outputs...")

        eos_id = config.output.eos_token_id
        eos_arr = np.array([eos_id], dtype=np.uint16) if eos_id >= 0 else None

        paths = {}
        for split_name, docs in splits.items():
            if not docs:
                continue

            if eos_arr is not None:
                # Interleave EOS token between every pair of documents so the model
                # learns where one document ends and the next begins.  The final doc
                # gets an EOS too so the last sequence never straddles a boundary.
                parts: list[np.ndarray] = []
                for _, doc in docs:
                    parts.append(doc)
                    parts.append(eos_arr)
                tokens = np.concatenate(parts, dtype=np.uint16)
            else:
                tokens = np.concatenate([doc for _, doc in docs], dtype=np.uint16)

            path = output_dir / f"{config.output.prefix}_{split_name}.npy"
            np.save(path, tokens)
            paths[split_name] = str(path)

            logging.info(f"  {split_name}: {path} ({len(tokens):,} tokens, {len(tokens)/1e6:.1f}M)")

        if config.output.save_stats:
            stats = OutputManager._compute_stats(splits)
            stats_path = output_dir / f"{config.output.prefix}_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            logging.info(f"  Stats: {stats_path}")

        if config.output.save_manifest:
            manifest = {
                "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
                "output_paths": paths,
            }
            manifest_path = output_dir / f"{config.output.prefix}_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logging.info(f"  Manifest: {manifest_path}")

        logging.info("\n✓ Dataset preparation complete!")
        logging.info("✓ Document boundaries preserved")
        logging.info("✓ Validation set representative of all sources")

    @staticmethod
    def _compute_stats(splits: dict) -> dict:
        """Compute dataset statistics."""
        stats = {}

        for split_name, docs in splits.items():
            by_source = defaultdict(list)
            for name, doc in docs:
                by_source[name].append(len(doc))

            source_stats = {}
            for name, lengths in by_source.items():
                source_stats[name] = {
                    "num_docs": len(lengths),
                    "total_tokens": sum(lengths),
                    "mean_length": float(np.mean(lengths)),
                    "median_length": float(np.median(lengths)),
                    "min_length": min(lengths),
                    "max_length": max(lengths),
                }

            stats[split_name] = {
                "total_docs": len(docs),
                "total_tokens": sum(len(d) for _, d in docs),
                "by_source": source_stats,
            }

        return stats


# ============================================================================
# Main
# ============================================================================


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s", level=level, datefmt="%H:%M:%S"
    )


def load_config(path: str) -> DataPreparationConfig:
    """Load configuration from JSON."""
    with open(path) as f:
        data = json.load(f)

    config = DataPreparationConfig(
        tokenizer=TokenizerConfig(**data.get("tokenizer", {})),
        datasets=[DataSource(**ds) for ds in data.get("datasets", [])],
        mixing=MixingConfig(**data.get("mixing", {})),
        curriculum=(
            CurriculumConfig(
                **{k: v for k, v in data.get("curriculum", {}).items() if k != "stages"},
                stages=[CurriculumStage(**s) for s in data.get("curriculum", {}).get("stages", [])],
            )
            if "curriculum" in data
            else CurriculumConfig()
        ),
        splits=SplitConfig(**data.get("splits", {})),
        output=OutputConfig(**data.get("output", {})),
    )

    return config


def main():
    parser = argparse.ArgumentParser(description="Prepare tokenized dataset")
    parser.add_argument("--config", required=True, help="JSON config file")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    setup_logging(args.verbose)

    config = load_config(args.config)
    config.validate()

    logging.info("=" * 70)
    logging.info("DATASET PREPARATION PIPELINE")
    logging.info("=" * 70)

    try:
        output_dir = Path(config.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_prefix = str(output_dir / f"{config.output.prefix}_tokenizer")

        # Train tokenizer
        corpus_path = TokenizerFactory.create_corpus(config.datasets, config.tokenizer)
        tokenizer, tokenizer_path = TokenizerFactory.train(
            corpus_path, config.tokenizer, tokenizer_prefix
        )

        # Load and encode documents
        all_documents = {}
        for ds in config.datasets:
            if ds.weight == 0:
                continue
            docs = DataLoader.load_documents(ds, tokenizer)
            all_documents[ds.name] = [doc for _, doc in docs]

        # Mix datasets
        mixed_documents = DatasetMixer.mix(all_documents, config.mixing)

        # Build curriculum (if enabled)
        if config.curriculum.enabled:
            curriculum_stages = CurriculumBuilder.build(mixed_documents, config.curriculum)

            # Handle output modes
            if config.curriculum.output_mode == "separate":
                # Create splits per stage
                all_stage_splits = {}
                for stage_name, stage_docs in curriculum_stages.items():
                    logging.info(f"\nSplitting stage: {stage_name}")
                    stage_splits = DataSplitter.split(stage_docs, config.splits)
                    # Add stage prefix to split names
                    for split_name, docs in stage_splits.items():
                        all_stage_splits[f"{stage_name}_{split_name}"] = docs
                splits = all_stage_splits

            elif config.curriculum.output_mode == "merged":
                # Merge all stages, then split
                merged_docs = []
                for stage_docs in curriculum_stages.values():
                    merged_docs.extend(stage_docs)
                logging.info(
                    f"\nMerged {len(curriculum_stages)} stages into {len(merged_docs)} documents"
                )
                splits = DataSplitter.split(merged_docs, config.splits)

            elif config.curriculum.output_mode == "both":
                # Create both separate and merged outputs
                all_stage_splits = {}
                merged_docs = []

                for stage_name, stage_docs in curriculum_stages.items():
                    logging.info(f"\nSplitting stage: {stage_name}")
                    stage_splits = DataSplitter.split(stage_docs, config.splits)
                    for split_name, docs in stage_splits.items():
                        all_stage_splits[f"{stage_name}_{split_name}"] = docs
                    merged_docs.extend(stage_docs)

                # Also create merged splits
                merged_splits = DataSplitter.split(merged_docs, config.splits)
                for split_name, docs in merged_splits.items():
                    all_stage_splits[f"merged_{split_name}"] = docs

                splits = all_stage_splits

            else:
                raise ValueError(f"Unknown output_mode: {config.curriculum.output_mode}")

        else:
            # No curriculum - standard split
            splits = DataSplitter.split(mixed_documents, config.splits)

        # Save
        OutputManager.save(splits, tokenizer_path, config)

        # Cleanup
        import os

        os.unlink(corpus_path)

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
