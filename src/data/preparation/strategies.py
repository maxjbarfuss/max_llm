"""Strategy components for data preparation."""

import json
import unicodedata
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np

from src.data.preparation.config import (
    CurriculumConfig,
    DataSource,
    MixingConfig,
    SplitConfig,
)


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...


def _token_array(tokens: list[int]) -> np.ndarray:
    if not tokens:
        return np.array([], dtype=np.uint16)
    dtype = np.uint16 if max(tokens) < 65536 else np.uint32
    return np.array(tokens, dtype=dtype)


T = TypeVar("T")


def _shuffle_with_rng(items: list[T], rng: np.random.RandomState) -> None:
    if len(items) < 2:
        return
    indices = rng.permutation(len(items))
    shuffled = [items[int(i)] for i in indices]
    items[:] = shuffled


def _normalize_text(text: str) -> str:
    """NFKC-normalize and strip non-printable control characters.

    NFKC collapses compatibility variants (fullwidth, ligatures, etc.) into
    their canonical forms.  The category-Cc filter removes ASCII and Latin-1
    control bytes that survive byte-level decoding but are meaningless to a
    language model, while preserving the three whitespace controls (\n \r \t)
    that carry real document structure.
    """
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\n\r\t")


def _filter_unk(
    tokens: list[int],
    unk_id: int = 0,
    max_unk_rate: float = 0.02,
) -> list[int]:
    """Strip unk tokens, or return [] (skip) if unk rate exceeds max_unk_rate. unk_id=-1 disables."""
    if unk_id < 0 or not tokens:
        return tokens
    n_unk = sum(1 for t in tokens if t == unk_id)
    if n_unk == 0:
        return tokens
    if n_unk / len(tokens) > max_unk_rate:
        return []
    return [t for t in tokens if t != unk_id]


class FormatReader(ABC):
    def iter_documents(
        self, source: DataSource, tokenizer: TokenizerLike
    ) -> Iterator[tuple[str, np.ndarray]]:
        """Yield documents one at a time (override for memory-efficient streaming)."""
        yield from self.read_documents(source, tokenizer)

    @abstractmethod
    def read_documents(
        self, source: DataSource, tokenizer: TokenizerLike
    ) -> list[tuple[str, np.ndarray]]: ...


class TextFormatReader(FormatReader):
    @staticmethod
    def _iter_text_docs(file_path: Path, delimiter: str = "\n\n"):
        buffer = ""
        with file_path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                buffer += line
                while delimiter in buffer:
                    idx = buffer.index(delimiter)
                    doc = buffer[:idx].strip()
                    buffer = buffer[idx + len(delimiter) :]
                    if doc:
                        yield doc
        if buffer.strip():
            yield buffer.strip()

    def iter_documents(
        self, source: DataSource, tokenizer: TokenizerLike
    ) -> Iterator[tuple[str, np.ndarray]]:
        path = Path(source.path)
        files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
        for file_path in files:
            for doc in self._iter_text_docs(file_path, source.delimiter):
                doc = _normalize_text(doc)
                if len(doc) < source.min_length:
                    continue
                tokens = _filter_unk(tokenizer.encode(doc))
                if not tokens:
                    continue
                if source.max_length and len(tokens) > source.max_length:
                    tokens = tokens[: source.max_length]
                encoded = _token_array(tokens)
                if len(encoded) < source.min_length:
                    continue
                yield (source.name, encoded)

    def read_documents(
        self, source: DataSource, tokenizer: TokenizerLike
    ) -> list[tuple[str, np.ndarray]]:
        return list(self.iter_documents(source, tokenizer))


class UTF8TokensReader(FormatReader):
    def iter_documents(
        self, source: DataSource, tokenizer: TokenizerLike
    ) -> Iterator[tuple[str, np.ndarray]]:
        path = Path(source.path)
        utf8_tokens = np.load(path, mmap_mode="r")
        total = len(utf8_tokens)
        stream_chunk = 10_000_000
        doc_chunk = source.chunk_size or 512
        for start in range(0, total, stream_chunk):
            raw = utf8_tokens[start : start + stream_chunk]
            text = _normalize_text(raw.astype(np.uint8).tobytes().decode("utf-8", errors="ignore"))
            tokens = _filter_unk(tokenizer.encode(text))
            for i in range(0, len(tokens), doc_chunk):
                chunk = tokens[i : i + doc_chunk]
                if len(chunk) >= source.min_length:
                    yield (source.name, _token_array(chunk))

    def read_documents(
        self, source: DataSource, tokenizer: TokenizerLike
    ) -> list[tuple[str, np.ndarray]]:
        return list(self.iter_documents(source, tokenizer))


class JsonlReader(FormatReader):
    def iter_documents(
        self, source: DataSource, tokenizer: TokenizerLike
    ) -> Iterator[tuple[str, np.ndarray]]:
        path = Path(source.path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                text = obj.get(source.text_field, "")
                if not text:
                    continue
                text = _normalize_text(text)
                tokens = _filter_unk(tokenizer.encode(text))
                if source.max_length:
                    tokens = tokens[: source.max_length]
                encoded = _token_array(tokens)
                if len(encoded) >= source.min_length:
                    yield (source.name, encoded)

    def read_documents(
        self, source: DataSource, tokenizer: TokenizerLike
    ) -> list[tuple[str, np.ndarray]]:
        return list(self.iter_documents(source, tokenizer))


class NpyReader(FormatReader):
    @staticmethod
    def _coerce_token_dtype(tokens: np.ndarray) -> np.ndarray:
        if tokens.size == 0:
            return tokens.astype(np.uint16)
        if int(tokens.min()) < 0:
            raise ValueError("Token arrays must contain non-negative IDs")
        dtype = np.uint16 if int(tokens.max()) < 65536 else np.uint32
        return tokens.astype(dtype, copy=False)

    def iter_documents(
        self, source: DataSource, tokenizer: TokenizerLike
    ) -> Iterator[tuple[str, np.ndarray]]:
        del tokenizer
        path = Path(source.path)
        tokens = np.load(path)
        if source.chunk_size:
            for i in range(0, len(tokens), source.chunk_size):
                chunk = self._coerce_token_dtype(tokens[i : i + source.chunk_size])
                chunk = chunk[chunk != 0]  # filter <unk>
                if len(chunk) >= source.min_length:
                    yield (source.name, chunk)
        else:
            coerced = self._coerce_token_dtype(tokens)
            yield (source.name, coerced[coerced != 0])

    def read_documents(
        self, source: DataSource, tokenizer: TokenizerLike
    ) -> list[tuple[str, np.ndarray]]:
        return list(self.iter_documents(source, tokenizer))


_FORMAT_READERS: dict[str, FormatReader] = {
    "text": TextFormatReader(),
    "utf8_tokens": UTF8TokensReader(),
    "jsonl": JsonlReader(),
    "npy": NpyReader(),
}


def resolve_format_reader(fmt: str) -> FormatReader:
    if fmt not in _FORMAT_READERS:
        raise ValueError(f"Unknown format: {fmt}")
    return _FORMAT_READERS[fmt]


class MixingStrategy(ABC):
    @abstractmethod
    def mix(
        self,
        all_documents: dict[str, Sequence[np.ndarray]],
        config: MixingConfig,
    ) -> list[tuple[str, np.ndarray]]: ...


def _apply_sampling(
    docs: dict[str, Sequence[np.ndarray]], config: MixingConfig
) -> dict[str, Sequence[np.ndarray]]:
    if not (config.upsample_to_max or config.downsample_to_min or config.target_total_docs):
        return docs

    rng = np.random.RandomState(config.seed)

    if config.target_total_docs:
        target = config.target_total_docs
    elif config.upsample_to_max:
        target = max(len(items) for items in docs.values())
    else:
        target = min(len(items) for items in docs.values())

    sampled: dict[str, Sequence[np.ndarray]] = {}
    for name, items in docs.items():
        if len(items) < target:
            indices = rng.choice(len(items), size=target, replace=True)
            sampled[name] = [items[i] for i in indices]
        elif len(items) > target:
            indices = rng.choice(len(items), size=target, replace=False)
            sampled[name] = [items[i] for i in indices]
        else:
            sampled[name] = list(items)

    return sampled


def _apply_ratio_resampling(
    docs: dict[str, list[np.ndarray]],
    ratios: dict[str, float],
    rng: np.random.RandomState,
) -> dict[str, list[np.ndarray]]:
    total_ratio = sum(ratios.values())
    normalized = {k: v / total_ratio for k, v in ratios.items()}

    current_counts = {name: len(items) for name, items in docs.items()}
    total_docs = sum(current_counts.values())

    target_counts: dict[str, int] = {}
    for name in docs:
        if name in normalized:
            target_counts[name] = int(total_docs * normalized[name])
        else:
            remaining_sources = [n for n in docs if n not in normalized]
            if remaining_sources:
                remaining_ratio = 1.0 - sum(normalized.values())
                target_counts[name] = int(total_docs * remaining_ratio / len(remaining_sources))
            else:
                target_counts[name] = 0

    resampled: dict[str, list[np.ndarray]] = {}
    for name, items in docs.items():
        target = target_counts.get(name, 0)
        if target == 0:
            continue
        if len(items) < target:
            indices = rng.choice(len(items), size=target, replace=True)
            resampled[name] = [items[i] for i in indices]
        elif len(items) > target:
            indices = rng.choice(len(items), size=target, replace=False)
            resampled[name] = [items[i] for i in indices]
        else:
            resampled[name] = list(items)

    return resampled


class ConcatenateMixer(MixingStrategy):
    def mix(
        self,
        all_documents: dict[str, Sequence[np.ndarray]],
        config: MixingConfig,
    ) -> list[tuple[str, np.ndarray]]:
        docs = _apply_sampling(all_documents, config)
        mixed: list[tuple[str, np.ndarray]] = []
        for name, items in docs.items():
            mixed.extend((name, item) for item in items)
        return mixed


class InterleaveMixer(MixingStrategy):
    def mix(
        self,
        all_documents: dict[str, Sequence[np.ndarray]],
        config: MixingConfig,
    ) -> list[tuple[str, np.ndarray]]:
        docs = _apply_sampling(all_documents, config)
        rng = np.random.RandomState(config.seed)

        source_docs = {name: list(items) for name, items in docs.items()}
        for items in source_docs.values():
            rng.shuffle(items)

        if config.source_ratios:
            source_docs = _apply_ratio_resampling(source_docs, config.source_ratios, rng)

        mixed: list[tuple[str, np.ndarray]] = []
        pointers = dict.fromkeys(source_docs, 0)

        while any(pointers[name] < len(source_docs[name]) for name in source_docs):
            for name in source_docs:
                if pointers[name] < len(source_docs[name]):
                    for _ in range(config.block_size):
                        if pointers[name] < len(source_docs[name]):
                            mixed.append((name, source_docs[name][pointers[name]]))
                            pointers[name] += 1

        return mixed


_MIXING_STRATEGIES: dict[str, MixingStrategy] = {
    "interleave": InterleaveMixer(),
    "concatenate": ConcatenateMixer(),
}


def resolve_mixing_strategy(config: MixingConfig) -> MixingStrategy:
    if config.strategy not in _MIXING_STRATEGIES:
        raise ValueError(f"Unknown strategy: {config.strategy}")
    return _MIXING_STRATEGIES[config.strategy]


class CurriculumStrategy(ABC):
    @abstractmethod
    def build(
        self,
        documents: list[tuple[str, np.ndarray]],
        config: CurriculumConfig,
    ) -> dict[str, list[tuple[str, np.ndarray]]]: ...


class NoCurriculum(CurriculumStrategy):
    def build(
        self,
        documents: list[tuple[str, np.ndarray]],
        config: CurriculumConfig,
    ) -> dict[str, list[tuple[str, np.ndarray]]]:
        del config
        return {"default": documents}


class LengthBasedCurriculum(CurriculumStrategy):
    def build(
        self,
        documents: list[tuple[str, np.ndarray]],
        config: CurriculumConfig,
    ) -> dict[str, list[tuple[str, np.ndarray]]]:
        if not documents:
            return {"default": []}

        sorted_docs = sorted(documents, key=lambda item: len(item[1]))
        stage_size = max(1, len(sorted_docs) // config.num_stages)
        stages: dict[str, list[tuple[str, np.ndarray]]] = {}

        for i in range(config.num_stages):
            start = i * stage_size
            end = start + stage_size if i < config.num_stages - 1 else len(sorted_docs)
            stage_docs = sorted_docs[start:end]
            if not stage_docs:
                continue
            min_len = min(len(doc) for _, doc in stage_docs)
            max_len = max(len(doc) for _, doc in stage_docs)
            stages[f"stage{i + 1}_len{min_len}-{max_len}"] = stage_docs

        return stages or {"default": documents}


class DomainProgressionCurriculum(CurriculumStrategy):
    def build(
        self,
        documents: list[tuple[str, np.ndarray]],
        config: CurriculumConfig,
    ) -> dict[str, list[tuple[str, np.ndarray]]]:
        del config

        by_source: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
        for name, doc in documents:
            by_source[name].append((name, doc))

        return {f"stage{i + 1}_{name}": docs for i, (name, docs) in enumerate(by_source.items())}


class CustomCurriculum(CurriculumStrategy):
    def build(
        self,
        documents: list[tuple[str, np.ndarray]],
        config: CurriculumConfig,
    ) -> dict[str, list[tuple[str, np.ndarray]]]:
        stages: dict[str, list[tuple[str, np.ndarray]]] = {}

        for stage in config.stages:
            stage_docs: list[tuple[str, np.ndarray]] = []
            for name, doc in documents:
                doc_len = len(doc)
                if stage.min_length and doc_len < stage.min_length:
                    continue
                if stage.max_length and doc_len > stage.max_length:
                    continue
                stage_docs.append((name, doc))

            if stage.sources:
                stage_docs = [(name, doc) for name, doc in stage_docs if name in stage.sources]

            if stage.source_weights:
                grouped: dict[str, list[np.ndarray]] = defaultdict(list)
                for name, doc in stage_docs:
                    grouped[name].append(doc)
                sampled = _apply_ratio_resampling(
                    grouped,
                    stage.source_weights,
                    np.random.RandomState(42),
                )
                stage_docs = [(name, doc) for name, docs in sampled.items() for doc in docs]

            if stage.fraction:
                stage_docs = stage_docs[: int(len(documents) * stage.fraction)]
            elif stage.num_docs:
                stage_docs = stage_docs[: stage.num_docs]

            stages[stage.name] = stage_docs

        return stages


_CURRICULUM_STRATEGIES: dict[str, CurriculumStrategy] = {
    "length_based": LengthBasedCurriculum(),
    "domain_progression": DomainProgressionCurriculum(),
    "custom": CustomCurriculum(),
}


def resolve_curriculum_strategy(config: CurriculumConfig) -> CurriculumStrategy:
    if not config.enabled:
        return NoCurriculum()
    if config.type not in _CURRICULUM_STRATEGIES:
        raise ValueError(f"Unknown curriculum type: {config.type}")
    return _CURRICULUM_STRATEGIES[config.type]


class SplitStrategy(ABC):
    @abstractmethod
    def split(
        self,
        documents: list[tuple[str, np.ndarray]],
        config: SplitConfig,
    ) -> dict[str, list[tuple[str, np.ndarray]]]: ...


class StratifiedSplitter(SplitStrategy):
    def split(
        self,
        documents: list[tuple[str, np.ndarray]],
        config: SplitConfig,
    ) -> dict[str, list[tuple[str, np.ndarray]]]:
        rng = np.random.RandomState(config.seed)

        by_source: dict[str, list[np.ndarray]] = defaultdict(list)
        for name, doc in documents:
            by_source[name].append(doc)

        if config.shuffle:
            for items in by_source.values():
                _shuffle_with_rng(items, rng)

        splits: dict[str, list[tuple[str, np.ndarray]]] = {"train": [], "val": [], "test": []}

        for name, items in by_source.items():
            count = len(items)
            train_end = int(count * config.train)
            val_end = int(count * (config.train + config.val))

            splits["train"].extend((name, d) for d in items[:train_end])
            splits["val"].extend((name, d) for d in items[train_end:val_end])
            splits["test"].extend((name, d) for d in items[val_end:])

        if config.shuffle:
            for split_name in splits:
                _shuffle_with_rng(splits[split_name], rng)

        return splits


class SimpleSplitter(SplitStrategy):
    def split(
        self,
        documents: list[tuple[str, np.ndarray]],
        config: SplitConfig,
    ) -> dict[str, list[tuple[str, np.ndarray]]]:
        rng = np.random.RandomState(config.seed)
        docs = list(documents)
        if config.shuffle:
            _shuffle_with_rng(docs, rng)

        count = len(docs)
        train_end = int(count * config.train)
        val_end = int(count * (config.train + config.val))

        return {
            "train": docs[:train_end],
            "val": docs[train_end:val_end],
            "test": docs[val_end:],
        }


_SPLIT_STRATEGIES: dict[str, SplitStrategy] = {
    "stratified": StratifiedSplitter(),
    "simple": SimpleSplitter(),
}


def resolve_split_strategy(config: SplitConfig) -> SplitStrategy:
    key = "stratified" if config.stratified else "simple"
    return _SPLIT_STRATEGIES[key]
