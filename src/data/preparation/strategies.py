"""Strategy components for data preparation."""

import json
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np

from src.data.preparation.config import (
    CurriculumConfig,
    DataSource,
    MixingConfig,
    SplitConfig,
)
from src.data.preparation.diagnostics import emit_prep_diagnostic


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


_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_LINE_END_PUNCTUATION = {".", "!", "?", ";", ":"}


def _punctuation_ended_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    ended = sum(1 for line in lines if line[-1] in _LINE_END_PUNCTUATION)
    return ended / len(lines)


def _duplicate_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return 0.0
    unique_count = len(set(lines))
    duplicate_count = len(lines) - unique_count
    return duplicate_count / len(lines)


def _symbol_to_word_ratio(text: str) -> float:
    words = _WORD_RE.findall(text)
    if not words:
        return float("inf")
    symbol_count = sum(
        1
        for ch in text
        if not ch.isspace() and (unicodedata.category(ch).startswith("S") or ch in "#$%^&*_=+<>|~`")
    )
    return symbol_count / len(words)


def _passes_text_hygiene(text: str, source: DataSource) -> bool:
    if source.min_punctuation_ended_line_ratio is not None:
        if _punctuation_ended_line_ratio(text) < source.min_punctuation_ended_line_ratio:
            return False

    if source.max_duplicate_line_ratio is not None:
        if _duplicate_line_ratio(text) > source.max_duplicate_line_ratio:
            return False

    if source.max_symbol_to_word_ratio is not None:
        if _symbol_to_word_ratio(text) > source.max_symbol_to_word_ratio:
            return False

    return True


# ── Language filter ──────────────────────────────────────────────────────────

_LANG_MODEL: Any = None


def load_lang_model(path: str) -> None:
    """Load the fastText language identification model (e.g. lid.176.bin)."""
    global _LANG_MODEL
    import contextlib
    import os

    import fasttext

    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        _LANG_MODEL = fasttext.load_model(path)


def _detect_language(text: str) -> str:
    """Return ISO 639-1 language code for the dominant language (e.g. 'en').

    Returns empty string if the model is not loaded or the sample is empty.
    Uses the first 500 characters to avoid passing very long strings to fastText.
    """
    if _LANG_MODEL is None:
        return ""
    sample = text[:500].replace("\n", " ").strip()
    if not sample:
        return ""
    try:
        labels, _ = _LANG_MODEL.predict(sample, k=1)
        # fastText labels are formatted as '__label__en'
        return labels[0].replace("__label__", "")
    except ValueError as exc:
        # fasttext-wheel can raise here with NumPy 2.x due to copy=False semantics.
        # Fall back to the underlying pybind API which returns plain Python tuples.
        if "Unable to avoid copy while creating an array as requested" not in str(exc):
            raise
        raw = _LANG_MODEL.f.predict(sample, 1, 0.0, "strict")
        if not raw:
            return ""
        return raw[0][1].replace("__label__", "")


def _passes_language_filter(text: str, source: DataSource) -> bool:
    """Return True if text passes the per-source language allowlist.

    If source.allowed_languages is None, all documents pass.
    Texts shorter than 50 characters are always passed to avoid false negatives
    from unreliable fastText predictions on tiny inputs.
    """
    if not source.allowed_languages:
        return True
    if _LANG_MODEL is None:
        raise RuntimeError(
            f"allowed_languages is set for source '{source.name}' but no fastText model "
            "is loaded. Set 'lang_model_path' in the prep config."
        )
    if len(text) < 50:
        return True
    lang = _detect_language(text)
    return lang in source.allowed_languages


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
                if not _passes_text_hygiene(doc, source):
                    continue
                if not _passes_language_filter(doc, source):
                    continue
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
                if not _passes_text_hygiene(text, source):
                    continue
                if not _passes_language_filter(text, source):
                    continue
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


class ParquetReader(FormatReader):
    def iter_documents(  # noqa: C901
        self, source: DataSource, tokenizer: TokenizerLike
    ) -> Iterator[tuple[str, np.ndarray]]:
        if isinstance(source.path, str) and source.path.startswith("hf://"):
            # Hugging Face dataset streaming
            from datasets import load_dataset

            # Parse URI: hf://namespace/dataset[/config][/split]
            uri = source.path[len("hf://") :]
            parts = uri.split("/")
            namespace = parts[0]
            dataset_name = parts[1]
            config = parts[2] if len(parts) > 2 else None
            split = parts[3] if len(parts) > 3 else "train"
            ds_id = f"{namespace}/{dataset_name}"
            kwargs = {"split": split, "streaming": True}
            if config:
                kwargs["name"] = config
            ds = load_dataset(ds_id, **kwargs)
            for example in ds:
                text = example.get(source.text_field, "")
                if not text:
                    continue
                text = _normalize_text(text)
                if len(text) < source.min_length:
                    continue
                if not _passes_text_hygiene(text, source):
                    continue
                if not _passes_language_filter(text, source):
                    continue
                tokens = _filter_unk(tokenizer.encode(text))
                if source.max_length:
                    tokens = tokens[: source.max_length]
                encoded = _token_array(tokens)
                if len(encoded) >= source.min_length:
                    yield (source.name, encoded)
        else:
            import pyarrow.parquet as pq

            path = Path(source.path)
            files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
            for file_path in files:
                parquet_file = pq.ParquetFile(file_path)
                for batch in parquet_file.iter_batches(
                    columns=[source.text_field], batch_size=1000
                ):
                    for val in batch.column(source.text_field):
                        text = val.as_py()
                        if not text:
                            continue
                        text = _normalize_text(text)
                        if len(text) < source.min_length:
                            continue
                        if not _passes_text_hygiene(text, source):
                            continue
                        if not _passes_language_filter(text, source):
                            continue
                        tokens = _filter_unk(tokenizer.encode(text))
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
    "parquet": ParquetReader(),
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


def _doc_lengths(items: Sequence[np.ndarray]) -> np.ndarray:
    lengths = getattr(items, "doc_lengths", None)
    if lengths is not None:
        return np.asarray(lengths, dtype=np.int64)
    return np.fromiter((len(doc) for doc in items), dtype=np.int64, count=len(items))


def _token_total(items: Sequence[np.ndarray]) -> int:
    total_tokens = getattr(items, "total_tokens", None)
    if total_tokens is not None:
        return int(total_tokens)
    return int(sum(len(doc) for doc in items))


def _index_dtype(size: int) -> Any:
    return np.uint32 if size <= np.iinfo(np.uint32).max else np.uint64


def _full_index_array(size: int) -> np.ndarray:
    return np.arange(size, dtype=_index_dtype(size))


def _sample_indices(size: int, target: int, rng: np.random.RandomState) -> np.ndarray:
    dtype = _index_dtype(size)
    if target <= 0:
        return np.array([], dtype=dtype)
    if size == target:
        return _full_index_array(size)
    if size < target:
        return rng.choice(size, size=target, replace=True).astype(dtype, copy=False)
    return rng.choice(size, size=target, replace=False).astype(dtype, copy=False)


def _apply_sampling(
    docs: dict[str, Sequence[np.ndarray]], config: MixingConfig
) -> dict[str, Sequence[np.ndarray]]:
    has_total_budget = (
        config.target_total_docs is not None or config.target_total_tokens is not None
    )
    if config.source_ratios and has_total_budget:
        return docs

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


def _target_counts_by_docs(
    docs: Mapping[str, Sequence[np.ndarray]],
    normalized: dict[str, float],
    total_docs_override: int | None = None,
) -> dict[str, int]:
    total_docs = (
        total_docs_override
        if total_docs_override is not None
        else sum(len(items) for items in docs.values())
    )
    remaining = [n for n in docs if n not in normalized]
    rem_ratio = max(0.0, 1.0 - sum(normalized.values()))
    result: dict[str, int] = {}
    for name in docs:
        if name in normalized:
            result[name] = int(total_docs * normalized[name])
        elif remaining:
            result[name] = int(total_docs * rem_ratio / len(remaining))
        else:
            result[name] = 0
    return result


def _target_counts_by_tokens(
    docs: Mapping[str, Sequence[np.ndarray]],
    normalized: dict[str, float],
    total_tokens_override: int | None = None,
) -> dict[str, int]:
    lengths = {name: _doc_lengths(items) for name, items in docs.items()}
    tok_totals = {name: int(lengths[name].sum()) for name in docs}
    total_tokens = (
        total_tokens_override if total_tokens_override is not None else sum(tok_totals.values())
    )
    avg_len = {name: tok_totals[name] / max(1, len(docs[name])) for name in docs}
    remaining = [n for n in docs if n not in normalized]
    rem_ratio = max(0.0, 1.0 - sum(normalized.values()))
    result: dict[str, int] = {}
    for name in docs:
        if name in normalized:
            target_tok = total_tokens * normalized[name]
        elif remaining:
            target_tok = total_tokens * rem_ratio / len(remaining)
        else:
            target_tok = 0.0
        result[name] = max(1, int(target_tok / max(1.0, avg_len[name]))) if target_tok > 0 else 0
    return result


def _target_counts_by_lengths(
    lengths_by_source: dict[str, np.ndarray],
    normalized: dict[str, float],
    total_docs_override: int | None = None,
    total_tokens_override: int | None = None,
    weight_by: str = "docs",
) -> dict[str, int]:
    if weight_by == "tokens":
        tok_totals = {name: int(lengths.sum()) for name, lengths in lengths_by_source.items()}
        total_tokens = (
            total_tokens_override if total_tokens_override is not None else sum(tok_totals.values())
        )
        avg_len = {
            name: tok_totals[name] / max(1, len(lengths_by_source[name]))
            for name in lengths_by_source
        }
        remaining = [n for n in lengths_by_source if n not in normalized]
        rem_ratio = max(0.0, 1.0 - sum(normalized.values()))
        token_result: dict[str, int] = {}
        for name in lengths_by_source:
            if name in normalized:
                target_tok = total_tokens * normalized[name]
            elif remaining:
                target_tok = total_tokens * rem_ratio / len(remaining)
            else:
                target_tok = 0.0
            token_result[name] = (
                max(1, int(target_tok / max(1.0, avg_len[name]))) if target_tok > 0 else 0
            )
        return token_result

    total_docs = (
        total_docs_override
        if total_docs_override is not None
        else sum(len(lengths) for lengths in lengths_by_source.values())
    )
    remaining = [n for n in lengths_by_source if n not in normalized]
    rem_ratio = max(0.0, 1.0 - sum(normalized.values()))
    result: dict[str, int] = {}
    for name in lengths_by_source:
        if name in normalized:
            result[name] = int(total_docs * normalized[name])
        elif remaining:
            result[name] = int(total_docs * rem_ratio / len(remaining))
        else:
            result[name] = 0
    return result


def _apply_ratio_resampling(
    docs: dict[str, list[np.ndarray]],
    ratios: dict[str, float],
    rng: np.random.RandomState,
    target_total_docs: int | None = None,
    target_total_tokens: int | None = None,
    weight_by: str = "docs",
) -> dict[str, list[np.ndarray]]:
    normalized = {k: v / sum(ratios.values()) for k, v in ratios.items()}
    target_counts = (
        _target_counts_by_tokens(docs, normalized, total_tokens_override=target_total_tokens)
        if weight_by == "tokens"
        else _target_counts_by_docs(docs, normalized, total_docs_override=target_total_docs)
    )

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


def select_source_indices(  # noqa: C901
    docs: dict[str, Sequence[np.ndarray]],
    config: MixingConfig,
    rng: np.random.RandomState | None = None,
    diagnostic_context: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    rng = rng or np.random.RandomState(config.seed)
    has_total_budget = (
        config.target_total_docs is not None or config.target_total_tokens is not None
    )

    if diagnostic_context is not None:
        emit_prep_diagnostic(
            diagnostic_context["output_dir"],
            diagnostic_context["prefix"],
            "select_source_indices_start",
            num_sources=len(docs),
            source_doc_counts={name: len(items) for name, items in docs.items()},
            source_token_totals={name: _token_total(items) for name, items in docs.items()},
            weight_by=config.weight_by,
            target_total_docs=config.target_total_docs,
            target_total_tokens=config.target_total_tokens,
            has_total_budget=has_total_budget,
        )

    selected = {name: _full_index_array(len(items)) for name, items in docs.items()}

    if diagnostic_context is not None:
        emit_prep_diagnostic(
            diagnostic_context["output_dir"],
            diagnostic_context["prefix"],
            "select_source_indices_full_index_arrays",
            index_bytes={name: int(indices.nbytes) for name, indices in selected.items()},
            total_index_bytes=int(sum(indices.nbytes for indices in selected.values())),
        )

    if not (config.source_ratios and has_total_budget):
        if config.target_total_docs is not None:
            target = config.target_total_docs
        elif config.upsample_to_max:
            target = max(len(items) for items in docs.values())
        elif config.downsample_to_min:
            target = min(len(items) for items in docs.values())
        else:
            target = None

        if target is not None:
            selected = {
                name: _sample_indices(len(items), target, rng) for name, items in docs.items()
            }
            if diagnostic_context is not None:
                emit_prep_diagnostic(
                    diagnostic_context["output_dir"],
                    diagnostic_context["prefix"],
                    "select_source_indices_sampling_applied",
                    sampling_target=int(target),
                    sampled_index_bytes={
                        name: int(indices.nbytes) for name, indices in selected.items()
                    },
                    total_sampled_index_bytes=int(
                        sum(indices.nbytes for indices in selected.values())
                    ),
                )

    if not config.source_ratios:
        if diagnostic_context is not None:
            emit_prep_diagnostic(
                diagnostic_context["output_dir"],
                diagnostic_context["prefix"],
                "select_source_indices_done",
                selected_doc_counts={name: int(len(indices)) for name, indices in selected.items()},
                selected_index_bytes={
                    name: int(indices.nbytes) for name, indices in selected.items()
                },
            )
        return selected

    normalized = {
        name: weight / sum(config.source_ratios.values())
        for name, weight in config.source_ratios.items()
    }
    selected_lengths = {
        name: _doc_lengths(items)[selected[name]]
        for name, items in docs.items()
        if len(selected[name]) > 0
    }
    target_counts = _target_counts_by_lengths(
        selected_lengths,
        normalized,
        total_docs_override=config.target_total_docs,
        total_tokens_override=config.target_total_tokens,
        weight_by=config.weight_by,
    )

    if diagnostic_context is not None:
        emit_prep_diagnostic(
            diagnostic_context["output_dir"],
            diagnostic_context["prefix"],
            "select_source_indices_target_counts",
            target_counts={name: int(count) for name, count in target_counts.items()},
            selected_length_bytes={
                name: int(lengths.nbytes) for name, lengths in selected_lengths.items()
            },
            total_selected_length_bytes=int(
                sum(lengths.nbytes for lengths in selected_lengths.values())
            ),
        )

    resampled: dict[str, np.ndarray] = {}
    for name, _items in docs.items():
        base = selected[name]
        target = target_counts.get(name, 0)
        if target <= 0:
            continue
        if len(base) == target:
            resampled[name] = base
            continue
        choice = _sample_indices(len(base), target, rng)
        resampled[name] = base[choice]

    if diagnostic_context is not None:
        emit_prep_diagnostic(
            diagnostic_context["output_dir"],
            diagnostic_context["prefix"],
            "select_source_indices_done",
            selected_doc_counts={name: int(len(indices)) for name, indices in resampled.items()},
            selected_index_bytes={name: int(indices.nbytes) for name, indices in resampled.items()},
            total_selected_index_bytes=int(sum(indices.nbytes for indices in resampled.values())),
        )

    return resampled


class ConcatenateMixer(MixingStrategy):
    def mix(
        self,
        all_documents: dict[str, Sequence[np.ndarray]],
        config: MixingConfig,
    ) -> list[tuple[str, np.ndarray]]:
        mixed: list[tuple[str, np.ndarray]] = []
        selected = select_source_indices(all_documents, config)
        for name, indices in selected.items():
            mixed.extend((name, all_documents[name][int(index)]) for index in indices)
        return mixed


class InterleaveMixer(MixingStrategy):
    def mix(
        self,
        all_documents: dict[str, Sequence[np.ndarray]],
        config: MixingConfig,
    ) -> list[tuple[str, np.ndarray]]:
        import sys

        rng = np.random.RandomState(config.seed)

        selected = select_source_indices(all_documents, config, rng)
        for indices in selected.values():
            rng.shuffle(indices)

        mixed: list[tuple[str, np.ndarray]] = []
        pointers = dict.fromkeys(selected, 0)
        total_docs = sum(len(lst) for lst in selected.values())
        print(
            f"[DEBUG] Mixing sources: {[f'{k}: {len(v)}' for k,v in selected.items()]}",
            file=sys.stderr,
            flush=True,
        )
        print(f"[DEBUG] Total docs to mix: {total_docs}", file=sys.stderr, flush=True)
        step = 0
        while any(pointers[name] < len(selected[name]) for name in selected):
            for name in selected:
                if pointers[name] < len(selected[name]):
                    for _ in range(config.block_size):
                        if pointers[name] < len(selected[name]):
                            doc_index = int(selected[name][pointers[name]])
                            mixed.append((name, all_documents[name][doc_index]))
                            pointers[name] += 1
                            step += 1
                            if step % 100000 == 0 or step == total_docs:
                                print(
                                    f"[DEBUG] Mixed {step}/{total_docs} docs. Pointers: {{ {', '.join(f'{k}: {pointers[k]}' for k in pointers)} }}",
                                    file=sys.stderr,
                                    flush=True,
                                )
        print(f"[DEBUG] Mixing complete: {len(mixed)} docs", file=sys.stderr, flush=True)
        return mixed


_MIXING_STRATEGIES: dict[str, MixingStrategy] = {
    "interleave": InterleaveMixer(),
    "concatenate": ConcatenateMixer(),
}


def resolve_mixing_strategy(config: MixingConfig) -> MixingStrategy:
    if config.strategy not in _MIXING_STRATEGIES:
        raise ValueError(f"Unknown strategy: {config.strategy}")
    return _MIXING_STRATEGIES[config.strategy]


def apply_curriculum_repetition_budget(
    stages: dict[str, list[tuple[str, np.ndarray]]],
    source_repetition_budget: dict[str, float] | None,
) -> tuple[dict[str, list[tuple[str, np.ndarray]]], dict[str, dict[str, float | int | None]]]:
    """Enforce per-source exposure caps across ordered curriculum stages.

    Budget semantics are source-specific max exposure factors. For a source with
    budget `b`, we keep at most `ceil(unique_docs_seen * b)` cumulative examples
    while iterating stages in order. A budget of 1.0 means "no repeated exposure"
    across stages (each unique document kept at most once).
    """

    if not source_repetition_budget:
        return stages, {}

    seen_doc_ids: dict[str, set[int]] = defaultdict(set)
    kept_exposures: dict[str, int] = defaultdict(int)
    dropped_exposures: dict[str, int] = defaultdict(int)
    trimmed_stages: dict[str, list[tuple[str, np.ndarray]]] = {}

    for stage_name, stage_docs in stages.items():
        kept_stage_docs: list[tuple[str, np.ndarray]] = []
        for source_name, doc in stage_docs:
            budget = source_repetition_budget.get(source_name)
            if budget is None:
                kept_stage_docs.append((source_name, doc))
                kept_exposures[source_name] += 1
                continue

            doc_id = id(doc)
            if doc_id not in seen_doc_ids[source_name]:
                seen_doc_ids[source_name].add(doc_id)
                kept_stage_docs.append((source_name, doc))
                kept_exposures[source_name] += 1
                continue

            max_allowed = int(np.ceil(len(seen_doc_ids[source_name]) * budget))
            if kept_exposures[source_name] < max_allowed:
                kept_stage_docs.append((source_name, doc))
                kept_exposures[source_name] += 1
            else:
                dropped_exposures[source_name] += 1

        trimmed_stages[stage_name] = kept_stage_docs

    tracked_sources = set(kept_exposures) | set(dropped_exposures) | set(source_repetition_budget)
    stats: dict[str, dict[str, float | int | None]] = {}
    for source_name in tracked_sources:
        unique_docs = len(seen_doc_ids[source_name])
        kept = kept_exposures[source_name]
        dropped = dropped_exposures[source_name]
        stats[source_name] = {
            "budget": source_repetition_budget.get(source_name),
            "unique_docs": unique_docs,
            "kept_exposures": kept,
            "dropped_exposures": dropped,
            "kept_repetition_ratio": float(kept / max(1, unique_docs)),
        }

    return trimmed_stages, stats


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
