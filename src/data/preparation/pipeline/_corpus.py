"""Tokenizer construction and training corpus utilities."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.data.preparation.config import DataPreparationConfig, DataSource
from src.data.preparation.strategies import _normalize_text
from src.tokenizer import TokenizerFactory, UnigramTokenizer

logger = logging.getLogger(__name__)


class _UTF8Tokenizer:
    def __init__(self) -> None:
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="ignore")


def build_tokenizer(
    config: DataPreparationConfig,
    tokenizer_prefix: str,
) -> tuple[Any, str | None]:
    tok_cfg = config.tokenizer

    if tok_cfg.type == "unigram":
        if tok_cfg.model_path:
            return UnigramTokenizer(tok_cfg.model_path), str(tok_cfg.model_path)

        corpus_path = _create_training_corpus(config.datasets, tok_cfg.max_corpus_mb)
        model_path = UnigramTokenizer.train_model(
            input_path=corpus_path,
            model_prefix=tokenizer_prefix,
            vocab_size=tok_cfg.vocab_size,
            character_coverage=tok_cfg.character_coverage,
        )
        Path(corpus_path).unlink(missing_ok=True)
        return UnigramTokenizer(model_path), str(model_path)

    if tok_cfg.type == "char":
        return TokenizerFactory.create("char", vocab_size=tok_cfg.vocab_size), None

    if tok_cfg.type == "bpe":
        return TokenizerFactory.create("bpe"), None

    if tok_cfg.type == "hf_bpe":
        return TokenizerFactory.create("hf_bpe"), None

    if tok_cfg.type == "utf8":
        return _UTF8Tokenizer(), None

    raise ValueError(f"Unknown tokenizer type: {tok_cfg.type}")


def load_source_as_text(source: DataSource) -> str:
    path = Path(source.path)

    if source.format == "text":
        if path.is_dir():
            parts = [
                p.read_text(encoding="utf-8", errors="ignore") for p in sorted(path.glob("*.txt"))
            ]
            return "\n\n".join(parts)
        return path.read_text(encoding="utf-8", errors="ignore")

    if source.format == "utf8_tokens":
        tokens = np.load(path)
        byte_tokens = np.asarray(tokens)
        if byte_tokens.size and (int(byte_tokens.min()) < 0 or int(byte_tokens.max()) > 255):
            raise ValueError(f"utf8_tokens source contains out-of-byte-range values: {source.path}")
        return byte_tokens.astype(np.uint8, copy=False).tobytes().decode("utf-8", errors="ignore")

    if source.format == "jsonl":
        lines: list[str] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                lines.append(obj.get(source.text_field, ""))
        return "\n\n".join(lines)

    raise ValueError(f"Cannot convert format '{source.format}' to text corpus")


def _create_training_corpus(
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
            written = _write_source_to_corpus(
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
                    delta = _write_with_limit(
                        handle,
                        _normalize_text(chunk),
                        current_bytes + written,
                        max_bytes,
                    )
                    written += delta
                    if delta < len(chunk):
                        return written

            sep_delta = _write_with_limit(
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
            raise ValueError(f"utf8_tokens source contains out-of-byte-range values: {source.path}")
        decoded = _normalize_text(
            byte_tokens.astype(np.uint8, copy=False).tobytes().decode("utf-8", errors="ignore")
        )
        return _write_with_limit(handle, decoded, current_bytes, max_bytes)

    if source.format == "jsonl":
        with path.open(encoding="utf-8") as src:
            for line in src:
                obj = json.loads(line)
                text = obj.get(source.text_field, "")
                if not text:
                    continue
                delta = _write_with_limit(
                    handle,
                    _normalize_text(text),
                    current_bytes + written,
                    max_bytes,
                )
                written += delta
                if delta < len(text):
                    return written

                sep_delta = _write_with_limit(
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
