"""Unigram tokenizer backed by sentencepiece.

Supports loading pre-trained `.model` files and training new models
for benchmarking/experimentation.
"""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm  # type: ignore[import-untyped]

from .tokenizer import Tokenizer


class UnigramTokenizer(Tokenizer):
    """SentencePiece unigram tokenizer wrapper."""

    def __init__(self, model_path: str | Path, **_kwargs: object) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Unigram model not found: {self._model_path}")

        self._processor = spm.SentencePieceProcessor()
        loaded = self._processor.Load(str(self._model_path))
        if not loaded:
            raise ValueError(f"Failed to load unigram model: {self._model_path}")

    @staticmethod
    def train_model(
        input_path: str | Path,
        model_prefix: str | Path,
        vocab_size: int = 8000,
        character_coverage: float = 1.0,
    ) -> Path:
        """Train a SentencePiece unigram model and return the model file path."""
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Training corpus not found: {input_file}")

        prefix = Path(model_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        spm.SentencePieceTrainer.Train(
            input=str(input_file),
            model_prefix=str(prefix),
            model_type="unigram",
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            hard_vocab_limit=False,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            unk_id=0,
        )
        return prefix.with_suffix(".model")

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return list(self._processor.EncodeAsIds(text))

    def decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        return str(self._processor.DecodeIds(tokens))

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._processor.EncodeAsIds(text))

    @property
    def vocab_size(self) -> int:
        return int(self._processor.GetPieceSize())
