"""Helpers for constructing tokenizers from runtime configuration."""

from pathlib import Path

from src.tokenizer.hf_bpe_tokenizer import HFBPETokenizer
from src.tokenizer.tokenizer import Tokenizer, TokenizerFactory

# tiktoken encodings use large vocabularies (>= 50k).
_TIKTOKEN_MIN_VOCAB = 10_000


def create_configured_tokenizer(
    tokenizer_name: str,
    tokenizer_mode: str | None = None,
    tokenizer_vocab_size: int | None = None,
    tokenizer_backend: str | None = None,
    unigram_model_path: str | None = None,
    tokenizer_vocab_path: str | None = None,
    bpe_encoding: str = "gpt2",
) -> Tokenizer:
    """Create a tokenizer using shared config semantics across entrypoints."""
    factory_name = tokenizer_name
    if tokenizer_backend == "gpt2_bpe":
        factory_name = "bpe"
    elif tokenizer_backend == "unigram":
        factory_name = "unigram"

    if factory_name == "bpe":
        # Optional explicit vocab path takes precedence for custom small-vocab BPE.
        if tokenizer_vocab_path:
            return HFBPETokenizer(tokenizer_vocab_path)

        if tokenizer_vocab_size is not None and tokenizer_vocab_size < _TIKTOKEN_MIN_VOCAB:
            vocab_path = Path("data/fast") / f"bpe_vocab_{tokenizer_vocab_size}.json"
            if not vocab_path.exists():
                raise FileNotFoundError(
                    f"Custom BPE vocab file not found: {vocab_path}. "
                    "Set tokenizer_vocab_path or run tokenization first."
                )
            return HFBPETokenizer(str(vocab_path))

        return TokenizerFactory.create("bpe", encoding=bpe_encoding)

    if factory_name == "unigram":
        if not unigram_model_path:
            raise ValueError("unigram_model_path is required for unigram tokenizer")
        return TokenizerFactory.create("unigram", model_path=unigram_model_path)

    mode = tokenizer_mode or "utf8"
    kwargs: dict[str, object] = {"mode": mode}
    if mode == "codepoint":
        kwargs["vocab_size"] = tokenizer_vocab_size or 256

    return TokenizerFactory.create(factory_name, **kwargs)
