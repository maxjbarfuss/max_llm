"""Tokenizer implementations."""

from .bpe_tokenizer import BPETokenizer
from .char_tokenizer import CharTokenizer
from .configured_tokenizer import create_configured_tokenizer
from .hf_bpe_tokenizer import HFBPETokenizer
from .tokenizer import Tokenizer, TokenizerFactory
from .unigram_tokenizer import UnigramTokenizer

# Register built-in tokenizers
if "char" not in TokenizerFactory.list_available():
    TokenizerFactory.register("char", CharTokenizer)
if "bpe" not in TokenizerFactory.list_available():
    TokenizerFactory.register("bpe", BPETokenizer)
if "unigram" not in TokenizerFactory.list_available():
    TokenizerFactory.register("unigram", UnigramTokenizer)
if "hf_bpe" not in TokenizerFactory.list_available():
    TokenizerFactory.register("hf_bpe", HFBPETokenizer)

__all__ = [
    "BPETokenizer",
    "CharTokenizer",
    "create_configured_tokenizer",
    "HFBPETokenizer",
    "Tokenizer",
    "TokenizerFactory",
    "UnigramTokenizer",
]
