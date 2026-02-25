"""Abstract tokenizer interface and factory.

The TokenizerFactory automatically registers the CharTokenizer and provides
a simple interface for creating tokenizer instances.

Examples:
    >>> from src.tokenizer import TokenizerFactory
    >>> tok = TokenizerFactory.create("char")
    >>> tokens = tok.encode("hello")
    >>> text = tok.decode(tokens)
"""

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.
        """

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back to text.

        Args:
            tokens: List of integer token IDs.

        Returns:
            Decoded text string.
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text.

        Args:
            text: Input text string.

        Returns:
            Number of tokens.
        """


class TokenizerFactory:
    """Factory for creating and managing tokenizer instances.

    Provides a registry mechanism for tokenizer implementations.
    Built-in tokenizers are registered via __init__.py imports.
    """

    _tokenizers: dict[str, type[Tokenizer]] = {}

    @classmethod
    def create(cls, name: str = "char", **kwargs) -> Tokenizer:
        """Create a tokenizer instance by name.

        Args:
            name: Tokenizer name (default: "char").
            **kwargs: Additional arguments passed to tokenizer constructor.

        Returns:
            A Tokenizer instance.

        Raises:
            ValueError: If tokenizer name is not registered.

        Examples:
            >>> tok = TokenizerFactory.create("char", vocab_size=128)
            >>> tok.encode("hello")
            [104, 101, 108, 108, 111]
        """
        if name not in cls._tokenizers:
            available = ", ".join(sorted(cls._tokenizers.keys()))
            raise ValueError(f"Unknown tokenizer '{name}'. Available: {available}")

        return cls._tokenizers[name](**kwargs)

    @classmethod
    def register(cls, name: str, tokenizer_class: type[Tokenizer]) -> None:
        """Register a custom tokenizer class.

        Args:
            name: Name to register the tokenizer under.
            tokenizer_class: Tokenizer class to register.

        Raises:
            ValueError: If name is already registered.
        """
        if name in cls._tokenizers:
            raise ValueError(f"Tokenizer '{name}' is already registered")

        cls._tokenizers[name] = tokenizer_class

    @classmethod
    def list_available(cls) -> list[str]:
        """List all available tokenizer names.

        Returns:
            Sorted list of registered tokenizer names.
        """
        return sorted(cls._tokenizers.keys())
