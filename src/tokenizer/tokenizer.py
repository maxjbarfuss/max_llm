"""Abstract tokenizer interface and factory."""

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract tokenizer interface."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass


class TokenizerFactory:
    """Factory for creating tokenizer instances."""

    _tokenizers: dict[str, type[Tokenizer]] = {}

    @classmethod
    def create(cls, name: str = "char") -> Tokenizer:
        """
        Create a tokenizer instance by name.

        Args:
            name: Tokenizer name (default: "char")

        Returns:
            A Tokenizer instance

        Raises:
            ValueError: If tokenizer name is not registered

        Examples:
            >>> tokenizer = TokenizerFactory.create("char")
            >>> tokens = tokenizer.encode("hello")
        """
        if name not in cls._tokenizers:
            available = ", ".join(cls._tokenizers.keys())
            raise ValueError(f"Unknown tokenizer: {name}. " f"Available: {available}")

        tokenizer_class = cls._tokenizers[name]
        return tokenizer_class()

    @classmethod
    def register(cls, name: str, tokenizer_class: type[Tokenizer]) -> None:
        """
        Register a new tokenizer class.

        Args:
            name: Name to register the tokenizer under
            tokenizer_class: Tokenizer class to register

        Raises:
            TypeError: If tokenizer_class is not a subclass of Tokenizer
        """
        if not issubclass(tokenizer_class, Tokenizer):
            raise TypeError(f"{tokenizer_class} must be a subclass of Tokenizer")

        cls._tokenizers[name] = tokenizer_class

    @classmethod
    def list_available(cls) -> list[str]:
        """
        List available tokenizer names.

        Returns:
            List of registered tokenizer names
        """
        return list(cls._tokenizers.keys())
