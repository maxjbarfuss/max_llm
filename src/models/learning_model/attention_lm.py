"""Minimal attention-based language model for testing CausalMultiHeadAttention."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from typing_extensions import Self

from src.config.model import ModelConfig
from src.models.attention.causal_mha import CausalMultiHeadAttention
from src.models.embeddings.token_embedding import TokenEmbedding
from src.models.learning_model.base import BaseLearningModel
from src.models.position.learned_position import LearnedPositionEmbedding


class AttentionLM(BaseLearningModel):
    """Minimal language model with token embedding + position + attention + LM head.

    This is a simplified 1-layer transformer without the feed-forward network,
    used to test the CausalMultiHeadAttention module in a training context.

    Args:
        vocab_size: Vocabulary size for token embeddings.
        d_model: Model dimension (embedding size).
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length for positional embeddings.
        dropout: Dropout probability (default: 0.0).
        attention_backend: Attention backend to use (default: "flash").
            Options: "flash", "sage", "xformers", "standard"
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        attention_backend: str = "flash",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = LearnedPositionEmbedding(max_seq_len, d_model)

        # Pre-norm layer normalization
        self.ln = nn.LayerNorm(d_model)

        # Attention
        self.attention = CausalMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attention_backend=attention_backend,
        )

        # LM head (weight-tied to token embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.embedding.weight

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Token indices of shape (batch, seq_len).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        # Embed tokens and add positional embeddings
        token_emb = self.token_embedding(x)  # (B, T, d_model)
        pos_emb = self.position_embedding(x)  # (1, T, d_model)
        x = token_emb + pos_emb
        x = self.dropout(x)

        # Pre-norm + attention + residual
        x = x + self.attention(self.ln(x))

        # LM head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Load state dict and re-establish weight tying after restore."""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.lm_head.weight = self.token_embedding.embedding.weight
        return result

    @classmethod
    def from_config(cls, config: ModelConfig, attention_backend: str = "flash") -> Self:
        """Construct an AttentionLM from a ModelConfig.

        Args:
            config: ModelConfig instance.
            attention_backend: Attention backend to use (default: "flash").
                Options: "flash", "sage", "xformers", "standard"

        Returns:
            AttentionLM instance.
        """
        return cls(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_length,
            dropout=config.dropout,
            attention_backend=attention_backend,
        )
