"""Phase 3 model: GPT-style decoder with transformer blocks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from typing_extensions import Self

from src.config.model import ModelConfig
from src.models.embeddings.token_embedding import TokenEmbedding
from src.models.position.learned_position import LearnedPositionEmbedding
from src.models.transformer.transformer_block import TransformerBlock

from .base import BaseLearningModel


class DecoderLM(BaseLearningModel):
    """GPT-style decoder-only language model with causal self-attention.

    Architecture:
        1. Token embedding + Learned positional embedding
        2. Stack of N transformer blocks (pre-norm attention + FFN with residuals)
        3. Final layer norm
        4. LM head (weight-tied to token embedding)

    Weight tying: lm_head shares its weight matrix with token_embedding,
    halving the parameter count and tying input/output representations.

    Args:
        vocab_size: Size of vocabulary.
        d_model: Model dimension (embedding and hidden size).
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        dropout: Dropout probability (default: 0.0).
        ff_expansion_ratio: Expansion ratio for FFN hidden dimension (default: 4).
        max_seq_len: Maximum sequence length (default: 2048).
        attention_backend: Attention backend to use (default: "flash").
            Options: "flash", "sage", "xformers", "standard"
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
        ff_expansion_ratio: int = 4,
        max_seq_len: int = 2048,
        attention_backend: str = "flash",
    ) -> None:
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = LearnedPositionEmbedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_expansion_ratio=ff_expansion_ratio,
                    attention_backend=attention_backend,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm (standard for GPT-style models)
        self.final_norm = nn.LayerNorm(d_model)

        # LM head with weight tying
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for token index sequences.

        Args:
            x: Token indices of shape (batch_size, seq_len).
               Values must be in [0, vocab_size).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        assert x.ndim == 2, f"DecoderLM expects 2-D input (batch, seq_len), got shape {x.shape}"
        assert x.dtype == torch.long, f"DecoderLM expects dtype=torch.long, got {x.dtype}"
        B, T = x.shape

        # Token and position embeddings
        tok_emb = self.token_embedding(x)  # (B, T, d_model)
        pos_emb = self.position_embedding(x)  # (1, T, d_model)
        h = tok_emb + pos_emb  # (B, T, d_model)

        # Apply transformer blocks
        for block in self.blocks:
            h = block(h)

        # Final layer norm
        h = self.final_norm(h)

        # LM head
        logits = self.lm_head(h)

        assert logits.shape == (
            B,
            T,
            self.vocab_size,
        ), f"DecoderLM output shape mismatch: expected {(B, T, self.vocab_size)}, got {logits.shape}"
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
        """Construct a DecoderLM from a ModelConfig.

        Args:
            config: ModelConfig instance.
            attention_backend: Attention backend to use (default: "flash").
                Options: "flash", "sage", "xformers", "standard"

        Returns:
            DecoderLM instance.
        """
        return cls(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            ff_expansion_ratio=4,
            max_seq_len=config.max_seq_length,
            attention_backend=attention_backend,
        )
