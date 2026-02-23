"""Phase 2 skeleton model: token embedding + learned position + linear FFN + LM head."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config.model import ModelConfig

from .base import BaseLearningModel


class SimpleLM(BaseLearningModel):
    """Minimal language model for Phase 2.

    Architecture:
        token_emb(x) + pos_emb(positions)  →  (B, T, hidden_size)
        linear FFN                          →  (B, T, hidden_size)
        LM head (weight-tied to token_emb)  →  (B, T, vocab_size)

    Weight tying: lm_head shares its weight matrix with token_emb, halving
    the parameter count for the output projection and tying input/output
    representations — standard practice for small LMs.
    """

    def __init__(self, vocab_size: int, hidden_size: int, max_seq_length: int) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_length, hidden_size)
        self.ffn = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for token index sequences.

        Args:
            x: Token indices of shape (batch_size, seq_len).
               Values must be in [0, vocab_size).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        _B, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.ffn(h)
        return self.lm_head(h)

    @classmethod
    def from_config(cls, config: ModelConfig) -> SimpleLM:
        """Construct a SimpleLM from a ModelConfig."""
        return cls(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_seq_length=config.max_seq_length,
        )
