"""Abstract base class for all learning models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.config.model import ModelConfig


class BaseLearningModel(nn.Module, ABC):
    """Abstract base for all max-llm model implementations.

    Defines the interface that evolves across phases:
      Phase 2: SimpleLM (embedding + linear FFN + LM head)
      Phase 3: TransformerLM (causal attention blocks)
      Phase 5+: Llama-style upgrades (RoPE, GQA, SwiGLU, ...)

    All concrete subclasses must implement forward() and from_config().
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for a sequence of token indices.

        Args:
            x: Token indices of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """

    @classmethod
    @abstractmethod
    def from_config(cls, config: ModelConfig) -> BaseLearningModel:
        """Construct a model from a ModelConfig."""
