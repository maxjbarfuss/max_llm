"""Abstract base class for all learning models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from typing_extensions import Self

from src.config.model import ModelConfig


class BaseLearningModel(nn.Module, ABC):
    """Abstract base for all max-llm model implementations."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: ModelConfig) -> Self: ...
