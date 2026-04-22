"""Shared types for benchmark tasks and evaluation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MCQExample:
    """Single multiple-choice example.

    Attributes:
        prompt: Prefix/context presented to the model.
        choices: Candidate completions/options.
        answer_index: Index of the correct choice in `choices`.
        example_id: Optional stable identifier from source dataset.
    """

    prompt: str
    choices: list[str]
    answer_index: int
    example_id: str | None = None


@dataclass(frozen=True)
class BenchmarkTaskResult:
    """Per-task evaluation summary."""

    task: str
    split: str
    num_examples: int
    num_correct: int
    accuracy: float


class BenchmarkTask(Protocol):
    """Task adapter protocol for loading MCQ examples."""

    name: str

    def load_examples(self, split: str, max_examples: int) -> list[MCQExample]:
        """Load and normalize examples into MCQExample form."""
