"""Unit tests for external benchmark harness core behavior."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.eval.runner import BenchmarkRunner, BenchmarkRunnerConfig
from src.eval.scoring import _fit_context_window
from src.eval.types import MCQExample


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        text = text.strip().lower()
        if text == "prompt":
            return [9, 9]
        if text == "good":
            return [1, 1]
        if text == "bad":
            return [2, 2]
        return [3]


class _FakeModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Strongly prefer token id 1 over others.
        bsz, seq = x.shape
        vocab = 16
        logits = torch.zeros((bsz, seq, vocab), dtype=torch.float32, device=x.device)
        logits[..., 1] = 5.0
        return logits


class _FakeTask:
    name = "fake"

    def load_examples(self, split: str, max_examples: int) -> list[MCQExample]:
        del split
        return [
            MCQExample(prompt="prompt", choices=["good", "bad"], answer_index=0),
            MCQExample(prompt="prompt", choices=["bad", "good"], answer_index=1),
        ][:max_examples]


def test_fit_context_window_keeps_choice_and_trims_prompt() -> None:
    prompt = [1, 2, 3, 4]
    choice = [5, 6, 7]
    p2, c2 = _fit_context_window(prompt, choice, max_seq_len=5)
    assert c2 == choice
    assert p2 == [3, 4]


def test_benchmark_runner_aggregates_accuracy() -> None:
    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(tasks=["fake"], split="validation", max_examples=2)
    )
    runner._registry = {"fake": _FakeTask()}  # inject test task registry

    result = runner.run(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        max_seq_len=32,
        device=torch.device("cpu"),
    )

    assert "fake" in result["per_task"]
    assert result["per_task"]["fake"] == 1.0
    assert result["macro_accuracy"] == 1.0
