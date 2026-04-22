"""Benchmark runner for periodic training-time external evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .cache import BenchmarkExampleCache
from .scoring import safe_accuracy, score_choice_logprob
from .tasks import task_registry
from .types import BenchmarkTaskResult


@dataclass(frozen=True)
class BenchmarkRunnerConfig:
    tasks: list[str]
    split: str = "validation"
    max_examples: int = 128
    length_normalize: bool = True
    cache_dir: str | None = "data/cache/benchmarks"


class BenchmarkRunner:
    """Run external MCQ benchmarks against a checkpoint/model state."""

    def __init__(self, config: BenchmarkRunnerConfig) -> None:
        if config.max_examples <= 0:
            raise ValueError("max_examples must be > 0")
        if not config.tasks:
            raise ValueError("tasks cannot be empty")
        self.config = config
        self._registry = task_registry()
        self._cache = BenchmarkExampleCache(config.cache_dir)

    def _eval_task(
        self,
        task_name: str,
        model: nn.Module,
        tokenizer: Any,
        max_seq_len: int,
        device: torch.device,
    ) -> BenchmarkTaskResult:
        if task_name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise ValueError(f"Unknown task '{task_name}'. Available: {available}")

        task = self._registry[task_name]
        examples = self._cache.load(task_name, self.config.split, self.config.max_examples)
        if examples is None:
            examples = task.load_examples(self.config.split, self.config.max_examples)
            self._cache.save(task_name, self.config.split, self.config.max_examples, examples)

        correct = 0
        for ex in examples:
            scores = [
                score_choice_logprob(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=ex.prompt,
                    choice=choice,
                    max_seq_len=max_seq_len,
                    length_normalize=self.config.length_normalize,
                    device=device,
                )
                for choice in ex.choices
            ]
            pred = max(range(len(scores)), key=lambda i: scores[i])
            if pred == ex.answer_index:
                correct += 1

        total = len(examples)
        return BenchmarkTaskResult(
            task=task_name,
            split=self.config.split,
            num_examples=total,
            num_correct=correct,
            accuracy=safe_accuracy(correct, total),
        )

    def run(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_seq_len: int,
        device: torch.device,
    ) -> dict[str, Any]:
        was_training = model.training
        model.eval()
        try:
            task_results = [
                self._eval_task(task_name, model, tokenizer, max_seq_len, device)
                for task_name in self.config.tasks
            ]
        finally:
            if was_training:
                model.train()

        per_task = {result.task: result.accuracy for result in task_results}
        macro = safe_accuracy(
            num_correct=sum(result.num_correct for result in task_results),
            num_examples=sum(result.num_examples for result in task_results),
        )
        return {
            "split": self.config.split,
            "max_examples": self.config.max_examples,
            "per_task": per_task,
            "macro_accuracy": macro,
            "details": [result.__dict__ for result in task_results],
        }
