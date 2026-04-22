"""MCQ benchmark harness: HellaSwag, PIQA, ARC-Easy.

Usage:
    from src.eval import BenchmarkRunner, BenchmarkRunnerConfig

CLI:
    python -m src.eval.run --config ... --checkpoint ... --tasks hellaswag,piqa,arc_easy
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCQExample:
    prompt: str
    choices: list[str]
    answer_index: int
    example_id: str | None = None


@dataclass(frozen=True)
class BenchmarkTaskResult:
    task: str
    split: str
    num_examples: int
    num_correct: int
    accuracy: float


# ---------------------------------------------------------------------------
# HF auth
# ---------------------------------------------------------------------------

_TOKEN_FILE = Path(".huggingface/.hf_token")


def _require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and _TOKEN_FILE.exists():
        token = _TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(
            "Missing HF token. Set HF_TOKEN/HUGGING_FACE_HUB_TOKEN or create .huggingface/.hf_token."
        )
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


# ---------------------------------------------------------------------------
# Example cache
# ---------------------------------------------------------------------------


class _ExampleCache:
    def __init__(self, cache_dir: str | Path | None) -> None:
        self._dir = Path(cache_dir) if cache_dir is not None else None

    def _path(self, task: str, split: str, n: int) -> Path | None:
        return None if self._dir is None else self._dir / f"{task}__{split}__{n}.json"

    def load(self, task: str, split: str, n: int) -> list[MCQExample] | None:
        p = self._path(task, split, n)
        if p is None or not p.exists():
            return None
        return [MCQExample(**item) for item in json.loads(p.read_text(encoding="utf-8"))]

    def save(self, task: str, split: str, n: int, examples: list[MCQExample]) -> None:
        p = self._path(task, split, n)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([asdict(ex) for ex in examples]), encoding="utf-8")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _fit_context_window(
    prompt_tokens: list[int], choice_tokens: list[int], max_seq_len: int
) -> tuple[list[int], list[int]]:
    """Trim left context to fit window while preserving completion tokens."""
    if max_seq_len <= 1:
        return prompt_tokens[-1:], choice_tokens[:1]
    if len(prompt_tokens) + len(choice_tokens) <= max_seq_len:
        return prompt_tokens, choice_tokens
    keep = max(0, max_seq_len - len(choice_tokens))
    if keep == 0:
        return [], choice_tokens[-max_seq_len:]
    return prompt_tokens[-keep:], choice_tokens


def _score_logprob(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    choice: str,
    max_seq_len: int,
    length_normalize: bool,
    device: torch.device,
) -> float:
    pt = tokenizer.encode(prompt)
    ct = tokenizer.encode(choice)
    if not ct:
        return float("-inf")
    pt, ct = _fit_context_window(pt, ct, max_seq_len)
    full = pt + ct
    if len(full) < 2:
        return float("-inf")
    ids = torch.tensor(full[:-1], dtype=torch.long, device=device).unsqueeze(0)
    tgt = torch.tensor(full[1:], dtype=torch.long, device=device)
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), ctx:
        lp = F.log_softmax(model(ids)[0], dim=-1)
    start = max(0, len(pt) - 1)
    if start >= tgt.shape[0]:
        return float("-inf")
    sl = tgt[start:]
    s = float(lp[start:].gather(1, sl.unsqueeze(1)).squeeze(1).sum().item())
    return s / max(1, sl.shape[0]) if length_normalize else s


def _safe_acc(correct: int, total: int) -> float:
    return float("nan") if total <= 0 else correct / total


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def _require_datasets() -> None:
    if find_spec("datasets") is None:  # pragma: no cover
        raise RuntimeError("datasets package required for benchmark evaluation.")


def _load_hf(*args: object, **kwargs: object) -> Any:
    token = _require_hf_token()
    kwargs.setdefault("token", token)
    return import_module("datasets").load_dataset(*args, **kwargs)


class _HellaSwag:
    name = "hellaswag"

    def load_examples(self, split: str, max_examples: int) -> list[MCQExample]:
        _require_datasets()
        examples: list[MCQExample] = []
        for item in _load_hf("hellaswag", split=split):
            try:
                label = int(item.get("label", "0"))
            except (TypeError, ValueError):
                continue
            endings = item.get("endings")
            if not isinstance(endings, list) or len(endings) < 2:
                continue
            prompt = (str(item.get("ctx_a", "")) + " " + str(item.get("ctx_b", ""))).strip()
            if not prompt or label < 0 or label >= len(endings):
                continue
            examples.append(
                MCQExample(
                    prompt, [str(c) for c in endings], label, str(item.get("ind", "")) or None
                )
            )
            if len(examples) >= max_examples:
                break
        return examples


_PIQA_URL = "hf://datasets/gimmaru/piqa/data/validation-00000-of-00001-26538eb75c618d24.parquet"


class _PIQA:
    name = "piqa"

    def load_examples(self, split: str, max_examples: int) -> list[MCQExample]:
        _require_datasets()
        if split != "validation":
            raise ValueError(f"PIQA only supports 'validation'; got '{split}'")
        examples: list[MCQExample] = []
        for item in _load_hf("parquet", data_files={split: _PIQA_URL}, split=split):
            goal = str(item.get("goal", "")).strip()
            sol1, sol2 = str(item.get("sol1", "")).strip(), str(item.get("sol2", "")).strip()
            try:
                label = int(item.get("label", 0))
            except (TypeError, ValueError):
                continue
            if not goal or not sol1 or not sol2 or label not in (0, 1):
                continue
            examples.append(
                MCQExample(
                    f"Question: {goal}\nAnswer:",
                    [sol1, sol2],
                    label,
                    str(item.get("id", "")) or None,
                )
            )
            if len(examples) >= max_examples:
                break
        return examples


def _arc_answer_index(labels: list[str], answer_key: str) -> int | None:
    if answer_key in labels:
        return labels.index(answer_key)
    mapped = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}.get(answer_key)
    return labels.index(mapped) if mapped and mapped in labels else None


class _ARCEasy:
    name = "arc_easy"

    def load_examples(self, split: str, max_examples: int) -> list[MCQExample]:
        _require_datasets()
        examples: list[MCQExample] = []
        for item in _load_hf("allenai/ai2_arc", "ARC-Easy", split=split):
            question = str(item.get("question", "")).strip()
            cb = item.get("choices", {})
            if not isinstance(cb, dict):
                continue
            texts, raw_labels = cb.get("text"), cb.get("label")
            if not isinstance(texts, list) or not isinstance(raw_labels, list) or len(texts) < 2:
                continue
            norms = [str(lb) for lb in raw_labels]
            idx = _arc_answer_index(norms, str(item.get("answerKey", "")).strip())
            if not question or idx is None:
                continue
            examples.append(
                MCQExample(
                    f"Question: {question}\nAnswer:",
                    [str(c) for c in texts],
                    idx,
                    str(item.get("id", "")) or None,
                )
            )
            if len(examples) >= max_examples:
                break
        return examples


def _default_registry() -> dict[str, Any]:
    return {"hellaswag": _HellaSwag(), "piqa": _PIQA(), "arc_easy": _ARCEasy()}


# ---------------------------------------------------------------------------
# Runner (public API)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkRunnerConfig:
    tasks: list[str]
    split: str = "validation"
    max_examples: int = 128
    length_normalize: bool = True
    cache_dir: str | None = "data/cache/benchmarks"


class BenchmarkRunner:
    """Run MCQ benchmarks against a model checkpoint."""

    def __init__(self, config: BenchmarkRunnerConfig) -> None:
        if config.max_examples <= 0:
            raise ValueError("max_examples must be > 0")
        if not config.tasks:
            raise ValueError("tasks cannot be empty")
        self.config = config
        self._registry = _default_registry()
        self._cache = _ExampleCache(config.cache_dir)

    def _eval_task(
        self,
        name: str,
        model: nn.Module,
        tokenizer: Any,
        max_seq_len: int,
        device: torch.device,
    ) -> BenchmarkTaskResult:
        if name not in self._registry:
            raise ValueError(
                f"Unknown task '{name}'. Available: {', '.join(sorted(self._registry))}"
            )
        task = self._registry[name]
        examples = self._cache.load(name, self.config.split, self.config.max_examples)
        if examples is None:
            examples = task.load_examples(self.config.split, self.config.max_examples)
            self._cache.save(name, self.config.split, self.config.max_examples, examples)
        ln = self.config.length_normalize

        def _best(ex: MCQExample) -> int:
            scores = [
                _score_logprob(model, tokenizer, ex.prompt, c, max_seq_len, ln, device)
                for c in ex.choices
            ]
            return max(range(len(scores)), key=lambda i: scores[i])

        correct = sum(1 for ex in examples if _best(ex) == ex.answer_index)
        return BenchmarkTaskResult(
            name, self.config.split, len(examples), correct, _safe_acc(correct, len(examples))
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
            results = [
                self._eval_task(t, model, tokenizer, max_seq_len, device) for t in self.config.tasks
            ]
        finally:
            if was_training:
                model.train()
        return {
            "split": self.config.split,
            "max_examples": self.config.max_examples,
            "per_task": {r.task: r.accuracy for r in results},
            "macro_accuracy": _safe_acc(
                sum(r.num_correct for r in results), sum(r.num_examples for r in results)
            ),
            "details": [r.__dict__ for r in results],
        }


__all__ = ["BenchmarkRunner", "BenchmarkRunnerConfig", "MCQExample"]
