"""Task adapters for external MCQ benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec

from .hf import require_hf_token
from .types import BenchmarkTask, MCQExample


def _require_datasets() -> None:
    if find_spec("datasets") is None:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "datasets package is required for benchmark evaluation. "
            "Install requirements and retry."
        )


def _load_dataset(*args: object, **kwargs: object):
    token = require_hf_token()
    datasets_mod = import_module("datasets")
    kwargs.setdefault("token", token)
    return datasets_mod.load_dataset(*args, **kwargs)


def _load_parquet_split(split_files: dict[str, str], split: str):
    token = require_hf_token()
    if split not in split_files:
        available = ", ".join(sorted(split_files))
        raise ValueError(f"Split '{split}' not available for this task. Available: {available}")

    return _load_dataset(
        "parquet",
        data_files={split: split_files[split]},
        split=split,
        token=token,
    )


@dataclass
class HellaSwagTask:
    name: str = "hellaswag"
    dataset_id: str = "hellaswag"

    def load_examples(self, split: str, max_examples: int) -> list[MCQExample]:
        _require_datasets()
        ds = _load_dataset(self.dataset_id, split=split)
        examples: list[MCQExample] = []
        for item in ds:
            label_raw = item.get("label", "0")
            try:
                label = int(label_raw)
            except (TypeError, ValueError):
                continue
            endings = item.get("endings")
            if not isinstance(endings, list) or len(endings) < 2:
                continue
            ctx_a = str(item.get("ctx_a", "")).strip()
            ctx_b = str(item.get("ctx_b", "")).strip()
            prompt = (ctx_a + " " + ctx_b).strip()
            if not prompt or label < 0 or label >= len(endings):
                continue
            examples.append(
                MCQExample(
                    prompt=prompt,
                    choices=[str(choice) for choice in endings],
                    answer_index=label,
                    example_id=str(item.get("ind", "")) or None,
                )
            )
            if len(examples) >= max_examples:
                break
        return examples


@dataclass
class PIQATask:
    name: str = "piqa"
    split_files: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.split_files is None:
            self.split_files = {
                "validation": (
                    "hf://datasets/gimmaru/piqa/"
                    "data/validation-00000-of-00001-26538eb75c618d24.parquet"
                )
            }

    def load_examples(self, split: str, max_examples: int) -> list[MCQExample]:
        _require_datasets()
        ds = _load_parquet_split(self.split_files, split)
        examples: list[MCQExample] = []
        for item in ds:
            goal = str(item.get("goal", "")).strip()
            sol1 = str(item.get("sol1", "")).strip()
            sol2 = str(item.get("sol2", "")).strip()
            label_raw = item.get("label", 0)
            try:
                label = int(label_raw)
            except (TypeError, ValueError):
                continue
            if not goal or not sol1 or not sol2 or label not in (0, 1):
                continue
            examples.append(
                MCQExample(
                    prompt=f"Question: {goal}\nAnswer:",
                    choices=[sol1, sol2],
                    answer_index=label,
                    example_id=str(item.get("id", "")) or None,
                )
            )
            if len(examples) >= max_examples:
                break
        return examples


def _arc_answer_index(labels: list[str], answer_key: str) -> int | None:
    # Common ARC forms: A/B/C/D or 1/2/3/4.
    if answer_key in labels:
        return labels.index(answer_key)
    map_num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    mapped = map_num_to_letter.get(answer_key)
    if mapped and mapped in labels:
        return labels.index(mapped)
    return None


@dataclass
class ARCEasyTask:
    name: str = "arc_easy"
    dataset_id: str = "allenai/ai2_arc"
    dataset_config: str = "ARC-Easy"

    def load_examples(self, split: str, max_examples: int) -> list[MCQExample]:
        _require_datasets()
        ds = _load_dataset(self.dataset_id, self.dataset_config, split=split)
        examples: list[MCQExample] = []
        for item in ds:
            question = str(item.get("question", "")).strip()
            choices_blob = item.get("choices", {})
            if not isinstance(choices_blob, dict):
                continue
            texts = choices_blob.get("text")
            labels = choices_blob.get("label")
            if not isinstance(texts, list) or not isinstance(labels, list) or len(texts) < 2:
                continue

            normalized_labels = [str(label) for label in labels]
            answer_key = str(item.get("answerKey", "")).strip()
            answer_index = _arc_answer_index(normalized_labels, answer_key)
            if not question or answer_index is None:
                continue

            examples.append(
                MCQExample(
                    prompt=f"Question: {question}\nAnswer:",
                    choices=[str(choice) for choice in texts],
                    answer_index=answer_index,
                    example_id=str(item.get("id", "")) or None,
                )
            )
            if len(examples) >= max_examples:
                break
        return examples


def task_registry() -> dict[str, BenchmarkTask]:
    """Return canonical task registry for MCQ benchmarks."""
    return {
        "hellaswag": HellaSwagTask(),
        "piqa": PIQATask(),
        "arc_easy": ARCEasyTask(),
    }
