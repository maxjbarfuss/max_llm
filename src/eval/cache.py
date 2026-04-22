"""Disk-backed cache for normalized benchmark examples."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .types import MCQExample


class BenchmarkExampleCache:
    """Persist normalized MCQ examples for fast repeated benchmark runs."""

    def __init__(self, cache_dir: str | Path | None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def _cache_path(self, task: str, split: str, max_examples: int) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_name = f"{task}__{split}__{max_examples}.json"
        return self.cache_dir / safe_name

    def load(self, task: str, split: str, max_examples: int) -> list[MCQExample] | None:
        path = self._cache_path(task, split, max_examples)
        if path is None or not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [MCQExample(**item) for item in payload]

    def save(
        self,
        task: str,
        split: str,
        max_examples: int,
        examples: list[MCQExample],
    ) -> None:
        path = self._cache_path(task, split, max_examples)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(example) for example in examples]
        path.write_text(json.dumps(payload), encoding="utf-8")
