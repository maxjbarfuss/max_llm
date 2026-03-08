#!/usr/bin/env python3
"""Fast local dataset locator for common experiment sources.

This avoids expensive whole-home scans by searching only explicit roots with
bounded depth and deterministic ranking.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_ROOTS = [
    "/mnt/d/dev/data",
    "/mnt/d/data",
    "/home/max/dev/data",
]

_SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}


def _iter_files(root: Path, max_depth: int):
    root = root.resolve()
    root_depth = len(root.parts)

    for dirpath, dirnames, filenames in os.walk(root):
        cur = Path(dirpath)
        depth = len(cur.parts) - root_depth

        # Prune deep recursion and noisy folders early.
        dirnames[:] = [
            d
            for d in dirnames
            if d not in _SKIP_DIR_NAMES and not d.startswith(".") and depth < max_depth
        ]

        for name in filenames:
            yield cur / name


def _score_tinystories(path: Path) -> int:
    p = str(path).lower()
    score = 0
    if p.endswith("/tinystories-gpt4-clean/train.txt"):
        score += 100
    if "tinystories-gpt4-clean" in p:
        score += 20
    if "/max_llm_cache/" in p:
        score += 10
    if p.endswith(".txt"):
        score += 5
    if "boundary" in p:
        score += 3
    if p.endswith(".arrow"):
        score -= 10
    return score


def find_tinystories(roots: list[Path], max_depth: int) -> list[Path]:
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in _iter_files(root, max_depth=max_depth):
            p = str(path).lower()
            if "tinystories" not in p:
                continue
            if path.suffix.lower() not in {".txt", ".json", ".jsonl", ".parquet", ".arrow", ".npy"}:
                continue
            candidates.append(path)

    candidates.sort(key=lambda p: (-_score_tinystories(p), str(p)))
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Locate local dataset source files quickly")
    parser.add_argument(
        "--dataset",
        default="tinystories-gpt4-clean",
        choices=["tinystories-gpt4-clean"],
        help="Dataset identifier (default: tinystories-gpt4-clean)",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Search root (repeatable). Defaults to known local data roots.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum directory depth under each root (default: 6)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print all matching candidates instead of only the best match",
    )

    args = parser.parse_args()

    roots = [Path(p) for p in (args.root or DEFAULT_ROOTS)]

    if args.dataset == "tinystories-gpt4-clean":
        matches = find_tinystories(roots=roots, max_depth=args.max_depth)
    else:
        matches = []

    if not matches:
        print("No dataset files found. Tried roots:")
        for root in roots:
            print(f"- {root}")
        return 1

    if args.all:
        for p in matches:
            print(p)
    else:
        print(matches[0])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
