"""Document boundary detection for dataset-aware text extraction.

Used by src.data.pipeline.extract_text to decide where documents start.
Each dataset gets a concrete BoundaryDetector; the factory resolves by name.

Adding a new dataset:
    1. Implement a BoundaryDetector subclass below
    2. Register it in _REGISTRY with the dataset's canonical name
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


class BoundaryDetector(ABC):
    """Detects the start of a new document within a text stream."""

    @abstractmethod
    def is_boundary(self, line: str) -> bool:
        """Return True if *line* is the first line of a new document."""


class WikiTextBoundary(BoundaryDetector):
    """Top-level article headers in WikiText-103.

    Matches ' = Title = ' (single = on each side, not section headers ' == ... == ').
    Example: ' = Valkyria Chronicles III = '
    """

    _PATTERN = re.compile(r"^\s*=\s+[^=]+\s+=\s*$")

    def is_boundary(self, line: str) -> bool:
        return bool(self._PATTERN.match(line))


class BlankLineBoundary(BoundaryDetector):
    """Blank or whitespace-only lines mark document boundaries.

    Used by datasets (e.g., karpathy/tinystories-gpt4-clean) where stories
    are separated by empty lines rather than explicit delimiters.
    """

    def is_boundary(self, line: str) -> bool:
        return line.strip() == ""


class TinyStoriesBoundary(BoundaryDetector):
    """Story separators in TinyStories datasets.

    Supports two formats:
    1. GPT-style with '<|endoftext|>' token on its own line
    2. karpathy/tinystories-gpt4-clean with blank line separators

    Checks for either format to be compatible with dataset variants.
    """

    def is_boundary(self, line: str) -> bool:
        stripped = line.strip()
        # GPT-style endoftext token
        if stripped == "<|endoftext|>":
            return True
        # Blank line separator (karpathy variant)
        if stripped == "":
            return True
        return False


class PatternBoundary(BoundaryDetector):
    """Arbitrary caller-supplied regex pattern."""

    def __init__(self, pattern: str) -> None:
        self._re = re.compile(pattern)

    def is_boundary(self, line: str) -> bool:
        return bool(self._re.match(line))


class NoBoundary(BoundaryDetector):
    """Disables boundary detection; extract_text cuts at the byte limit."""

    def is_boundary(self, line: str) -> bool:
        return False


_REGISTRY: dict[str, type[BoundaryDetector]] = {
    "wikitext": WikiTextBoundary,
    "tinystories": TinyStoriesBoundary,
    "blank_line": BlankLineBoundary,
}


def get_boundary_detector(
    dataset: str | None = None,
    pattern: str | None = None,
) -> BoundaryDetector:
    """Return the appropriate BoundaryDetector.

    Resolution order (first match wins):
      1. Explicit *pattern* string  →  PatternBoundary
      2. Known *dataset* name       →  registered detector
      3. Neither                    →  NoBoundary (extract at byte limit)

    Raises ValueError for an unrecognised dataset name so callers fail
    loudly rather than silently extracting without boundaries.
    """
    if pattern is not None:
        return PatternBoundary(pattern)
    if dataset is not None:
        if dataset not in _REGISTRY:
            known = ", ".join(sorted(_REGISTRY))
            raise ValueError(f"Unknown dataset '{dataset}'. Known: {known}")
        return _REGISTRY[dataset]()
    return NoBoundary()
