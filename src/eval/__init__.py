"""Reusable external benchmark harness for max_llm.

This package provides:
- Task adapters (HellaSwag, PIQA, ARC-Easy)
- Shared multiple-choice log-likelihood scoring
- A runner suitable for periodic in-training evaluation
"""

from .runner import BenchmarkRunner, BenchmarkRunnerConfig

__all__ = ["BenchmarkRunner", "BenchmarkRunnerConfig"]
