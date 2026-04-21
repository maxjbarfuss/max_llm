"""Lightweight runtime diagnostics for large data-preparation runs."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


def _process_snapshot() -> dict[str, float]:
    process = psutil.Process()
    memory = process.memory_info()
    virtual = psutil.virtual_memory()
    return {
        "rss_mb": round(memory.rss / 1024**2, 2),
        "vms_mb": round(memory.vms / 1024**2, 2),
        "system_available_mb": round(virtual.available / 1024**2, 2),
        "system_used_percent": round(float(virtual.percent), 2),
    }


def emit_prep_diagnostic(
    output_dir: str | Path,
    prefix: str,
    phase: str,
    **details: Any,
) -> None:
    """Append a structured diagnostic event for long-running prep jobs.

    The event is written to a JSONL file in the output directory and echoed to
    the logger so the last successful phase is recoverable after a crash.
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "phase": phase,
            **_process_snapshot(),
            **details,
        }
        diag_path = output_path / f"{prefix}_prep_diagnostics.jsonl"
        with diag_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        logger.info("[prep-diagnostic] %s", json.dumps(event, sort_keys=True))
    except Exception as exc:
        logger.warning("Failed to write prep diagnostic for phase=%s: %s", phase, exc)
