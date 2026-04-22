"""Shared Hugging Face auth helpers for benchmark loading."""

from __future__ import annotations

import os
from pathlib import Path

_TOKEN_FILE = Path(".huggingface/.hf_token")


def require_hf_token() -> str:
    """Load Hugging Face token from env or repo-local token file.

    Always mirrors the discovered token into both common environment variables
    so downstream `datasets` / `huggingface_hub` calls run authenticated.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and _TOKEN_FILE.exists():
        token = _TOKEN_FILE.read_text(encoding="utf-8").strip()

    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN/HUGGING_FACE_HUB_TOKEN or save one to "
            ".huggingface/.hf_token."
        )

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token
