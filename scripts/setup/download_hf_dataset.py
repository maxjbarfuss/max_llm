#!/usr/bin/env python3
"""Download Hugging Face dataset repos to a persistent D-drive staging area.

This is a rerunnable helper for repeated corpus acquisition before data prep.
It keeps raw dataset snapshots under `/mnt/d/dev/data/hf/` by default so the fast
local training artifact area (`data/fast/`) remains reserved for prepared
tokenized outputs.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DEFAULT_DOWNLOAD_ROOT = Path("/mnt/d/dev/data/hf")
DEFAULT_CACHE_DIR = Path("/mnt/d/dev/data/hf_cache")
TOKEN_FILE = Path(".huggingface/.hf_token")
_BRACE_EXPR = re.compile(r"\{([^{}]+)\}")


def _load_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()

    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()

    raise SystemExit(
        "Missing Hugging Face token. Set HF_TOKEN/HUGGING_FACE_HUB_TOKEN or save one to "
        ".huggingface/.hf_token."
    )


def _dataset_dir(root: Path, dataset_id: str, revision: str | None) -> Path:
    safe_id = dataset_id.replace("/", "__")
    if revision:
        safe_rev = revision.replace("/", "__")
        return root / f"{safe_id}__{safe_rev}"
    return root / safe_id


def _expand_brace_pattern(pattern: str) -> list[str]:
    match = _BRACE_EXPR.search(pattern)
    if not match:
        return [pattern]

    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    expr = match.group(1)
    range_match = re.fullmatch(r"(-?\d+)\.\.(-?\d+)", expr)

    expansions: list[str] = []
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        width = max(
            len(range_match.group(1).lstrip("-")),
            len(range_match.group(2).lstrip("-")),
        )
        step = 1 if end >= start else -1
        for value in range(start, end + step, step):
            sign = "-" if value < 0 else ""
            digits = f"{abs(value):0{width}d}"
            expansions.extend(_expand_brace_pattern(f"{prefix}{sign}{digits}{suffix}"))
        return expansions

    for option in expr.split(","):
        expansions.extend(_expand_brace_pattern(f"{prefix}{option}{suffix}"))
    return expansions


def _expand_patterns(patterns: list[str] | None) -> list[str] | None:
    if not patterns:
        return None

    expanded: list[str] = []
    for pattern in patterns:
        expanded.extend(_expand_brace_pattern(pattern))
    return expanded


def _filter_paths(paths: list[str], patterns: list[str] | None) -> list[str]:
    if not patterns:
        return paths
    return [path for path in paths if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)]


def _print_paths(paths: list[str], limit: int) -> None:
    for path in paths[:limit]:
        print(path)
    if len(paths) > limit:
        print(f"... ({len(paths) - limit} more)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset repo into /mnt/d/dev/data for repeated prep runs"
    )
    parser.add_argument("dataset", help="Dataset repo id, e.g. wikimedia/wikipedia")
    parser.add_argument(
        "--revision",
        default=None,
        help="Dataset revision / branch / tag / commit, e.g. 20231101.en",
    )
    parser.add_argument(
        "--download-root",
        default=str(DEFAULT_DOWNLOAD_ROOT),
        help="Root directory for raw dataset snapshots (default: /mnt/d/dev/data/hf)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Hugging Face cache directory (default: /mnt/d/dev/data/hf_cache)",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=None,
        help=(
            "Glob pattern to include (repeatable), e.g. 'data/*.parquet'. "
            "Supports brace expansion such as 'data/train-{0000..0007}.parquet'."
        ),
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=None,
        help="Glob pattern to exclude (repeatable)",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List repo files after applying allow filters instead of downloading",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max files to print with --list-files (default: 200)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh download instead of reusing existing files when possible",
    )

    args = parser.parse_args()

    token = _load_hf_token()
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    api = HfApi(token=token)
    repo_files = api.list_repo_files(
        repo_id=args.dataset,
        repo_type="dataset",
        revision=args.revision,
    )

    expanded_allow = _expand_patterns(args.allow)
    expanded_ignore = _expand_patterns(args.ignore)
    filtered_files = _filter_paths(sorted(repo_files), expanded_allow)
    if expanded_ignore:
        filtered_files = [
            path
            for path in filtered_files
            if not any(fnmatch.fnmatch(path, pattern) for pattern in expanded_ignore)
        ]

    if expanded_allow:
        print(f"Matched {len(filtered_files)} files for download.")
        if len(filtered_files) > 25 and any("*" in pattern for pattern in args.allow or []):
            print(
                "Warning: allow pattern matched many files. Prefer explicit brace ranges like "
                "'data/CC-MAIN-2023-14/000_000{00..07}.parquet' for shard subsets.",
                file=sys.stderr,
            )

    if args.list_files:
        _print_paths(filtered_files, args.limit)
        return 0

    download_root = Path(args.download_root)
    cache_dir = Path(args.cache_dir)
    local_dir = _dataset_dir(download_root, args.dataset, args.revision)

    download_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        revision=args.revision,
        token=token,
        local_dir=str(local_dir),
        cache_dir=str(cache_dir),
        allow_patterns=expanded_allow,
        ignore_patterns=expanded_ignore,
        force_download=args.force,
    )

    print(f"Downloaded to: {snapshot_path}")
    if expanded_allow:
        print("Included patterns:")
        for pattern in expanded_allow:
            print(f"- {pattern}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
