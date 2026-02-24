#!/usr/bin/env python3
"""Run data preparation from a YAML config.

This script wraps the existing CLI tools and standardizes:
- Cutoff mode (article/row/delimiter/none)
- Output naming with encoding metadata
- Metadata sidecar JSON
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping at top level")
    return data


def get_nested(cfg: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = cfg
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


def write_first_n_lines(src: Path, dst: Path, rows: int) -> None:
    ensure_parent(dst)
    with open(src, encoding="utf-8") as f_in, open(dst, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if i >= rows:
                break
            f_out.write(line)


def build_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    source = Path(get_nested(cfg, ["paths", "source"]))
    output_dir = Path(get_nested(cfg, ["paths", "output_dir"], "data/fast"))

    source_dir = source.parent
    source_stem = source.stem

    tokenizer_mode = get_nested(cfg, ["tokenizer", "mode"], "codepoint")
    vocab_size = int(get_nested(cfg, ["tokenizer", "vocab_size"], 128))
    cutoff_mode = str(get_nested(cfg, ["cutoff", "mode"], "article"))
    cutoff_size = str(get_nested(cfg, ["cutoff", "size"], "10M"))

    if tokenizer_mode == "codepoint":
        token_suffix = f"cp{vocab_size}"
    else:
        token_suffix = tokenizer_mode

    normalized_path = get_nested(cfg, ["paths", "normalized_path"])
    if normalized_path:
        normalized_path = Path(normalized_path)
    else:
        normalized_path = source_dir / f"{source_stem}_normalized.txt"

    token_cache_path = get_nested(cfg, ["paths", "token_cache_path"])
    if token_cache_path:
        token_cache_path = Path(token_cache_path)
    else:
        if cutoff_mode == "none":
            token_cache_path = source_dir / f"{source_stem}_normalized_tokens__{token_suffix}.npy"
        else:
            token_cache_path = (
                output_dir
                / f"{source_stem}_subset__{cutoff_mode}_{cutoff_size}_tokens__{token_suffix}.npy"
            )

    subset_tokens_path = get_nested(cfg, ["paths", "subset_tokens_path"])
    subset_size = str(get_nested(cfg, ["subset", "size"], "5M"))
    if subset_tokens_path:
        subset_tokens_path = Path(subset_tokens_path)
    else:
        subset_tokens_path = output_dir / f"{source_stem}_{subset_size}_tokens__{token_suffix}.npy"

    metadata_path = get_nested(cfg, ["paths", "metadata_path"])
    if metadata_path:
        metadata_path = Path(metadata_path)
    else:
        metadata_path = subset_tokens_path.with_suffix(subset_tokens_path.suffix + ".meta.json")

    text_subset_path: Path | None = None
    if cutoff_mode != "none":
        text_subset_path = output_dir / f"{source_stem}_subset__{cutoff_mode}_{cutoff_size}.txt"

    return {
        "source": source,
        "output_dir": output_dir,
        "normalized": normalized_path,
        "token_cache": token_cache_path,
        "subset_tokens": subset_tokens_path,
        "metadata": metadata_path,
        "text_subset": text_subset_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data prep from YAML config")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML config file",
    )

    args = parser.parse_args()
    cfg = load_yaml(args.config)

    workflow = get_nested(cfg, ["workflow"], {}) or {}
    normalize = bool(workflow.get("normalize", True))
    tokenize = bool(workflow.get("tokenize", True))
    force = bool(workflow.get("force", False))

    tokenizer_name = str(get_nested(cfg, ["tokenizer", "name"], "char"))
    tokenizer_mode = str(get_nested(cfg, ["tokenizer", "mode"], "codepoint"))
    vocab_size = int(get_nested(cfg, ["tokenizer", "vocab_size"], 128))

    cutoff_mode = str(get_nested(cfg, ["cutoff", "mode"], "article"))
    cutoff_size = str(get_nested(cfg, ["cutoff", "size"], "10M"))
    cutoff_rows = get_nested(cfg, ["cutoff", "rows"], None)
    cutoff_pattern = get_nested(cfg, ["cutoff", "pattern"], None)
    cutoff_dataset = get_nested(cfg, ["cutoff", "dataset"], None)

    subset_size = str(get_nested(cfg, ["subset", "size"], "5M"))

    paths = build_paths(cfg)

    if cutoff_mode not in {"article", "row", "delimiter", "none"}:
        raise ValueError("cutoff.mode must be one of: article, row, delimiter, none")
    if tokenizer_mode not in {"codepoint", "utf8", "utf16", "utf32"}:
        raise ValueError("tokenizer.mode must be one of: codepoint, utf8, utf16, utf32")

    # Step 1: Normalize
    if normalize:
        if paths["normalized"].exists() and not force:
            print(f"Using cached normalized text: {paths['normalized']}")
        else:
            print(f"Normalizing: {paths['source']} -> {paths['normalized']}")
            run_cmd(
                [
                    sys.executable,
                    "src/data/datasets/wikitext/normalize.py",
                    str(paths["source"]),
                    str(paths["normalized"]),
                ]
            )
    else:
        print("Skipping normalization (workflow.normalize=false)")

    # Step 2: Cutoff (optional)
    token_input_path = paths["normalized"]
    if cutoff_mode != "none":
        if paths["text_subset"] is None:
            raise ValueError("text_subset path is required for cutoff mode")

        if paths["text_subset"].exists() and not force:
            print(f"Using cached text subset: {paths['text_subset']}")
        elif cutoff_mode == "row":
            if cutoff_rows is None:
                raise ValueError("cutoff.rows is required when cutoff.mode=row")
            print(f"Extracting rows: {cutoff_rows} -> {paths['text_subset']}")
            write_first_n_lines(paths["normalized"], paths["text_subset"], int(cutoff_rows))
        else:
            args = [
                sys.executable,
                "-m",
                "src.data.pipeline.extract_text",
                "--input",
                str(paths["normalized"]),
                "--output",
                str(paths["text_subset"]),
                "--size",
                cutoff_size,
            ]
            if cutoff_mode == "article":
                if cutoff_dataset:
                    args += ["--dataset", str(cutoff_dataset)]
            elif cutoff_mode == "delimiter":
                if not cutoff_pattern:
                    raise ValueError("cutoff.pattern is required when cutoff.mode=delimiter")
                args += ["--boundary-pattern", str(cutoff_pattern)]
            print(f"Extracting text subset ({cutoff_mode}): {paths['text_subset']}")
            run_cmd(args)

        token_input_path = paths["text_subset"]

    # Step 3: Tokenize
    if tokenize:
        token_cache = paths["token_cache"]
        if token_cache.exists() and not force:
            print(f"Using cached token file: {token_cache}")
        else:
            print(f"Tokenizing: {token_input_path} -> {token_cache}")
            args = [
                sys.executable,
                "-m",
                "src.data.pipeline.tokenize",
                "--input",
                str(token_input_path),
                "--output",
                str(token_cache),
                "--tokenizer",
                tokenizer_name,
                "--mode",
                tokenizer_mode,
            ]
            if tokenizer_mode == "codepoint":
                args += ["--vocab-size", str(vocab_size)]
            run_cmd(args)
    else:
        if not paths["token_cache"].exists():
            raise FileNotFoundError("token_cache_path does not exist and workflow.tokenize=false")

    # Step 4: Extract token subset
    subset_tokens = paths["subset_tokens"]
    if subset_tokens.exists() and not force:
        print(f"Using existing token subset: {subset_tokens}")
    else:
        print(f"Extracting token subset: {subset_size} -> {subset_tokens}")
        run_cmd(
            [
                sys.executable,
                "-m",
                "src.data.pipeline.extract_tokens",
                "--input",
                str(paths["token_cache"]),
                "--output",
                str(subset_tokens),
                "--size",
                subset_size,
                "-v",
            ]
        )

    # Step 5: Metadata
    metadata = {
        "source": str(paths["source"]),
        "normalized": str(paths["normalized"]),
        "cutoff_mode": cutoff_mode,
        "cutoff_size": cutoff_size,
        "cutoff_rows": cutoff_rows,
        "cutoff_pattern": cutoff_pattern,
        "subset_size": subset_size,
        "tokenizer": tokenizer_name,
        "tokenizer_mode": tokenizer_mode,
        "tokenizer_vocab_size": vocab_size,
        "token_input": str(token_input_path),
        "token_cache": str(paths["token_cache"]),
        "subset_tokens": str(paths["subset_tokens"]),
        "generated_at_utc": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    ensure_parent(paths["metadata"])
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"Metadata written: {paths['metadata']}")


if __name__ == "__main__":
    main()
