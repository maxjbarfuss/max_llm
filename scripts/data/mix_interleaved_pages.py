#!/usr/bin/env python3
"""Build an interleaved mixed text corpus from multiple boundary-aware sources.

The mixer:
- Splits each input into complete documents/pages using boundary detectors
- Shuffles page order within each source (seeded)
- Interleaves across N sources with weighted randomized selection
- Writes only whole documents/pages (never cuts mid-page)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.datasets.boundary import get_boundary_detector

# Dataset registry: Maps dataset name to (default_input_path, boundary_dataset, description)
DATASET_REGISTRY: dict[str, tuple[str, str, str]] = {
    "wikitext-103": (
        "/mnt/d/dev/data/wikitext-103-raw/train.txt",
        "wikitext",
        "WikiText-103 raw corpus (article-level boundaries)",
    ),
    "tinystories": (
        "/mnt/d/dev/data/tinystories-gpt4-clean/train.txt",
        "tinystories",
        "TinyStories GPT-4 clean corpus (story-level boundaries)",
    ),
}


def parse_size(size_str: str) -> int:
    size = size_str.strip().upper()
    multipliers = {"K": 1024, "M": 1024 * 1024, "G": 1024 * 1024 * 1024}
    for suffix, multiplier in multipliers.items():
        if size.endswith(suffix):
            return int(float(size[:-1]) * multiplier)
    return int(size)


def parse_corpus_mix(corpus_mix_args: list[str]) -> list[dict[str, Any]]:
    """Parse --corpus-mix dataset_name percent ... into source specs.

    Example:
        ['wikitext-103', '50', 'tinystories', '50']
        → [
            {'name': 'wikitext-103', 'dataset': 'wikitext', 'percent': 50.0, 'weight': 1.0, ...},
            {'name': 'tinystories', 'dataset': 'tinystories', 'percent': 50.0, 'weight': 1.0, ...},
          ]
    """
    if len(corpus_mix_args) < 2 or len(corpus_mix_args) % 2 != 0:
        raise ValueError(
            "--corpus-mix requires pairs of dataset_name and percent. "
            "Example: --corpus-mix wikitext-103 50 tinystories 50"
        )

    sources: list[dict[str, Any]] = []
    total_percent = 0.0

    for i in range(0, len(corpus_mix_args), 2):
        dataset_name = corpus_mix_args[i]
        try:
            percent = float(corpus_mix_args[i + 1])
        except ValueError:
            raise ValueError(f"Percent value must be numeric, got: {corpus_mix_args[i + 1]}")

        if dataset_name not in DATASET_REGISTRY:
            known = ", ".join(sorted(DATASET_REGISTRY))
            raise ValueError(f"Unknown dataset '{dataset_name}'. Known: {known}")

        if percent <= 0:
            raise ValueError(f"Dataset '{dataset_name}' has non-positive percent: {percent}")

        input_path, boundary_dataset, _desc = DATASET_REGISTRY[dataset_name]

        sources.append(
            {
                "name": dataset_name,
                "input": input_path,
                "dataset": boundary_dataset,
                "percent": percent,
                "weight": percent,  # Normalized to sum=100 later
                "force_line_boundaries": False,
                "fallback_line_boundaries": False,
            }
        )
        total_percent += percent

    if abs(total_percent - 100.0) > 0.01:
        raise ValueError(f"Corpus mix percentages must sum to 100, got: {total_percent}")

    # Normalize weights so they sum to 1.0 (for internal weighted_choice)
    for source in sources:
        source["weight"] = source["percent"] / 100.0

    return sources


def split_documents(
    text: str,
    dataset: str,
) -> list[str]:
    """Split text into documents using the dataset's boundary detector."""
    detector = get_boundary_detector(dataset=dataset)
    documents: list[str] = []
    current_lines: list[str] = []

    for line in text.splitlines(keepends=True):
        if detector.is_boundary(line):
            if current_lines:
                documents.append("".join(current_lines))
                current_lines = []
        current_lines.append(line)

    if current_lines:
        documents.append("".join(current_lines))

    return [doc for doc in documents if doc.strip()]


def split_documents_by_ratio(
    docs: list[str],
    train_pct: float,
    val_pct: float,
    test_pct: float,
    rng: random.Random,
) -> tuple[list[str], list[str], list[str]]:
    """Split documents into train/val/test sets while preserving order within each split.

    Args:
        docs: List of documents to split
        train_pct: Percentage for training set (0-100)
        val_pct: Percentage for validation set (0-100)
        test_pct: Percentage for test set (0-100)
        rng: Random number generator (for reproducibility)

    Returns:
        (train_docs, val_docs, test_docs)
    """
    if abs((train_pct + val_pct + test_pct) - 100.0) > 0.01:
        raise ValueError(
            f"Split percentages must sum to 100, got: {train_pct + val_pct + test_pct}"
        )

    # Create list of (index, doc) pairs and shuffle by indices
    indexed_docs = list(enumerate(docs))
    rng.shuffle(indexed_docs)

    total = len(docs)
    train_count = max(1, int(total * train_pct / 100.0))
    val_count = max(1, int(total * val_pct / 100.0)) if val_pct > 0 else 0
    # test_count gets remainder to handle rounding
    test_count = total - train_count - val_count

    train_indices = set(idx for idx, _doc in indexed_docs[:train_count])
    val_indices = set(idx for idx, _doc in indexed_docs[train_count : train_count + val_count])
    test_indices = set(idx for idx, _doc in indexed_docs[train_count + val_count :])

    # Reconstruct each split preserving original order
    train = [doc for i, doc in enumerate(docs) if i in train_indices]
    val = [doc for i, doc in enumerate(docs) if i in val_indices]
    test = [doc for i, doc in enumerate(docs) if i in test_indices]

    return train, val, test


def write_split(output_path: Path, docs: list[str], split_name: str) -> tuple[int, int]:
    """Write a split to disk and return (doc_count, bytes_written)."""
    text = "".join(docs)
    output_path.write_text(text, encoding="utf-8")
    return len(docs), len(text.encode("utf-8"))


def weighted_choice(
    names: list[str],
    weights: list[float],
    rng: random.Random,
    last_name: str | None,
) -> str:
    candidates = [
        (name, weight)
        for name, weight in zip(names, weights)
        if weight > 0 and (name != last_name or len(names) == 1)
    ]
    if not candidates:
        candidates = [(name, weight) for name, weight in zip(names, weights) if weight > 0]
    pick_names = [name for name, _weight in candidates]
    pick_weights = [weight for _name, weight in candidates]
    return rng.choices(pick_names, weights=pick_weights, k=1)[0]


def interleave_documents(
    sources_docs: dict[str, list[str]],
    source_weights: dict[str, float],
    rng: random.Random,
    target_bytes: int,
) -> tuple[list[str], dict[str, Any]]:
    """Interleave documents and return them as a list for later splitting.

    Returns:
        (interleaved_docs, stats) where interleaved_docs is a list of full documents
        and stats contains metadata about the interleaving process.
    """
    out_docs: list[str] = []
    consumed = dict.fromkeys(sources_docs, 0)
    pointers = dict.fromkeys(sources_docs, 0)
    last_name: str | None = None
    bytes_written = 0

    while True:
        available = [name for name in sources_docs if pointers[name] < len(sources_docs[name])]
        if not available:
            break

        chosen = weighted_choice(
            names=available,
            weights=[source_weights[name] for name in available],
            rng=rng,
            last_name=last_name,
        )

        doc = sources_docs[chosen][pointers[chosen]]
        pointers[chosen] += 1
        consumed[chosen] += 1

        if not doc.endswith("\n"):
            doc = f"{doc}\n"
        doc = f"{doc}\n"

        out_docs.append(doc)
        bytes_written += len(doc.encode("utf-8"))
        last_name = chosen

        if bytes_written >= target_bytes:
            break

    return out_docs, {
        "docs_used": consumed,
        "bytes_written": bytes_written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mix corpora with randomized interleaved pages.",
        epilog="Example: %(prog)s --corpus-mix wikitext-103 50 tinystories 50 --output mixed.txt --target-size 100M",
    )
    parser.add_argument(
        "--corpus-mix",
        nargs="+",
        type=str,
        required=True,
        metavar="DATASET PERCENT",
        help="Dataset names and percentages (must sum to 100). "
        "Example: wikitext-103 50 tinystories 50",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file for mixed corpus (or base name if using --splits)",
    )
    parser.add_argument(
        "--target-size",
        type=str,
        default="100M",
        help="Target size of mixed corpus (default: 100M). Supports K/M/G suffixes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for document shuffling and interleaving (default: 42)",
    )
    parser.add_argument(
        "--splits",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=None,
        help="Generate train/val/test splits with specified percentages (must sum to 100). "
        "Example: --splits 80 10 10. If omitted, generates single output file.",
    )
    args = parser.parse_args()

    # Parse corpus mix and build source specs from dataset registry
    source_specs = parse_corpus_mix(args.corpus_mix)
    source_docs: dict[str, list[str]] = {}
    source_weights: dict[str, float] = {}
    source_inputs: dict[str, str] = {}
    source_meta: dict[str, dict[str, Any]] = {}

    for spec in source_specs:
        name = str(spec["name"])
        input_path = Path(str(spec["input"]))
        dataset = str(spec["dataset"])
        weight = float(spec["weight"])
        percent = float(spec["percent"])

        if not input_path.exists():
            raise FileNotFoundError(f"Source '{name}' input not found: {input_path}")

        text = input_path.read_text(encoding="utf-8")
        docs = split_documents(text, dataset=dataset)
        if not docs:
            raise ValueError(f"Failed to extract documents from source '{name}'")

        source_docs[name] = docs
        source_weights[name] = weight
        source_inputs[name] = str(input_path)
        source_meta[name] = {
            "dataset": dataset,
            "percent": percent,
        }

    # Interleave with shared RNG
    rng = random.Random(args.seed)
    for docs in source_docs.values():
        rng.shuffle(docs)

    target_bytes = parse_size(args.target_size)
    interleaved_docs, stats = interleave_documents(
        source_docs,
        source_weights=source_weights,
        rng=rng,
        target_bytes=target_bytes,
    )

    # Build common metadata
    docs_used: dict[str, int] = stats["docs_used"]
    meta_sources: list[dict[str, Any]] = []
    for name in source_docs:
        meta_sources.append(
            {
                "name": name,
                "input": source_inputs[name],
                "total_docs": len(source_docs[name]),
                "docs_used": docs_used.get(name, 0),
                **source_meta[name],
            }
        )

    # Handle output: either single file or train/val/test splits
    args.output.parent.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "seed": args.seed,
        "target_size": args.target_size,
        "target_bytes": target_bytes,
        "corpus_mix": [{s["name"]: s["percent"]} for s in meta_sources],
        "sources": meta_sources,
    }

    if args.splits is None:
        # Single output file
        merged_text = "".join(interleaved_docs)
        args.output.write_text(merged_text, encoding="utf-8")

        print(f"Wrote mixed corpus: {args.output}")
        print(f"Total documents: {len(interleaved_docs)} | bytes={stats['bytes_written']:,}")
        print(f"Corpus mix: {', '.join(f'{s['name']}={s['percent']}%' for s in source_specs)}")

        # Add single-file metadata
        meta.update(stats)
        meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote metadata: {meta_path}")
        usage = ", ".join(f"{name}={count}" for name, count in docs_used.items())
        print(f"Docs used: {usage}")
    else:
        # Generate train/val/test splits
        train_pct, val_pct, test_pct = args.splits
        train_docs, val_docs, test_docs = split_documents_by_ratio(
            interleaved_docs,
            train_pct=train_pct,
            val_pct=val_pct,
            test_pct=test_pct,
            rng=rng,
        )

        # Determine output paths
        output_stem = args.output.stem
        output_dir = args.output.parent
        train_path = output_dir / f"{output_stem}_train.txt"
        val_path = output_dir / f"{output_stem}_val.txt"
        test_path = output_dir / f"{output_stem}_test.txt"

        train_count, train_bytes = write_split(train_path, train_docs, "train")
        val_count, val_bytes = write_split(val_path, val_docs, "val")
        test_count, test_bytes = write_split(test_path, test_docs, "test")

        print(f"Wrote train split: {train_path} ({train_count} docs, {train_bytes:,} bytes)")
        print(f"Wrote val split: {val_path} ({val_count} docs, {val_bytes:,} bytes)")
        print(f"Wrote test split: {test_path} ({test_count} docs, {test_bytes:,} bytes)")
        print(f"Corpus mix: {', '.join(f'{s['name']}={s['percent']}%' for s in source_specs)}")

        # Add split-specific metadata
        meta.update(
            {
                "splits": {
                    "train": {
                        "percent": train_pct,
                        "docs": train_count,
                        "bytes": train_bytes,
                        "path": str(train_path),
                    },
                    "val": {
                        "percent": val_pct,
                        "docs": val_count,
                        "bytes": val_bytes,
                        "path": str(val_path),
                    },
                    "test": {
                        "percent": test_pct,
                        "docs": test_count,
                        "bytes": test_bytes,
                        "path": str(test_path),
                    },
                },
                "total": {
                    "docs": len(interleaved_docs),
                    "bytes": train_bytes + val_bytes + test_bytes,
                    "docs_used": docs_used,
                },
            }
        )

        meta_path = output_dir / f"{output_stem}.splits.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
