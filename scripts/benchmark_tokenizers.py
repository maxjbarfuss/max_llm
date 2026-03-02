#!/usr/bin/env python3
"""Benchmark BPE vs Unigram tokenization on identical corpus slices.

Usage:
    source .venv/bin/activate && python scripts/benchmark_tokenizers.py \
    --input data/fast/<corpus_sample>.txt \
        --sample-chars 500000 \
        --unigram-vocab-size 16000 \
    --output outputs/<benchmark_run>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tokenizer.benchmark import benchmark_tokenizer
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.unigram_tokenizer import UnigramTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BPE vs Unigram tokenizers")
    parser.add_argument("--input", type=Path, required=True, help="Input UTF-8 text file")
    parser.add_argument(
        "--sample-chars",
        type=int,
        default=500_000,
        help="Number of initial characters to benchmark (default: 500000)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of repeated encode runs for throughput averaging",
    )
    parser.add_argument(
        "--unigram-vocab-size",
        type=int,
        default=16_000,
        help="Vocab size for SentencePiece Unigram model training",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON report path",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts") / "tokenizer_benchmark",
        help="Directory for temporary benchmark artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.sample_chars <= 0:
        raise ValueError("sample-chars must be > 0")

    text = args.input.read_text(encoding="utf-8")[: args.sample_chars]
    if not text:
        raise ValueError("Input text is empty after slicing sample-chars")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.work_dir / "tokenizer_benchmark_sample.txt"
    sample_path.write_text(text, encoding="utf-8")

    unigram_model_path = UnigramTokenizer.train_model(
        input_path=sample_path,
        model_prefix=args.work_dir / "unigram_benchmark",
        vocab_size=args.unigram_vocab_size,
    )

    bpe = BPETokenizer(encoding="gpt2")
    unigram = UnigramTokenizer(model_path=unigram_model_path)

    bpe_result = benchmark_tokenizer("bpe_gpt2", bpe, text, runs=args.runs)
    unigram_result = benchmark_tokenizer("unigram_spm", unigram, text, runs=args.runs)

    report = {
        "input": str(args.input),
        "sample_chars": len(text),
        "runs": args.runs,
        "unigram_model_path": str(unigram_model_path),
        "results": [asdict(bpe_result), asdict(unigram_result)],
        "decision_hint": (
            "Prefer the tokenizer that improves at least one axis "
            "(chars_per_token or encode_tokens_per_sec) without severe regressions."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote benchmark report: {args.output}")
    for result in (bpe_result, unigram_result):
        print(
            f"{result.tokenizer_name}: chars/token={result.chars_per_token:.3f}, "
            f"unique_ratio={result.unique_token_ratio:.3f}, "
            f"encode_toks/s={result.encode_tokens_per_sec:.0f}"
        )


if __name__ == "__main__":
    main()
