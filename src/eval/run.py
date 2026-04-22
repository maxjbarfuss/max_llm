"""CLI entrypoint for external benchmark harness.

Usage:
  python -m src.eval.run --config ... --checkpoint ... --tasks hellaswag,piqa,arc_easy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config.experiment import ExperimentConfig
from src.eval import BenchmarkRunner, BenchmarkRunnerConfig
from src.inference.utils import (
    create_tokenizer_from_data_config,
    load_checkpoint_into_model,
    resolve_device,
)
from src.models.learning_model import LearningModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external MCQ benchmarks")
    parser.add_argument("--config", required=True, help="Path to experiment TOML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument(
        "--tasks", default="hellaswag,piqa,arc_easy", help="Comma-separated task list"
    )
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--max-examples", type=int, default=128, help="Max examples per task")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output",
        help="Optional path to write JSON results in addition to stdout",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache/benchmarks",
        help="Directory for cached normalized benchmark examples",
    )
    parser.add_argument(
        "--no-length-normalize",
        action="store_true",
        help="Use sum log-probability instead of length-normalized score",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = ExperimentConfig.from_toml(args.config)
    device = resolve_device(args.device)

    attn_backend = getattr(cfg.training, "attention_backend", "standard")
    if device.type == "cpu" and attn_backend != "standard":
        attn_backend = "standard"

    model = LearningModel.from_config(cfg.model, attention_backend=attn_backend).to(device)
    load_checkpoint_into_model(model, args.checkpoint, device)

    tokenizer = create_tokenizer_from_data_config(cfg.data)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

    runner = BenchmarkRunner(
        BenchmarkRunnerConfig(
            tasks=tasks,
            split=args.split,
            max_examples=args.max_examples,
            length_normalize=not args.no_length_normalize,
            cache_dir=args.cache_dir,
        )
    )
    result = runner.run(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=cfg.model.max_seq_length,
        device=device,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
