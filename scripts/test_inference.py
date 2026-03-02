#!/usr/bin/env python3
"""Smoke test chat inference using configurable experiment inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chat inference smoke prompts")
    parser.add_argument("--config", required=True, help="Path to experiment config TOML")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--max-tokens", type=int, default=40, help="Max generated tokens")
    parser.add_argument("--timeout", type=int, default=30, help="Per-prompt timeout seconds")
    parser.add_argument(
        "--prompts-file",
        help="Optional JSON file containing array of prompt strings",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to test (repeatable); uses defaults if omitted",
    )
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts:
        return args.prompts
    if args.prompts_file:
        prompts_path = Path(args.prompts_file)
        parsed = json.loads(prompts_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("prompts-file must contain a JSON array of strings")
        return parsed
    return [
        "Once upon a time",
        "In the forest",
        "The dragon",
        "She walked",
        "The story",
        "Chapter",
    ]


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args)

    print("Testing trained model inference...")
    print("=" * 70)

    for prompt in prompts:
        print(f"\nPrompt: {prompt!r}")
        print("-" * 40)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.inference.chat",
                "--config",
                args.config,
                "--checkpoint",
                args.checkpoint,
                "--max-tokens",
                str(args.max_tokens),
            ],
            input=prompt + "\n",
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )

        lines = result.stdout.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("You:") and i + 1 < len(lines):
                output = lines[i + 1]
                if output.startswith("🤖"):
                    text = output[3:].strip()
                    print(f"Output: {text}")
                break

    print("\n" + "=" * 70)
    print("✓ Testing complete")


if __name__ == "__main__":
    main()
