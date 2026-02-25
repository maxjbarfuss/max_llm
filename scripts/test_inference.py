#!/usr/bin/env python3
"""Test chat interface with interactive prompts."""

import subprocess
import sys

prompts = [
    "Once upon a time",
    "In the forest",
    "The dragon",
    "She walked",
    "The story",
    "Chapter",
]

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
            "config/experiment_curriculum.toml",
            "--checkpoint",
            "outputs/curriculum-alternating/checkpoint.pt",
            "--max-tokens",
            "40",
        ],
        input=prompt + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Extract model output (skip system messages)
    lines = result.stdout.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("You:") and i + 1 < len(lines):
            output = lines[i + 1]
            if output.startswith("🤖"):
                # Extract text after emoji
                text = output[3:].strip()
                print(f"Output: {text}")
            break

print("\n" + "=" * 70)
print("✓ Testing complete")
