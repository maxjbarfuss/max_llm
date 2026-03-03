#!/usr/bin/env python
"""Convert CSV loss curves to TensorBoard event files for visualization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Error: torch.utils.tensorboard not available. Install with: pip install tensorboard")
    exit(1)


def csv_to_tensorboard(csv_path: str | Path, output_dir: str | Path | None = None) -> None:
    """Convert a loss_curve.csv to TensorBoard events.

    Args:
        csv_path: Path to loss_curve.csv file
        output_dir: Output directory for events (defaults to parent of csv + /events)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return

    if output_dir is None:
        output_dir = csv_path.parent / "events"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(output_dir))

    print(f"📊 Converting {csv_path.name}")

    step_count = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            loss = float(row["loss"])
            ppl = float(row["perplexity"])
            lr = float(row["lr"])

            # Log scalars
            writer.add_scalar("loss/train", loss, global_step=step)
            writer.add_scalar("perplexity/train", ppl, global_step=step)
            writer.add_scalar("learning_rate", lr, global_step=step)

            # Optional: log val/test losses if present
            if row.get("val_loss") and row["val_loss"]:
                val_loss = float(row["val_loss"])
                writer.add_scalar("loss/validation", val_loss, global_step=step)

            if row.get("test_loss") and row["test_loss"]:
                test_loss = float(row["test_loss"])
                writer.add_scalar("loss/test", test_loss, global_step=step)

            # Log throughput if available
            if row.get("tokens_per_sec") and row["tokens_per_sec"]:
                tps = float(row["tokens_per_sec"])
                writer.add_scalar("throughput/tokens_per_sec", tps, global_step=step)

            step_count += 1

    writer.close()

    print(f"   ✓ Logged {step_count} steps")
    print(f"   ✓ Events saved to: {output_dir}")
    print(f"\n   View with: tensorboard --logdir={output_dir.parent}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", nargs="+", help="Path(s) to loss_curve.csv file(s)")
    parser.add_argument(
        "--output-dir", help="Where to save event files (defaults to csv_dir/events)"
    )
    args = parser.parse_args()

    for csv_file in args.csv_files:
        csv_to_tensorboard(csv_file, args.output_dir)


if __name__ == "__main__":
    main()
