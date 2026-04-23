"""Unified run-status reporter.

Usage:
    python -m src.status
    python -m src.status --data-dir data/fast --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────


def _fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} PB"


def _bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total) if total else 0
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total}"


# ── data-prep section ─────────────────────────────────────────────────────────


def _report_dataprep(data_dir: Path) -> None:
    print("── Data Prep ─────────────────────────────────────────────────────────")
    spill_dir = data_dir / ".prep_spill"
    if not spill_dir.exists():
        print("  No active prep session (no .prep_spill directory)")
    else:
        bins = sorted(spill_dir.glob("*.bin"))
        if not bins:
            print("  .prep_spill exists but no source .bin files yet")
        else:
            for bin_path in bins:
                name = bin_path.stem
                size = bin_path.stat().st_size
                offsets_path = bin_path.with_suffix(".bin.offsets.npy")
                meta_path = bin_path.with_suffix(".bin.meta.json")
                cached = offsets_path.exists() and meta_path.exists()
                status = "cached" if cached else "partial (no offsets — incomplete or in-progress)"
                print(f"  {name:<20} {_fmt_size(size):>10}  {status}")

    # Completed datasets
    npy_files = sorted(data_dir.glob("*_train.npy"))
    if npy_files:
        print()
        print("  Completed datasets:")
        for npy in npy_files:
            prefix = npy.name.replace("_train.npy", "")
            stats_path = data_dir / f"{prefix}_stats.json"
            token_info = ""
            if stats_path.exists():
                try:
                    stats = json.loads(stats_path.read_text())
                    train_toks = stats.get("train", {}).get("total_tokens", 0)
                    val_toks = stats.get("val", {}).get("total_tokens", 0)
                    token_info = f"  train={train_toks/1e6:.1f}M  val={val_toks/1e6:.1f}M tokens"
                except Exception:
                    pass
            print(f"  {prefix}{token_info}")


# ── training section ──────────────────────────────────────────────────────────


def _print_status_json(status_path: Path) -> None:
    try:
        status = json.loads(status_path.read_text())
        step = status.get("step", "?")
        max_steps = status.get("max_steps", "?")
        done = status.get("done", False)
        val_loss = status.get("val_loss")
        updated = status.get("updated_at", "")
        ckpt = status.get("checkpoint", "")
        pct = (
            f"{100 * step / max_steps:.1f}%"
            if isinstance(step, int) and isinstance(max_steps, int)
            else ""
        )
        state = "DONE" if done else "running"
        val_str = f"  val_loss={val_loss:.4f}" if val_loss is not None else ""
        print(f"    step {step}/{max_steps} ({pct})  [{state}]{val_str}")
        if updated:
            print(f"    last update: {updated}")
        if ckpt:
            print(f"    checkpoint:  {ckpt}")
    except Exception as exc:
        print(f"    (error reading status: {exc})")


def _print_csv_tail(csv_path: Path) -> None:
    try:
        lines = csv_path.read_text().splitlines()
        if len(lines) <= 1:
            return
        header = lines[0].split(",")
        row = dict(zip(header, lines[-1].split(","), strict=False))
        parts = []
        if step_csv := row.get("step", ""):
            parts.append(f"step={step_csv}")
        if loss_csv := row.get("loss", ""):
            parts.append(f"loss={float(loss_csv):.4f}")
        if lr_csv := row.get("lr", ""):
            parts.append(f"lr={float(lr_csv):.2e}")
        if tps := row.get("tokens_per_sec", ""):
            parts.append(f"tok/s={float(tps):.0f}")
        if parts:
            print(f"    latest CSV:  {', '.join(parts)}")
    except Exception:
        pass


def _report_run(run_dir: Path) -> None:
    status_path = run_dir / "training_status.json"
    csv_path = run_dir / "loss_curve.csv"

    if not status_path.exists() and not csv_path.exists():
        return

    print(f"  {run_dir.name}")
    if status_path.exists():
        _print_status_json(status_path)
    if csv_path.exists():
        _print_csv_tail(csv_path)


def _report_training(output_dir: Path) -> None:
    print()
    print("── Training Runs ─────────────────────────────────────────────────────")
    run_dirs = (
        sorted(
            (d for d in output_dir.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if output_dir.exists()
        else []
    )

    if not run_dirs:
        print("  No output directories found")
        return

    for run_dir in run_dirs:
        _report_run(run_dir)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Show current run status")
    parser.add_argument(
        "--data-dir", default="data/fast", help="Data directory (default: data/fast)"
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Training output directory (default: outputs)"
    )
    args = parser.parse_args()

    _report_dataprep(Path(args.data_dir))
    _report_training(Path(args.output_dir))
    print()


if __name__ == "__main__":
    main()
