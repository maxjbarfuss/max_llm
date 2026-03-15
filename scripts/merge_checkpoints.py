#!/usr/bin/env python3
"""
Checkpoint merge (model soup) utility.

Computes a weighted average of model weights across multiple checkpoints.
Optimizer state is taken from the primary (highest-weight) checkpoint.
Useful for SWA-style merging across runs that share the same architecture.

Usage examples
--------------
# Equal-weight merge of two checkpoints:
python scripts/merge_checkpoints.py \
    --checkpoints path/a.pt path/b.pt \
    --output outputs/ephemeral/merged.pt

# Explicit per-checkpoint weights (need not sum to 1 — auto-normalised):
python scripts/merge_checkpoints.py \
    --checkpoints path/a.pt path/b.pt path/c.pt \
    --weights 0.6 0.3 0.1 \
    --output outputs/ephemeral/merged.pt

# Quality-softmax weights from val losses (lower = higher weight):
python scripts/merge_checkpoints.py \
    --checkpoints path/a.pt path/b.pt \
    --val-losses 3.58 3.75 \
    --softmax-temp 0.05 \
    --output outputs/ephemeral/merged.pt

# Force a specific checkpoint to donate the optimizer state:
python scripts/merge_checkpoints.py \
    --checkpoints path/a.pt path/b.pt \
    --weights 0.8 0.2 \
    --optimizer-source 0 \
    --output outputs/ephemeral/merged.pt
"""

import argparse
import json
import math
from pathlib import Path

import torch


def softmax_weights(val_losses: list[float], temperature: float) -> list[float]:
    """Lower val_loss → higher weight via softmax(-loss/T)."""
    logits = [-v / temperature for v in val_losses]
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return [w / total for w in weights]


def load_checkpoint(path: str) -> dict:
    ck = torch.load(path, map_location="cpu")
    if not isinstance(ck, dict) or "model_state" not in ck:
        raise ValueError(f"{path} does not look like a max_llm checkpoint (missing 'model_state').")
    return ck


def weighted_average_state_dicts(
    state_dicts: list[dict],
    weights: list[float],
) -> dict:
    assert len(state_dicts) == len(weights)
    merged = {}
    for key in state_dicts[0]:
        tensors = [sd[key] for sd in state_dicts]
        if not tensors[0].is_floating_point():
            # Integer tensors (e.g. token type ids) — take from primary (highest weight)
            primary = weights.index(max(weights))
            merged[key] = tensors[primary].clone()
        else:
            merged[key] = sum(w * t for w, t in zip(weights, tensors))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge (soup) multiple checkpoints by weighted averaging."
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True, metavar="PATH", help="Checkpoint paths to merge."
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Per-checkpoint weights (auto-normalised). Defaults to equal weights.",
    )
    parser.add_argument(
        "--val-losses",
        nargs="+",
        type=float,
        default=None,
        help="Validation losses for quality-softmax weighting (requires --softmax-temp).",
    )
    parser.add_argument(
        "--softmax-temp",
        type=float,
        default=0.05,
        help="Temperature for softmax over val losses (default: 0.05).",
    )
    parser.add_argument(
        "--optimizer-source",
        type=int,
        default=None,
        help="Index of checkpoint whose optimizer state to use. "
        "Defaults to the checkpoint with highest weight.",
    )
    parser.add_argument(
        "--output", required=True, metavar="PATH", help="Output path for merged checkpoint."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print weights and exit without writing anything."
    )
    args = parser.parse_args()

    n = len(args.checkpoints)

    # Resolve weights
    if args.val_losses is not None:
        if len(args.val_losses) != n:
            parser.error("--val-losses must have same length as --checkpoints.")
        weights = softmax_weights(args.val_losses, args.softmax_temp)
        weight_source = f"softmax over val losses (T={args.softmax_temp})"
    elif args.weights is not None:
        if len(args.weights) != n:
            parser.error("--weights must have same length as --checkpoints.")
        weights = normalize(args.weights)
        weight_source = "manual (normalised)"
    else:
        weights = normalize([1.0] * n)
        weight_source = "equal"

    optimizer_idx = (
        args.optimizer_source if args.optimizer_source is not None else weights.index(max(weights))
    )

    print(f"Merge plan ({weight_source}):")
    for path, w in zip(args.checkpoints, weights):
        flag = "  [optimizer source]" if args.checkpoints.index(path) == optimizer_idx else ""
        print(f"  {w:.4f}  {path}{flag}")
    print(f"Output: {args.output}")

    if args.dry_run:
        print("Dry run — exiting.")
        return

    # Validate all checkpoints have compatible model keys
    print("\nLoading checkpoints...")
    checkpoints = [load_checkpoint(p) for p in args.checkpoints]
    ref_keys = set(checkpoints[0]["model_state"].keys())
    for i, ck in enumerate(checkpoints[1:], 1):
        keys = set(ck["model_state"].keys())
        if keys != ref_keys:
            extra = keys - ref_keys
            missing = ref_keys - keys
            raise ValueError(
                f"Checkpoint {i} ({args.checkpoints[i]}) has incompatible model_state keys.\n"
                f"  Extra: {extra}\n  Missing: {missing}"
            )

    print("Averaging model weights...")
    merged_model_state = weighted_average_state_dicts(
        [ck["model_state"] for ck in checkpoints],
        weights,
    )

    primary_ck = checkpoints[optimizer_idx]
    merged_ck = {
        "model_state": merged_model_state,
        "optimizer_state": primary_ck.get("optimizer_state"),
        "step": primary_ck.get("step"),
        "config_versions": primary_ck.get("config_versions"),
        "_merge_manifest": {
            "checkpoints": list(args.checkpoints),
            "weights": weights,
            "weight_source": weight_source,
            "optimizer_source_idx": optimizer_idx,
            "optimizer_source_path": args.checkpoints[optimizer_idx],
            "primary_step": primary_ck.get("step"),
            "val_losses": args.val_losses,
            "softmax_temp": args.softmax_temp if args.val_losses else None,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_ck, out_path)
    print(f"\nSaved merged checkpoint → {out_path}")

    # Also write human-readable manifest alongside
    manifest_path = out_path.with_suffix(".merge_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(merged_ck["_merge_manifest"], f, indent=2)
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
