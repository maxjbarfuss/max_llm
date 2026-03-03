"""Analyze K/V projection cosine similarity across transformer layers.

Measures pairwise cosine similarity between K and V projection weight matrices
across all layers in a trained DecoderLM checkpoint. High similarity between
adjacent layers is the prerequisite for CommonKV-style shared-KV training to
be effective (if layers naturally converge to similar K/V, forcing sharing
during training is less likely to hurt convergence).

Interpretation:
  cosine_sim > 0.90  → very high; shared KV almost certainly safe
  cosine_sim 0.70-0.90 → moderate; shared KV worth trying
  cosine_sim < 0.70  → low; layers are using K/V space differently, sharing risky

Usage:
    python scripts/analyze_kv_similarity.py <checkpoint.pt>
    python scripts/analyze_kv_similarity.py outputs/ephemeral/p3-512bpe-noshare-mixed-10k/checkpoint.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two weight matrices (flattened)."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def analyze(checkpoint_path: str | Path) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"Checkpoint not found: {path}")
        sys.exit(1)

    print(f"Loading: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt["model_state"]
    step = ckpt.get("step", "?")
    print(f"Step: {step}\n")

    # Find all layer indices
    layer_indices = sorted(
        {int(k.split(".")[1]) for k in state if k.startswith("blocks.") and "k_proj" in k}
    )
    num_layers = len(layer_indices)

    if num_layers < 2:
        print("Only 1 layer — no cross-layer similarity to compute.")
        return

    # Extract K and V weight matrices
    k_weights = [state[f"blocks.{i}.attention.k_proj.weight"] for i in layer_indices]
    v_weights = [state[f"blocks.{i}.attention.v_proj.weight"] for i in layer_indices]
    q_weights = [state[f"blocks.{i}.attention.q_proj.weight"] for i in layer_indices]

    print(f"{'Layer pair':<18} {'K sim':>8} {'V sim':>8} {'Q sim':>8}  {'Verdict'}")
    print("-" * 60)

    k_sims, v_sims, q_sims = [], [], []
    for i in range(num_layers - 1):
        ks = cosine_sim(k_weights[i], k_weights[i + 1])
        vs = cosine_sim(v_weights[i], v_weights[i + 1])
        qs = cosine_sim(q_weights[i], q_weights[i + 1])
        k_sims.append(ks)
        v_sims.append(vs)
        q_sims.append(qs)

        avg = (ks + vs) / 2
        if avg > 0.90:
            verdict = "✓ high — sharing safe"
        elif avg > 0.70:
            verdict = "~ moderate — try sharing"
        else:
            verdict = "✗ low — sharing risky"

        print(f"  L{i} → L{i+1}          {ks:>8.4f} {vs:>8.4f} {qs:>8.4f}  {verdict}")

    print()

    # All-pairs (not just adjacent)
    print(f"{'Layer pair':<18} {'K sim':>8} {'V sim':>8}  (all pairs)")
    print("-" * 45)
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            ks = cosine_sim(k_weights[i], k_weights[j])
            vs = cosine_sim(v_weights[i], v_weights[j])
            print(f"  L{i} → L{j}          {ks:>8.4f} {vs:>8.4f}")

    print()
    print("Summary:")
    print(f"  Adjacent K sim: mean={sum(k_sims)/len(k_sims):.4f}  "
          f"min={min(k_sims):.4f}  max={max(k_sims):.4f}")
    print(f"  Adjacent V sim: mean={sum(v_sims)/len(v_sims):.4f}  "
          f"min={min(v_sims):.4f}  max={max(v_sims):.4f}")
    print(f"  Adjacent Q sim: mean={sum(q_sims)/len(q_sims):.4f}  "
          f"min={min(q_sims):.4f}  max={max(q_sims):.4f}")
    print()

    avg_kv = (sum(k_sims) + sum(v_sims)) / (len(k_sims) + len(v_sims))
    avg_q = sum(q_sims) / len(q_sims)
    print(f"  Mean K/V sim: {avg_kv:.4f}   Mean Q sim: {avg_q:.4f}")

    if avg_kv > avg_q:
        delta = avg_kv - avg_q
        print(f"  K/V is {delta:.4f} more similar than Q across layers")
        print("  → CommonKV hypothesis supported: K/V converges more than Q")
    else:
        delta = avg_q - avg_kv
        print(f"  Q is {delta:.4f} more similar than K/V — CommonKV hypothesis NOT supported here")

    print()
    if avg_kv > 0.90:
        print("  VERDICT: Shared-KV training is very likely safe for this model.")
    elif avg_kv > 0.70:
        print("  VERDICT: Shared-KV training is worth experimenting with.")
    else:
        print("  VERDICT: Layers use K/V space differently — shared-KV training is risky.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <checkpoint.pt>")
        print(f"Example: python {sys.argv[0]} outputs/ephemeral/p3-512bpe-noshare-mixed-10k/checkpoint.pt")
        sys.exit(1)
    analyze(sys.argv[1])
