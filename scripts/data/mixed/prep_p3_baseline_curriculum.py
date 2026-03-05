#!/usr/bin/env python3
"""Build the Phase 3 curriculum training dataset from pre-tokenised sources.

Interleaves complete documents (stories / articles / web docs) — never
chopping mid-sentence — across three curriculum phases that progressively
shift the data distribution from simple prose toward complex text.

Phase weights (TinyStories / FineWeb-Edu / WikiText-103):
  Phase 1  (first 40% of tokens): 25% / 58% / 17%  – front-load simple stories
  Phase 2  (next  40% of tokens):  5% / 65% / 30%  – transition to mixed
  Phase 3  (last  20% of tokens):  0% / 55% / 45%  – back-load formal text

Within each phase documents from all sources are randomly shuffled together
so every batch sees a mixture of domains — no block-level loss spikes.

Prerequisites (run first):
  python scripts/data/train_bpe_8k.py
  python scripts/data/retokenize_with_bpe_8k.py

Outputs:
  data/fast/p3_coherence_train_bpe8k.npy   – interleaved training curriculum
  data/fast/p3_coherence_val_bpe8k.npy     – WikiText-103 validation tokens
  data/fast/p3_coherence_curriculum.meta.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

FAST_DIR = Path("data/fast")
SEED = 42

# Phase proportions by token count (fractions of total train tokens)
PHASE_FRACS = [0.40, 0.40, 0.20]  # phase1, phase2, phase3

# Sampling weights [TinyStories, FineWeb-Edu, WikiText] per phase
PHASE_WEIGHTS = [
    [0.25, 0.58, 0.17],  # phase 1 – heavy TinyStories
    [0.05, 0.65, 0.30],  # phase 2 – transition
    [0.00, 0.55, 0.45],  # phase 3 – heavy WikiText
]


def load_source(
    tokens_path: Path, bounds_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load a tokenised source and its document boundary array."""
    tokens = np.load(tokens_path, mmap_mode="r")
    bounds = np.load(bounds_path)  # small enough to keep in RAM
    return tokens, bounds


def sample_docs(
    tokens: np.ndarray,
    bounds: np.ndarray,
    token_budget: int,
    rng: random.Random,
) -> np.ndarray:
    """Return a concatenated array of complete documents totalling ≤ token_budget.

    Documents are sampled in a random order (shuffled index) so that consecutive
    docs in the output are not from the same region of the source file.
    Stops when token_budget is reached or all docs are exhausted (whichever
    comes first).
    """
    n_docs = len(bounds) - 1
    order = list(range(n_docs))
    rng.shuffle(order)

    chunks: list[np.ndarray] = []
    total = 0
    for idx in order:
        start, end = int(bounds[idx]), int(bounds[idx + 1])
        doc_len = end - start
        if total + doc_len > token_budget:
            # Include this doc only if it fits; otherwise skip (don't truncate)
            if not chunks:  # edge case: first doc already exceeds budget
                chunks.append(np.array(tokens[start:end], dtype=np.int32))
                total += doc_len
            break
        chunks.append(np.array(tokens[start:end], dtype=np.int32))
        total += doc_len

    return np.concatenate(chunks) if chunks else np.array([], dtype=np.int32)


def build_phase(
    ts_tokens: np.ndarray, ts_bounds: np.ndarray,
    fw_tokens: np.ndarray, fw_bounds: np.ndarray,
    wt_tokens: np.ndarray, wt_bounds: np.ndarray,
    phase_token_budget: int,
    weights: list[float],
    rng: random.Random,
    phase_num: int,
) -> np.ndarray:
    """Sample from all three sources according to weights, then interleave.

    Each source receives a token budget proportional to its weight.  We then
    concatenate the per-source document lists, shuffle the combined document
    list, and concatenate to form the phase token array.  This guarantees
    every contiguous segment (and therefore every batch) contains a mix of
    source domains.
    """
    w_ts, w_fw, w_wt = weights
    budget_ts = int(phase_token_budget * w_ts)
    budget_fw = int(phase_token_budget * w_fw)
    budget_wt = int(phase_token_budget * w_wt)

    print(
        f"  Phase {phase_num}: budget {phase_token_budget/1e6:.0f}M  "
        f"TS={budget_ts/1e6:.0f}M  FW={budget_fw/1e6:.0f}M  WT={budget_wt/1e6:.0f}M"
    )

    # Collect per-source documents as individual arrays
    def _collect_docs(
        tokens: np.ndarray, bounds: np.ndarray, budget: int
    ) -> list[np.ndarray]:
        if budget == 0:
            return []
        n_docs = len(bounds) - 1
        order = list(range(n_docs))
        rng.shuffle(order)
        docs: list[np.ndarray] = []
        total = 0
        for idx in order:
            start, end = int(bounds[idx]), int(bounds[idx + 1])
            doc = np.array(tokens[start:end], dtype=np.int32)
            docs.append(doc)
            total += len(doc)
            if total >= budget:
                break
        return docs

    ts_docs = _collect_docs(ts_tokens, ts_bounds, budget_ts)
    fw_docs = _collect_docs(fw_tokens, fw_bounds, budget_fw)
    wt_docs = _collect_docs(wt_tokens, wt_bounds, budget_wt)

    all_docs = ts_docs + fw_docs + wt_docs
    rng.shuffle(all_docs)  # interleave: random order of whole documents

    phase_tokens = np.concatenate(all_docs) if all_docs else np.array([], dtype=np.int32)
    actual_ts = sum(len(d) for d in ts_docs)
    actual_fw = sum(len(d) for d in fw_docs)
    actual_wt = sum(len(d) for d in wt_docs)
    print(
        f"    → {len(phase_tokens)/1e6:.1f}M tokens  "
        f"({len(ts_docs)} TS + {len(fw_docs)} FW + {len(wt_docs)} WT docs)  "
        f"actual TS={actual_ts/1e6:.1f}M FW={actual_fw/1e6:.1f}M WT={actual_wt/1e6:.1f}M"
    )
    return phase_tokens


def main() -> None:
    rng = random.Random(SEED)

    # ------------------------------------------------------------------
    # Check prerequisites
    # ------------------------------------------------------------------
    required = [
        FAST_DIR / "tinystories_bpe8k_train.npy",
        FAST_DIR / "tinystories_bpe8k_train_bounds.npy",
        FAST_DIR / "wikitext_bpe8k_train.npy",
        FAST_DIR / "wikitext_bpe8k_train_bounds.npy",
        FAST_DIR / "wikitext_bpe8k_val.npy",
        FAST_DIR / "fineweb_bpe8k_train.npy",
        FAST_DIR / "fineweb_bpe8k_train_bounds.npy",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("❌ Missing prerequisite files:")
        for p in missing:
            print(f"   {p}")
        print("   Run scripts/data/retokenize_with_bpe_8k.py first.")
        return

    out_train = FAST_DIR / "p3_coherence_train_bpe8k.npy"
    out_val = FAST_DIR / "p3_coherence_val_bpe8k.npy"
    if out_train.exists() and out_val.exists():
        t = np.load(out_train, mmap_mode="r")
        v = np.load(out_val, mmap_mode="r")
        print(
            f"✓ Already exists: {out_train.name} ({len(t)/1e6:.0f}M tokens), "
            f"{out_val.name} ({len(v)/1e6:.0f}M tokens)"
        )
        return

    # ------------------------------------------------------------------
    # Load sources
    # ------------------------------------------------------------------
    print("Loading tokenised sources...")
    ts_tokens, ts_bounds = load_source(
        FAST_DIR / "tinystories_bpe8k_train.npy",
        FAST_DIR / "tinystories_bpe8k_train_bounds.npy",
    )
    wt_tokens, wt_bounds = load_source(
        FAST_DIR / "wikitext_bpe8k_train.npy",
        FAST_DIR / "wikitext_bpe8k_train_bounds.npy",
    )
    fw_tokens, fw_bounds = load_source(
        FAST_DIR / "fineweb_bpe8k_train.npy",
        FAST_DIR / "fineweb_bpe8k_train_bounds.npy",
    )

    total_available = len(ts_tokens) + len(fw_tokens) + len(wt_tokens)
    print(
        f"  TinyStories: {len(ts_tokens)/1e6:.1f}M tokens  {len(ts_bounds)-1:,} docs\n"
        f"  FineWeb-Edu: {len(fw_tokens)/1e6:.1f}M tokens  {len(fw_bounds)-1:,} docs\n"
        f"  WikiText-103: {len(wt_tokens)/1e6:.1f}M tokens  {len(wt_bounds)-1:,} docs\n"
        f"  Total available: {total_available/1e6:.1f}M tokens"
    )

    # ------------------------------------------------------------------
    # Build three curriculum phases
    # ------------------------------------------------------------------
    # Use WikiText as the scarcest-quality resource to size the total budget:
    # WikiText (30% target share) defines total = WT_available / 0.30.
    # Cap at actual available tokens to avoid heavy oversampling.
    wt_share_target = sum(w[2] for w in PHASE_WEIGHTS) / len(PHASE_WEIGHTS)  # avg WT weight
    total_budget = min(int(len(wt_tokens) / wt_share_target), total_available)
    print(f"\nCurriculum budget: {total_budget/1e6:.0f}M tokens (limited by WikiText availability)")

    phases: list[np.ndarray] = []
    print("\nBuilding curriculum phases...")
    for i, (frac, weights) in enumerate(zip(PHASE_FRACS, PHASE_WEIGHTS), 1):
        phase_budget = int(total_budget * frac)
        phase_arr = build_phase(
            ts_tokens, ts_bounds,
            fw_tokens, fw_bounds,
            wt_tokens, wt_bounds,
            phase_token_budget=phase_budget,
            weights=weights,
            rng=rng,
            phase_num=i,
        )
        phases.append(phase_arr)

    # ------------------------------------------------------------------
    # Concatenate phases and save
    # ------------------------------------------------------------------
    print("\nConcatenating phases and saving...")
    train_tokens = np.concatenate(phases)
    np.save(out_train, train_tokens)
    print(f"  ✓ {out_train.name}: {len(train_tokens)/1e6:.1f}M tokens")

    # Validation: use WikiText-103 val set (clean, held-out domain)
    import shutil
    shutil.copy(FAST_DIR / "wikitext_bpe8k_val.npy", out_val)
    val_tokens = np.load(out_val, mmap_mode="r")
    print(f"  ✓ {out_val.name}: {len(val_tokens)/1e6:.1f}M tokens (WikiText-103 val)")

    # ------------------------------------------------------------------
    # Metadata + step estimate
    # ------------------------------------------------------------------
    # 2-GPU DDP: batch=64/GPU × 2 GPUs × grad_accum=2 × seq_len=1024
    tokens_per_step = 64 * 2 * 2 * 1024
    steps_per_pass = int(len(train_tokens) / tokens_per_step)
    steps_2pass = steps_per_pass * 2

    meta = {
        "total_train_tokens": int(len(train_tokens)),
        "total_val_tokens": int(len(val_tokens)),
        "total_train_tokens_M": round(len(train_tokens) / 1e6, 1),
        "phase_fracs": PHASE_FRACS,
        "phase_weights_ts_fw_wt": PHASE_WEIGHTS,
        "seed": SEED,
        "estimated_steps_per_pass": steps_per_pass,
        "estimated_steps_2pass": steps_2pass,
        "tokens_per_step_2gpu": tokens_per_step,
    }
    (FAST_DIR / "p3_coherence_curriculum.meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  Steps/pass (2-GPU, batch=64, accum=2, seq=1024): {steps_per_pass:,}")
    print(f"  Steps for 2 passes: {steps_2pass:,}  →  set max_steps ≈ {steps_2pass}")
    print(f"\n  Metadata: {FAST_DIR / 'p3_coherence_curriculum.meta.json'}")
    print("\n✓ Curriculum dataset ready.")


if __name__ == "__main__":
    main()
