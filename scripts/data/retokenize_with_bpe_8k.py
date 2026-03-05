#!/usr/bin/env python3
"""Retokenize all three sources with the 8192-vocab BPE tokenizer.

Reads from slow drive, writes to data/fast/.  Preserves document boundaries
so the curriculum builder can interleave whole stories/articles (never
chopping mid-sentence).

Outputs per source (train split):
  data/fast/tinystories_bpe8k_train.npy        – flat int32 token array
  data/fast/tinystories_bpe8k_train_bounds.npy – int64 doc-start offsets
  data/fast/wikitext_bpe8k_train.npy
  data/fast/wikitext_bpe8k_train_bounds.npy
  data/fast/wikitext_bpe8k_val.npy             – WikiText-103 validation set
  data/fast/fineweb_bpe8k_train.npy
  data/fast/fineweb_bpe8k_train_bounds.npy

Token budget (single pass of each source):
  TinyStories  ~45M tokens  (10% of curriculum target; file has 2.1B chars)
  WikiText-103 ~110M tokens (whole normalised train set, 496M chars)
  FineWeb-Edu  ~211M tokens (whole 1B-char slow-drive cache)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import numpy as np
from tokenizers import Tokenizer  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SLOW_TS = Path("/mnt/d/dev/data/tinystories-gpt4-clean/train.txt")
SLOW_WT_TRAIN = Path("/mnt/d/dev/data/wikitext-103-raw/train_normalized.txt")
SLOW_WT_VAL = Path("/mnt/d/dev/data/wikitext-103-raw/validation.txt")
SLOW_FW = Path("/mnt/d/dev/data/fineweb-edu/tokens_utf8_1000m.npy")

FAST_DIR = Path("data/fast")
TOKENIZER_PATH = FAST_DIR / "bpe_vocab_8192.json"

# TinyStories token budget: 45M keeps it at ≈10% of the curriculum
TS_TOKEN_BUDGET = 45_000_000
# FineWeb chunk size for streaming decode (10M bytes at a time)
FW_CHUNK_BYTES = 10_000_000


# ---------------------------------------------------------------------------
# Document iterators
# ---------------------------------------------------------------------------

def ts_doc_iter(path: Path, token_budget: int | None = None) -> Generator[str, None, None]:
    """Yield one TinyStories story per line.  Stop at token_budget if set.

    The GPT-4-clean dataset stores one complete story per line (no blank-line
    separators).  Each line averages ~180 chars ≈ 1 complete story.
    """
    approx_chars = (token_budget * 4) if token_budget else None  # 4 chars/token rough estimate
    chars_read = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            story = line.rstrip("\n").strip()
            if story:
                yield story
                chars_read += len(story)
                if approx_chars and chars_read >= approx_chars:
                    break


def wt_doc_iter(path: Path) -> Generator[str, None, None]:
    """Yield one WikiText article per call.

    Articles start with a level-1 heading of the form ' = Title = ' and are
    separated by blank lines.  We accumulate lines until we see a new level-1
    heading (or EOF), then yield the accumulated article.
    """
    current: list[str] = []

    def _emit(lines: list[str]) -> str | None:
        text = "\n".join(lines).strip()
        return text if len(text) > 100 else None  # skip stubs

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            # Level-1 heading: exactly two ' = ' markers on the line
            is_article_start = (
                line.startswith(" = ") and line.endswith(" = ") and line.count(" = ") == 2
            )
            if is_article_start and current:
                doc = _emit(current)
                if doc:
                    yield doc
                current = [line]
            else:
                current.append(line)

    if current:
        doc = _emit(current)
        if doc:
            yield doc


def fw_doc_iter(path: Path) -> Generator[str, None, None]:
    """Yield one FineWeb document per call by streaming the byte array.

    The prep script stored documents separated by b'\\n\\n'.  We decode in
    10M-byte chunks with carryover so we never split a document mid-byte.
    """
    arr = np.load(path, mmap_mode="r")
    total = len(arr)
    carryover = ""

    for start in range(0, total, FW_CHUNK_BYTES):
        end = min(start + FW_CHUNK_BYTES, total)
        chunk = bytes(arr[start:end].astype(np.uint8)).decode("utf-8", errors="replace")
        text = carryover + chunk
        parts = text.split("\n\n")

        if end < total:
            carryover = parts[-1]
            complete = parts[:-1]
        else:
            carryover = ""
            complete = parts

        for doc in complete:
            doc = doc.strip()
            if len(doc) > 20:
                yield doc


# ---------------------------------------------------------------------------
# Core tokenisation helper
# ---------------------------------------------------------------------------

def tokenize_source(
    doc_iter: Generator[str, None, None],
    tokenizer: Tokenizer,
    eos_id: int,
    token_budget: int | None = None,
    source_name: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenise documents, appending EOS after each one.

    Returns:
        tokens: int32 flat token array
        bounds: int64 array of document-start offsets (len = n_docs + 1,
                last entry == len(tokens) for convenient slicing)
    """
    chunks: list[np.ndarray] = []
    bounds: list[int] = [0]
    total = 0
    n_docs = 0

    for doc in doc_iter:
        ids = tokenizer.encode(doc).ids
        ids.append(eos_id)
        arr = np.array(ids, dtype=np.int32)
        chunks.append(arr)
        total += len(arr)
        bounds.append(total)
        n_docs += 1

        if n_docs % 50_000 == 0:
            print(f"  [{source_name}] {n_docs:,} docs  {total/1e6:.1f}M tokens", flush=True)

        if token_budget and total >= token_budget:
            break

    tokens = np.concatenate(chunks) if chunks else np.array([], dtype=np.int32)
    return tokens, np.array(bounds, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not TOKENIZER_PATH.exists():
        print(f"❌ Tokenizer not found: {TOKENIZER_PATH}")
        print("   Run scripts/data/train_bpe_8k.py first.")
        return

    FAST_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos_id: int = tokenizer.token_to_id("<eos>")
    print(f"Loaded tokenizer: vocab={tokenizer.get_vocab_size()}  eos_id={eos_id}\n")

    results: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # TinyStories — budget 45M tokens (≈10% of curriculum)
    # ------------------------------------------------------------------
    ts_out = FAST_DIR / "tinystories_bpe8k_train.npy"
    ts_bounds_out = FAST_DIR / "tinystories_bpe8k_train_bounds.npy"
    if ts_out.exists() and ts_bounds_out.exists():
        print(f"✓ TinyStories already done: {ts_out.name}")
        t = np.load(ts_out, mmap_mode="r")
        b = np.load(ts_bounds_out, mmap_mode="r")
        results["tinystories"] = {"tokens": int(len(t)), "docs": int(len(b)) - 1}
    else:
        print("Tokenising TinyStories (budget 45M tokens)...")
        tokens, bounds = tokenize_source(
            ts_doc_iter(SLOW_TS, token_budget=TS_TOKEN_BUDGET),
            tokenizer, eos_id,
            token_budget=TS_TOKEN_BUDGET,
            source_name="TinyStories",
        )
        np.save(ts_out, tokens)
        np.save(ts_bounds_out, bounds)
        results["tinystories"] = {"tokens": int(len(tokens)), "docs": int(len(bounds)) - 1}
        print(f"  ✓ {len(tokens)/1e6:.1f}M tokens  {len(bounds)-1:,} docs → {ts_out.name}")

    # ------------------------------------------------------------------
    # WikiText-103 train
    # ------------------------------------------------------------------
    wt_out = FAST_DIR / "wikitext_bpe8k_train.npy"
    wt_bounds_out = FAST_DIR / "wikitext_bpe8k_train_bounds.npy"
    if wt_out.exists() and wt_bounds_out.exists():
        print(f"✓ WikiText train already done: {wt_out.name}")
        t = np.load(wt_out, mmap_mode="r")
        b = np.load(wt_bounds_out, mmap_mode="r")
        results["wikitext_train"] = {"tokens": int(len(t)), "docs": int(len(b)) - 1}
    else:
        print("Tokenising WikiText-103 train...")
        tokens, bounds = tokenize_source(
            wt_doc_iter(SLOW_WT_TRAIN),
            tokenizer, eos_id,
            source_name="WikiText-train",
        )
        np.save(wt_out, tokens)
        np.save(wt_bounds_out, bounds)
        results["wikitext_train"] = {"tokens": int(len(tokens)), "docs": int(len(bounds)) - 1}
        print(f"  ✓ {len(tokens)/1e6:.1f}M tokens  {len(bounds)-1:,} docs → {wt_out.name}")

    # ------------------------------------------------------------------
    # WikiText-103 validation
    # ------------------------------------------------------------------
    wt_val_out = FAST_DIR / "wikitext_bpe8k_val.npy"
    if wt_val_out.exists():
        print(f"✓ WikiText val already done: {wt_val_out.name}")
        t = np.load(wt_val_out, mmap_mode="r")
        results["wikitext_val"] = {"tokens": int(len(t)), "docs": -1}
    else:
        print("Tokenising WikiText-103 validation...")
        tokens, _ = tokenize_source(
            wt_doc_iter(SLOW_WT_VAL),
            tokenizer, eos_id,
            source_name="WikiText-val",
        )
        np.save(wt_val_out, tokens)
        results["wikitext_val"] = {"tokens": int(len(tokens)), "docs": -1}
        print(f"  ✓ {len(tokens)/1e6:.1f}M tokens → {wt_val_out.name}")

    # ------------------------------------------------------------------
    # FineWeb-Edu — stream all 1B chars from slow cache
    # ------------------------------------------------------------------
    fw_out = FAST_DIR / "fineweb_bpe8k_train.npy"
    fw_bounds_out = FAST_DIR / "fineweb_bpe8k_train_bounds.npy"
    if fw_out.exists() and fw_bounds_out.exists():
        print(f"✓ FineWeb already done: {fw_out.name}")
        t = np.load(fw_out, mmap_mode="r")
        b = np.load(fw_bounds_out, mmap_mode="r")
        results["fineweb"] = {"tokens": int(len(t)), "docs": int(len(b)) - 1}
    else:
        print("Tokenising FineWeb-Edu (streaming 1B-char cache)...")
        tokens, bounds = tokenize_source(
            fw_doc_iter(SLOW_FW),
            tokenizer, eos_id,
            source_name="FineWeb",
        )
        np.save(fw_out, tokens)
        np.save(fw_bounds_out, bounds)
        results["fineweb"] = {"tokens": int(len(tokens)), "docs": int(len(bounds)) - 1}
        print(f"  ✓ {len(tokens)/1e6:.1f}M tokens  {len(bounds)-1:,} docs → {fw_out.name}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Retokenisation complete")
    print("=" * 60)
    total_train = 0
    for name, info in results.items():
        docs_str = f"  {info['docs']:,} docs" if info["docs"] >= 0 else ""
        print(f"  {name:20s}: {info['tokens']/1e6:7.1f}M tokens{docs_str}")
        if "val" not in name:
            total_train += info["tokens"]
    print(f"  {'TRAIN TOTAL':20s}: {total_train/1e6:7.1f}M tokens")

    meta = {k: {**v, "tokens_M": round(v["tokens"] / 1e6, 1)} for k, v in results.items()}
    (FAST_DIR / "bpe8k_retokenize.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  Metadata: {FAST_DIR / 'bpe8k_retokenize.meta.json'}")


if __name__ == "__main__":
    main()
