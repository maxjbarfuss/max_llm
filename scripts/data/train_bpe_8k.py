#!/usr/bin/env python3
"""Train a custom BPE tokenizer with 8192 vocabulary from local slow-drive text.

Sources (no download required — already on slow drive):
  TinyStories:  /mnt/d/dev/data/tinystories-gpt4-clean/train.txt
  WikiText-103: /mnt/d/dev/data/wikitext-103-raw/train_normalized.txt
  FineWeb-Edu:  /mnt/d/dev/data/fineweb-edu/tokens_utf8_1000m.npy

Uses 35M chars per source (105M total) for fast, representative tokenizer training.
No lowercase normaliser — preserves case for better language modelling.
Output: data/fast/bpe_vocab_8192.json  (convention: bpe_vocab_{vocab_size}.json)
"""

import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers

SLOW_TS = Path("/mnt/d/dev/data/tinystories-gpt4-clean/train.txt")
SLOW_WT = Path("/mnt/d/dev/data/wikitext-103-raw/train_normalized.txt")
SLOW_FW = Path("/mnt/d/dev/data/fineweb-edu/tokens_utf8_1000m.npy")

FAST_DIR = Path("data/fast")
VOCAB_SIZE = 8192
SAMPLE_CHARS = 35_000_000  # 35M chars per source → 105M total


def _read_text_sample(path: Path, n: int) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read(n)


def _decode_fw_sample(path: Path, n: int) -> str:
    """Decode the first n bytes from the FineWeb UTF-8 int32 byte array."""
    arr = np.load(path, mmap_mode="r")
    return bytes(arr[:n].astype(np.uint8)).decode("utf-8", errors="replace")


def main() -> None:
    out = FAST_DIR / "bpe_vocab_8192.json"
    if out.exists():
        print(f"✓ Already exists: {out}")
        return

    FAST_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training samples (35M chars each)...")
    ts = _read_text_sample(SLOW_TS, SAMPLE_CHARS)
    wt = _read_text_sample(SLOW_WT, SAMPLE_CHARS)
    fw = _decode_fw_sample(SLOW_FW, SAMPLE_CHARS)
    print(
        f"  TinyStories: {len(ts)/1e6:.1f}M  "
        f"WikiText: {len(wt)/1e6:.1f}M  "
        f"FineWeb: {len(fw)/1e6:.1f}M chars"
    )

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    # ByteLevel pre-tokenizer: handles any Unicode via byte encoding (GPT-2 style).
    # No lowercase normaliser — preserves case information for coherent generation.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        min_frequency=2,
        show_progress=True,
    )

    print(f"Training BPE (vocab_size={VOCAB_SIZE}) on 105M chars total...")
    tok.train_from_iterator(iter([ts, wt, fw]), trainer=trainer, length=3)
    tok.post_processor = processors.ByteLevel(trim_offsets=True)

    tok.save(str(out))
    eos_id = tok.token_to_id("<eos>")
    print(f"✓ Saved: {out}  vocab={tok.get_vocab_size()}  eos_id={eos_id}")

    meta = {
        "vocab_size": tok.get_vocab_size(),
        "eos_id": eos_id,
        "bos_id": tok.token_to_id("<bos>"),
        "unk_id": tok.token_to_id("<unk>"),
        "pad_id": tok.token_to_id("<pad>"),
        "sources": ["tinystories-gpt4-clean", "wikitext-103-raw", "fineweb-edu"],
        "sample_chars_per_source": SAMPLE_CHARS,
    }
    (FAST_DIR / "bpe_vocab_8192.meta.json").write_text(json.dumps(meta, indent=2))

    test = "Once upon a time there was a little girl who loved to read."
    enc = tok.encode(test)
    print(f"Test encode: {len(enc.ids)} tokens → '{tok.decode(enc.ids)}'")


if __name__ == "__main__":
    main()
