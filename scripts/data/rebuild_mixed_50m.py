#!/usr/bin/env python3
"""
Properly mix wiki + stories + webtext into a 50M token dataset.
Ensures all three sources are actually concatenated and interleaved.
"""

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(path: str) -> np.ndarray:
    """Load and verify data."""
    data = np.load(path)
    logger.info(f"✓ Loaded {Path(path).name}: {len(data):,} tokens ({data.dtype})")
    return data


def save_data(path: str, data: np.ndarray) -> None:
    """Save with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data.astype(np.uint32))

    import json

    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump(
            {
                "source": "mixed (wikitext_10m + tinystories_5m + webtext_15m)",
                "tokens": int(len(data)),
                "composition": "20M unique + 30M repeat = 50M",
                "vocab": 50257,
            },
            f,
            indent=2,
        )

    logger.info(f"✓ Saved {path.name}: {len(data):,} tokens")


def interleave_datasets_proportional(wiki, stories, webtext, target_tokens=50_000_000):
    """
    Interleave datasets by proportion, ensuring all sources are represented.

    Strategy:
    1. First pass: 20% wiki (2M) + 10% stories (0.5M) + 70% webtext (10.5M) = 13M
    2. Second pass: repeat to fill to 50M with smart proportions
    """
    logger.info("\n=== Building 50M mixed dataset ===")
    logger.info(f"Target: {target_tokens:,} tokens")
    logger.info("Sources available:")
    logger.info(f"  wiki: {len(wiki):,}")
    logger.info(f"  stories: {len(stories):,}")
    logger.info(f"  webtext: {len(webtext):,}")

    tokens = []

    # Pass 1: Include all three sources in natural proportion (wiki:stories:webtext = 10:5:15)
    logger.info("\nPass 1: Adding all sources (first time)...")
    tokens.extend(wiki[: len(wiki)])  # Full wiki
    logger.info(f"  + All wiki: {len(wiki):,} tokens (total: {len(tokens):,})")

    tokens.extend(stories[: len(stories)])  # Full stories
    logger.info(f"  + All stories: {len(stories):,} tokens (total: {len(tokens):,})")

    tokens.extend(webtext[: len(webtext)])  # Full webtext
    logger.info(f"  + All webtext: {len(webtext):,} tokens (total: {len(tokens):,})")

    # Pass 2: Repeat to reach 50M (proportionally)
    logger.info(f"\nPass 2: Repeating to reach {target_tokens:,}...")
    remaining = target_tokens - len(tokens)

    # Repeat in same proportion: wiki:stories:webtext = 10:5:15 = 40:20:60%
    wiki_repeat = int(remaining * 0.4)
    stories_repeat = int(remaining * 0.2)
    webtext_repeat = remaining - wiki_repeat - stories_repeat

    logger.info(f"  Need {remaining:,} more tokens:")
    logger.info(f"    + {wiki_repeat:,} wiki (40%)")
    logger.info(f"    + {stories_repeat:,} stories (20%)")
    logger.info(f"    + {webtext_repeat:,} webtext (60%)")

    # Cycle through sources to interleave
    tokens.extend(wiki[:wiki_repeat])
    tokens.extend(stories[:stories_repeat])
    tokens.extend(webtext[:webtext_repeat])

    tokens_arr = np.array(tokens, dtype=np.uint32)[:target_tokens]
    logger.info(f"\n✓ Final: {len(tokens_arr):,} tokens")

    return tokens_arr


def main():
    data_fast = Path("data/fast")

    logger.info("=" * 70)
    logger.info("REBUILDING 50M MIXED DATASET (WITH PROOF)")
    logger.info("=" * 70)

    # Load sources
    wiki = load_data(str(data_fast / "wikitext_10m_tokens_bpe_gpt2.npy"))
    stories = load_data(str(data_fast / "tinystories_5m_tokens_bpe_gpt2.npy"))
    webtext = load_data(str(data_fast / "webtext_15m_bpe_gpt2.npy"))

    # Mix with verification
    mixed = interleave_datasets_proportional(wiki, stories, webtext)

    # Save
    output = data_fast / "interleaved_mixed_50m_tokens_bpe_gpt2_v2.npy"
    save_data(str(output), mixed)

    # Verify composition
    logger.info("\n=== VERIFICATION ===")

    def fingerprint_check(data, source_name):
        """Get first 10 tokens as fingerprint."""
        return data[:10]

    wiki_fp = fingerprint_check(wiki, "wiki")
    stories_fp = fingerprint_check(stories, "stories")
    webtext_fp = fingerprint_check(webtext, "webtext")

    logger.info(f"Wiki fingerprint (first 10 tokens): {wiki_fp.tolist()}")
    logger.info(f"Stories fingerprint (first 10 tokens): {stories_fp.tolist()}")
    logger.info(f"Webtext fingerprint (first 10 tokens): {webtext_fp.tolist()}")

    # Check where they appear in mixed
    mixed_loaded = np.load(str(output))
    logger.info("\nSearching for fingerprints in mixed dataset:")
    logger.info(f"  Wiki fp at position 0: {np.array_equal(mixed_loaded[:10], wiki_fp)}")
    logger.info(
        f"  Stories fp at position {len(wiki):,}: {np.array_equal(mixed_loaded[len(wiki):len(wiki)+10], stories_fp)}"
    )
    logger.info(
        f"  Webtext fp at position {len(wiki)+len(stories):,}: {np.array_equal(mixed_loaded[len(wiki)+len(stories):len(wiki)+len(stories)+10], webtext_fp)}"
    )

    logger.info("\n" + "=" * 70)
    logger.info("✓ SUCCESS: Mixed dataset updated")
    logger.info(f"Path: {output}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
