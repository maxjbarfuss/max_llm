#!/usr/bin/env python3
"""
Mix datasets to create a ~50M token dataset for 50m_data re-run.

Sources:
- wikitext_10m (10M tokens)
- tinystories_5m (5M tokens)
- openwebtext_15m (from prepare_code_github_webtext.py)
= 30M unique tokens

Then interleave to 50M by selective repetition (60% rule):
- Primary pass: all 30M unique
- Secondary pass: top 20M (66%) repeated = 20M
- Total: ~50M tokens

Usage:
    source .venv/bin/activate
    python scripts/data/mix_for_50m_rerun.py
"""

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(path: str, max_tokens: int = None) -> np.ndarray:
    """Load numpy array, optionally limit to max_tokens."""
    data = np.load(path)
    if max_tokens and len(data) > max_tokens:
        data = data[:max_tokens]
    logger.info(f"✓ Loaded {Path(path).name}: {len(data):,} tokens")
    return data


def save_data(path: str, data: np.ndarray) -> None:
    """Save numpy array and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data)

    import json

    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump(
            {
                "source": "mixed (wikitext+tinystories+openwebtext)",
                "tokens": int(len(data)),
                "method": "30M primary + 20M secondary (interleaved)",
            },
            f,
        )

    logger.info(f"✓ Saved {path.name}: {len(data):,} tokens ({len(data)/1e6:.1f}M)")


def main():
    logger.info("=" * 70)
    logger.info("MIXING DATASETS FOR 50M RE-RUN")
    logger.info("=" * 70)

    data_fast = Path("data/fast")

    # Check for required files
    required_files = [
        "wikitext_10m_tokens_bpe_gpt2.npy",
        "tinystories_5m_tokens_bpe_gpt2.npy",
        "webtext_15m_bpe_gpt2.npy",  # From prepare_code_github_webtext.py
    ]

    missing = [f for f in required_files if not (data_fast / f).exists()]
    if missing:
        logger.error(f"Missing files: {missing}")
        logger.error("Waiting for data_prep to complete...")
        return False

    # Load primary sources
    logger.info("\nLoading data sources:")
    wiki = load_data(str(data_fast / "wikitext_10m_tokens_bpe_gpt2.npy"))
    stories = load_data(str(data_fast / "tinystories_5m_tokens_bpe_gpt2.npy"))
    webtext = load_data(str(data_fast / "webtext_15m_bpe_gpt2.npy"))

    total_unique = len(wiki) + len(stories) + len(webtext)
    logger.info(f"Total unique tokens: {total_unique:,} ({total_unique/1e6:.1f}M)")

    # Interleave: go through each dataset in order, then repeat top sources
    logger.info("\nBuilding 50M token dataset:")
    tokens = []

    # First pass: all unique data (30M)
    logger.info("  Pass 1: Adding all unique data...")
    tokens.extend(wiki)
    tokens.extend(stories)
    tokens.extend(webtext)
    current = len(tokens)
    logger.info(f"    → {current:,} tokens")

    # Second pass: supplement to reach 50M
    # Add wiki + stories again (10M + 5M = 15M more) = 45M total
    # Then top 5M of webtext
    logger.info("  Pass 2: Adding selective repetition to reach 50M...")
    tokens.extend(wiki)  # +10M → 40M
    tokens.extend(stories)  # +5M → 45M
    tokens.extend(webtext[:5_000_000])  # +5M → 50M

    tokens_arr = np.array(tokens, dtype=np.uint32)
    tokens_arr = tokens_arr[:50_000_000]  # Trim to exactly 50M

    logger.info(f"    → {len(tokens_arr):,} tokens ({len(tokens_arr)/1e6:.1f}M)")

    # Save
    logger.info("\nSaving:")
    output = data_fast / "interleaved_mixed_50m_tokens_bpe_gpt2.npy"
    save_data(str(output), tokens_arr)

    logger.info("\n" + "=" * 70)
    logger.info("✓ SUCCESS: 50M dataset ready")
    logger.info("=" * 70)
    logger.info("\nNext: Update config and re-run 50m_data:")
    logger.info(f'  dataset_path = "{output}"')
    logger.info("  bash scripts/run_p3_50m_data.sh")

    return True


if __name__ == "__main__":
    main()
