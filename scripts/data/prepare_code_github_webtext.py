#!/usr/bin/env python3
"""
Prepare GitHub Code + Web Text datasets for training.

Uses proven public datasets:
- GitHub code: aistudio-projects/code_python_github
- Web text: openwebtext (fallback: wikitext)

Pattern: Download from HuggingFace → Tokenize with BPE → Extract to data/fast/

Usage:
    source .venv/bin/activate
    python scripts/data/prepare_code_github_webtext.py --source all
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_token_size(size_str: str) -> int:
    """Parse token size string (e.g., '20m', '500k')."""
    size_str = size_str.lower().strip()
    multipliers = {"k": 1_000, "m": 1_000_000}
    for suffix, mult in multipliers.items():
        if size_str.endswith(suffix):
            try:
                return int(float(size_str[:-1]) * mult)
            except ValueError:
                pass
    return int(size_str)


def prepare_github_code(output_path: str, token_limit: int = 20_000_000) -> bool:
    """Download and tokenize GitHub Python code."""
    logger.info("=" * 70)
    logger.info("PREPARING GITHUB PYTHON CODE")
    logger.info("=" * 70)

    try:
        import numpy as np
        from datasets import load_dataset
    except ImportError:
        logger.error("Missing: pip install datasets numpy")
        return False

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.tokenizer import TokenizerFactory

    try:
        tokenizer = TokenizerFactory.create("bpe", encoding="gpt2")
        logger.info(f"✓ Tokenizer: BPE (vocab={tokenizer.vocab_size})")
    except Exception as e:
        logger.error(f"Tokenizer load failed: {e}")
        return False

    logger.info("Downloading: aistudio-projects/code_python_github")
    try:
        ds = load_dataset(
            "aistudio-projects/code_python_github",
            split="train",
            streaming=False,
            trust_remote_code=False,
        )
        logger.info(f"✓ Loaded: {len(ds):,} code samples")
    except Exception as e:
        logger.error(f"Dataset load failed: {e}")
        return False

    tokens = []
    logger.info(f"Tokenizing (target: {token_limit:,} tokens)...")

    for i, item in enumerate(ds):
        if i % 5000 == 0 and i > 0:
            logger.info(
                f"  Step {i:,} → {len(tokens):,} tokens [{100*min(1, len(tokens)/token_limit):.0f}%]"
            )

        if len(tokens) >= token_limit:
            break

        code = item.get("code", "") or item.get("text", "")
        if not code:
            continue

        try:
            tokens.extend(tokenizer.encode(code))
        except:
            pass

    tokens = tokens[:token_limit]
    if not tokens:
        logger.error("No tokens generated")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array(tokens, dtype=np.uint32))

    import json

    with open(output_path.with_suffix(".meta.json"), "w") as f:
        json.dump(
            {
                "source": "aistudio/code_python_github",
                "tokens": len(tokens),
                "vocab": tokenizer.vocab_size,
            },
            f,
        )

    logger.info(f"✓ GitHub Code: {output_path} ({len(tokens):,} tokens)\n")
    return True


def prepare_webtext(output_path: str, token_limit: int = 15_000_000) -> bool:
    """Download and tokenize web text (OpenWebText)."""
    logger.info("=" * 70)
    logger.info("PREPARING WEB TEXT")
    logger.info("=" * 70)

    try:
        import numpy as np
        from datasets import load_dataset
    except ImportError:
        logger.error("Missing: pip install datasets numpy")
        return False

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.tokenizer import TokenizerFactory

    try:
        tokenizer = TokenizerFactory.create("bpe", encoding="gpt2")
        logger.info(f"✓ Tokenizer: BPE (vocab={tokenizer.vocab_size})")
    except Exception as e:
        logger.error(f"Tokenizer load failed: {e}")
        return False

    # Try openwebtext first, fallback to wikitext
    ds = None
    source_name = None

    for source, config in [("openwebtext", None), ("wikitext", "wikitext-103-v1")]:
        logger.info(f"Attempting to download: {source}")
        try:
            if config:
                ds = load_dataset(source, config, split="train")
            else:
                ds = load_dataset(source, split="train")
            source_name = source
            logger.info(f"✓ Loaded: {len(ds):,} documents")
            break
        except Exception as e:
            logger.warning(f"  Failed: {e}")

    if ds is None:
        logger.error("All text sources failed")
        return False

    tokens = []
    logger.info(f"Tokenizing (target: {token_limit:,} tokens)...")

    for i, item in enumerate(ds):
        if i % 5000 == 0 and i > 0:
            logger.info(
                f"  Step {i:,} → {len(tokens):,} tokens [{100*min(1, len(tokens)/token_limit):.0f}%]"
            )

        if len(tokens) >= token_limit:
            break

        text = item.get("text", "") or item.get("article", "")
        if not text:
            continue

        try:
            tokens.extend(tokenizer.encode(text))
        except:
            pass

    tokens = tokens[:token_limit]
    if not tokens:
        logger.error("No tokens generated")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array(tokens, dtype=np.uint32))

    import json

    with open(output_path.with_suffix(".meta.json"), "w") as f:
        json.dump({"source": source_name, "tokens": len(tokens), "vocab": tokenizer.vocab_size}, f)

    logger.info(f"✓ Web Text: {output_path} ({len(tokens):,} tokens)\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare GitHub Code + Web Text")
    parser.add_argument("--source", choices=["github", "webtext", "all"], default="all")
    parser.add_argument("--tokens", default="github:20m,webtext:15m")
    parser.add_argument("--output-dir", default="data/fast")
    args = parser.parse_args()

    # Parse tokens
    limits = {"github": 20_000_000, "webtext": 15_000_000}
    if ":" in args.tokens:
        for pair in args.tokens.split(","):
            k, v = pair.split(":")
            limits[k.strip()] = parse_token_size(v.strip())

    logger.info("\nData Preparation: GitHub + WebText")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"GitHub: {limits['github']:,} tokens")
    logger.info(f"WebText: {limits['webtext']:,} tokens\n")

    success = True

    if args.source in ["github", "all"]:
        out = os.path.join(
            args.output_dir, f"github_code_{limits['github']//1_000_000}m_bpe_gpt2.npy"
        )
        success &= prepare_github_code(out, limits["github"])

    if args.source in ["webtext", "all"]:
        out = os.path.join(args.output_dir, f"webtext_{limits['webtext']//1_000_000}m_bpe_gpt2.npy")
        success &= prepare_webtext(out, limits["webtext"])

    if success:
        logger.info("=" * 70)
        logger.info("✓ SUCCESS: Data files ready in data/fast/")
        logger.info("=" * 70)
        logger.info("\nNext: Mix with existing WikiText + TinyStories")
        logger.info("  python scripts/data/mix_interleaved_pages.py ...")
    else:
        logger.error("❌ Failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
