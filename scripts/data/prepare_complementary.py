#!/usr/bin/env python3
"""
Prepare complementary datasets for mixing with WikiText/TinyStories.

Usage:
    python scripts/data/prepare_arxiv.py --output data/fast/arxiv_abstracts_2m_bpe.npy
    python scripts/data/prepare_stack_exchange.py --output data/fast/stackexchange_qa_3m_bpe.npy
    python scripts/data/prepare_code.py --output data/fast/github_python_5m_bpe.npy
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tokenizer.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def prepare_arxiv_abstracts(output_path: str, token_limit: int = 2_000_000):
    """
    Download and tokenize ArXiv abstracts.

    Source: HuggingFace datasets (arxiv-abstracts)
    Format: Scientific abstracts, high quality, standardized structure
    Size: ~2M tokens for ~50K abstracts
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not found. Install with: pip install datasets")
        return False

    logger.info("Loading ArXiv abstracts dataset...")
    try:
        # Load from HuggingFace
        ds = load_dataset("vilsonsk/arxiv-abstracts", split="train", streaming=False)
        logger.info(f"Loaded {len(ds)} abstracts")
    except Exception as e:
        logger.error(f"Failed to load ArXiv dataset: {e}")
        logger.info(
            "Alternative: Use 'allenai/scientific_papers' or download manually from arxiv.org"
        )
        return False

    # Get tokenizer
    tokenizer = get_tokenizer("bpe", "gpt2")

    tokens = []
    for i, item in enumerate(ds):
        if i % 10000 == 0:
            logger.info(f"Processed {i}/{len(ds)} abstracts ({len(tokens)} tokens)")

        # Extract abstract text
        abstract = item.get("abstract", "")
        if not abstract:
            continue

        # Tokenize
        encoded = tokenizer.encode(abstract)
        tokens.extend(encoded)

        if len(tokens) >= token_limit:
            tokens = tokens[:token_limit]
            break

    # Save
    logger.info(f"Saving {len(tokens)} tokens to {output_path}")
    import numpy as np

    np.save(output_path, np.array(tokens, dtype=np.uint32))

    # Save metadata
    metadata_path = output_path.replace(".npy", ".meta.json")
    import json

    with open(metadata_path, "w") as f:
        json.dump(
            {
                "source": "arxiv-abstracts",
                "num_tokens": len(tokens),
                "tokenizer": "gpt2_bpe",
                "vocab_size": tokenizer.vocab_size,
            },
            f,
        )

    logger.info(f"✓ ArXiv dataset ready: {output_path}")
    return True


def prepare_stack_exchange(output_path: str, token_limit: int = 3_000_000):
    """
    Download and tokenize Stack Exchange Q&A.

    Source: HuggingFace datasets (stack-exchange-dump)
    Format: Question + Answer pairs, high-quality technical content
    Size: ~3M tokens for ~100K Q&A pairs
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not found. Install with: pip install datasets")
        return False

    logger.info("Loading Stack Exchange dataset...")
    try:
        # Load from HuggingFace
        ds = load_dataset("rishavnayan/stack-exchange-qa", split="train", streaming=False)
        logger.info(f"Loaded {len(ds)} Q&A pairs")
    except Exception as e:
        logger.error(f"Failed to load Stack Exchange dataset: {e}")
        logger.info("Alternative: Download from https://archive.org/download/stackexchange")
        return False

    # Get tokenizer
    tokenizer = get_tokenizer("bpe", "gpt2")

    tokens = []
    for i, item in enumerate(ds):
        if i % 5000 == 0:
            logger.info(f"Processed {i}/{len(ds)} Q&A pairs ({len(tokens)} tokens)")

        # Format: "Q: [question]\nA: [answer]\n"
        question = item.get("question", "")
        answer = item.get("answer", "")

        if not question or not answer:
            continue

        text = f"Q: {question}\nA: {answer}\n"
        encoded = tokenizer.encode(text)
        tokens.extend(encoded)

        if len(tokens) >= token_limit:
            tokens = tokens[:token_limit]
            break

    # Save
    logger.info(f"Saving {len(tokens)} tokens to {output_path}")
    import numpy as np

    np.save(output_path, np.array(tokens, dtype=np.uint32))

    # Save metadata
    metadata_path = output_path.replace(".npy", ".meta.json")
    import json

    with open(metadata_path, "w") as f:
        json.dump(
            {
                "source": "stack-exchange-dump",
                "num_tokens": len(tokens),
                "tokenizer": "gpt2_bpe",
                "vocab_size": tokenizer.vocab_size,
            },
            f,
        )

    logger.info(f"✓ Stack Exchange dataset ready: {output_path}")
    return True


def prepare_code_samples(output_path: str, token_limit: int = 5_000_000):
    """
    Download and tokenize code samples from GitHub.

    Source: HuggingFace datasets (github-code)
    Format: Python + JavaScript code, diverse projects
    Size: ~5M tokens for diverse code samples
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not found. Install with: pip install datasets")
        return False

    logger.info("Loading GitHub code dataset...")
    try:
        # Load from HuggingFace (Python + JavaScript)
        ds = load_dataset("codeparetum/github-code", split="train", streaming=False)
        logger.info(f"Loaded {len(ds)} code samples")
    except Exception as e:
        logger.error(f"Failed to load GitHub code dataset: {e}")
        logger.info(
            "Alternative: Download from https://huggingface.co/datasets/codeparetum/github-code"
        )
        return False

    # Get tokenizer
    tokenizer = get_tokenizer("bpe", "gpt2")

    tokens = []
    language_counts = {}

    for i, item in enumerate(ds):
        if i % 5000 == 0:
            logger.info(f"Processed {i}/{len(ds)} samples ({len(tokens)} tokens)")

        code = item.get("code", "")
        language = item.get("language", "unknown")

        # Prefer Python and JavaScript
        if language not in ["python", "javascript"]:
            continue

        language_counts[language] = language_counts.get(language, 0) + 1

        # Add code with optional language marker
        text = f"# {language}\n{code}\n"
        encoded = tokenizer.encode(text)
        tokens.extend(encoded)

        if len(tokens) >= token_limit:
            tokens = tokens[:token_limit]
            break

    # Save
    logger.info(f"Saving {len(tokens)} tokens to {output_path}")
    logger.info(f"Languages: {language_counts}")

    import numpy as np

    np.save(output_path, np.array(tokens, dtype=np.uint32))

    # Save metadata
    metadata_path = output_path.replace(".npy", ".meta.json")
    import json

    with open(metadata_path, "w") as f:
        json.dump(
            {
                "source": "github-code",
                "num_tokens": len(tokens),
                "languages": language_counts,
                "tokenizer": "gpt2_bpe",
                "vocab_size": tokenizer.vocab_size,
            },
            f,
        )

    logger.info(f"✓ Code dataset ready: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare complementary datasets")
    parser.add_argument(
        "--source",
        type=str,
        choices=["arxiv", "stack-exchange", "code", "all"],
        default="all",
        help="Which dataset to prepare",
    )
    parser.add_argument("--output-dir", type=str, default="data/fast", help="Output directory")
    parser.add_argument("--arxiv-tokens", type=int, default=2_000_000, help="ArXiv token limit")
    parser.add_argument(
        "--stackexchange-tokens", type=int, default=3_000_000, help="Stack Exchange token limit"
    )
    parser.add_argument("--code-tokens", type=int, default=5_000_000, help="Code token limit")

    args = parser.parse_args()

    # Ensure output dir exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    success = True

    if args.source in ["arxiv", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("PREPARING ARXIV ABSTRACTS")
        logger.info("=" * 60)
        output = os.path.join(
            args.output_dir, f"arxiv_abstracts_{args.arxiv_tokens//1_000_000}m_bpe_gpt2.npy"
        )
        success &= prepare_arxiv_abstracts(output, args.arxiv_tokens)

    if args.source in ["stack-exchange", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("PREPARING STACK EXCHANGE Q&A")
        logger.info("=" * 60)
        output = os.path.join(
            args.output_dir,
            f"stackexchange_qa_{args.stackexchange_tokens//1_000_000}m_bpe_gpt2.npy",
        )
        success &= prepare_stack_exchange(output, args.stackexchange_tokens)

    if args.source in ["code", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("PREPARING GITHUB CODE")
        logger.info("=" * 60)
        output = os.path.join(
            args.output_dir, f"github_code_{args.code_tokens//1_000_000}m_bpe_gpt2.npy"
        )
        success &= prepare_code_samples(output, args.code_tokens)

    if success:
        logger.info("\n" + "=" * 60)
        logger.info("✓ All datasets prepared successfully!")
        logger.info("=" * 60)
    else:
        logger.warning("\nSome datasets failed to prepare. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
