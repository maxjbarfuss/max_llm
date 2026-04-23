#!/usr/bin/env python3
"""
Download Spanish-English parallel corpora for multilingual training.

Sources:
  - JW300: Jehovah's Witness corpus (14M pairs, clean, ~2.2GB)
  - Europarl: EU Parliament proceedings (~2M pairs, formal)
  - UN Corpus: UN documents (~600K pairs, technical)

Total target: ~500M tokens (1.85% of 27B training set)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def download_jw300(output_dir: str) -> None:
    """Download JW300 from HuggingFace."""
    print("📥 Downloading JW300 (Jehovah's Witness corpus)...")
    print("   ~14M Spanish-English sentence pairs, highly aligned, ~2.2GB")

    # Use the existing downloader for HF datasets
    cmd = [
        sys.executable,
        "scripts/setup/download_hf_dataset.py",
        "jonathandinu/jw300",
        "--download-root",
        output_dir,
    ]

    result = subprocess.run(cmd, cwd="/home/max/dev/max_llm")
    if result.returncode != 0:
        print(f"⚠️  JW300 download failed with code {result.returncode}")
        print("   Try manual download: huggingface-cli download jonathandinu/jw300")
    else:
        print("✅ JW300 downloaded successfully")


def download_europarl(output_dir: str) -> None:
    """Download Europarl from HuggingFace or direct source."""
    print("\n📥 Downloading Europarl (EU Parliament proceedings)...")
    print("   ~2M Spanish-English pairs, formal/technical, ~500MB")

    # Try fetching from OPUS on HuggingFace first
    try:
        cmd = [
            sys.executable,
            "scripts/setup/download_hf_dataset.py",
            "Helsinki-NLP/opus-europarl",
            "--download-root",
            output_dir,
            "--allow",
            "*/es-en/*",
        ]

        result = subprocess.run(cmd, cwd="/home/max/dev/max_llm", capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Europarl downloaded successfully from OPUS HF")
            return
    except Exception as e:
        print(f"   HuggingFace source failed: {e}")

    print("⚠️  Europarl download could not be completed")
    print("   Alternative: Use OPUS multilingual corpus on HuggingFace")
    print("   Or download manually: https://object.pouta.csc.fi/OPUS-Europarl/v3/")


def download_un_corpus(output_dir: str) -> None:
    """Download UN Parallel Corpus from HuggingFace OPUS."""
    print("\n📥 Downloading UN Parallel Corpus...")
    print("   ~600K Spanish-English pairs, technical/formal, ~150MB")

    try:
        # Try OPUS UN corpus on HuggingFace
        cmd = [
            sys.executable,
            "scripts/setup/download_hf_dataset.py",
            "Helsinki-NLP/opus-un",
            "--download-root",
            output_dir,
            "--allow",
            "*/es-en/*",
        ]

        result = subprocess.run(cmd, cwd="/home/max/dev/max_llm", capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ UN Corpus downloaded successfully from OPUS HF")
            return
    except Exception as e:
        print(f"   HuggingFace source failed: {e}")

    print("⚠️  UN Corpus download could not be completed")
    print("   Alternative: Use OPUS UN corpus available on HuggingFace")


def main():
    parser = argparse.ArgumentParser(description="Download Spanish-English parallel corpora")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/d/dev/data/parallel",
        help="Output directory for corpora (default: /mnt/d/dev/data/parallel)",
    )
    parser.add_argument("--jw300", action="store_true", help="Download JW300 only")
    parser.add_argument("--europarl", action="store_true", help="Download Europarl only")
    parser.add_argument("--un", action="store_true", help="Download UN Corpus only")

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # If specific corpus is requested, only download that
    if args.jw300 or args.europarl or args.un:
        if args.jw300:
            download_jw300(output_dir)
        if args.europarl:
            download_europarl(output_dir)
        if args.un:
            download_un_corpus(output_dir)
    else:
        # Download all
        download_jw300(output_dir)
        download_europarl(output_dir)
        download_un_corpus(output_dir)

    print("\n" + "=" * 60)
    print("📊 Parallel Corpora Summary")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("\nDownloaded sources:")
    for corpus in ["jw300", "europarl", "un_corpus"]:
        corpus_dir = Path(output_dir) / corpus
        if corpus_dir.exists():
            size_mb = sum(f.stat().st_size for f in corpus_dir.rglob("*") if f.is_file()) / (
                1024**2
            )
            file_count = len(list(corpus_dir.rglob("*")))
            print(f"  ✅ {corpus:12s}: {size_mb:6.1f}MB ({file_count} files)")
        else:
            print(f"  ❌ {corpus:12s}: not downloaded")

    print("\n💡 Next: Update data prep config with these corpus paths")
    print("   Paths to add:")
    print(f"   - {output_dir}/jw300")
    print(f"   - {output_dir}/europarl")
    print(f"   - {output_dir}/un_corpus")


if __name__ == "__main__":
    main()
