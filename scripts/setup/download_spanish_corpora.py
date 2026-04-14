#!/usr/bin/env python3
"""
Download Spanish language corpora for multilingual training.

Target: ~15% of 27B tokens = ~4.05B tokens

Sources:
  - Spanish Wikipedia (~1.2B tokens)
  - Spanish CommonCrawl (mC4) (~2B tokens)
  - Spanish Children's Stories (~0.85B tokens)
"""

import os
import subprocess
import sys
from pathlib import Path


def run_download(dataset_id: str, output_dir: str, allow_patterns: list[str] = None) -> bool:
    """Run HF dataset downloader."""
    cmd = [
        sys.executable,
        "scripts/setup/download_hf_dataset.py",
        dataset_id,
        "--download-root",
        output_dir,
    ]

    if allow_patterns:
        for pattern in allow_patterns:
            cmd.extend(["--allow", pattern])

    result = subprocess.run(cmd, cwd="/home/max/dev/max_llm")
    return result.returncode == 0


def main():
    spanish_dir = "/mnt/d/dev/data/spanish"
    os.makedirs(spanish_dir, exist_ok=True)

    print("=" * 70)
    print("📥 Spanish Language Corpus Download")
    print("=" * 70)

    # 1. Spanish Wikipedia
    print("\n1️⃣ Spanish Wikipedia (~1.2B tokens)...")
    if run_download(
        "wikimedia/wikipedia",
        f"{spanish_dir}/wikipedia",
        allow_patterns=["*/es/*", "*/es_*.parquet"],
    ):
        print("✅ Spanish Wikipedia downloaded")
    else:
        print("⚠️  Spanish Wikipedia download had issues")

    # 2. Spanish mC4 (CommonCrawl)
    print("\n2️⃣ Spanish CommonCrawl (mC4) (~2B tokens)...")
    if run_download("allenai/mc4", f"{spanish_dir}/mc4", allow_patterns=["es/*"]):
        print("✅ Spanish mC4 downloaded")
    else:
        print("⚠️  Spanish mC4 download had issues")

    # 3. Spanish children's literature datasets
    print("\n3️⃣ Spanish Children's Stories (~0.85B tokens)...")

    # Try Spanish fairy tales dataset
    print("   Attempting Spanish fairy tales/literature...")
    if run_download("jhu-clsp/spanish-children", f"{spanish_dir}/stories"):
        print("✅ Spanish children's stories downloaded")
    else:
        print("   (Spanish children's stories dataset not available on HF)")

    # Alternative: Try OpenWebText Spanish equivalent if available
    print("\n4️⃣ Looking for Spanish OpenWebText equivalent...")
    try:
        if run_download(
            "togethercomputer/RedPajama-Data-1T",
            f"{spanish_dir}/redpajama",
            allow_patterns=["*spanish*", "*es*"],
        ):
            print("✅ RedPajama Spanish subset downloaded")
    except:
        print("   (RedPajama Spanish may have restrictions)")

    print("\n" + "=" * 70)
    print("📊 Spanish Corpus Summary")
    print("=" * 70)
    print(f"\nOutput directory: {spanish_dir}")

    # Calculate sizes
    for corpus_type in ["wikipedia", "mc4", "stories", "redpajama"]:
        corpus_dir = Path(spanish_dir) / corpus_type
        if corpus_dir.exists():
            size_bytes = sum(f.stat().st_size for f in corpus_dir.rglob("*") if f.is_file())
            size_mb = size_bytes / (1024**2)
            size_gb = size_bytes / (1024**3)
            files = len(list(corpus_dir.rglob("*")))
            print(f"  {corpus_type:12s}: {size_gb:6.2f}GB ({files} files)")

    print("\n💡 Configuration will be updated to include Spanish paths:")
    print(f"   - {spanish_dir}/wikipedia")
    print(f"   - {spanish_dir}/mc4")
    print(f"   - {spanish_dir}/stories")


if __name__ == "__main__":
    main()
