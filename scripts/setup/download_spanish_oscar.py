#!/usr/bin/env python3
"""
Download Spanish language corpus from OSCAR-2301.

Target: ~15% of 27B tokens = ~4.05B tokens

Source:
  - OSCAR-2301 Spanish subset (~4B+ tokens, web text)
"""

import os
import subprocess
import sys


def main():
    print("=" * 70)
    print("📥 Spanish OSCAR-2301 Corpus Download")
    print("=" * 70)

    spanish_oscar_dir = "/mnt/d/dev/data/spanish/oscar"
    os.makedirs(spanish_oscar_dir, exist_ok=True)

    print("\n📥 Downloading OSCAR-2301 Spanish subset...")
    print("   ~4B+ tokens of Spanish web text")
    print("   Source: oscar-corpus/OSCAR-2301")
    print("   Filter: Spanish language ('es' subset)\n")

    cmd = [
        sys.executable,
        "scripts/setup/download_hf_dataset.py",
        "oscar-corpus/OSCAR-2301",
        "--download-root",
        spanish_oscar_dir,
        "--allow",
        "*es*",  # Match Spanish language subset
        "--limit",
        "100",  # Limit to 100 files to keep download manageable
    ]

    result = subprocess.run(cmd, cwd="/home/max/dev/max_llm")

    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("✅ Spanish OSCAR-2301 download completed")
        print("\nNext steps:")
        print("1. Update data prep config with Spanish corpus paths")
        print("2. Run data preparation to tokenize Spanish content")
        print("3. Create final training config with 65/20/15 split")
        print("\nNew dataset mix (27B tokens):")
        print("  - English broad (OpenWebText, FineWeb, Cosmopedia, Wiki, WikiText): ~65% (17.55B)")
        print("  - Specialist (code/math/logic): ~5% (1.35B)")
        print("  - Spanish web text + stories: ~15% (4.05B)")
        print("  - English-Spanish translation: ~15% (4.05B)")
    else:
        print("⚠️  Download had issues, may need manual intervention")
    print("=" * 70)


if __name__ == "__main__":
    main()
