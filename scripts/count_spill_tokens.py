#!/usr/bin/env python3
"""
Count documents and tokens in .bin spill files using the _SpilledDocs loader.
Usage: python count_spill_tokens.py data/fast/.prep_spill/openwebtext.bin [...]
"""
import sys
from pathlib import Path

# Allow running as a script from the repo root
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data.preparation.pipeline._spill import _load_cached_spill

if len(sys.argv) < 2:
    print("Usage: python3 count_spill_tokens.py <file1.bin> [file2.bin ...]")
    sys.exit(1)

for bin_path in sys.argv[1:]:
    spill = _load_cached_spill(Path(bin_path))
    if spill is None:
        print(f"{bin_path}: [ERROR] Could not load spill/cache.")
        continue
    print(f"{bin_path}: {len(spill):,} docs, {spill.total_tokens:,} tokens")
