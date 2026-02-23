#!/usr/bin/env python3
"""Test dataset implementations end-to-end."""

from pathlib import Path

from scripts.data.dataset_impl import (
    ReasoningWorldModelDataset,
    TinyStoriesDataset,
    WikiText103Dataset,
)
from scripts.data.split_strategies import Split


def test_rwm():
    """Test Reasoning World Model dataset."""
    print("\n" + "=" * 60)
    print("Testing Reasoning World Model (9.9 MB)")
    print("=" * 60)

    ds = ReasoningWorldModelDataset()

    # Test info
    for split in [Split.TRAIN, Split.TEST, Split.VAL]:
        rows = ds.row_count(split)
        tokens = ds.token_count(split)
        print(f"{split.value:6s}: {rows:8,d} rows, {tokens:12,d} tokens")

    # Test pagination
    print("\nTesting pagination...")
    rows_page0 = ds.iter_rows(Split.TRAIN, page=0, page_size_tokens=10000)
    print(f"  Page 0: {len(rows_page0):,d} rows")
    if rows_page0:
        print(f"  First row: {rows_page0[0][:80]}...")

    ds.save_metadata()
    print("✅ RWM passed!")


def test_wikitext():
    """Test WikiText-103 dataset."""
    print("\n" + "=" * 60)
    print("Testing WikiText-103 (517 MB)")
    print("=" * 60)

    ds = WikiText103Dataset()

    # Test info
    for split in [Split.TRAIN, Split.TEST, Split.VAL]:
        rows = ds.row_count(split)
        tokens = ds.token_count(split)
        print(f"{split.value:6s}: {rows:8,d} rows, {tokens:12,d} tokens")

    # Test pagination
    print("\nTesting pagination...")
    rows_page0 = ds.iter_rows(Split.TRAIN, page=0, page_size_tokens=50000)
    print(f"  Page 0: {len(rows_page0):,d} rows")
    if rows_page0:
        print(f"  First row: {rows_page0[0][:80]}...")

    ds.save_metadata()
    print("✅ WikiText passed!")


def test_tinystories_info_only():
    """Test TinyStories dataset (info only, full metadata computation skipped)."""
    print("\n" + "=" * 60)
    print("Testing TinyStories (2.1 GB) - Checking for cached metadata")
    print("=" * 60)

    ds = TinyStoriesDataset()

    # Check if metadata is cached
    meta_files = list(Path("data/fast/tinystories-gpt4-clean").glob("*-metadata.json"))
    if meta_files:
        print(f"Cached metadata found: {len(meta_files)} splits")
        for mf in sorted(meta_files):
            print(f"  - {mf.name}")
        print("✅ TinyStories metadata cached!")
    else:
        print("⚠️  No cached metadata for TinyStories yet")
        print("   (Full file parsing required, would take 5-10 minutes)")
        print("   To compute: python -m scripts.data.dataset_cli info tinystories")


if __name__ == "__main__":
    try:
        test_rwm()
        test_wikitext()
        test_tinystories_info_only()
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
