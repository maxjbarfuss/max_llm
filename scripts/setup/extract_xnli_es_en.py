#!/usr/bin/env python3
"""
Extract Spanish-English translation pairs from the XNLI dataset using the Hugging Face datasets API.
- Output: TSV file with columns: [spanish, english, label]
- Only pairs where language is 'es' and 'en' are included.
- Designed to be lightweight and not CPU lock.
"""

import os

from datasets import load_dataset


# Always set HF_TOKEN from .huggingface/.hf_token or ~/.huggingface/token if available
def set_hf_token():
    token = None
    for path in [os.path.expanduser("~/.huggingface/token"), ".huggingface/.hf_token"]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                token = f.read().strip()
            break
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("[INFO] Hugging Face token loaded for datasets API.")
    else:
        print("[WARN] No Hugging Face token found. You may hit rate limits.")


set_hf_token()

OUTPUT_PATH = "/mnt/d/dev/data/parallel/xnli/xnli_es_en.tsv"

if os.path.exists(OUTPUT_PATH):
    print(f"✅ Output already exists: {OUTPUT_PATH}")
    exit(0)

print("Loading XNLI dataset from Hugging Face...")
print("Loading XNLI dataset from Hugging Face (all_languages config)...")
ds = load_dataset("xnli", "all_languages")

# Debug: print the first example from train split
print("[DEBUG] First example from train split:")
print(ds["train"][0])


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("spanish\tenglish\tlabel\ttype\n")
    count = 0
    for split in ["train", "validation", "test"]:
        for ex in ds[split]:
            try:
                # Extract premise
                es_premise = ex["premise"].get("es")
                en_premise = ex["premise"].get("en")
                if es_premise and en_premise:
                    f.write(f"{es_premise}\t{en_premise}\t{ex['label']}\tpremise\n")
                    count += 1
                # Extract hypothesis
                es_hyp = ex["hypothesis"].get("es")
                en_hyp = ex["hypothesis"].get("en")
                if es_hyp and en_hyp:
                    f.write(f"{es_hyp}\t{en_hyp}\t{ex['label']}\thypothesis\n")
                    count += 1
            except Exception as e:
                print(f"[WARN] Exception for example: {e}")
                continue
    print(f"[INFO] Wrote {count} Spanish-English pairs to {OUTPUT_PATH}")
print(f"✅ Extracted Spanish-English pairs to {OUTPUT_PATH}")
