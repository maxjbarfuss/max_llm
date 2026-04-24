"""Download and preprocess NuminaMath-CoT for pretraining.

Combines 'problem' and 'solution' fields into a single 'text' field:

    Problem: {problem}

    Solution: {solution}

This format preserves the reasoning chain as continuous prose suitable for
pretraining while keeping problem context visible to the model.

Output: /mnt/d/Dev/data/numina_math_cot/train/*.parquet
        (directory of parquet shards, readable by the parquet prep reader)

Usage:
    cd <repo root>
    source .venv/bin/activate
    python -m src.data.datasets.numina_math.preprocess
"""

import logging
import os
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_HF_TOKEN_PATH = pathlib.Path(".huggingface/.hf_token")
_OUTPUT_DIR = pathlib.Path("/mnt/d/Dev/data/numina_math_cot/train")
_SHARD_SIZE = 50_000


def _setup_hf_env() -> None:
    token = _HF_TOKEN_PATH.read_text().strip()
    os.environ.setdefault("HF_HOME", "/mnt/d/Dev/data/hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/d/Dev/data/hf_cache/datasets")
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/d/Dev/data/hf_cache/hub")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def _format_row(problem: str, solution: str) -> str:
    return f"Problem: {problem.strip()}\n\nSolution: {solution.strip()}"


def _write_shard(
    shard_idx: int,
    texts: list[str],
    sources: list[str],
    output_dir: pathlib.Path,
) -> None:
    schema = pa.schema([("text", pa.string()), ("source", pa.string())])
    table = pa.table({"text": texts, "source": sources}, schema=schema)
    path = output_dir / f"numina_math_cot_{shard_idx:05d}.parquet"
    pq.write_table(table, path)
    log.info("  shard %d: %d rows → %s", shard_idx, len(texts), path.name)


def main() -> None:
    _setup_hf_env()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Import here so HF env vars are set before the library initializes caches.
    from datasets import load_dataset  # noqa: PLC0415

    log.info("Loading AI-MO/NuminaMath-CoT (train split) …")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    log.info("  %d examples loaded", len(ds))

    batch_texts: list[str] = []
    batch_sources: list[str] = []
    shard_idx = 0
    total = 0

    log.info("Writing shards to %s …", _OUTPUT_DIR)
    for row in ds:
        batch_texts.append(_format_row(row["problem"], row["solution"]))
        batch_sources.append(row["source"])
        total += 1

        if len(batch_texts) >= _SHARD_SIZE:
            _write_shard(shard_idx, batch_texts, batch_sources, _OUTPUT_DIR)
            shard_idx += 1
            batch_texts.clear()
            batch_sources.clear()

    if batch_texts:
        _write_shard(shard_idx, batch_texts, batch_sources, _OUTPUT_DIR)

    log.info("Done. %d total examples in %d shards.", total, shard_idx + 1)


if __name__ == "__main__":
    main()
