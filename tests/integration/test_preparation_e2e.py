"""Integration tests for data preparation pipeline."""

from __future__ import annotations

import json

import numpy as np

from src.data.preparation.config import (
    DataPreparationConfig,
    DataSource,
    OutputConfig,
    TokenizerConfig,
)
from src.data.preparation.pipeline import PreparationPipeline


def test_preparation_pipeline_multisource_e2e(tmp_path):
    text_path = tmp_path / "a.txt"
    text_path.write_text("hello world\n\nsecond doc", encoding="utf-8")

    jsonl_path = tmp_path / "b.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "json doc one"}) + "\n")
        handle.write(json.dumps({"text": "json doc two"}) + "\n")

    npy_path = tmp_path / "c.npy"
    np.save(npy_path, np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16))

    out_dir = tmp_path / "out"
    cfg = DataPreparationConfig(
        tokenizer=TokenizerConfig(type="char", vocab_size=256),
        datasets=[
            DataSource(name="textsrc", path=str(text_path), format="text", min_length=1),
            DataSource(name="jsonsrc", path=str(jsonl_path), format="jsonl", min_length=1),
            DataSource(name="npysrc", path=str(npy_path), format="npy", min_length=1, chunk_size=3),
        ],
        output=OutputConfig(dir=str(out_dir), prefix="e2e", eos_token_id=-1),
    )

    manifest = PreparationPipeline().run(cfg)

    assert "train" in manifest["output_paths"]
    assert (out_dir / "e2e_stats.json").exists()

    stats = json.loads((out_dir / "e2e_stats.json").read_text(encoding="utf-8"))
    assert stats["train"]["total_docs"] > 0
    assert "textsrc" in stats["train"]["by_source"] or "jsonsrc" in stats["train"]["by_source"]
