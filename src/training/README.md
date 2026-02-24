# Training Entry Point

Phase 2 training entrypoint lives in this folder.

## Run Training

```bash
python -m src.training.train --config config/experiment.toml
```

## Notes

- The entrypoint supports `.npy` token arrays for fast iteration.
- The config path must include `[model]`, `[training]`, and `[data]` sections.
- For data preparation, see [scripts/data/README.md](../../scripts/data/README.md).
