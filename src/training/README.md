# Training Entry Point

Phase 2 training entrypoint lives in this folder.

## Run Training

```bash
python -m src.training.train --config config/experiment.toml
```

## Checkpoint

A checkpoint is saved automatically to `output_dir/checkpoint.pt` at the end of training. The file contains `{ model_state, optimizer_state, step }` and is directly loadable by `src.inference.run`.

## Notes

- The entrypoint supports `.npy` token arrays for fast iteration.
- The config path must include `[model]`, `[training]`, and `[data]` sections.
- For data preparation, see [scripts/data/README.md](../../scripts/data/README.md).
