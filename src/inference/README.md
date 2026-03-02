# Inference Entry Point

Minimal inference entrypoint for Phase 2 models.

## Run Inference

```bash
python -m src.inference.run \
    --config config/milestones/p2_baseline.toml \
    --checkpoint /path/to/checkpoint.pt \
    --prompt "Hello"
```

## Notes

- If no checkpoint is provided, inference runs with random weights.
- Uses tokenizer settings from the experiment config.
- Sampling controls can be overridden via CLI flags.
