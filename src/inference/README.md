# Inference Entry Point

Two modes: interactive chat and single-prompt generation.

## Interactive chat (recommended)

```bash
source .venv/bin/activate
python -m src.inference.chat \
    --config config/ephemeral/p4_relu2.toml \
    --checkpoint outputs/ephemeral/p4_relu2/checkpoint.pt
```

Type prompts at the `>` prompt. The model continues your text. `Ctrl-C` to exit.

## Single prompt

```bash
source .venv/bin/activate
python -m src.inference.run \
    --config config/milestones/p3_final_unigram.toml \
    --checkpoint outputs/milestones/p3_final_unigram/checkpoint.pt \
    --prompt "Once upon a time"
```

## CLI flags

| Flag | Default | Notes |
|------|---------|-------|
| `--config` | required | TOML config file (used to load tokenizer + model architecture) |
| `--checkpoint` | required | Path to `checkpoint.pt` |
| `--prompt` | `""` | Seed text for generation (run mode only) |
| `--temperature` | from config | Sampling temperature; `0.0` = greedy |
| `--top_p` | from config | Nucleus sampling; `0.0` = disabled |
| `--max_new_tokens` | from config | Tokens to generate |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |

## Notes

- For the latest checkpoint in any run: `outputs/<run>/checkpoint.pt`
- Inference automatically falls back to `standard` attention backend on CPU
- Config reference: [config/README.md](../../config/README.md)
