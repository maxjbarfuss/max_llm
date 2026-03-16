# Training Entry Point

Single entrypoint for all phases. Pass a TOML config and optionally a checkpoint to resume.

## Single-GPU

```bash
source .venv/bin/activate
python -m src.training.train --config config/milestones/p3_final_unigram.toml
```

## Multi-GPU (DDP) — recommended for Phase 3+

```bash
source .venv/bin/activate
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m src.training.train \
    --config config/ephemeral/p4_relu2.toml \
    --distributed
```

## Resume from checkpoint

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m src.training.train \
    --config config/ephemeral/p4_relu2.toml \
    --distributed \
    --resume outputs/ephemeral/p4_relu2/checkpoint.pt
```

## Outputs

Each run writes to `output_dir` (from config):

```
output_dir/
  checkpoint.pt           Latest checkpoint (model + optimizer + step)
  checkpoint_step_N.pt    Periodic snapshots (every checkpoint_interval steps)
  loss_curve.csv          step, loss, perplexity, lr, tokens_per_sec, gpu_memory_mb
  tensorboard/            TensorBoard event files
```

## Notes

- Config reference: [config/README.md](../../config/README.md)
- Data preparation: [src/data/README.md](../data/README.md)
- The `--distributed` flag enables DDP; always launch with `torchrun` when using it
- `attention_backend = "flash"` is recommended for Phase 3+; `use_torch_compile = true` for further throughput
