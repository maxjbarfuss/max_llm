Launch a training run, always using both GPUs (RTX 4090 + RTX 3090 Ti) via torchrun DDP for maximum throughput.

## Steps

1. **Resolve the config** — if the user passed an argument, treat it as:
   - A full path if it contains `/`
   - Otherwise check `config/ephemeral/<arg>.toml`, then `config/milestones/<arg>.toml`
   - If no argument, list available configs in `config/ephemeral/` and `config/milestones/` and ask the user to pick one

2. **Always launch with torchrun across both GPUs**:
   ```
   source .venv/bin/activate && torchrun --standalone --nnodes=1 --nproc_per_node=2 \
     -m src.training.train --config <config_path> --distributed
   ```

3. **Confirm** — print the resolved config path and command before running.
