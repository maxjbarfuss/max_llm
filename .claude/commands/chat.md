Resolve the latest checkpoint and print the command to start a chat session. Do NOT execute it — just print it so the user can paste it into their terminal.

## Steps

1. **Resolve the run** — if the user passed an argument (e.g. `/chat p4_norm_ab_rms`), use that run name. Otherwise, find the most recently modified subdirectory under `outputs/ephemeral/` that contains a `checkpoint.pt`.

2. **Resolve the config** — look for `config/ephemeral/<run_name>.toml`, then `config/milestones/<run_name>.toml`.

3. **Print the command** for the user to paste into their terminal:
   ```
   source .venv/bin/activate && python -m src.inference.chat \
     --config <config_path> \
     --checkpoint outputs/ephemeral/<run_name>/checkpoint.pt
   ```

4. **If ambiguous** — list available runs with last-modified times and ask the user to pick one.
