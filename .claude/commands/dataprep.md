Run the data preparation pipeline to tokenize and stage a dataset.

## Steps

1. **Resolve the config** — if the user passed an argument, treat it as:
   - A full path if it contains `/`
   - Otherwise search `config/ephemeral/<arg>.{json,toml}`, then `config/data_prep/<arg>.{json,toml}`
   - If no argument, list available data-prep configs and ask the user to pick one

2. **Run the pipeline**:
   ```
   source .venv/bin/activate && python -m src.data.preparation --config <config_path> -v
   ```

3. **After completion** — report the output artifact paths and token counts from the stats JSON written alongside the .npy files in `data/fast/`.
