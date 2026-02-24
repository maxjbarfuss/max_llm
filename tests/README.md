# Tests

All test commands are run from the repository root.

## Activate the environment

From the repository root:

```bash
source .venv/bin/activate
```

## Test commands (repo root)

```bash
make test
make test-quick
make test-py
make test-py-quick
make test-cpp
make test-cov
make test-report
```

## Expectations

- If data/tools changed: run targeted tests and a quick data pipeline run.
- If model/training changed: run a short training sanity check.
- If docs-only: no tests required, but still update PLAN/SESSION if status changes.

## Optional acceleration probes

These may fail if dependencies are not installed:

```bash
python3 -c "import flash_attn; print(flash_attn.__version__)"
python3 -c "import torchao; print(torchao.__version__)"
python3 -c "import xformers; print(xformers.__version__)"
```
