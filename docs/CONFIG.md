# Config System API Reference

> **Design**: See [DESIGN.md](DESIGN.md#config-system-version-aware-evolution) for architecture, principles, and when to increment `__version__`.

---

## Adding Config Fields

### Optional Field (Non-Breaking)

```python
# model.py
@dataclass(frozen=True)
class ModelConfig:
    attention_dropout: float = 0.0  # Has default → no __version__ increment
```

Result: Old TOML files work, tests auto-update, old checkpoints still valid.

### Required Field (Breaking)

```python
@dataclass(frozen=True)
class ModelConfig:
    __version__: ClassVar[int] = 2  # Increment!
    new_required_param: int          # No default
```

Result: Old TOML files fail validation, old checkpoints rejected with error.

---

## Checkpoint Versioning

```python
from src.training.train import save_checkpoint, load_checkpoint

# Save (auto-includes config versions)
path = save_checkpoint(model, optimizer, step=100, output_dir="outputs")

# Load with version validation
step = load_checkpoint(path, model, optimizer, strict_version_check=True)
# Raises ConfigVersionMismatchError if versions don't match

# Load old checkpoint (v0 without versions)
step = load_checkpoint(path, model, optimizer, strict_version_check=False)
# Warns but loads successfully
```

---

## Validation Utilities

```python
from src.config import (
    validate_checkpoint_config_compatibility,
    get_checkpoint_config_versions,
    ConfigVersionMismatchError,
)

# Check compatibility (raises on mismatch if strict=True)
try:
    versions = validate_checkpoint_config_compatibility("ckpt.pt", strict=True)
except ConfigVersionMismatchError as e:
    print(e)  # Includes remediation steps

# Get versions without raising
versions = get_checkpoint_config_versions("ckpt.pt")
# Returns: {"model": 1, "training": 1, "data": 1, "inference": 1}
# Returns None if checkpoint predates versioning system
```

---

## Test Fixtures

```python
from tests.conftest import build_model_config, build_training_config, build_data_config

# ❌ Brittle (updates needed whenever fields added)
config = ModelConfig(hidden_size=64, num_layers=1, ...) # 15+ fields!

# ✅ Resilient (auto-adapts to new fields)
config = build_model_config(hidden_size=64, num_layers=1)
# Other fields get sensible test defaults automatically
```

All builders in [tests/conftest.py](../tests/conftest.py):
- `build_model_config(**overrides)`
- `build_training_config(**overrides)`
- `build_data_config(**overrides)`
- `build_inference_config(**overrides)`

---

## Migration: Old Checkpoints

Checkpoints created before versioning system (no `config_versions` key):

```python
# This will warn but load successfully
step = load_checkpoint(
    "old_checkpoint.pt",
    model, optimizer,
    strict_version_check=False
)
```

Message: `"Checkpoint at ... was created before config versioning. Cannot verify compatibility."`

---

## Files

| File | Purpose |
|------|---------|
| [src/config/model.py](../src/config/model.py), [training.py](../src/config/training.py), [data.py](../src/config/data.py), [inference.py](../src/config/inference.py) | Config dataclasses with `__version__` |
| [src/config/validation.py](../src/config/validation.py) | Version validation and error handling |
| [src/training/train.py](../src/training/train.py) | `save_checkpoint()`, `load_checkpoint()` with versions |
| [tests/conftest.py](../tests/conftest.py) | Schema-aware config builders |
| [tests/unit/test_config_versioning.py](../tests/unit/test_config_versioning.py) | Versioning system tests (9 tests) |
