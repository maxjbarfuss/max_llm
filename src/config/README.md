# src/config — Config System

Typed, versioned configuration dataclasses for all training runs. Loaded from TOML via `ExperimentConfig.from_toml(path)`.

For the full TOML field reference see [config/README.md](../../config/README.md).

## Structure

```
src/config/
├── experiment.py   ExperimentConfig — top-level composition; validates cross-section constraints
├── model.py        ModelConfig (frozen)
├── training.py     TrainingConfig
├── inference.py    InferenceConfig
├── data.py         DataConfig — TOML null-coercion + tokenizer validation
├── toml_utils.py   load_toml, require_section, section_or_root
└── __init__.py     Public re-exports
```

## Loading a Config

```python
from src.config import ExperimentConfig

cfg = ExperimentConfig.from_toml("config/milestones/p3_final.toml")
print(cfg.model.hidden_size, cfg.training.learning_rate)
```

## Adding a Field

**Optional (non-breaking)** — add with a default; old TOML files still load:

```python
@dataclass
class TrainingConfig:
    new_option: bool = False   # no __version__ bump needed
```

**Required (breaking)** — bump `__version__` and omit the default:

```python
@dataclass
class TrainingConfig:
    __version__: ClassVar[int] = 2   # increment
    new_required_field: int          # no default → old TOML files fail loudly
```

## Test Fixtures

Always use the schema-aware builders from `tests/conftest.py` — they auto-fill defaults and survive field additions without refactoring:

```python
from tests.conftest import build_model_config, build_training_config

config = build_model_config(hidden_size=64, num_layers=2)
```

Available: `build_model_config`, `build_training_config`, `build_data_config`, `build_inference_config`.
