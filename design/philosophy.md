# Design Philosophy

This project is designed to be maintained by both human developers and AI coding assistants. Our philosophy emphasizes clarity, modularity, and context efficiency to enable seamless collaboration across sessions and contributors.

## Core Principles

### 1. **KISS (Keep It Simple, Stupid) - Priority #1**
- Prefer straightforward solutions over clever ones
- One responsibility per function/class
- Avoid premature optimization
- If it needs extensive comments to explain, it's too complex

### 2. **Test-Driven Development (TDD)**
Write tests before implementation. This ensures code is testable, well-designed, and correct from the start.

**TDD Workflow:**
1. **Red** - Write a failing test that defines desired behavior
2. **Green** - Write minimal code to make the test pass
3. **Refactor** - Improve code while keeping tests green

```python
# Step 1: Write test first (Red)
def test_mla_reduces_kv_cache():
    """MLA should compress KV cache by 75%."""
    config = ModelConfig(hidden_size=768, mla_latent_dim=512)
    attn = MLAAttention(config)
    
    x = torch.randn(2, 128, 768)  # batch=2, seq=128, hidden=768
    kv_cache_size = attn.forward(x).kv_cache.numel()
    
    # Expected: 75% reduction from full KV cache
    full_kv_size = 2 * 128 * 768  # K + V
    expected_size = 2 * 128 * 512  # Compressed to latent_dim
    assert kv_cache_size == expected_size
    assert kv_cache_size < full_kv_size * 0.5  # >50% reduction

# Step 2: Implement minimal code (Green)
class MLAAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        self.kv_compress = nn.Linear(config.hidden_size, config.mla_latent_dim * 2)
    
    def forward(self, x: Tensor) -> AttentionOutput:
        kv = self.kv_compress(x)  # Compress to latent space
        # ... rest of implementation

# Step 3: Refactor while keeping tests green
# Add Flash Attention, optimize kernels, improve naming
```

**Benefits:**
- **Testable design** - Forces you to think about interfaces
- **Regression prevention** - Tests catch breaking changes
- **Documentation** - Tests show how to use the code
- **Confidence** - Refactor without fear

**For AI Assistants:**
- Always check for existing tests before modifying code
- If no tests exist, write them before making changes
- Test edge cases: empty inputs, max sizes, invalid configs
- Use fixtures for common test data (see `tests/fixtures/`)

### 3. **Context Window Efficiency**
AI assistants work within token limits. Our code structure minimizes context requirements:

#### Vertical Feature Slicing
```python
# ✓ Good: Self-contained feature module
src/models/attention/mla.py          # Complete MLA implementation
src/models/attention/__init__.py     # Public API exports only

# ✗ Bad: Horizontal layering requiring multiple files
src/layers/attention.py              # Generic attention
src/optimizations/attention.py       # Attention optimizations
src/utils/attention_helpers.py       # Scattered utilities
```

#### Factory & Builder Patterns
```python
# ✓ Reduces coupling, clear construction
model = ModelFactory.create(config)
trainer = TrainerBuilder().with_precision("fp8").with_ddp().build()

# ✗ Requires reading entire class hierarchy
model = ModelA(ModelB(config), optimizer=OptimizerC(...), ...)
```

#### Decorator Pattern for Cross-Cutting Concerns
```python
@profile_memory
@checkpoint_activations
@compile_with_torch
def transformer_forward(x: Tensor) -> Tensor:
    """Single-purpose function with composed behaviors."""
    return self.attention(x) + x
```

### 4. **Type Hints Everywhere**
Type hints are documentation that IDEs and static analyzers understand:

```python
from typing import Protocol, TypeVar, Generic
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model configuration with validated defaults."""
    hidden_size: int = 768
    num_layers: int = 12
    vocab_size: int = 50304
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        assert self.hidden_size % 64 == 0, "hidden_size must be divisible by 64"

class Attention(Protocol):
    """Protocol defines interface without inheritance."""
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor: ...

def train_step(
    model: nn.Module,
    batch: dict[str, Tensor],
    optimizer: Optimizer,
) -> tuple[float, dict[str, float]]:
    """Clear inputs and outputs. Returns: (loss, metrics)"""
    ...
```

**Benefits:**
- Static analysis catches errors before runtime
- AI assistants understand types without reading implementations
- IDEs provide accurate autocomplete
- Self-documenting code reduces comment needs

### 5. **SOLID Principles**

#### Single Responsibility Principle (SRP)
```python
# Each class has ONE reason to change
class MLAAttention:           # Attention mechanism only
class AttentionCache:         # KV cache management only
class AttentionMetrics:       # Metric collection only
```

#### Open/Closed Principle (OCP)
```python
# Extend behavior without modifying existing code
class BasePrecisionScheduler(ABC):
    @abstractmethod
    def get_precision(self, step: int) -> str: ...

class ProgressivePrecisionScheduler(BasePrecisionScheduler):
    """FP4 → FP8 → BF16 schedule."""
    ...
```

#### Liskov Substitution Principle (LSP)
```python
# Subclasses must be substitutable for base classes
def train_model(attention: Attention, data: DataLoader) -> None:
    """Works with any Attention implementation."""
    ...  # MLA, MHA, or future variants
```

#### Interface Segregation Principle (ISP)
```python
# Clients shouldn't depend on interfaces they don't use
class Trainable(Protocol):
    def train_step(self, batch) -> float: ...

class Evaluable(Protocol):
    def eval_step(self, batch) -> dict[str, float]: ...

# Classes implement only what they need
```

#### Dependency Inversion Principle (DIP)
```python
# Depend on abstractions, not concretions
class Trainer:
    def __init__(
        self,
        model: Trainable,           # Protocol, not concrete class
        data: Iterable[Batch],      # Abstract iterable
        logger: LoggerProtocol,     # Protocol for logging
    ): ...
```

### 6. **Modern Python Idioms**

#### Use dataclasses for configuration and state
```python
from dataclasses import dataclass, field

@dataclass(frozen=True)  # Immutable config
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    enabled_optimizations: list[str] = field(default_factory=list)
```

#### Use structural pattern matching (Python 3.10+)
```python
def handle_precision(precision: str) -> torch.dtype:
    match precision:
        case "fp4": return torch.float8_e4m3fn
        case "fp8": return torch.float8_e5m2
        case "bf16": return torch.bfloat16
        case _: raise ValueError(f"Unknown precision: {precision}")
```

#### Use context managers for resource management
```python
@contextmanager
def precision_context(dtype: torch.dtype):
    """Temporarily change autocast precision."""
    old_dtype = torch.get_autocast_dtype()
    torch.set_autocast_dtype(dtype)
    try:
        yield
    finally:
        torch.set_autocast_dtype(old_dtype)
```

### 7. **Documentation Strategy**

#### Comprehensive Headers, Minimal Inline Comments

```python
class MLAAttention(nn.Module):
    """Multi-head Latent Attention with KV compression.
    
    Reduces KV cache by 75% through latent space projection while maintaining
    full query representation. Uses RoPE for position encoding and Flash Attention
    for memory efficiency.
    
    Args:
        hidden_size: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        latent_dim: Compressed dimension for K/V (default: 512)
        rope_base: Base frequency for rotary embeddings (default: 10000)
    
    Example:
        >>> attn = MLAAttention(hidden_size=768, num_heads=12)
        >>> out = attn(x, mask=causal_mask)
    
    References:
        - "Multi-head Latent Attention" (Paper Link)
        - Flash Attention: https://arxiv.org/abs/2205.14135
    """
    
    def __init__(self, hidden_size: int, num_heads: int, latent_dim: int = 512) -> None:
        """Initialize MLA attention layer."""
        super().__init__()
        # Code is self-explanatory with good names, no inline comments needed
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.kv_compress = nn.Linear(hidden_size, latent_dim * 2)
        self.rope = RotaryEmbedding(dim=latent_dim // num_heads)
    
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply MLA attention. Returns attended features."""
        q = self.q_proj(x)
        k, v = self.kv_compress(x).chunk(2, dim=-1)
        # Implementation is clear from variable names and types
        return self._flash_attention(q, k, v, mask)
```

#### Update Documentation Atomically with Code
- Treat docstrings as part of the implementation
- Update README.md architecture diagrams when structure changes
- Keep design/plan.md synchronized with actual implementation

### 8. **MLOps and Monitoring**

#### Built-in Observability
```python
@dataclass
class TrainingMetrics:
    """Tracked metrics for drift detection."""
    loss: float
    perplexity: float
    gradient_norm: float
    expert_utilization: dict[int, float]  # MoE balance
    precision_errors: dict[str, float]     # Quantization impact
    
class DriftDetector:
    """Monitors for data and model drift.
    
    Alerts when:
    - Loss distribution shifts significantly
    - Expert utilization becomes imbalanced
    - Gradient norms spike or vanish
    - Quantization error exceeds thresholds
    """
    
    def check_data_drift(self, current: Batch, reference: Batch) -> DriftReport:
        """Detect distribution shift in input data."""
        ...
    
    def check_model_drift(self, metrics: TrainingMetrics) -> DriftReport:
        """Detect performance degradation over time."""
        ...
```

#### Checkpoint Everything
```python
@dataclass
class CheckpointState:
    """Complete state for reproducible training."""
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict
    rng_state: dict  # PyTorch, NumPy, Python random
    precision_phase: str
    training_step: int
    metrics_history: list[TrainingMetrics]
```

#### Continuous Validation
```python
# Regular generation samples to catch degeneration
if step % config.eval_interval == 0:
    samples = generate_text(model, prompts=validation_prompts)
    quality_metrics = evaluate_quality(samples)
    if quality_metrics.repetition_ratio > 0.5:
        logger.alert("High repetition detected - possible degeneration")
```

### 9. **For AI Coding Assistants**

#### Before Starting Work:
1. **READ `design/plan-checklist.md` FIRST** - This is your primary scratch pad
   - Check "Next Actions" for what to work on
   - Read "Last Handoff" for current state and context
   - Review "Broad Plan" for overall direction
2. Read `design/plan.md` for detailed architecture specifications
3. Read relevant module docstrings (not implementations)
4. Review recent commits for additional context

#### While Working:

**1. Make atomic commits** - One feature/fix per commit

**2. Write descriptive commit messages:**
```
feat(attention): Implement MLA with latent compression

- Add MLAAttention module with Q full-size, KV compressed
- Integrate RoPE in latent space
- Add Flash Attention 2 backend
- Update ModelConfig with mla_latent_dim parameter

Files changed:
- src/models/attention/mla.py (new)
- src/models/transformer.py (import MLA)
- configs/model.yaml (add latent_dim config)
- tests/test_mla.py (new)

Next steps: Integrate MLA into transformer blocks, benchmark vs MHA
```

**3. Update documentation immediately:**
- Docstrings for new classes/functions
- README.md if architecture changes
- design/plan-checklist.md to mark completed items

**4. Leave breadcrumbs for the next assistant:**
```python
# TODO(priority=high): Add attention mask support for variable length sequences
# Context: Currently assumes fixed length. Need to handle padding tokens.
# See: design/plan.md section 3 for mask requirements
# Files to modify: src/models/attention/mla.py, src/data/collate.py
```

**5. Run validation before committing:**
```bash
# Type checking
mypy src/

# Linting
ruff check src/

# Format
black src/ tests/

# Tests
pytest tests/ -v
```

#### Before Finishing Work:
**update `design/plan-checklist.md`:**
- What you completed this session
- Current state of the codebase
- What to work on next (update "Next Actions" if needed)
- Any blockers or issues encountered
- Cross check with the [`plan.md`](plan.md) and update if necessary and [`../README.md`](../README.md) files

This is the primary handoff mechanism for AI assistants. Future sessions start by reading this section.

### 10. **Code Organization Rules**

#### File Size Limits
- Max 500 lines per file (prefer 200-300)
- If larger, split by feature vertically
- Use `__init__.py` to provide clean public API

#### Import Organization
```python
# Standard library
import os
from pathlib import Path
from typing import Protocol

# Third-party
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Local - absolute imports from project root
from src.models.attention import MLAAttention
from src.config import ModelConfig
from src.utils.logging import get_logger
```

#### Test Organization
```
tests/
├── unit/              # Fast, isolated tests
│   ├── test_mla.py
│   └── test_moe.py
├── integration/       # Multi-component tests
│   ├── test_training_loop.py
│   └── test_distributed.py
└── fixtures/          # Shared test data
    └── sample_data.py
```

---

## Summary Checklist for Contributors

- [ ] Code follows KISS principle
- [ ] Tests written BEFORE implementation (TDD)
- [ ] All tests passing (unit + integration if needed)
- [ ] All functions/classes have type hints
- [ ] Docstrings explain WHAT and WHY, not HOW
- [ ] No inline comments unless truly necessary
- [ ] Changes are vertically sliced (complete feature in minimal files)
- [ ] SOLID principles applied
- [ ] Monitoring/metrics added for new components
- [ ] Documentation updated (docstrings + README if needed)
- [ ] Static analysis passes (mypy, ruff)
- [ ] Code coverage maintained or improved
- [ ] Commit message includes context for next developer
- [ ] design/plan-checklist.md updated

**Remember:** Future you (or the next AI assistant) will thank you for clear, self-contained code with excellent headers and minimal cognitive load.
