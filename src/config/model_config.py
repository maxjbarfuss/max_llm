"""Configuration dataclasses for Max LLM.

This module defines all configuration objects used throughout the project.
Following the design philosophy of using typed dataclasses with validation.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration.
    
    Defines the structure of the Max LLM hybrid architecture including
    MLA attention, MoE layers, and GRU output components.
    
    Args:
        hidden_size: Model dimension (must be divisible by num_heads)
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        vocab_size: Vocabulary size (padded to multiple of 64)
        max_seq_length: Maximum sequence length
        mla_latent_dim: Latent dimension for MLA compression
        intermediate_size: FFN intermediate dimension
        num_experts: Number of MoE experts
        experts_per_token: Top-K experts per token (default: 2)
        moe_frequency: Apply MoE every N layers (0 = no MoE)
        gru_hidden_size: GRU output layer hidden size
        rope_base: RoPE frequency base
        dropout: Dropout probability
        
    Example:
        >>> config = ModelConfig(hidden_size=768, num_layers=12)
        >>> assert config.head_dim == 64  # Computed property
    """
    
    # Core dimensions
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50304
    max_seq_length: int = 2048
    
    # MLA attention
    mla_latent_dim: int = 512
    rope_base: int = 10000
    
    # FFN
    intermediate_size: int | None = None  # Defaults to 4 * hidden_size
    
    # MoE
    num_experts: int = 16
    experts_per_token: int = 2
    moe_frequency: int = 2  # Every 2nd layer
    
    # GRU output
    gru_hidden_size: int | None = None  # Defaults to hidden_size
    
    # Regularization
    dropout: float = 0.1
    
    # Precision
    attention_dtype: torch.dtype = torch.bfloat16
    ffn_dtype: torch.dtype = torch.float8_e5m2
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Dimension checks
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        assert self.hidden_size % 64 == 0, (
            f"hidden_size ({self.hidden_size}) must be multiple of 64"
        )
        assert self.vocab_size % 64 == 0, (
            f"vocab_size ({self.vocab_size}) must be multiple of 64"
        )
        
        # MLA checks
        assert self.mla_latent_dim > 0, "mla_latent_dim must be positive"
        assert self.mla_latent_dim % self.num_heads == 0, (
            f"mla_latent_dim ({self.mla_latent_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        
        # MoE checks
        if self.moe_frequency > 0:
            assert self.num_experts >= 2, "Need at least 2 experts for MoE"
            assert 1 <= self.experts_per_token <= self.num_experts, (
                f"experts_per_token ({self.experts_per_token}) must be "
                f"between 1 and num_experts ({self.num_experts})"
            )
        
        # Set defaults for optional fields using object.__setattr__ (frozen dataclass)
        if self.intermediate_size is None:
            object.__setattr__(self, "intermediate_size", 4 * self.hidden_size)
        if self.gru_hidden_size is None:
            object.__setattr__(self, "gru_hidden_size", self.hidden_size)
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_heads
    
    @property
    def latent_head_dim(self) -> int:
        """Dimension per head in MLA latent space."""
        return self.mla_latent_dim // self.num_heads


@dataclass
class TrainingConfig:
    """Training configuration.
    
    Defines training hyperparameters, precision schedule, and optimization settings.
    
    Args:
        batch_size: Per-device batch size
        gradient_accumulation_steps: Steps to accumulate before optimizer step
        max_steps: Maximum training steps
        learning_rate: Peak learning rate
        warmup_steps: Linear warmup steps
        weight_decay: AdamW weight decay
        gradient_clip_norm: Max gradient norm (0 = no clipping)
        precision_schedule: Progressive precision phases
        distributed_backend: DDP or FSDP
        checkpoint_interval: Steps between checkpoints
        eval_interval: Steps between evaluations
        log_interval: Steps between logging
        
    Example:
        >>> config = TrainingConfig(batch_size=32, max_steps=100000)
        >>> assert config.effective_batch_size == 32  # No accumulation
    """
    
    # Batch configuration
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Training schedule
    max_steps: int = 100000
    warmup_steps: int = 2000
    
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-8
    gradient_clip_norm: float = 1.0
    
    # Precision schedule: (start_step, end_step, precision)
    precision_schedule: list[tuple[int, int, str]] = field(default_factory=lambda: [
        (0, 30000, "fp4"),      # Phase 1: FP4 for FFN/RNN
        (30000, 80000, "fp8"),  # Phase 2: FP8 for FFN/RNN
        (80000, -1, "mixed"),   # Phase 3: Mixed BF16/FP8
    ])
    
    # MoE loss weighting
    moe_balance_loss_weight: float = 0.01
    
    # Distributed training
    distributed_backend: Literal["ddp", "fsdp"] = "ddp"
    
    # Checkpointing and logging
    checkpoint_interval: int = 5000
    eval_interval: int = 1000
    log_interval: int = 100
    keep_last_n_checkpoints: int = 3
    
    # Optimization features
    use_torch_compile: bool = True
    use_flash_attention: bool = True
    selective_checkpointing: bool = True  # Checkpoint attention only
    
    # Monitoring
    track_expert_utilization: bool = True
    track_gradient_norms: bool = True
    detect_anomalies: bool = True
    
    def __post_init__(self) -> None:
        """Validate training configuration."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.max_steps > 0, "max_steps must be positive"
        assert 0 < self.learning_rate < 1, "learning_rate must be in (0, 1)"
        assert 0 <= self.weight_decay < 1, "weight_decay must be in [0, 1)"
        assert self.warmup_steps <= self.max_steps, "warmup_steps must be <= max_steps"
        
        # Validate precision schedule
        for start, end, precision in self.precision_schedule:
            assert precision in ["fp4", "fp8", "bf16", "mixed"], (
                f"Invalid precision: {precision}"
            )
            if end != -1:
                assert start < end, f"Invalid schedule interval: ({start}, {end})"
    
    @property
    def effective_batch_size(self) -> int:
        """Total batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class DataConfig:
    """Data pipeline configuration.
    
    Args:
        dataset_path: Path to dataset or HuggingFace dataset name
        tokenizer_name: HuggingFace tokenizer name
        max_length: Maximum sequence length (truncation)
        num_workers: DataLoader worker processes
        prefetch_factor: Batches to prefetch per worker
        streaming: Use streaming mode for large datasets
        cache_dir: Directory for caching tokenized data
        validation_split: Fraction or number of samples for validation
        seed: Random seed for reproducibility
    """
    
    # Dataset
    dataset_path: str = "openwebtext"
    tokenizer_name: str = "gpt2"
    max_length: int = 2048
    
    # DataLoader
    num_workers: int = 6
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Streaming and caching
    streaming: bool = True
    cache_dir: str = "./data/cache"
    num_shards: int = 64
    
    # Validation
    validation_split: float | int = 0.01  # 1% or absolute number
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self) -> None:
        """Validate data configuration."""
        assert self.max_length > 0, "max_length must be positive"
        assert self.num_workers >= 0, "num_workers must be non-negative"
        
        if isinstance(self.validation_split, float):
            assert 0 < self.validation_split < 1, "validation_split fraction must be in (0, 1)"
        else:
            assert self.validation_split > 0, "validation_split must be positive"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all sub-configs.
    
    This is the top-level configuration object that bundles model, training,
    and data configurations together.
    
    Args:
        name: Experiment name
        output_dir: Directory for checkpoints and logs
        model: Model configuration
        training: Training configuration
        data: Data configuration
        
    Example:
        >>> config = ExperimentConfig(
        ...     name="maxllm-100m",
        ...     model=ModelConfig(hidden_size=768),
        ...     training=TrainingConfig(batch_size=32),
        ...     data=DataConfig(dataset_path="openwebtext")
        ... )
    """
    
    name: str
    output_dir: str = "./outputs"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def __post_init__(self) -> None:
        """Validate experiment configuration."""
        assert self.name, "Experiment name cannot be empty"
        assert self.data.max_length <= self.model.max_seq_length, (
            f"data.max_length ({self.data.max_length}) must be <= "
            f"model.max_seq_length ({self.model.max_seq_length})"
        )
