"""ModelConfig definition."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .toml_utils import load_toml, section_or_root

_VALID_NORM_TYPES = {"layer", "rms", "flash", "dyt", "crms"}
_VALID_FFN_TYPES = {"gelu", "swiglu", "relu2", "xielu"}
_VALID_POS_TYPES = {"learned", "rope", "add_rope", "alibi", "rel_pos"}
_VALID_ATTN_TYPES = {"mha", "swa", "rla", "mla"}
_VALID_RES_TYPES = {"standard", "full_attn", "block_attn"}


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration.

    Required: hidden_size, vocab_size, max_seq_length.
    Everything else has sensible defaults (num_layers=0 for embedding-only, num_heads=4, etc).
    """

    __version__: ClassVar[int] = 2

    # Required: core dimensions
    hidden_size: int
    vocab_size: int
    max_seq_length: int

    # Optional: transformer stack (defaults to Phase 2: no layers)
    num_layers: int = 0
    num_heads: int = 4
    num_kv_heads: int | None = None  # None = MHA; 1 = MQA; N = GQA (must divide num_heads)

    # Optional: Phase 3+ features (MLA, MoE, GRU)
    mla_latent_dim: int = -1  # Defaults to hidden_size in __post_init__
    rope_base: int | None = None  # None = no RoPE; any int = enable RoPE with that frequency base
    rope_scaling_factor: float = 1.0  # YaRN: s = target_ctx / train_ctx (1.0 = disabled)
    rope_low_freq_factor: float = 1.0  # YaRN: low-frequency threshold α
    rope_high_freq_factor: float = 4.0  # YaRN: high-frequency threshold β
    rope_original_max_seq_len: int | None = None  # YaRN: training context length for thresholds
    intermediate_size: int | None = None  # Defaults to 4*hidden_size
    ffn_chunk_size: int | None = None  # None = disabled; positive chunks sequence FFN path
    num_experts: int = 1
    experts_per_token: int = 1
    moe_frequency: int = 0
    gru_hidden_size: int | None = None  # Defaults to hidden_size
    dropout: float = 0.0

    # Optional: advanced features
    embedding_dim: int | None = None
    share_layer_weights: bool = False
    looped_num_blocks: int | None = None  # None = one block per logical layer (default)
    norm_type: str = "layer"  # "layer" = LayerNorm (Phase 3); "rms" = RMSNorm (Phase 4+ Llama)
    ffn_type: str = "gelu"  # "gelu" | "swiglu" | "relu2" | "xielu"
    pos_type: str = "learned"  # "learned" | "rope" | "add_rope" | "alibi" | "rel_pos"
    rel_pos_num_buckets: int = 32  # used when pos_type == "rel_pos"
    attn_type: str = "mha"  # "mha" | "swa" | "rla" | "mla"
    swa_window_size: int = 256  # window width used when attn_type == "swa"
    res_type: str = "standard"  # "standard" | "full_attn" | "block_attn" (Attention Residuals)
    attn_res_num_blocks: int = (
        8  # block count N for res_type="block_attn"; num_layers % N must == 0
    )

    def __post_init__(self) -> None:
        """Validate model configuration."""
        self._apply_back_compat()
        self._validate_basic()
        self._set_defaults()
        self._validate_dims()
        self._validate_moe()

    def _apply_back_compat(self) -> None:
        """Apply backwards-compatibility promotions before validation."""
        # rope_base set but pos_type still "learned" → promote to "rope"
        if self.rope_base is not None and self.pos_type == "learned":
            object.__setattr__(self, "pos_type", "rope")

    def _validate_basic(self) -> None:
        """Validate basic scalar constraints."""
        self._validate_positive("hidden_size", self.hidden_size)
        self._validate_positive("vocab_size", self.vocab_size)
        self._validate_positive("max_seq_length", self.max_seq_length)
        if self.num_layers < 0:
            raise ValueError(
                "num_layers must be non-negative (0 for embed-only, >0 for transformer)"
            )
        if self.num_layers > 0 and self.num_heads <= 0:
            raise ValueError("num_heads must be positive when num_layers > 0")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be in [0, 1)")
        if self.norm_type not in _VALID_NORM_TYPES:
            raise ValueError(
                f"norm_type must be one of {_VALID_NORM_TYPES}, got '{self.norm_type}'"
            )
        if self.ffn_type not in _VALID_FFN_TYPES:
            raise ValueError(f"ffn_type must be one of {_VALID_FFN_TYPES}, got '{self.ffn_type}'")
        self._validate_pos_type()
        self._validate_attn_type()
        self._validate_res_type()
        self._validate_looped()

    @staticmethod
    def _validate_positive(name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    def _validate_pos_type(self) -> None:
        """Validate positional encoding type and required co-fields."""
        if self.pos_type not in _VALID_POS_TYPES:
            raise ValueError(f"pos_type must be one of {_VALID_POS_TYPES}, got '{self.pos_type}'")
        if self.pos_type in {"rope", "add_rope"} and self.rope_base is None:
            raise ValueError(
                f"pos_type='{self.pos_type}' requires rope_base to be set (e.g. rope_base = 10000)"
            )
        if self.rope_scaling_factor < 1.0:
            raise ValueError(
                f"rope_scaling_factor must be >= 1.0 (1.0 = disabled), got {self.rope_scaling_factor}"
            )
        if self.rope_high_freq_factor <= self.rope_low_freq_factor:
            raise ValueError(
                f"rope_high_freq_factor ({self.rope_high_freq_factor}) must be greater than "
                f"rope_low_freq_factor ({self.rope_low_freq_factor})"
            )
        if self.rope_original_max_seq_len is not None and self.rope_original_max_seq_len <= 0:
            raise ValueError(
                f"rope_original_max_seq_len must be positive, got {self.rope_original_max_seq_len}"
            )

    def _validate_attn_type(self) -> None:
        """Validate attention type and cross-field constraints."""
        if self.attn_type not in _VALID_ATTN_TYPES:
            raise ValueError(
                f"attn_type must be one of {_VALID_ATTN_TYPES}, got '{self.attn_type}'"
            )
        if self.attn_type == "swa" and self.swa_window_size < 1:
            raise ValueError(
                f"swa_window_size must be >= 1 when attn_type='swa', got {self.swa_window_size}"
            )
        if self.attn_type == "rla" and self.pos_type in {"rope", "add_rope"}:
            raise ValueError(
                "attn_type='rla' is incompatible with RoPE — the ELU+1 kernel breaks "
                "rotation equivariance. Use pos_type='learned' or 'alibi' instead."
            )
        if self.attn_type == "mla" and self.pos_type not in {"rope", "add_rope"}:
            raise ValueError(
                "attn_type='mla' requires decoupled RoPE; use pos_type='rope' or 'add_rope'."
            )

    def _validate_res_type(self) -> None:
        """Validate residual connection type and block count."""
        if self.res_type not in _VALID_RES_TYPES:
            raise ValueError(f"res_type must be one of {_VALID_RES_TYPES}, got '{self.res_type}'")
        if self.res_type == "block_attn":
            if self.attn_res_num_blocks < 1:
                raise ValueError(
                    f"attn_res_num_blocks must be >= 1, got {self.attn_res_num_blocks}"
                )
            if self.num_layers > 0 and self.num_layers % self.attn_res_num_blocks != 0:
                raise ValueError(
                    f"num_layers ({self.num_layers}) must be divisible by "
                    f"attn_res_num_blocks ({self.attn_res_num_blocks}) "
                    f"when res_type='block_attn'"
                )

    def _validate_looped(self) -> None:
        """Validate looped block execution field."""
        if self.looped_num_blocks is not None:
            if self.looped_num_blocks <= 0:
                raise ValueError(
                    f"looped_num_blocks must be positive when set, got {self.looped_num_blocks}"
                )
            if self.looped_num_blocks > self.num_layers:
                raise ValueError(
                    f"looped_num_blocks ({self.looped_num_blocks}) must be <= "
                    f"num_layers ({self.num_layers})"
                )

    def _validate_dims(self) -> None:
        """Validate dimension alignments and divisibility."""
        if self.hidden_size % 64 != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be multiple of 64")
        if self.vocab_size % 64 != 0:
            raise ValueError(f"vocab_size ({self.vocab_size}) must be multiple of 64")

        # Only validate attention dimensions when actually using transformer layers
        if self.num_layers > 0:
            if self.hidden_size % self.num_heads != 0:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) must be divisible by "
                    f"num_heads ({self.num_heads})"
                )
            if self.mla_latent_dim % self.num_heads != 0:
                raise ValueError(
                    f"mla_latent_dim ({self.mla_latent_dim}) must be divisible by "
                    f"num_heads ({self.num_heads})"
                )
            if self.num_kv_heads is not None:
                self._validate_num_kv_heads()

        if self.intermediate_size is not None and self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive when provided")
        if self.ffn_chunk_size is not None and self.ffn_chunk_size <= 0:
            raise ValueError("ffn_chunk_size must be positive when provided")
        if self.gru_hidden_size is not None and self.gru_hidden_size <= 0:
            raise ValueError("gru_hidden_size must be positive when provided")

    def _validate_num_kv_heads(self) -> None:
        """Validate num_kv_heads for GQA/MQA."""
        assert self.num_kv_heads is not None
        if not (1 <= self.num_kv_heads <= self.num_heads):
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) must be between 1 and "
                f"num_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

    def _validate_moe(self) -> None:
        """Validate mixture-of-experts configuration."""
        if self.moe_frequency > 0:
            if self.num_experts < 2:
                raise ValueError("Need at least 2 experts for MoE")
            if not (1 <= self.experts_per_token <= self.num_experts):
                raise ValueError(
                    f"experts_per_token ({self.experts_per_token}) must be "
                    f"between 1 and num_experts ({self.num_experts})"
                )

    def _set_defaults(self) -> None:
        """Set default values for optional fields."""
        if self.mla_latent_dim <= 0:
            object.__setattr__(self, "mla_latent_dim", self.hidden_size)
        if self.intermediate_size is None:
            object.__setattr__(self, "intermediate_size", 4 * self.hidden_size)
        if self.gru_hidden_size is None:
            object.__setattr__(self, "gru_hidden_size", self.hidden_size)

    @classmethod
    def from_toml(cls, file_path: str | Path) -> "ModelConfig":
        """Create a model config from TOML."""
        raw = load_toml(file_path)
        return cls(**section_or_root(raw, "model"))

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_heads

    @property
    def latent_head_dim(self) -> int:
        """Dimension per head in MLA latent space."""
        return self.mla_latent_dim // self.num_heads
