# config — TOML Field Reference

All training runs use a single TOML file with five sections. Pass it with `--config`.

For the Python config module (adding fields, versioning, test fixtures) see [src/config/README.md](../src/config/README.md).

```
[experiment]   name, output directory
[model]        architecture dimensions and features
[training]     optimizer, scheduler, precision, distributed, logging
[inference]    sampling parameters for generation
[data]         dataset paths, tokenizer, dataloader settings
```

---

## [experiment]

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `name` | string | ✅ | Run identifier; used in logs and TensorBoard |
| `output_dir` | string | ✅ | Directory for checkpoints, CSV, TensorBoard events |

---

## [model]

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `hidden_size` | int | ✅ required | Model dimension; must be a multiple of 64 |
| `vocab_size` | int | ✅ required | Vocabulary size; must be a multiple of 64 |
| `max_seq_length` | int | ✅ required | Maximum context length; must be ≥ `data.max_length` |
| `num_layers` | int | `0` | Transformer depth; `0` = embedding-only (Phase 2 baseline) |
| `num_heads` | int | `4` | Attention heads; `hidden_size` must be divisible by `num_heads` |
| `num_kv_heads` | int \\| `null` | `null` | K/V head count for grouped-query attention: `null` = MHA (`num_kv_heads = num_heads`), `1` = MQA, `1 < N < num_heads` = GQA; must divide `num_heads` |
| `dropout` | float | `0.0` | Dropout probability applied in attention and FFN |
| `norm_type` | `"layer"` \| `"rms"` \| `"flash"` \| `"dyt"` \| `"crms"` | `"layer"` | Normalization variant: `"layer"` = LayerNorm (P3 default); `"rms"` = RMSNorm (P4+ recommended); `"flash"` = parameter-free RMSNorm (scale absorbed into adjacent linear weights); `"dyt"` = Dynamic Tanh `γ⊙tanh(α·x)`, replaces norm with bounded nonlinearity (Zhai et al. 2025); `"crms"` = Centered RMSNorm (mean-subtract then RMSNorm, bridges RMSNorm and LayerNorm) |
| `ffn_type` | `"gelu"` \| `"swiglu"` \| `"relu2"` \| `"xielu"` | `"gelu"` | FFN activation variant: `"swiglu"` (Llama, no bias, hidden=4d×2/3); `"relu2"` (sparse ~50%, no params); `"xielu"` (piecewise quad/exp, 2 trainable scalars) |
| `intermediate_size` | int | `4 × hidden_size` | FFN hidden dimension; overrides the 4× default; for SwiGLU use `swiglu_intermediate_size(hidden_size)` (rounds to 256 multiple) |
| `ffn_chunk_size` | int | `null` | Split the sequence into chunks of this size through each FFN to cap peak `(B,T,intermediate_size)` activation memory; `null` = disabled; trades throughput for memory |
| `pos_type` | string | `"learned"` | Positional encoding: `"learned"` (additive table), `"rope"` (rotary), `"add_rope"` (additive sinusoidal on Q/K), `"alibi"` (linear bias, no params), `"rel_pos"` (learned T5-style bucket bias) |
| `rope_base` | int | `null` | Frequency base for `pos_type = "rope"` or `"add_rope"` (e.g. `10000`); ignored for other variants; back-compat: setting this without `pos_type` auto-selects `pos_type = "rope"` |
| `rope_scaling_factor` | float | `1.0` | YaRN NTK-by-parts context extension scale s = target_ctx / train_ctx; `1.0` = disabled; set to `target_len / train_len` to extend effective context |
| `rope_low_freq_factor` | float | `1.0` | YaRN α — low-frequency wavelength threshold; dims whose wavelength exceeds α × `rope_original_max_seq_len` are scaled fully by 1/s |
| `rope_high_freq_factor` | float | `4.0` | YaRN β — high-frequency wavelength threshold; dims below β × `rope_original_max_seq_len` are left unchanged; dims between α and β are blended |
| `rope_original_max_seq_len` | int | `null` | YaRN training context length used to compute wavelength thresholds; required when `rope_scaling_factor > 1.0` |
| `rel_pos_num_buckets` | int | `32` | Distance buckets for `pos_type = "rel_pos"`; ignored for other variants |
| `attn_type` | `"mha"` \| `"swa"` \| `"rla"` \| `"mla"` | `"mha"` | Attention variant: `"mha"` = standard causal MHA; `"swa"` = Sliding Window — each token attends to the last `swa_window_size` tokens (Flash Attention 2 native; otherwise band mask); `"rla"` = Residual Linear Attention — O(T·D²) ELU+1 kernel + learned input residual; incompatible with RoPE; `"mla"` = Multi-head Latent Attention — low-rank KV compression via `mla_latent_dim`; requires `pos_type = "rope"` or `"add_rope"` |
| `swa_window_size` | int | `256` | Window width for `attn_type = "swa"`; must be ≥ 1; ignored for other variants |
| `res_type` | `"standard"` \| `"full_attn"` \| `"block_attn"` | `"standard"` | Residual connection variant: `"standard"` = fixed additive residual (default); `"full_attn"` = Full Attention Residuals — each sublayer input is softmax attention over all prior sublayer outputs (O(L²), practical for small models); `"block_attn"` = Block Attention Residuals — layers grouped into `attn_res_num_blocks` blocks, attention over N block-level summaries (O(N²), recommended for scale; paper shows 1.25× compute advantage at N≈8). Queries zero-initialized so initial weights are uniform. Ref: Kimi Team 2025 |
| `attn_res_num_blocks` | int | `8` | Block count N for `res_type = "block_attn"`; `num_layers` must be divisible by N; N=1 collapses to standard residuals, N=num_layers ≈ full_attn; ignored for `"standard"` and `"full_attn"` |
| `embedding_dim` | int | `null` | Factorized embedding dimension; `null` = no factorization |
| `share_layer_weights` | bool | `false` | Backward-compatible alias for `looped_num_blocks = 1` — one physical block shared across all logical layers. Drastically cuts capacity; prefer `looped_num_blocks` for new configs |
| `looped_num_blocks` | int \| `null` | `null` | Number of physical transformer blocks. `null` = one block per logical layer (default). When set to `K` (1 ≤ K ≤ `num_layers`), constructs K physical blocks and maps logical layer `i` to physical block `i % K`. Reduces parameter count by `num_layers / K`× while preserving logical depth. Per-logical-layer KV caches are always allocated regardless of sharing. Use with caution in training; watch for loss degradation vs the full-depth baseline |
| `mla_latent_dim` | int | `hidden_size` | MLA latent KV dimension; set < `hidden_size` to compress KV size; only meaningful when `attn_type = "mla"` |
| `num_experts` | int | `1` | Total MoE experts (Phase 6+); `1` = dense |
| `experts_per_token` | int | `1` | Top-k experts per token (Phase 6+) |
| `moe_frequency` | int | `0` | Insert MoE every N blocks (Phase 6+); `0` = dense throughout |
| `gru_hidden_size` | int | `hidden_size` | GRU reasoning stream size (Phase 7+) |

---

## [training]

### Optimizer

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `optimizer_type` | `"adamw"` \| `"muon"` | `"adamw"` | `"adamw"` = standard AdamW for all params; `"muon"` = Newton-Schulz orthogonalized gradient for non-embedding 2-D matrices + AdamW for embeddings, lm_head, biases, and 1-D params |
| `learning_rate` | float | ✅ required | Peak LR; `3e-4`–`4e-3` typical for P3; used for AdamW params (and Muon if `muon_lr` is unset) |
| `muon_lr` | float | `null` | Separate peak LR for Muon-managed parameters; `null` inherits `learning_rate` for all groups |
| `muon_momentum` | float | `0.95` | Nesterov momentum for Muon parameter groups |
| `muon_ns_steps` | int | `5` | Newton-Schulz iteration steps; more = closer to true orthogonalization at higher compute cost |
| `weight_decay` | float | ✅ required | L2 regularization; `0.05` recommended; applied to all params with `ndim ≥ 2` |
| `betas` | `[float, float]` | ✅ required | AdamW momentum; `[0.9, 0.95]` recommended |
| `epsilon` | float | ✅ required | AdamW stability term; `1e-8` standard |
| `gradient_clip_norm` | float | ✅ required | Global gradient norm clip; `0.5`–`1.0` typical |

### Schedule

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `max_steps` | int | ✅ required | Total training steps |
| `warmup_steps` | int | ✅ required | Linear LR warmup steps; must be ≤ `max_steps` |
| `scheduler_type` | `"cosine"` \| `"wsd"` \| `"sgdr"` | `"cosine"` | LR schedule: `"cosine"` = single cosine decay; `"wsd"` = Warmup-Stable-Decay; `"sgdr"` = cosine restarts (preferred for new experiments) |
| `min_lr_ratio` | float | `0.1` | LR floor as fraction of peak; e.g. `0.05` → decay to 5% of peak |
| `wsd_stable_fraction` | float | `0.7` | Fraction of total steps held at peak LR (WSD only) |
| `wsd_decay_fraction` | float | `0.2` | Fraction of total steps spent decaying (WSD only) |
| `wsd_decay_shape` | `"linear"` \| `"sqrt"` \| `"lowered_linear"` | `"sqrt"` | Decay curve shape (WSD only); `"sqrt"` recommended |
| `wsd_lowered_linear_alpha` | float | `0.7` | Exponent for `lowered_linear` shape: higher = stays high longer (WSD only) |
| `sgdr_num_cycles` | int | `4` | Number of cosine restart cycles (SGDR only) |
| `sgdr_cycle_decay` | float | `0.8` | Per-restart cycle length multiplier (SGDR only); `< 1.0` shortens successive cycles |

### Batching and Precision

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `batch_size` | int | ✅ required | Micro-batch size per GPU per step |
| `gradient_accumulation_steps` | int | ✅ required | Steps to accumulate before optimizer update; effective batch = `batch_size × accum_steps × num_gpus` |
| `precision_schedule` | `[[start, end, dtype], ...]` | ✅ required | List of `[start_step, end_step, dtype]` tuples; `-1` for end = "until end of run"; dtype: `"bf16"`, `"fp8"`, `"mixed"` |

### Distributed Training

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `use_distributed` | bool | `false` | Enable multi-GPU training; requires launching with `torchrun` |
| `distributed_backend` | `"ddp"` \| `"fsdp"` | `"ddp"` | DDP = data parallel (model replicated per GPU); FSDP = model sharded across GPUs (use when model > per-GPU VRAM) |

### Attention and Compilation

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `attention_backend` | string | `"standard"` | `"flash"` (recommended), `"sage"`, `"xformers"`, `"standard"`; see [OPTIMIZATION.md](../docs/OPTIMIZATION.md) |
| `use_torch_compile` | bool | `false` | AOT kernel fusion via `torch.compile`; incompatible with `flash` and `sage` backends |
| `torch_compile_mode` | string \| `null` | `null` | `torch.compile` mode: `null` = framework default; `"reduce-overhead"` = CUDA graphs; `"max-autotune"` = kernel auto-tuning (slow first run); `"max-autotune-no-cudagraphs"` |
| `torch_compile_fullgraph` | bool | `false` | Require a single contiguous graph (errors on graph breaks); leave `false` for partial compilation |
| `torch_compile_dynamic` | bool | `false` | Track dynamic shapes; useful for variable-length sequences at the cost of reduced kernel specialization |
| `selective_checkpointing` | bool | `false` | Activation checkpointing — recompute activations during backward to trade compute for memory |
| `selective_checkpointing_mode` | `"full"` \| `"ffn"` \| `"attn_res_fused"` | `"full"` | `"full"` recomputes each sublayer (norm + op + residual add); `"ffn"` runs attention eagerly and recomputes only the FFN sublayer; `"attn_res_fused"` is `block_attn`-only — fuses ar() + sublayer + residual add per sublayer so the mixer output and sublayer delta are both recomputed, saving ~2×(B,T,D) per sublayer beyond `"full"` |
| `selective_checkpointing_interval` | int | `1` | Checkpoint every N-th block; `2` = every other block, halving recompute cost while keeping savings on checkpointed blocks |

### Checkpointing and Logging

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `checkpoint_interval` | int | `1000` | Save a checkpoint every N steps |
| `keep_last_n_checkpoints` | int | `3` | Delete older checkpoints; keeps the N most recent |
| `resume_from_checkpoint` | string | `null` | Path to checkpoint `.pt` file to resume from |
| `resume_optimizer_state` | bool | `true` | Restore optimizer state (momentum buffers, step counters) from checkpoint; set `false` to restart optimizer fresh while keeping model weights |
| `resume_scheduler_state` | bool | `true` | Restore LR scheduler state from checkpoint; set `false` to restart schedule from step 0 |
| `resume_lr_hold_steps` | int | `0` | Extra warmup-hold steps inserted after resume before the scheduler resumes decay; useful when adapting a checkpoint to new data |
| `eval_interval` | int | `100` | Run validation every N steps |
| `eval_max_batches` | int | `0` | Cap validation batches per eval; `0` = no limit; set to ~64 for large val sets |
| `eval_on_test` | bool | `false` | Also evaluate on `data.test_dataset_path` at each eval interval |
| `log_interval` | int | `10` | Print training stats every N steps |

### Memory Optimization

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `use_chunked_loss` | bool | `false` | Chunked LM-head + CE: projects hidden states per time-chunk inside `torch.utils.checkpoint` so peak logit memory is `O(B·chunk_size·V)` instead of `O(B·T·V)`; numerically equivalent to the standard path |
| `loss_chunk_size` | int | `256` | Time-axis chunk size for `use_chunked_loss`; smaller = lower peak memory, more checkpoint recompute |
| `z_loss_weight` | float | `0.0` | PaLM-style logit-scale regularizer: adds `z_loss_weight × mean(logsumexp(logits)²)` to training loss to prevent logit explosion; `1e-4` is a good starting value; excluded from validation loss |

### External Benchmarks (MCQ Harness)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `benchmark_tasks` | `[string, ...]` | `[]` | Task names to run periodically (currently: `"hellaswag"`, `"piqa"`, `"arc_easy"`) |
| `benchmark_eval_interval` | int | `0` | Run benchmarks every N train steps; `0` disables periodic benchmark runs |
| `benchmark_max_examples` | int | `128` | Max examples per task per run (keep small during training for low overhead) |
| `benchmark_split` | string | `"validation"` | Split passed to dataset loader (e.g. `validation`) |
| `benchmark_length_normalize` | bool | `true` | Use average token log-probability per option (reduces length bias) |

When enabled, benchmark snapshots are appended to `output_dir/benchmark_curve.jsonl` and also logged to TensorBoard under `eval_benchmark/*`.

### Early Stopping and Regularization

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `early_stopping_patience` | int | `null` | Stop if val loss doesn't improve for N eval intervals; `null` = disabled |
| `early_stopping_min_delta` | float | `0.0` | Minimum improvement to reset patience counter |
| `label_smoothing` | float | `0.0` | Smooth CE targets; `0.1` typical; reduces overconfidence |
| `moe_balance_loss_weight` | float | `0.0` | Auxiliary load-balance loss weight for MoE routing (Phase 6+) |

---

## [inference]

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `max_new_tokens` | int | `64` | Tokens to generate per prompt |
| `temperature` | float | `0.0` | Sampling temperature; `0.0` = greedy argmax |
| `top_p` | float | `0.0` | Nucleus sampling threshold; `0.0` = disabled |
| `top_k` | int | `0` | Top-k sampling; `0` = disabled |
| `device` | `"auto"` \| `"cpu"` \| `"cuda"` | `"auto"` | Inference device; `"auto"` picks CUDA if available |
| `use_kv_cache` | bool | `true` | Enable KV-cache for autoregressive generation (Phase 5+) |
| `kv_cache_dtype` | `"fp8"` \| `"bf16"` | `"bf16"` | KV-cache precision (Phase 5+) |

---

## [data]

### Dataset Paths

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `dataset_path` | string | ✅ required | Path to training `.npy` token file |
| `validation_dataset_path` | string | `null` | Explicit val set; if omitted, `validation_split` carves one from train |
| `test_dataset_path` | string | `null` | Optional held-out test set (used when `training.eval_on_test = true`) |
| `packing_metadata_path` | string | `null` | Optional train split `*_packing_meta.npz`; enables packed document-boundary masks and boundary-crossing loss masking |
| `validation_packing_metadata_path` | string | `null` | Optional validation split packing metadata; use with `validation_dataset_path` |
| `test_packing_metadata_path` | string | `null` | Optional test split packing metadata; use with `test_dataset_path` |
| `validation_split` | float | `0.1` | Fraction of train tokens used as val when no explicit val path given; ignored if `validation_dataset_path` is set |

### Tokenizer

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `tokenizer_name` | string | ✅ required | Logical name; used for display and factory dispatch |
| `tokenizer_backend` | `"unigram"` \| `"gpt2_bpe"` \| `"char"` \| `"char_utf8"` | `"char"` | Tokenizer implementation; `"unigram"` is canonical for P3+ |
| `tokenizer_mode` | `"utf8"` \| `"utf16"` \| `"utf32"` \| `"codepoint"` | ✅ required | Character encoding mode (used by char/char_utf8 backends) |
| `tokenizer_vocab_size` | int | ✅ required | Vocabulary size; must match `model.vocab_size` |
| `unigram_model_path` | string | `null` | Path to SentencePiece `.model` file (required when `tokenizer_backend = "unigram"`) |
| `tokenizer_vocab_path` | string | `null` | Path to custom BPE vocab `.json` (used when `tokenizer_backend = "gpt2_bpe"` with custom vocab) |
| `seed` | int | `42` | Seed for validation split and sequence offset augmentation |

### DataLoader

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `max_length` | int | `512` | Sequence length per sample; must be ≤ `model.max_seq_length` |
| `num_workers` | int | `4` | DataLoader worker processes; `4` per GPU recommended |
| `prefetch_factor` | int | `2` | Batches prefetched per worker; increase for I/O-bound training |
| `pin_memory` | bool | `true` | Pin CPU tensors for faster CPU→GPU transfer (CUDA only) |
| `persistent_workers` | bool | `true` | Keep workers alive between epochs; set `false` for single-pass runs |
| `streaming` | bool | `false` | Reserved for future streaming datasets (Phase 4+) |
| `cache_dir` | string | `"./data/cache"` | Reserved for future chunked token cache (Phase 4+) |
| `num_shards` | int | `1` | Reserved for future sharded datasets (Phase 4+) |

---

## Complete Example

### Phase 3 baseline

```toml
[experiment]
name = "maxllm-p3-unigram-1024h-10l"
output_dir = "./outputs/ephemeral/p3-example"

[model]
hidden_size = 1024
num_heads = 16
vocab_size = 8192
max_seq_length = 2048
num_layers = 10
dropout = 0.0

[training]
batch_size = 12
gradient_accumulation_steps = 10
max_steps = 12000
warmup_steps = 500
learning_rate = 0.004
weight_decay = 0.05
betas = [0.9, 0.95]
epsilon = 1e-8
gradient_clip_norm = 0.5
precision_schedule = [[0, -1, "bf16"]]
scheduler_type = "wsd"
wsd_stable_fraction = 0.55
wsd_decay_fraction = 0.35
wsd_decay_shape = "sqrt"
min_lr_ratio = 0.05
use_distributed = true
distributed_backend = "ddp"
attention_backend = "flash"
use_torch_compile = false
checkpoint_interval = 500
eval_interval = 100
eval_max_batches = 64
log_interval = 25

[inference]
temperature = 0.9
top_p = 0.95
max_new_tokens = 1024

[data]
dataset_path = "data/fast/p3_train.npy"
validation_dataset_path = "data/fast/p3_val.npy"
tokenizer_name = "unigram"
tokenizer_mode = "utf8"
tokenizer_vocab_size = 8192
tokenizer_backend = "unigram"
unigram_model_path = "data/fast/p3_tokenizer.model"
max_length = 2048
num_workers = 4
prefetch_factor = 4
```

### Phase 4 — Llama-style (RMSNorm + SwiGLU + RoPE + GQA)

```toml
[experiment]
name = "p4-swiglu-rope"
output_dir = "./outputs/ephemeral/p4-swiglu-rope"

[model]
hidden_size = 1024
num_heads = 16
num_kv_heads = 4               # GQA: 4 KV heads shared across 16 query heads
vocab_size = 8192
max_seq_length = 1024
num_layers = 6
norm_type = "rms"
ffn_type = "swiglu"
intermediate_size = 2816        # swiglu_intermediate_size(1024): 4*1024*2/3 rounded to 256
pos_type = "rope"
rope_base = 10000
attn_type = "mha"               # standard MHA (default; explicit for clarity)

[training]
batch_size = 16
gradient_accumulation_steps = 4
max_steps = 2000
warmup_steps = 400
learning_rate = 0.0018          # RoPE needs lower LR than standard (0.004 causes grad norm spike)
weight_decay = 0.05
betas = [0.9, 0.95]
epsilon = 1e-8
gradient_clip_norm = 1.0
precision_schedule = [[0, -1, "bf16"]]
scheduler_type = "cosine"
min_lr_ratio = 0.1
use_distributed = true
attention_backend = "flash"
use_torch_compile = true
checkpoint_interval = 2000
eval_interval = 100
eval_max_batches = 32
log_interval = 25

[inference]
temperature = 0.9
top_p = 0.95
max_new_tokens = 256

[data]
dataset_path = "data/fast/p3_tiny10_wiki100_owt12_fineweb_unigram8192_20260312_train.npy"
validation_dataset_path = "data/fast/p3_tiny10_wiki100_owt12_fineweb_unigram8192_20260312_val.npy"
tokenizer_name = "unigram"
tokenizer_mode = "utf8"
tokenizer_vocab_size = 8192
tokenizer_backend = "unigram"
unigram_model_path = "data/fast/p3_tiny10_wiki100_owt12_fineweb_unigram8192_20260312_tokenizer.model"
max_length = 1024
num_workers = 4
prefetch_factor = 4
seed = 42
```

### Phase 4 — Sliding Window Attention (Mistral-style)

```toml
[model]
hidden_size = 1024
num_heads = 16
num_kv_heads = 4
vocab_size = 8192
max_seq_length = 4096
num_layers = 6
norm_type = "rms"
ffn_type = "swiglu"
intermediate_size = 2816
pos_type = "rope"
rope_base = 10000
attn_type = "swa"
swa_window_size = 512           # each token attends to at most 512 preceding tokens
```

Flash Attention 2 (`attention_backend = "flash"`) handles `swa_window_size` natively and is
strongly recommended; the fallback constructs an explicit band-diagonal mask in O(T²) memory.

### Phase 4 — Residual Linear Attention

```toml
[model]
hidden_size = 1024
num_heads = 16
num_kv_heads = 4
vocab_size = 8192
max_seq_length = 4096
num_layers = 6
norm_type = "rms"
ffn_type = "swiglu"
intermediate_size = 2816
pos_type = "learned"            # RoPE is incompatible with RLA (use "learned" or "alibi")
attn_type = "rla"               # O(T·D²) ELU+1 kernel + learned input residual
```

RLA is O(T·D²) time vs O(T²·D) for full attention — faster for long sequences when
`T > D`. Memory: O(B·T·H·D²) for the KV outer-product table (~512 MB at B=4, T=2048, H=8,
head_dim=64 in bf16). Not compatible with `pos_type = "rope"` or `"add_rope"`.

---

## Directory Layout

### Phase 5 Wave 0 Policy

- New Phase 5 Wave 0 data-preparation configs should include **full OpenWebText** and **full FineWeb-Edu** as core web sources.
- Include selected companion datasets (for example Wikipedia/Cosmopedia) as explicitly configured for each run.
- Treat Cosmopedia as **synthetic LLM-generated** companion data and keep its contribution explicit in source-level prep stats.
- Apply corpus hygiene thresholds on OpenWebText, English language filtering on OWT/FineWeb-Edu, and cross-source MinHash near-dedup before training.
- Enable sequence packing for short-document corpora in Wave 0 prep runs and record packing metrics (`packed_sequences`, `fill_ratio`) in stats outputs.
- For curriculum-enabled Wave 0 runs, set `curriculum.source_repetition_budget` to cap per-source cross-stage re-exposure.
- Keep existing Phase 3/4 milestone configs unchanged as historical records.

```
config/
  milestones/       Committed, validated configs — one per phase milestone
  ephemeral/        Scratch/experimental configs — gitignored, throw away freely
  README.md         This file
```

**Rule**: Never commit a config to `milestones/` until the results are validated and the user explicitly confirms it. All in-progress configs belong in `ephemeral/`. See [LESSONS.md](../.github/LESSONS.md#l009) L009 and L013.

---

## Config System API

### Adding Fields

**Optional field (non-breaking)** — add with a default; old TOML files and checkpoints still work:

```python
@dataclass(frozen=True)
class ModelConfig:
    new_field: float = 0.0  # Has default → no __version__ increment
```

**Required field (breaking)** — increment `__version__` and omit default:

```python
@dataclass(frozen=True)
class ModelConfig:
    __version__: ClassVar[int] = 2  # Increment!
    new_required_field: int          # No default → old TOML files fail validation
```

See [DESIGN.md](../docs/DESIGN.md#config-system-version-aware-evolution) for when to increment.

### Checkpoint Versioning

```python
from src.training.train import save_checkpoint, load_checkpoint

# Save (auto-includes config versions)
path = save_checkpoint(model, optimizer, step=100, output_dir="outputs")

# Load
step = load_checkpoint(path, model, optimizer)
```

### Test Fixtures

Always use the schema-aware builders from `tests/conftest.py`; they auto-fill defaults and survive field additions without test refactoring:

```python
from tests.conftest import build_model_config, build_training_config, build_data_config

# ❌ Brittle — breaks every time a field is added
config = ModelConfig(hidden_size=64, num_layers=1, ...)

# ✅ Resilient — only specify what you're testing
config = build_model_config(hidden_size=64, num_layers=1)
```

Available: `build_model_config`, `build_training_config`, `build_data_config`, `build_inference_config`.

### Source Files

| File | Purpose |
|------|---------|
| [src/config/model.py](../src/config/model.py) | `ModelConfig` |
| [src/config/training.py](../src/config/training.py) | `TrainingConfig` |
| [src/config/data.py](../src/config/data.py) | `DataConfig` |
| [src/config/inference.py](../src/config/inference.py) | `InferenceConfig` |
| [src/training/train.py](../src/training/train.py) | `save_checkpoint()`, `load_checkpoint()` |
| [tests/conftest.py](../tests/conftest.py) | Schema-aware config builders |
