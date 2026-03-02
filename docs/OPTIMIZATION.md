# Training Optimization Guide

**Purpose**: Maximize training throughput and minimize memory usage on dual-GPU hardware (RTX 4090 + RTX 3090 Ti, 48GB VRAM).

## Quick Start

```toml
[training]
attention_backend = "flash"           # flash|sage|xformers|standard
use_torch_compile = false            # incompatible with flash/sage
batch_size = 8
gradient_accumulation_steps = 2
precision_schedule = [[0, -1, "bf16"]]

[data]
num_workers = 4                      # 8 for dual-GPU
prefetch_factor = 2
pin_memory = true
persistent_workers = false           # false for single-pass LLM training
```

**Single-GPU**: `python -m src.training.train --config config/your_config.toml`
**Multi-GPU**: `./scripts/train_ddp.sh config/your_config.toml 2`

## Attention Backends

Choose one based on your constraints:

| Backend | Speed | Memory | torch.compile | Use When |
|---------|-------|--------|---------------|----------|
| **flash** | 2-4x | Best | ❌ No | Default for training (recommended) |
| **sage** | 2-4x | Best | ❌ No | A/B test vs Flash Attention |
| **xformers** | 1.5-2x | Good | ✅ Yes | Need torch.compile compatibility |
| **standard** | 1x | Baseline | ✅ Yes | Debugging, CPU, fallback |

**Critical constraint**: `flash` and `sage` cannot use `torch.compile`. Choose one path:
- **Path A**: `attention_backend = "flash"` + `use_torch_compile = false` (recommended)
- **Path B**: `attention_backend = "xformers"` + `use_torch_compile = true`

### Installation

```bash
pip install flash-attn --no-build-isolation  # Flash Attention 2
pip install sageattention                    # Sage Attention (optional)
pip install xformers                         # xFormers (already installed)
```

All backends auto-fallback to `standard` if unavailable. Training never fails on missing dependencies.

### Benchmark Backends

```bash
python scripts/benchmark_attention.py \
  --d-model 256 \
  --num-heads 8 \
  --seq-len 256 \
  --batch-size 8 \
  --backends flash,xformers,standard
```

## Multi-GPU Training (DDP)

**Status**: ✅ Implemented, tested with 2 GPUs

Enables near-linear scaling via data parallelism. Expected: ~1.8x throughput (30-35% sync overhead).

### Usage

```bash
# Using launcher script (recommended, baseline run)
./scripts/train_ddp.sh config/milestones/p3_baseline.toml 2

# Optional validation-only config
./scripts/train_ddp.sh config/tests/p3/p3_ddp.toml 2

# Or manually with torchrun
source .venv/bin/activate
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m src.training.train \
    --config config/your_config.toml \
    --distributed
```

### Config Adjustments

```toml
[data]
num_workers = 8  # 4 per GPU (2 × 4)
```

### How It Works

1. `torchrun` spawns one process per GPU
2. Each process gets unique rank (0, 1, ..., N-1)
3. Model replicated to each GPU
4. Data automatically sharded across processes
5. Gradients synchronized after backward pass
6. Only rank 0 saves checkpoints

### Expected Performance

- **Single GPU** (RTX 4090): ~150-170k tokens/sec
- **Dual GPU** (4090 + 3090 Ti): ~270-300k tokens/sec total (~1.8x)

### Troubleshooting

**"Address already in use"**: Change port with `export MASTER_PORT=29501`

**Training hangs at init**: Ensure both GPUs are idle (`nvidia-smi`)

**Loss divergence**: Verify same seed across all processes

**OOM**: Reduce `batch_size` or increase `gradient_accumulation_steps`

## DataLoader Optimization

**Purpose**: Eliminate CPU data loading bottleneck by parallelizing with GPU training.

**Status**: ✅ Production-ready

```toml
[data]
num_workers = 4              # Start with 4, increase if GPU util < 95%
prefetch_factor = 2          # Batches to prefetch per worker
pin_memory = true            # Faster CPU→GPU transfer (CUDA only)
persistent_workers = false   # true for multi-epoch, false for single-pass
```

**Impact**: ~1.2-1.5x throughput, prevents training hangs

**Guidelines**:
- Single-GPU: `num_workers = 4`
- Dual-GPU: `num_workers = 8`(4 per GPU)
- Reduce if OOM or high CPU usage

## torch.compile

**Purpose**: Kernel fusion and graph optimization for 30-40% speedup.

**Status**: ✅ Implemented (xformers/standard only)

```toml
[training]
attention_backend = "xformers"  # or "standard"
use_torch_compile = true
```

**Constraints**:
- ❌ **Cannot** be used with `flash` or `sage` attention
- Adds ~10-30s compilation overhead on first step
- Requires PyTorch 2.x

## Gradient Accumulation

**Purpose**: Simulate larger batch sizes without OOM.

```toml
[training]
batch_size = 8                      # Micro-batch per step
gradient_accumulation_steps = 4     # Effective batch = 8 × 4 = 32
```

Accumulates gradients across multiple forward/backward passes before optimizer step.

## Mixed Precision (AMP)

**Purpose**: Reduce memory usage and increase throughput via bf16 computation.

**Status**: ✅ Automatic when precision_schedule specified

```toml
[training]
precision_schedule = [[0, -1, "bf16"]]  # bf16 from step 0 to end
```

**Expected**: ~1.5-2x memory reduction, ~1.2-1.4x throughput (requires Ampere+ GPU for bf16)

## Recommended Configurations

### Phase 3: Single-GPU Training
```toml
[training]
attention_backend = "flash"
use_torch_compile = false
batch_size = 8
gradient_accumulation_steps = 2
precision_schedule = [[0, -1, "bf16"]]

[data]
num_workers = 4
prefetch_factor = 2
pin_memory = true
persistent_workers = false
```
**Expected**: ~150-180k tokens/sec (RTX 4090, 256 dim, 4 layers)

### Phase 4: Multi-GPU Training
```toml
[training]
attention_backend = "flash"
distributed_backend = "ddp"
batch_size = 16
gradient_accumulation_steps = 2

[data]
num_workers = 8  # 4 per GPU
```
**Expected**: ~270-300k tokens/sec total (2 GPUs)

### Alternative: torch.compile Path
```toml
[training]
attention_backend = "xformers"
use_torch_compile = true
```
**When**: Flash Attention unavailable or validating compile benefits

## Profiling

Enable detailed profiling for model analysis:

```bash
python -m src.training.train \
    --config config/your_config.toml \
    --profile
```

Output: Model size, parameter count, memory usage per layer

## Performance Hierarchy

Measured on RTX 4090 (256 hidden, 4 layers, seq_len=256):

1. **Flash Attention + Workers**: ~175k tokens/sec ⚡⚡⚡
2. **xFormers + compile + Workers**: ~145k tokens/sec ⚡⚡
3. **Standard + compile + Workers**: ~110k tokens/sec ⚡
4. **Baseline (no optimizations)**: ~80k tokens/sec ⚫

*Speedups increase with larger models and longer sequences*

## Common Issues

### Training hangs after initialization
**Cause**: `num_workers = 0` blocks on CPU
**Fix**: Set `num_workers = 4`

### "Flash Attention requested but not available"
**Cause**: Not installed
**Fix**: `pip install flash-attn --no-build-isolation`
**Note**: Auto-falls back to standard (training continues)

### torch.compile error with flash
**Cause**: Known incompatibility
**Fix**: Set `use_torch_compile = false` or switch to `attention_backend = "xformers"`

### OOM with num_workers > 0
**Cause**: Each worker allocates memory
**Fix**: Reduce `num_workers`, `batch_size`, or increase `gradient_accumulation_steps`

### DDP: NCCL error
**Fix**: Ensure CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`

## Future Optimizations

- [ ] Memory-mapped datasets (100M+ tokens, Phase 4)
- [ ] FSDP for model sharding (1B+ params, Phase 5)
- [ ] KV-cache for inference
- [ ] Automatic backend selection based on hardware

## References

- Flash Attention: https://arxiv.org/abs/2205.14135
- Sage Attention: https://github.com/thu-ml/SageAttention
- xFormers: https://github.com/facebookresearch/xformers
- torch.compile: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- DDP: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
