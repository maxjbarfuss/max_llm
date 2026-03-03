# Phase 3 Hyperparameter Exploration & Scaling Strategy

**Goal**: Improve upon the sealed baseline (loss 4.31) by systematically exploring architecture scale, data size, and training dynamics.

---

## Executive Summary: The Experiments

| Experiment | Model | Data | Effective Batch | Max Steps | GPU Hours | Expected Loss | Test Purpose |
|---|---|---|---|---|---|---|---|
| **Baseline** | 256H x 4L | 10M | 16 | 5K | ~6 | 4.31 | Reference |
| **512h_8l** | 512H x 8L | 10M | 16 | 5K | ~25 | 3.8-4.1? | **Is model capacity the bottleneck?** |
| **50m_data** | 256H x 4L | 50M | 16 | 10K | ~30 | 3.8-4.1? | **Does more data alone help convergence?** |
| **512h_50m** | 512H x 8L | 50M | 32 | 20K | ~100+ | 3.5-3.9? | **Scaling law: combined effect?** |

---

## Hypothesis: Why These Experiments?

### Core Question: Where is the bottleneck?

```
Performance can be limited by:
  1. Model capacity (too small to model the data)
  2. Data quantity (too small to regularize model)
  3. Training dynamics (LR schedule, batch size, regularization)
  4. Optimization ceiling (fundamental limit of architecture on this task)
```

**Current state (baseline)**:
- Loss: 4.31 (perplexity ~74)
- Still declining ~10 steps, suggesting undertrained
- Could converge further if: we increase steps, reduce regularization, or add data

### Experiment 1: `512h_8l` — Model Scaling Test

**Hypothesis**: "We're capacity-limited. Larger model will converge to lower loss."

**What changes**:
- Hidden size: 256 → 512 (2x)
- Layers: 4 → 8 (2x)
- Parameters: ~14M → ~110M (7.8x increase)
- Intermediate FFN: 1024 → 2048 (scales with hidden)

**Why these numbers**:
- 512H is a common architecture size (GPT-2 medium, modern small models)
- 8 layers gives more representational capacity
- Must scale intermediate_size to maintain model quality

**Expected outcome**:
- **If loss improves 0.3-0.5 points**: Model was capacity-limited; larger architectures will help Phase 4
- **If loss stays same or worsens**: Model size isn't the issue; problem is data, learning rate, or architecture type

**Cost**: 25 GPU hours (4x baseline). **Decision point**: If this wins decisively, do 512h_50m.

---

### Experiment 2: `50m_data` — Data Scaling Test

**Hypothesis**: "We're data-limited. More data improves convergence without model changes."

**What changes**:
- Data: 10M tokens → 50M tokens (5x more)
- Max steps: 5K → 10K (doubled to allow learning rate to decay further)
- Early stopping enabled to prevent wasteful training

**Why 50M**:
- 5x increase reveals if loss scales with data quantity
- Available: `interleaved_wikitext_tinystories_50m_pages_bpe_gpt2.npy`
- Standard scaling law: loss ≈ scale^(-α) where α ≈ 0.08 for language models

**Expected outcome from scaling law**:
```
Data scaling:   10M → 50M  (5x)
Predicted loss drop: 10% per 2x → ~4.31 * 0.9 * 0.9 * 0.95 ≈ 3.7
```

- **If final loss ≤ 3.9**: Data was a significant bottleneck; scaling data is high-ROI
- **If final loss stays ≥ 4.0**: Model capacity is the real bottleneck

**Cost**: 30 GPU hours. Enables early stopping to be smarter (stop when validation plateaus).

---

### Experiment 3: `512h_50m` — Full Scaling Experiment

**Hypothesis**: "Combining larger model + more data gives compounding improvements."

**What changes**:
- Model: 512H x 8L (from experiment 1)
- Data: 50M tokens (from experiment 2)
- Batch size: 16 → 32 effective (8 x 4 accumulation) for stability with larger model
- Warmup: 200 → 500 steps (larger models need smoother LR ramp)
- Max steps: 20K (more data + more complexity requires more iterations)

**Why this combination**:
- Tests if improvements compound (better than either alone)
- If 512h saves 0.2 loss and 50m saves 0.3 loss, do we get 0.5 total?
- Or do improvements saturate (diminishing returns)?

**Expected outcome**:
```
If additive:        loss ≈ 4.31 - 0.2 - 0.3 = 3.8
If compounding:     loss ≈ 3.5-3.7 (better than either alone)
If diminishing:     loss ≈ 3.9 (worse than hoped)
```

**Cost**: 100+ GPU hours (longest run). **Use case**: Kick off as background job; check results tomorrow.

---

## Scaling Laws & Theory

### Chinchilla Scaling (compute-optimal model size)

For language models, optimal allocation of compute is roughly:
$$\text{Compute} \propto N^{1.0} \times D^{1.0}$$

Where N = parameters, D = data tokens.

**Current imbalance**:
```
Baseline:
  Model params: ~14M
  Training tokens: 10M × 256 seq_len = 2.56B tokens
  Token/param ratio: 182× (severely data-limited)

Chin...chilla optimum for 2.56B tokens: ~12M params (we're at 14M ✓)
Chinchilla optimum for 12.8B tokens (50M × 256): ~60M params (should scale to 2-4x)
```

**Implication**: 512H model (110M params) is over-parameterized for 50M data, but under-parameterized without it.

---

## Detailed Hyperparameter Rationale

### Learning Rate Tuning

**Baseline**: lr=0.0003, warmup=200

For larger models and more data, consider:
```toml
# Safe (conservative):
warmup_steps = 500   # Longer ramp for 512H model
learning_rate = 0.0002  # Slightly lower for stability

# Aggressive (if baseline diverges):
warmup_steps = 1000
learning_rate = 0.0001  # Much lower, longer training
```

**Why**: Larger models have sharper loss landscape. Longer warmup helps avoid early local minima. Lower LR compensates for larger gradient values in attention heads.

### Batch Size Scaling

**Baseline**: batch=8, grad_accum=2 → effective=16

For 512h_50m run, scaled to:
```toml
batch_size = 8
gradient_accumulation_steps = 4  # ← 32 effective tokens per update
```

**Why**: Larger models benefit from larger batch sizes (more stable gradients). However, GPU memory is still limited, so we use accumulation instead of larger micro-batch.

### Dropout & Regularization

**Current**: dropout=0.1 (reasonable across all experiments)

If 512h_50m overfits:
```toml
dropout = 0.15          # Increase dropout
weight_decay = 0.05     # Increase weight decay (selective)
label_smoothing = 0.1   # Add to reduce overconfidence
```

---

## How to Run Experiments

### Option 1: Run all three (takes ~60+ GPU hours)

```bash
python scripts/run_p3_experiments.py
```

### Option 2: Run sequentially (recommended for debugging)

```bash
# Test 1: Can we beat baseline with just more capacity?
python -m src.training.train --config config/milestones/p3_optimized_512h_8l.toml

# Test 2: How much does data help?
python -m src.training.train --config config/milestones/p3_optimized_50m_data.toml

# Test 3 (optional, if tests 1 & 2 both show gains):
python -m src.training.train --config config/milestones/p3_optimized_512h_50m.toml --distributed
```

### Option 3: Start with quick baseline check

Verify current setup works:
```bash
python -m src.training.train --config config/milestones/p3_bpe_convergence.toml
```

---

## Interpreting Results

### Loss Curve Patterns to Watch

**Good convergence** (target):
```
Step    Loss   Interpretation
100:  8.5     Rapid initial drop (expected)
500:  5.0     Accelerating convergence
1000: 4.5     Smooth decline
5000: 4.0     Plateau with minor fluctuations
```

**Underfitting** (model too small):
```
Step    Loss   Interpretation
1000: 5.5     Slow initial drop
5000: 4.7     Doesn't reach baseline (still improving)
→ Need larger model or more data
```

**Overfitting** (data too small, model too large):
```
Step    Loss   Interpretation
1000: 3.9     Very fast convergence
5000: 4.1     Loss increases after step 3000
Val diverges from train
→ Need more data or regularization
```

**Divergence** (learning rate too high, gradient explosion):
```
Step 100: 5.0
Step 200: NaN
→ Reduce learning rate, increase warmup
```

### Quick Decision Tree

```
Does 512h_8l beat baseline by > 0.2 loss?
  YES → Model was capacity-limited. Do 512h_50m next.
  NO  → Architecture scale isn't the bottleneck.
       → Check: is learning rate schedule optimal?
       → Try longer warmup, lower LR

Does 50m_data beat baseline by > 0.2 loss?
  YES → Data is high-ROI. Expand to 100M+ for Phase 4.
  NO  → Diminishing returns on data. Focus on architecture.

Does 512h_50m beat both individually by compound effect?
  YES → Scaling law works. Scale to Phase 4: 1B+ params, 1T+ tokens
  NO  → Reaching optimization ceiling. Need architecture changes (RMSNorm, RoPE, etc.)
```

---

## File Organization

```
config/milestones/
├── p3_bpe_convergence.toml          ← Sealed baseline
├── p3_optimized_512h_8l.toml        ← Experiment 1
├── p3_optimized_50m_data.toml       ← Experiment 2
└── p3_optimized_512h_50m.toml       ← Experiment 3

outputs/
├── p3-bpe-convergence/              ← Baseline checkpoint
├── p3-optimized-512h-8l/            ← Experiment 1 results
├── p3-optimized-50m-data/           ← Experiment 2 results
├── p3-optimized-512h-50m/           ← Experiment 3 results
└── p3_experiments_results_*.json     ← Summary of all runs
```

---

## Next Steps After Experiments

### If 512h_8l wins (capacity breakthrough):
```
→ Design Phase 4 with 1B+ params
→ Plan multi-GPU training (FSDP)
→ Implement modern layers (RMSNorm, RoPE, SwiGLU)
→ Target 50B+ tokens for Chinchilla optimality
```

### If 50m_data wins (data is key):
```
→ Collect larger datasets (WikiText-103, Common Crawl subset)
→ Increase to 100M-1B tokens for current architecture
→ Focus on data quality and diversity
→ Plan multi-node DDP training
```

### If 512h_50m wins decisively (scaling works):
```
→ Confirm scaling laws hold
→ Push to larger scale: 500M params, 500B tokens
→ Implement advanced techniques (tensor parallelism)
→ Plan for high-performance cluster training
```

### If all plateau below 3.8:
```
→ Current architecture may have hit ceiling
→ Investigate: wrong LR schedule, insufficient regularization, data saturation
→ Consider Phase 4 structural changes: RMSNorm, RoPE, GQA
→ Run ablation studies on attention backends
```

---

## Logging & Monitoring

Each experiment logs to `output_dir/loss_curve.csv`:
```
step,loss,perplexity,lr,tokens_per_sec,gpu_memory_mb,val_loss,test_loss
1,10.8,53388.9,3.00e-05,146153,1567,,
100,8.92,7491.5,1.50e-04,143558,1567,,
...
```

**Plot convergence**:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/p3-optimized-512h-8l/loss_curve.csv')
plt.plot(df['step'], df['loss'], label='Train Loss')
if 'val_loss' in df.columns:
    plt.plot(df['step'], df['val_loss'], label='Val Loss')
plt.xscale('log')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.savefig('convergence.png')
```

---

## Recommended Sequence

1. **Verify baseline works** (30 min)
   ```bash
   python -m src.training.train --config config/milestones/p3_bpe_convergence.toml --max-steps 100
   ```

2. **Run 512h_8l test** (Start now, check in 24 hours)
   ```bash
   python -m src.training.train --config config/milestones/p3_optimized_512h_8l.toml
   ```

3. **Run 50m_data in parallel if possible** (Multi-GPU or second machine)
   ```bash
   python -m src.training.train --config config/milestones/p3_optimized_50m_data.toml --distributed
   ```

4. **Analyze results** (Compare loss curves, decide next direction)

5. **Run 512h_50m if both show gains** (Kick off as 100-hour background job)

---

**Good luck! Track progress at**: `outputs/p3_experiments_results_*.json`
