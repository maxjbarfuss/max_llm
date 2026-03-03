# Data Preparation Checklist — Ready When 50m_data Completes

## ✅ Already Created (Ready to Use)

### Documentation
- ✅ `DATA_SCALING_STRATEGY.md` — Full strategy, rationale, decision tree
- ✅ `PHASE_3_QUICK_START.md` — Baseline & training reference
- ✅ `PHASE_3_HYPERPARAMETER_EXPLORATION.md` — Theory & scaling laws

### Scripts & Infrastructure
- ✅ `scripts/data/prepare_complementary.py` — Downloads & tokenizes:
  - ArXiv abstracts (~2M tokens, technical domain)
  - Stack Exchange Q&A (~3M tokens, high-quality discussions)
  - GitHub code (~5M tokens, programming paradigms)
- ✅ `scripts/data/mix_interleaved_pages.py` — Mix datasets at arbitrary ratios
- ✅ `scripts/data/run_data_prep.py` — General data prep pipeline
- ✅ `scripts/data/prepare_training_data.py` — Full pipeline automation

### Experiment Configs
- ✅ `config/milestones/p3_optimized_50m_data.toml` — (now running)
- ✅ `config/milestones/p3_optimized_100m_diverse.toml` — (ready if 50m succeeds)
- ✅ `config/milestones/p3_optimized_512h_50m.toml` — (for later, combined scaling)

### Launcher Scripts
- ✅ `scripts/run_p3_50m_data.sh` — (running now)
- ✅ `scripts/run_p3_100m_diverse.sh` — (ready for Stage 2)
- ✅ `scripts/run_p3_512h_50m.sh` — (ready for Stage 3, if needed)

---

## 🎯 Decision Tree (When 50m_data Completes)

### If 50m_data achieves **loss < 3.9** ✅

→ **DATA WAS BOTTLENECK** — Proceed with Stage 2

**Immediate actions**:
1. Check final loss in `outputs/p3-optimized-50m-data/loss_curve.csv`
2. Document improvement: `(4.31 - final_loss) / 4.31 × 100%`
3. Prepare Stage 2:
   ```bash
   python scripts/data/prepare_complementary.py --source all
   # Wait ~30-60 min for downloads + tokenization

   python scripts/data/mix_interleaved_pages.py \
     --wikitext data/fast/wikitext_50m_tokens_bpe_gpt2.npy \
     --tinystories data/fast/tinystories_5m_tokens_bpe_gpt2.npy \
     --output data/fast/interleaved_wikitext_tinystories_arxiv_stackexchange_code_100m_tokens_bpe_gpt2.npy

   bash scripts/run_p3_100m_diverse.sh
   ```
4. Monitor Stage 2 (100M diverse): target loss < 3.5

**Parallel option** (if GPU available):
- While Stage 2 runs, prepare Stage 3 config
- Stage 3: 512H×8L model + 50M data (combined scaling)

---

### If 50m_data achieves **loss 3.9-4.1** ⚠️

→ **MARGINAL IMPROVEMENT** — Data helps but isn't sole bottleneck

**Next options** (priority order):
1. Run 100M diverse anyway (diversity might help more than quantity)
2. Investigate hyperparameter interactions (LR schedules, warmup for larger data)
3. Test architecture tweaks (layer norm placement, activation functions)
4. Consider curriculum learning (easy → hard examples)

**Experiment**: `scripts/run_p3_100m_diverse.sh` with lower learning rate (try 0.0002)

---

### If 50m_data achieves **loss ≥ 4.1** ❌

→ **DATA NOT PRIMARY BOTTLENECK** — Problem is architecture/training dynamics

**Investigate**:
1. Gradient flow issues (check gradient statistics in output)
2. Attention mechanism quality (consider alternative attention types)
3. Layer normalization placement (try pre-LN vs post-LN)
4. Initialization strategies
5. Learning rate schedules (try different warmup profiles)

**Don't** proceed with more data until root cause found.

---

## 📊 Monitoring 50m_data (Now Running)

**Current directory**: `outputs/p3-optimized-50m-data/`

**Check progress**:
```bash
# Live monitoring (every 10 seconds)
watch -n 10 'tail -3 outputs/p3-optimized-50m-data/loss_curve.csv'

# Quick analysis
python << 'EOF'
import re, numpy as np
with open('outputs/p3-optimized-50m-data/loss_curve.csv', 'r') as f:
    lines = [l.strip() for l in f if l.strip() and re.match(r'^\d+', l.strip())]
    if len(lines) > 1:
        latest = lines[-1].split(',')
        early = lines[0].split(',')
        print(f"Step {latest[0]}: loss={float(latest[1]):.4f}, ppl={float(latest[2]):.1f}")
        print(f"Progress: {len(lines)} steps logged")
        baseline_loss = 4.31
        improvement = 100 * (baseline_loss - float(latest[1])) / baseline_loss
        print(f"vs baseline: {improvement:+.1f}%")
EOF
```

**Expected behavior**:
- Initial loss: ~10.9 (random init)
- After warmup (100-200 steps): ~6-7 (fast drop)
- Main training (500-5000 steps): gradual descent toward plateau
- Early stopping: triggered when validation loss plateaus (~5-10 steps without improvement)

**Typical timeline**: 30-40 GPU hours total (RTX 4090/3090 Ti pair)

---

## 🚀 Files Summary

```
Project Root/
├── DATA_SCALING_STRATEGY.md              ← Full strategy doc (START HERE)
├── DATA_PREP_CHECKLIST.md               ← This file
├── scripts/
│   └── data/
│       ├── prepare_complementary.py      ← New: ArXiv, StackEx, Code
│       ├── mix_interleaved_pages.py      ← Existing: Dataset mixer
│       └── run_data_prep.py              ← Existing: General prep
├── scripts/run_p3_100m_diverse.sh        ← New: Stage 2 launcher
├── config/milestones/
│   ├── p3_optimized_50m_data.toml        ← Now running
│   ├── p3_optimized_100m_diverse.toml    ← Ready: Stage 2
│   └── p3_optimized_512h_50m.toml        ← Ready: Stage 3
└── outputs/p3-optimized-50m-data/        ← Live outputs
    ├── loss_curve.csv                    ← Monitor this
    └── checkpoint.pt                     ← Final weights
```

---

## ⏱️ Time Estimates

| Stage | Config | Data | Duration | GPU Hours | Status |
|-------|--------|------|----------|-----------|--------|
| Baseline | `p3_bpe_convergence.toml` | 10M | ~5h | ~10 | ✅ Done |
| Stage 1 | `p3_optimized_50m_data.toml` | 50M | ~15h | ~30 | 🟢 Running |
| Stage 2 | `p3_optimized_100m_diverse.toml` | 100M | ~25h | ~50 | ⏳ Ready (gate: loss<3.9) |
| Stage 3 | `p3_optimized_512h_50m.toml` | 50M | ~25h | ~50 | ⏳ Ready (combined test) |
| **Total** | | 100M | ~70h | ~140 | (both Stage 2 paths) |

**GPU efficiency**: ~1.4 GPU hours per 1M tokens (RTX 4090/3090 Ti pair)

---

## 🎓 Key Insights So Far

1. **Model scaling ≠ guaranteed improvement** (512H×8L failed → loss 6.75)
2. **Data quantity matters** (10M → 50M should show improvement)
3. **Data diversity also matters** (theory: ArXiv+StackEx+Code boost by 5-15%)
4. **Hyperparameters are model-size dependent** (LR, warmup must scale)
5. **Early stopping saves wasted compute** (validation plateau detection)

---

## Next Agent Instructions

1. **Monitor 50m_data** (check loss every 30 min)
2. **When loss plateaus** (early stopping triggered):
   - Check final loss vs baseline
   - Update MEMORY.md with results
   - Decide: Stage 2 or pivot?
3. **If Stage 2 launching**:
   - Run `prepare_complementary.py` first
   - Mix datasets with proper ratios
   - Launch `scripts/run_p3_100m_diverse.sh`
   - Document decision process

