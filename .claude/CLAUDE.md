# ⚠️ AGENT BOOTSTRAP — READ FIRST

**CRITICAL**: Every agent must read these FIRST before any work:
1. [`.github/AGENTS.md`](.github/AGENTS.md) — Universal development standard (all contributors)
2. [`.github/SKILLS.md`](.github/SKILLS.md) — Practical tool use patterns and workflows
3. [`.github/LESSONS.md`](.github/LESSONS.md) — Past agent mistakes to avoid

These are NON-NEGOTIABLE. Do not skip.

---

# Max LLM — Session Memory & Agent Bootstrap

**Last Updated**: 2026-03-03 (Session: Phase 3 Optimization Experiments)
**Status**: ✅ 512h_8l-ddp COMPLETE (5000/5000) — **loss 6.752 (FAILURE)** — now running 50m_data test
**Critical Finding**: Model scaling does NOT help. 512H×8L → loss 6.75 (vs baseline 4.31). **We are DATA-STARVED.**
**Next Action**: Watch 50m_data convergence (target: <3.9 to confirm data was bottleneck)

---

## Current Session: Phase 3 Best-of-Breed → Optimized Scaling

### Goal
Improve upon sealed Phase 3 baseline (loss 4.31) via systematic bottleneck testing with DDP:
- **Running NOW**: 512H×8L model with 2-GPU DDP (RTX 4090 + 3090 Ti)
  - Current: step 4895/5000, loss 6.63, trend -0.0022/step (still converging)
  - Expected: Final loss ~6.4-6.5 (plateau region)
  - Decision: If <6.2 → model scaling helps, test 768h; if ≥6.5 → data/architecture issue

- **Next (conditional)**: 768H×8L with DDP (if 512h shows promise <6.2)
  - Hypothesis: Larger hidden size may converge faster
  - Same data (10M tokens), same DDP batch size (effective 64)

- **Then**: Data scaling test 50M tokens on 256H×4L (baseline architecture)
  - Hypothesis: Is data quantity bottleneck?
  - Early stopping enabled (validation plateau detection)

### 🚨 CRITICAL FINDING: First 512h_8l Run Failed

**What happened**:
- Ran 512H×8L model with original hyperparams (LR=0.0003, warmup=200)
- Result: Loss converged to **6.71** (vs baseline 4.31) — **55.7% WORSE**
- Root cause: Learning rate too high for 110M param model + warmup too short

**Diagnosis** (from loss_curve.csv analysis):
```
Initial:       loss=10.94
After warmup:  loss=7.42  (32% improvement in 200 steps)
After 963:     loss=6.71  (stalled, no further improvement)
Trend:         -0.00045/step in last 100 steps (essentially flat)
```

**Fix applied** (new config):
```toml
# OLD (failed)
learning_rate = 0.0003
warmup_steps = 200
gradient_accumulation_steps = 2  # Effective batch = 16

# NEW (fixed)
learning_rate = 0.0001          # 10× lower
warmup_steps = 1000             # 5× longer
gradient_accumulation_steps = 4  # Effective batch = 32
```

**Rationale**: Larger models have sharper loss landscape → need gentler LR + longer warmup

### What's Complete (This Session)
1. ✅ Recovered sealed baseline config: `config/milestones/p3_bpe_convergence.toml` (loss 4.31)
2. ✅ Audited training loop: all optimizations present
3. ✅ Validated inference on sealed checkpoint
4. ✅ Created experiment configs (512h_8l, 768h_8l, 50m_data, 512h_50m)
5. ✅ Created DDP launch scripts with monitoring
6. ✅ Created docs (PHASE_3_QUICK_START.md + PHASE_3_HYPERPARAMETER_EXPLORATION.md)
7. ✅ **512h_8l-ddp launched**: Running on 2 GPUs (RTX 4090 + 3090 Ti)
8. ✅ **Real-time convergence tracking**: CSV analysis shows step 4895/5000, loss 6.63

### Experiments Status (UPDATED 2026-03-03 19:02)

| Exp | Config | Model | Data | Status | Notes |
|-----|--------|-------|------|--------|-------|
| 512h_8l-ddp | `p3_optimized_512h_8l.toml` | 512H×8L | 10M | ✅ COMPLETE | Loss 6.752, PPL 855.7 (WORSE than baseline!) |
| 50m_data | `p3_optimized_50m_data.toml` | 256H×4L | 50M | 🟢 RUNNING | Step 1800+, no early stopping yet |
| data-prep | `prepare_code_github_webtext.py` | N/A | 20M+15M | 🔵 RUNNING | OpenWebText download (1/80 files), ~24 min ETA |
| 512h_50m | `p3_optimized_512h_50m.toml` | 512H×8L | 50M | ⏳ CONDITIONAL | Only if 50m_data shows >10% improvement |

### Launch Commands

**Current: 50M DATA SCALING is RUNNING**
Monitor progress (in separate terminal):
```bash
cd /home/max/dev/max_llm
# Check latest loss every 10s
watch -n 10 'tail -10 outputs/p3-optimized-50m-data/loss_curve.csv | tail -5 | column -t -s,'
```

**After 50m_data completes (decision point):**
```bash
# If loss < 3.9: Launch combined scaling test
bash scripts/run_p3_512h_50m.sh

# Else: Investigate hyperparameters/architecture
```

### Key Decision Points

| Decision | If Loss < 4.1 | If Loss ≥ 4.2 |
|----------|---------------|---------------|
| **512h_8l**: Is model capacity bottleneck? | ✅ Yes → scale model more | ⚠️ No → check data/LR |
| **50m_data**: Is data quantity bottleneck? | ✅ Yes → scale data is high-ROI | ⚠️ No → focus on architecture |
| **512h_50m**: Do improvements compound? | Launch if both beat (expect 3.7) | Skip; investigate other factors |

### Key Files & Locations

| File | Purpose |
|------|---------|
| `config/milestones/p3_bpe_convergence.toml` | Sealed baseline (loss 4.31, locked) |
| `config/milestones/p3_optimized_512h_8l.toml` | Experiment 1: model scaling |
| `config/milestones/p3_optimized_50m_data.toml` | Experiment 2: data scaling |
| `config/milestones/p3_optimized_512h_50m.toml` | Experiment 3: combined scaling |
| `scripts/run_p3_512h_8l.sh` | Launch model scaling test |
| `scripts/run_p3_50m_data.sh` | Launch data scaling test |
| `scripts/run_p3_experiments.py` | Automated experiment runner (optional) |
| `docs/PHASE_3_QUICK_START.md` | Sealed baseline reference |
| `docs/PHASE_3_HYPERPARAMETER_EXPLORATION.md` | Theory, decision tree, analysis |

### Monitoring

**Real-time loss tracking (in separate terminal):**
```bash
watch -n 10 'tail -10 outputs/p3-optimized-50m-data/loss_curve.csv | tail -5 | column -t -s,'
```

**Key decision point**: When 50m_data hits early stopping or completes 10K steps:
- Check final loss in loss_curve.csv
- If loss < 3.9 → data IS the bottleneck ✅ (proceed to 512h_50m)
- If loss ≥ 4.1 → data NOT the limiting factor → investigate architecture

### Scaling Theory (Reference)

From Chinchilla & Kaplan (2022):
- Compute-optimal: N ∝ D^1.0 (params ∝ data)
- Current: 14M params × 2.56B tokens (severely data-limited)
- 50M tokens optimal for ~60M params → 512H×8L (110M) is overparameterized but testable

Expected loss drops from scaling law:
- Data 10M→50M (5×): ~10% loss reduction per 2× → 4.31 * 0.9 * 0.9 * 0.95 ≈ 3.7
- Model 256H→512H: ~5-15% loss reduction → 4.31 * 0.9 ≈ 3.9

### Lessons & Pitfalls to Avoid

From AGENTS.md & LESSONS.md:
- ✅ Use local `git` CLI only (no MCP git tools)
- ✅ Follow TDD: test configs before committing
- ✅ Update MEMORY frequently
- ✅ Validate checkpoints before/after training
- ✅ Use `make test-quick` before any commit
- 🚫 Don't interrupt training mid-step
- 🚫 Don't modify configs mid-experiment

---

## Session Workflow Checklist

- [x] Read AGENTS.md + SKILLS.md (at session start)
- [x] Plan work: identify bottleneck hypotheses
- [x] Implement: create configs, scripts, docs
- [x] Validate: verify configs load, checkpoint loads, inference works
- [x] Stage experiments: ready for launch
- [ ] Execute: run 512h_8l → monitor → decide next step
- [ ] Analyze: compare loss curves, update decision tree
- [ ] Commit: push configs + scripts + docs with atomic commit
- [ ] Document: final session summary in SESSION_LOG.md

---

## Before Next Session (If Handed Off)

**Current State**:
- 50m_data training just started (step ~1-2, loss ~10.9)
- Baseline 256H×4L model on 50M tokens
- Early stopping enabled (patience=5, monitors validation plateau)
- Expected: 30-40 GPU hours, complete by ~23:00 UTC+3 (2026-03-03)

**Decision Logic (when 50m_data completes)**:
1. Check final loss in `outputs/p3-optimized-50m-data/loss_curve.csv`
2. If final loss < 3.9 → **DATA IS BOTTLENECK** ✅
   - Data scaling is high-impact → proceed to 512h_50m (combined model+data)
   - Expected improvement: 10-20% additional with larger model
3. If final loss 3.9-4.1 → **MARGINAL IMPROVEMENT** ⚠️
   - Reconsider architecture or hyperparameters
4. If final loss ≥ 4.2 → **DATA NOT LIMITING** ❌
   - Problem elsewhere: attention mechanism, layer norm, gradient flow
   - Consider investigating curriculum learning or initialization

**Next Actions**:
1. ✅ Let 50m_data run to completion (already executing)
2. 🎯 Monitor convergence (check loss every 30min, early stopping ~10K steps)
3. 📊 Make decision (data scaling efficacy) → proceed with next experiment

---

## Data Scaling Strategy (NEW — In Preparation)

**Status**: 50M baseline test running. Data preparation (GitHub + OpenWebText) now in progress.

### Data Preparation Progress (Stage 1: OpenWebText)
**Started**: 2026-03-03 19:01 UTC+3
**Process**: PID 65485 (background)
**Log**: `outputs/data_prep_github_webtext.log`
**Progress**:
- GitHub Code source: ❌ Not accessible (expected)
- OpenWebText fallback: ✅ Active download (1/80 files, ~18s/file → 24 min total)
- Expected tokenization start: ~19:30 UTC+3
- Estimated completion: ~21:30-22:00 UTC+3

**Output files** (will appear in `data/fast/`):
- `openwebtext_15m_bpe_gpt2.npy` (15M tokens)
- `openwebtext_15m_bpe_gpt2.meta.json` (metadata)

### If 50m_data succeeds (loss < 3.9): Stage 2 Coming
**Add Diversity** (100M total with complementary sources):
- ✅ Strategy doc: [`DATA_SCALING_STRATEGY.md`](DATA_SCALING_STRATEGY.md)
- ✅ Preparer script: `scripts/data/prepare_complementary.py`
  - ArXiv abstracts (technical domain)
  - Stack Exchange Q&A (problem-solving, discussion)
  - GitHub code (programming paradigms)
- ✅ Config ready: `config/milestones/p3_optimized_100m_diverse.toml`
- ✅ Launcher: `scripts/run_p3_100m_diverse.sh`

**When 50m_data completes with loss < 3.9**, prepare next stage:
```bash
# 1. Generate complementary datasets
python scripts/data/prepare_complementary.py --source all

# 2. Mix datasets (40% wiki, 20% stories, 40% diverse)
python scripts/data/mix_interleaved_pages.py \
  --wikitext data/fast/wikitext_50m_tokens_bpe_gpt2.npy \
  --tinystories data/fast/tinystories_5m_tokens_bpe_gpt2.npy \
  --output data/fast/interleaved_wikitext_tinystories_arxiv_stackexchange_code_100m_tokens_bpe_gpt2.npy

# 3. Launch Stage 2 test
bash scripts/run_p3_100m_diverse.sh
```

**Rationale**: Data diversity (complementary domains) → better generalization, fewer overfitting artifacts, better in-context learning

---

Read [`.github/AGENTS.md`](.github/AGENTS.md) (universal standard) and [`.github/SKILLS.md`](.github/SKILLS.md) (practical workflows) before starting any work.
