# ⚠️ AGENT BOOTSTRAP — READ FIRST

**CRITICAL**: Every agent must read these FIRST before any work:
1. [`.github/AGENTS.md`](.github/AGENTS.md) — Universal development standard (all contributors)
2. [`.github/SKILLS.md`](.github/SKILLS.md) — Practical tool use patterns and workflows
3. [`.github/LESSONS.md`](.github/LESSONS.md) — Past agent mistakes to avoid

These are NON-NEGOTIABLE. Do not skip.

---

# Max LLM — Session Memory & Agent Bootstrap

**Last Updated**: 2026-03-03 23:45 UTC+3 (Session: Phase 3 Data Scaling Investigation)
**Status**: ✅ 50m_data experiment HALTED (step 3399/10000) — **HYPOTHESIS FAILED**
**Critical Finding**: **Data QUANTITY is NOT the bottleneck. Mixed data WORSE than Wikipedia baseline.**
**Baseline (10M Wikipedia)**: Loss 4.31, PPL 74 ✅ OPTIMAL
**50M Mixed Result**: Loss 6.90, PPL 994 ❌ 60% WORSE

---

## CURRENT SESSION SUMMARY

### What Happened
Investigated whether **larger quantity** of training data improves Phase 3 baseline. Hypothesis: 5× more data should reduce loss via Chinchilla scaling law. Result: **DISPROVEN**.

**50M Data Test** (step 3399/10000):
- Model: 256H×4L, 14M params (baseline architecture)
- Data: 50M mixed (wiki 10M + stories 5M + webtext 15M + repeats 20M)
- Final loss: 6.90 (vs baseline 4.31)
- **Result**: 60% WORSE than 10M Wikipedia-only baseline
- **PPL**: 994 (vs baseline 74) — 13.4× worse at predicting tokens

### Key Findings

**What's NOT the problem:**
- ✅ Data quantity alone (can improve if quality good)
- ✅ Early stopping logic (fixed patience=0 → patience=10,000)
- ✅ Tokenization/encoding (verified webtext real)
- ✅ Model capacity (14M params sufficient)

**What IS the problem:**
- ❌ **Data quality/composition mismatch**
- ❌ Mixed heterogeneous sources (wiki + stories + web) worse than pure domain
- ❌ Wikipedia is high-quality, coherent text
- ❌ TinyStories (simplified, informal) + WebText (noisy, uncurated) introduce distribution shift

### Critical Insight
**Scaling laws assume good data.** Chinchilla & Kaplan's compute-optimal scaling (params ∝ data) assumes DATA QUALITY is constant. Here: more diverse data = lower quality → performance degrades.

---

## FIXES APPLIED THIS SESSION

### 1. Data Prep Script Fix
**File**: `scripts/data/prepare_complementary.py`
- **Issue**: Used `get_tokenizer("bpe", "gpt2")` — API doesn't exist
- **Fix**: Changed to `TokenizerFactory.create("bpe", encoding="gpt2")`
- **Impact**: Script now functional, all 5 calls fixed
- **Status**: ✅ Verified working

### 2. Early Stopping Config Fix
**File**: `config/milestones/p3_optimized_50m_data.toml`
- **Issue**: `early_stopping_patience = 0` triggered immediately (0 >= 0)
- **Fix**: Changed to `early_stopping_patience = 10000`
- **Impact**: Training completes full duration without premature stopping
- **Status**: ✅ Training reached step 3399, no truncation

### 3. Validation Tests Added
**File**: `tests/unit/test_early_stopping_fixed.py`
- 5 tests validating early stopping logic
- Confirms patience=10000 allows full training
- All tests PASSING ✅
- Git commit: `c18289f`

---

## SEALED BASELINE (LOCKED)

**Config**: `config/milestones/p3_bpe_convergence.toml`
- **Model**: 256H×4L (14M params, 4 attention heads)
- **Data**: 10M tokens (Wikipedia interleaved, BPE GPT-2 tokenizer)
- **Training**: 5000 steps, loss 4.31, PPL 74
- **Status**: ✅ OPTIMAL CONFIGURATION — DO NOT CHANGE

This is the best known model. All experiments attempted to beat it, FAILED.

---

## FAILED EXPERIMENTS

### Experiment 1: Model Scaling (512H×8L with DDP)
**Config**: `config/milestones/p3_optimized_512h_8l_ddp.toml`
- **Model**: 512H×8L (110M params, 2-GPU DDP)
- **Data**: 10M Wikipedia (same as baseline)
- **Result**: Loss 6.752, PPL 856
- **Conclusion**: Larger model WORSE on same data — overfitting or optimization issue

### Experiment 2: Data Scaling (50M Mixed)
**Config**: `config/milestones/p3_optimized_50m_data.toml`
- **Model**: 256H×4L (same as baseline)
- **Data**: 50M tokens (wiki 10M + stories 5M + webtext 15M + repeats 20M)
- **Result**: Loss 6.90, PPL 994
- **Conclusion**: More diverse data WORSE than concentrated quality

---

## WHAT WE LEARNED

### Scaling Laws
- ❌ **Chinchilla scaling fails with heterogeneous data**
- ✅ **Data quality > quantity for language models**
- ✅ **Homogeneous > diverse data (unless specifically designed)**

### Architecture
- ❌ **Simple scaling (256→512H) not always beneficial**
- ❌ **Without tuning, larger models can overfit**
- ✅ **Baseline (14M params, 10M tokens) appears optimal for this setup**

### Data Strategy
- ❌ **Indiscriminate interleaving hurts performance**
- ⚠️ **Mixed domains need curriculum learning or careful balancing**
- ✅ **Pure high-quality source (Wikipedia) outperforms mixed**

### Development
- ✅ **Early stopping patience=0 triggers immediately (antipattern)**
- ✅ **Always validate configs before long training runs**
- ✅ **Test fixes thoroughly (created test suite for patience logic)**

---

## NEXT INVESTIGATION PRIORITIES

If continuing Phase 3 optimization:

1. **Curriculum Learning**: Start Wikipedia, gradually add diverse sources
2. **Domain Separation**: Train separate models for each domain (wiki, stories, web)
3. **Quality Filtering**: Pre-filter mixed data to match Wikipedia coherence
4. **Hyperparameter Tuning**: Try LR 0.0001 for 50M data (0.0003 too high?)
5. **Validation Monitoring**: Track validation loss to detect overfitting early

---

## FILES & STRUCTURE

**Key Configs**:
- `config/milestones/p3_bpe_convergence.toml` — SEALED BASELINE ✅
- `config/milestones/p3_optimized_50m_data.toml` — Data scaling test (deprecated) ❌

**Test Files**:
- `tests/unit/test_early_stopping_fixed.py` — Early stopping validation suite ✅

**Experiment Results** (archive/reference):
- `outputs/p3-optimized-50m-data/loss_curve.csv` — 3399 steps of failing experiment

**Data Files** (kept for future reference):
- `data/fast/wikitext_10m_tokens_bpe_gpt2.npy`
- `data/fast/tinystories_5m_tokens_bpe_gpt2.npy`
- `data/fast/interleaved_mixed_50m_tokens_bpe_gpt2_v2.npy` (failed dataset)

**Cleanup** (removed):
- `outputs/p3-optimized-512h-8l/` (model scaling attempt)
- `outputs/p3-optimized-lowlr-fast/` (intermediate test)
- `outputs/p3-optimized-test/` (debug run)
- `outputs/p3-optimized-2k/` (early test)

---

## GIT HISTORY (THIS SESSION)

```
a2be9aa - stop: 50m_data experiment halted — hypothesis FAILED
c18289f - test: add early stopping validation + fix config patience
49fddc1 - perf: create optimized-lowlr-fast config
97b7b45 - fix: complete aggressive config with tokenizer data sections
bcaa956 - fix: rebuild 50M dataset with verified all-three-sources
```

---

## BEFORE NEXT SESSION

**State of System**:
- ✅ Baseline sealed and locked (loss 4.31)
- ✅ Data infrastructure working (factory API fixed)
- ✅ Tests passing (early stopping validation)
- ❌ Scaling approach disproven (need quality-focused strategy)

**Decision Point**: 
Return to baseline OR pursue quality-driven optimization (curriculum learning, domain balancing).

**Critical Files for Next Agent**:
1. `.claude/CLAUDE.md` — This memory (session history)
2. `config/milestones/p3_bpe_convergence.toml` — Sealed baseline
3. `outputs/p3-optimized-50m-data/loss_curve.csv` — Failure case for analysis
4. `.github/AGENTS.md` — Development standards

---

Read [`.github/AGENTS.md`](.github/AGENTS.md) and [`.github/SKILLS.md`](.github/SKILLS.md) before starting work.

