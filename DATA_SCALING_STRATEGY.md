# Data Scaling Strategy — Phase 3 Optimization

## Current Status
- **Baseline data**: 10M tokens (70% WikiText, 30% TinyStories)
- **Currently testing**: 50M tokens (same mix) — running `p3-optimized-50m-data`
- **Decision point**: If loss < 3.9 → data IS bottleneck, proceed to next stage

## Data Scaling Roadmap

### Stage 1: Extend Current Mix (10M → 50M → 100M) ✅ IN PROGRESS
**Files**:
- `data/fast/interleaved_wikitext_tinystories_10m_tokens_bpe_gpt2.npy` (baseline)
- `data/fast/interleaved_wikitext_tinystories_50m_tokens_bpe_gpt2.npy` (testing now)
- `data/fast/interleaved_wikitext_tinystories_100m_tokens_bpe_gpt2.npy` (next if needed)

**Diversity**: Current mix provides:
- 70% encyclopedic text (WikiText) — long-range dependencies, facts
- 30% narrative (TinyStories) — storytelling, dialogue

### Stage 2: Inject Complementary Sources (100M+) — PLANNED

#### Option A: Code + Technical Writing (High Diversity Impact)
**Why**: LLMs benefit from code understanding + technical discourse
**Sources**:
- ArXiv abstracts (1.5M+ papers, technical domain)
- GitHub code (diverse programming paradigms)
- Stack Exchange Q&A (10M+ high-quality Q&A)

**Impact**: +15-25% diversity, better reasoning capabilities

#### Option B: PILE Subset (Universal Text Distribution)
**Why**: PILE is "bytes" of internet + books + code (proven benchmark)
**Sources**:
- Common Crawl (10% to start, ~20M tokens)
- Books3 subset (public domain literature)
- GitHub code

**Impact**: +20-50% diversity, closer to real-world distribution

#### Option C: Mixed Domain Balance (Recommended)
**Ratio suggested**:
- 40% WikiText-103 (encyclopedic, verified facts)
- 20% TinyStories (narrative coherence)
- 15% ArXiv abstracts (technical understanding)
- 15% Stack Exchange (problem-solving, discussion)
- 10% Code (Python/C++, algorithmic thinking)

**Expected Quality**: Higher diversity → better generalization, better in-context learning

### Stage 3: Data Augmentation (Synthetic Quality)
**If bottleneck unresolved**: Consider
- Paraphrasing existing content (increase effective dataset)
- Curriculum learning (easy → hard)
- Domain-specific fine-tuning data (finance, science, code)

## Quick-Access Preparation Scripts

### Current Setup (Already Available)
```bash
# 50M WikiText-103 only
python scripts/data/run_data_prep.py \
  --config scripts/data/wikitext-103/wikitext-103_bpe_gpt2_50m.yaml

# Mix WikiText (70%) + TinyStories (30%)
python scripts/data/mix_interleaved_pages.py \
  --wikitext data/fast/wikitext_50m_tokens_bpe_gpt2.npy \
  --tinystories data/fast/tinystories_5m_tokens_bpe_gpt2.npy \
  --ratio 0.7 0.3 \
  --output data/fast/interleaved_wikitext_tinystories_50m_tokens_bpe_gpt2.npy
```

### Next Steps (To Prepare If 50m_data Succeeds)

#### Prepare Code Dataset
```bash
# Clone GitHub public dataset + tokenize
# Target: 5-10M tokens from diverse repos
# Recommendation: Python + JavaScript (most common, easiest to learn)
```

#### Prepare ArXiv Abstracts
```bash
# Download from HuggingFace: arxiv-abstracts
# ~500K abstracts, technical + clear writing
# Estimated: 2-3M tokens after tokenization
```

#### Prepare Stack Exchange
```bash
# Download from HuggingFace: stack-exchange-dump
# Q&A format, high information density
# Estimated: 5-7M tokens
```

## Decision Tree

**If 50m_data loss < 3.9**: ✅ DATA WAS BOTTLENECK
1. Prioritize data quality/diversity over quantity
2. Prepare Option C (Mixed Domain Balance) config
3. Launch 100M+ mixed dataset test
4. Target: loss < 3.5

**If 50m_data loss 3.9-4.1**: ⚠️ MARGINAL IMPROVEMENT
1. Data helps but not fully sufficient
2. Combine data + architecture improvements
3. Investigate larger models (768H×12L) with data
4. Prepare curriculum learning pipeline

**If 50m_data loss ≥ 4.1**: ❌ DATA NOT PRIMARY BOTTLENECK
1. Architecture/hyperparameter issue (not data)
2. Investigate: attention mechanisms, layer norm, learning rate schedules
3. Keep current data size, focus on model improvements
4. Consider ablation studies on model components

## Files Ready to Use

Current data in `data/fast/`:
- ✅ `interleaved_wikitext_tinystories_10m_tokens_bpe_gpt2.npy` (39M)
- ✅ `interleaved_wikitext_tinystories_50m_tokens_bpe_gpt2.npy` (44M)
- ✅ `wikitext_50m_tokens_bpe_gpt2.npy` (191M) — WikiText only variant

Available scripts:
- ✅ `scripts/data/run_data_prep.py` — Generic data prep pipeline
- ✅ `scripts/data/mix_interleaved_pages.py` — Mix multiple datasets
- ✅ `scripts/data/prepare_training_data.py` — Full pipeline automation

## Monitoring

Check 50m_data progress:
```bash
watch -n 30 'tail -3 outputs/p3-optimized-50m-data/loss_curve.csv'
```

When early stopping triggers (~8-12K steps), check loss:
- If < 3.9: Start preparing Stage 2 data immediately
- If ≥ 4.1: Pivot to architecture investigation

