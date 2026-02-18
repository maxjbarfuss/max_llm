# Max LLM Development Checklist

**Purpose:** This is the coding "scratch pad" for tracking progress. AI assistants should:
1. Read this file at the start of every session
2. Update "Last Handoff" when completing work
3. Check "Next Actions" for what to work on next

---

## 🎯 Broad Plan

### Phase 1: Foundation (Current)
**Goal:** Core architecture and configuration system
- [x] Project structure and configuration system
- [x] Design philosophy and development guidelines
- [ ] Base model components (embeddings, RoPE)
- [ ] MLA attention with Flash Attention
- [ ] Transformer blocks with Pre-LN
- [ ] Basic forward pass validation

### Phase 2: Advanced Components
**Goal:** MoE, GRU, and quantization
- [ ] MoE layers with Top-2 routing
- [ ] Expert load balancing
- [ ] GRU output layer
- [ ] Progressive precision scheduler (FP4→FP8→BF16)
- [ ] Quantized linear layers

### Phase 3: Training Infrastructure
**Goal:** Complete training loop with monitoring
- [ ] Data pipeline with streaming
- [ ] Training loop with gradient accumulation
- [ ] Distributed training (DDP/FSDP)
- [ ] Checkpointing and resume
- [ ] Monitoring and drift detection

### Phase 4: Optimization & Testing
**Goal:** Performance optimization and validation
- [ ] torch.compile integration
- [ ] Selective gradient checkpointing
- [ ] Smoke test (10M model overfit)
- [ ] Multi-GPU synchronization tests
- [ ] Memory and throughput benchmarks

### Phase 5: Training Runs
**Goal:** Pre-train, fine-tune, and deployment
- [ ] Pre-training on 5TB dataset
- [ ] Fine-tuning on domain data
- [ ] Post-training (SFT/RLHF)
- [ ] Model export and optimization
- [ ] Evaluation benchmarks

---

## 📋 Next Actions

### Immediate (Do These Next)
1. **Create base embeddings module** (`src/models/embeddings.py`)
   - TokenEmbedding with weight tying support
   - Scaling by sqrt(hidden_size)
   - Write tests first (TDD)
   - See: design/plan.md section 2 for specs

2. **Implement RoPE (Rotary Position Embeddings)** (`src/models/position.py`)
   - RoPE cache for efficiency
   - Support for latent space application (MLA requirement)
   - Base frequency = 10000
   - Write tests for rotation correctness

3. **Create MLA attention module** (`src/models/attention/mla.py`)
   - Q full-size, KV compressed to latent_dim
   - RoPE in latent space
   - Flash Attention backend
   - KV cache management
   - Tests: verify 75% cache reduction

### Soon After
4. **Transformer block** (`src/models/transformer.py`)
   - Pre-LayerNorm architecture
   - Residual connections
   - Selective checkpointing support

5. **Basic forward pass integration**
   - Combine embeddings → attention → projection
   - Shape validation tests
   - Simple generation test (no MoE/GRU yet)

### Blocked/Waiting
- None currently

---

## 🔄 Last Handoff

### Session Date: 2026-02-18

#### ✅ Completed This Session
1. **Created design/philosophy.md** 
   - Full design philosophy with KISS, TDD, SOLID principles
   - Context window efficiency strategies
   - AI assistant collaboration guidelines
   - Comprehensive contributor checklist

2. **Updated README.md**
   - Added Design Philosophy section with key principles
   - Included TDD as core principle
   - Links to detailed documentation

3. **Implemented configuration system** (`src/config/model_config.py`)
   - `ModelConfig` with MLA, MoE, GRU parameters
   - `TrainingConfig` with progressive precision schedule
   - `DataConfig` for data pipeline
   - `ExperimentConfig` as top-level config
   - Full validation in `__post_init__`
   - Computed properties (head_dim, effective_batch_size)

4. **Created comprehensive test suite** (`tests/unit/test_config.py`)
   - Tests for all config validation logic
   - Tests for computed properties
   - Tests for frozen dataclasses
   - All tests passing

5. **Project structure created**
   ```
   src/
   ├── models/
   │   ├── attention/
   │   ├── moe/
   │   └── rnn/
   ├── config/     ✓ Complete
   ├── training/
   ├── data/
   ├── monitoring/
   └── utils/
   tests/
   ├── unit/       ✓ Config tests
   ├── integration/
   └── fixtures/
   ```

#### 📝 Notes for Next Developer
- **Configuration is production-ready** - All configs have validation, type hints, and tests
- **Following TDD strictly** - Write tests before implementation
- **Next up: Base model components** - Start with embeddings (simple, well-defined)
- **Type hints are mandatory** - All code must be fully typed
- **Max 500 lines/file** - Split if larger

#### 🚧 Current State
- **Environment:** `.venv` with PyTorch 2.10.0, CUDA 12.1, DeepSpeed 0.18.6
- **No model code yet** - Only configuration system implemented
- **Ready for embeddings** - Config supports all needed parameters

#### 🎯 Recommended Next Step
Start with embeddings module following TDD:
1. Write test for TokenEmbedding shape and initialization
2. Implement minimal TokenEmbedding class
3. Add weight tying test and implementation
4. Add scaling test and implementation
5. Refactor for clarity

See `src/config/model_config.py` for reference on code style and documentation.

---

## 📖 Architecture Reference

**Core Flow:**
```
Input Text 
  → Tokenizer (GPT-2 BPE)
  → Token Embeddings (BF16, scaled, tied weights)
  → Input FFN (enrichment)
  → Transformer Blocks (MLA + Pre-LN)
  → MoE Layers (every 2nd block)
  → GRU Output (sequential continuity)
  → Projection (tied weights)
  → Output Logits
```

**Key Design Decisions (Locked):**
- Vocab size: 50304 (padded to 64)
- Hidden size: 768 (100M), 1024 (300M), 1280 (500M)
- MLA latent_dim: 512 (75% KV cache reduction)
- MoE: 16 experts, Top-2 routing
- Precision: BF16 (attn/MoE/emb), FP8 (FFN/RNN)
- Max sequence: 2048 tokens

**Files to Read:**
- `design/plan.md` - Detailed architecture
- `design/philosophy.md` - Development guidelines
- `src/config/model_config.py` - Configuration reference

---

## ✅ Quick Checklist (Copy for Commits)

Before committing:
- [ ] Tests written BEFORE implementation (TDD)
- [ ] All tests passing
- [ ] Type hints on all functions/classes
- [ ] Docstrings (WHAT/WHY, not HOW)
- [ ] Static analysis passes: `mypy src/`, `ruff check src/`
- [ ] Code formatted: `black src/ tests/`
- [ ] Max 500 lines per file
- [ ] Updated this checklist (Last Handoff section)
- [ ] Commit message includes context for next developer

