# MEMORY

**Working session memory**. Update frequently. For project progress, see [PLAN.md](../docs/PLAN.md). For history, see [SESSION_LOG.md](SESSION_LOG.md).

---

## Current Work

(Empty — session completed and committed)

---

## Thinking Notes

**Technical Checkpoint** (crash recovery info):

**Dataset & Config Status:**
- 512-vocab BPE tokenizer: `/home/max/dev/max_llm/data/fast/` (tinystories + wikitext BPE, 512 vocab)
- Pretrain config: `config/ephemeral/p3_tinystories_pretrain_512_combined_single.toml` — 5K steps, 768 hidden, 5 layers (1 shared block reused), 256 embedding_dim (projection to 768), share_layer_weights=true, embedding_dim=256
- Finetune config: `config/ephemeral/p3_mixed_wikitext_tinystories_finetune_512_combined_single.toml` — 10K steps, same arch, resume from pretrain checkpoint, 3:1 alternating wikitext:tinystories data
- All optimizations wired: torch.compile, Flash Attention, label_smoothing=0.1, BF16 AMP, early_stopping_patience=10, selective weight decay

**Tests Passing**: 432 unit tests, 0 failures
- New tests added: test_cross_layer_parameter_sharing_reuses_one_block(), test_factorized_embeddings_create_projection_and_disable_weight_tying(), test_from_config_wires_sharing_and_factorized_embedding()
- All layer-sharing + factorized embedding behaviors validated in DecoderLM

**CUDA Status:**
- CUDA 12.9 installed at `/usr/local/cuda-12.9` (verified: nvcc present, 27MB, executable)
- torch.compile succeeds when CUDA_HOME=/usr/local/cuda-12.9 and PATH prepended (`/usr/local/cuda-12.9/bin:$PATH`)
- **Current Issue**: CUDA exports in train_ddp.sh only, not persistent in venv activation
- **User Request**: "put CUDA in the path during activate" — integrate into `.venv/bin/activate` script

**Next Action**:
1. ✅ COMPLETED: Integrated CUDA_HOME/PATH into venv activate script
2. ✅ COMPLETED: Verified torch.compile works with CUDA
3. ✅ COMPLETED: Launched pretrain (5000 steps on TinyStories)
4. MONITOR: Watch pretraining progress in terminal 98e3304b-f50b-4a48-9c81-8564f02479ad
5. ON COMPLETION: Check logs at `outputs/p3-tinystories-pretrain-512-combined-single/`
6. THEN: Launch fine-tune from checkpoint with `./scripts/train_ddp.sh config/ephemeral/p3_mixed_wikitext_tinystories_finetune_512_combined_single.toml 2`

**Files Modified Recent Session**:
- [scripts/train_ddp.sh](scripts/train_ddp.sh) — absolute path resolution for venv sourcing + CUDA exports (now redundant after venv integration)
- [src/models/learning_model/decoder_lm.py](src/models/learning_model/decoder_lm.py) — runtime assertions for layer sharing
- [tests/unit/test_decoder_lm.py](tests/unit/test_decoder_lm.py) — 3 new tests for layer sharing + factorized embeddings
- 2 config files created in `config/ephemeral/`

## MEMORY vs SESSION_LOG Pattern (MUST UNDERSTAND)

**MEMORY.md** = Working brain for **CURRENT session ONLY**
- "Current Work" = task **started in THIS session**
- "Thinking Notes" = state for crash recovery in THIS session
- Gets **CLEARED at commit** (content moves to SESSION_LOG)
- If stale info is here, previous agent made a mistake

**SESSION_LOG.md** = Permanent append-only log of completed work
- One row per commit = one session's result
- Never edited, only appended
- Use to understand past work chains

**⚠️ AGENT MISTAKE PREVENTION**:
- ❌ WRONG: Leaving old session's work in "Current Work" when starting new task
- ❌ WRONG: Copying SESSION_LOG content into MEMORY (gets confusing)
- ✅ RIGHT: At commit time: (1) Copy MEMORY sections to SESSION_LOG as new row, (2) Clear MEMORY sections completely, (3) Current Work empty for next agent

---

## Agent Checklist

**EVERY SESSION START** (do in order):
- [ ] Read "Current Work" and "Thinking Notes" — are these CURRENT or STALE?
- [ ] **If stale** (from old sessions): This is a bug. Alert user, then clear them for fresh start
- [ ] **If current**: Understand what task was in progress, read those notes to resume
- [ ] If "Current Work" is empty: Read [PLAN.md](../docs/PLAN.md), pick next task, fill it in with agent name
- [ ] `git status && source .venv/bin/activate && make test-quick` (verify baseline)

**When context compaction detected** (conversation-summary block present, or references to unseen work):
- [ ] **RE-READ MEMORY.md completely** — summary may be incomplete or outdated
- [ ] Update "Thinking Notes" with current state based on summary + conversation context
- [ ] Verify "Current Work" matches what you're actually doing
- [ ] Cross-check: Do file states match what summary claims? (`git status`, check test results)
- [ ] **This is a checkpoint moment** — synchronize MEMORY with reality before proceeding

**When user gives new instruction**:
- [ ] This is a NEW session/task: Clear "Thinking Notes" section completely
- [ ] Update "Current Work": NEW task, NEW agent name, fresh checkpoint
- [ ] Do NOT mix multiple sessions' work in "Current Work"

**EVERY 30min-1hr during work**:
- [ ] Update "Thinking Notes" with immediate state (if crash now, what's the next line of code?)
- [ ] After completing logical chunk: update "Current Work" checkpoint
- [ ] When blockers appear: update both sections immediately

**BEFORE COMMIT** (ABSOLUTELY CRITICAL):
1. [ ] Verify "Current Work" has ONLY this session's accomplishments (check dates/scope)
2. [ ] Verify "Thinking Notes" has: files changed, tests status, build status, next immediate action
3. [ ] **Summarize and move** to [SESSION_LOG.md](SESSION_LOG.md) — add new row with date | branch | concise summary of session's work
4. [ ] **DELETE content** from "Current Work" and "Thinking Notes" sections (but keep headers)
5. [ ] Commit
6. [ ] Next agent sees empty MEMORY, reads SESSION_LOG to understand context

**If crash/hang**:
- [ ] Read "Thinking Notes" → resume from that immediate state
- [ ] Read "Current Work" → understand broader task context
- [ ] If both empty: read recent SESSION_LOG entries

**Quick Links**:
- [PLAN.md](../docs/PLAN.md) — project progress
- [DESIGN.md](../docs/DESIGN.md) — architecture
- [SESSION_LOG.md](SESSION_LOG.md) — completed sessions
- [SKILLS.md](SKILLS.md#commit-workflow-required) — full before-commit checklist

