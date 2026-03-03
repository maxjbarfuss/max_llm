# MEMORY

**Working session memory**. Update frequently. For project progress, see [PLAN.md](../docs/PLAN.md). For history, see [SESSION_LOG.md](SESSION_LOG.md).

---

## Current Work

**Session**: Phase 2-3 Closeout + Ephemeral Testing (2026-03-03)

**Completed** ✅:
- Phase 2 closeout report: SimpleLM validation (UTF-8 works, BPE fails on 50K vocab)
- Phase 3 closeout report: DecoderLM proven (49% improvement: 4.31 vs 8.50)
- Training code: Implemented early stopping, label smoothing, validation tracking
- Committed: c5b4133 (training code) + 593979e (doc cleanup) + 6b48557 (closeouts) + f4f8331 (config)
- Doc sync verified: All evidence self-contained in phase closeouts, supporting materials in outputs/
- Convergence run: ~15K+ steps ongoing, expected to early-stop around 20K

**Next**: 
- Ephemeral testing: Validate training infrastructure on small runs
- Ready for Phase 4: Llama-style architecture upgrades (RMSNorm, RoPE, SwiGLU, GQA)

---

## Thinking Notes

**Key Findings**:
- SimpleLM + 256 vocab (UTF-8): Works well, loss 2.84
- SimpleLM + 50K vocab (BPE): Fails (stuck at 8.50), architectural rank bottleneck (128 < 50K)
- DecoderLM + 50K vocab (BPE): Proven superior, loss 4.31, 49% improvement
- Root cause: Multi-head attention distributes info across heads (implicit high rank)

**Tech Stack Validated** (Phase 3):
1. Flash Attention 2 ✓
2. BF16 mixed precision ✓
3. Gradient accumulation (batch 32) ✓
4. Early stopping framework ✓
5. Label smoothing (0.1) ✓
6. Selective weight decay ✓
7. Cosine LR schedule ✓
8. torch.compile ready ✓
9. Validation/test tracking ✓
10. Reproducibility ✓

**Ephemeral Approach**: Quick 100-500 step runs to validate code changes, configs, data loading before committing to long runs

---

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

