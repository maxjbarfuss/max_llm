# MEMORY

**Working session memory**. Update frequently. For project progress, see [PLAN.md](../docs/PLAN.md). For recent history, see [SESSION_LOG.md](SESSION_LOG.md); for older history, see [SESSION_LOG_ARCHIVE.md](SESSION_LOG_ARCHIVE.md).

---

## Current Work

Phase 5.2 Wave 1 new sources: NuminaMath preprocessing complete (859K→18 parquet shards), 4 production prep configs written (Gutenberg, ir_python, owm, NuminaMath), 32K tokenizer training config written. PLAN.md + README updated.

## Thinking Notes

Next step: run tokenizer config first (`p5_wave1_tokenizer_32k_20260424.toml`), then run 4 source prod configs. After all preps complete, build curriculum mixing config to blend Wave 0 + Wave 1 datasets.


---

## MEMORY vs SESSION_LOG Pattern (MUST UNDERSTAND)

**MEMORY.md** = Working brain for **CURRENT session ONLY**
- "Current Work" = task **started in THIS session**
- "Thinking Notes" = state for crash recovery in THIS session
- Gets **CLEARED at commit** (content moves to SESSION_LOG)
- If stale info is here, previous agent made a mistake

**SESSION_LOG.md** = Recent append-only log of completed work
- One row per commit = one session's result
- Older contiguous blocks may be moved to `SESSION_LOG_ARCHIVE.md` to keep the live log readable
- Use `SESSION_LOG.md` first, then the archive when older context is needed

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
6. [ ] Next agent sees empty MEMORY, reads SESSION_LOG (and archive if needed) to understand context

**If crash/hang**:
- [ ] Read "Thinking Notes" → resume from that immediate state
- [ ] Read "Current Work" → understand broader task context
- [ ] If both empty: read recent SESSION_LOG entries, then the archive if needed

**Quick Links**:
- [PLAN.md](../docs/PLAN.md) — project progress
- [DESIGN.md](../docs/DESIGN.md) — architecture
- [SESSION_LOG.md](SESSION_LOG.md) — recent completed sessions
- [SESSION_LOG_ARCHIVE.md](SESSION_LOG_ARCHIVE.md) — older completed sessions
- [SKILLS.md](SKILLS.md#commit-workflow-required) — full before-commit checklist

