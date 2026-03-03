# MEMORY

**Working session memory**. Update frequently. For project progress, see [PLAN.md](../docs/PLAN.md). For history, see [SESSION_LOG.md](SESSION_LOG.md).

---

## Current Work

**TRAINING RUNNING** — `p3_512bpe_noshare_mixed_10k.toml` (10K steps, flash+compile)
- Config: `config/ephemeral/p3_512bpe_noshare_mixed_10k.toml` (in gitignore, ephemeral)
- Output: `outputs/ephemeral/p3-512bpe-noshare-mixed-10k/loss_curve.csv`
- Check val progress: `python3 -c "import csv,math; rows=list(csv.DictReader(open('outputs/ephemeral/p3-512bpe-noshare-mixed-10k/loss_curve.csv'))); [print(f'step={r[\"step\"]} PPL={math.exp(float(r[\"val_loss\"])):.1f}') for r in rows if r.get('val_loss') and r['val_loss'].strip()]"`
- Relaunch if dead: `./scripts/train_ddp.sh config/ephemeral/p3_512bpe_noshare_mixed_10k.toml 2 > outputs/ephemeral/p3-512bpe-noshare-mixed-10k-launch.log 2>&1 &`

**Val loss trajectory (as of 2026-03-03):**
- Step  500: PPL=28.5
- Step 1000: PPL=24.0
- Step 1500: PPL=22.1
- Step 2000: PPL=18.5  ← already below target PPL=20
- Currently at step ~2500/10000, LR still near peak (4.7e-4 of 5e-4 max)
- Realistic end-of-training target: PPL=10-14

**Root cause of all prior PPL=195 plateaus (SOLVED):**
- `share_layer_weights=true` caused gradient conflict across depth levels → weights couldn't specialize
- Fixed by: share_layer_weights=false, label_smoothing=0.0, mixed 3:1 wikitext:stories data (17.5M tok)

**Key decisions made this session:**
- attention_backend="flash" (sage breaks DDP; single-GPU sage+compile=667K tok/s, flash+compile=543K)
- use_torch_compile=true, dropout=0.0, no label smoothing
- CommonKV hypothesis NOT supported at this scale (adjacent K/V cosine sim ≈ 0)
- Overfitting NOT a concern yet (tokens/param=0.49, severely underfit by Chinchilla)
- User open to increasing training data if PPL plateaus before ~PPL=10

---

## Thinking Notes

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

