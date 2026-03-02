# MEMORY

**Working session memory**. Update frequently. For project progress, see [PLAN.md](../docs/PLAN.md). For history, see [SESSION_LOG.md](SESSION_LOG.md).

---

## Current Work

- **Task**: Config topology cleanup (milestones/tests/templates + git-ignored ephemeral) with hard cutover of references.
- **Agent**: GitHub Copilot (GPT-5.3-Codex)
- **Scope in progress**:
	- Reorganized `config/` tree and pruned stale P3 configs.
	- Completed runtime/test/doc reference cutover from legacy `config/experiment*.toml` paths.
	- Completed final genericization pass to remove hardcoded experiment details from executable helper scripts.
	- Completed strict code-only sweep for remaining hardcoded machine/experiment paths and refactored data tooling hotspots.
	- Consolidated weird/legacy P3 variants into canonical milestone baselines for important Phase 2/3 experiments.
	- Validation complete (targeted tests + quick suite passing).
- **Checkpoint**:
	- Milestones now focus on important reproducible experiments: `p2_baseline.toml`, `p2_tinystories_baseline.toml`, `p2_curriculum.toml`, `p3_baseline.toml`, `p3_data_ramp_10m_bpe.toml`.
	- P3 tests reduced to essential validation config: `config/tests/p3/p3_ddp.toml`.
	- Path rewrites completed across scripts/tests/docs; legacy references remain only in historical logs and migration notes.

---

## Thinking Notes

**UPDATE CONSTANTLY during work.** This section answers: *If we crash RIGHT NOW, what's essential to resume?*


- Completed edits across scripts/tests/docs:
	- Runtime/examples: `src/inference/run.py`, `src/inference/chat.py`, `src/inference/utils.py`, `scripts/evaluate_p2.py`, `scripts/train_ddp.sh`, `scripts/test_inference.py`, `scripts/train_curriculum.py`.
	- Tests: `tests/unit/test_config.py`, `tests/unit/test_inference_utils.py`.
	- Docs/READMEs: `src/*/README.md`, `scripts/data/*README*`, `docs/OPTIMIZATION.md`, `docs/PLAN.md`, and `.gitignore`.
	- Final pass (generic code):
	  - `scripts/train_curriculum.py`: removed hardcoded TinyStories/WikiText dataset paths; now round-robin schedule from `--datasets` (or config fallback) and launches training via subprocess.
	  - `scripts/test_inference.py`: removed hardcoded config/checkpoint; now fully CLI-driven (`--config`, `--checkpoint`, prompts via args/JSON).
	  - `scripts/train_ddp.sh`: removed experiment-specific default config; now requires explicit config path.
	  - `train.py`, `src/training/train.py`, `src/inference/run.py`, `src/inference/chat.py`, `src/inference/utils.py`: generalized usage examples to placeholder patterns.
	  - `scripts/data/mix_interleaved_pages.py`: removed machine-specific absolute dataset paths; added overrideable dataset registry via env (`MAXLLM_DATASET_<NAME>_SOURCE`) and CLI (`--dataset-source`).
	  - `scripts/data/prepare_training_data.py`: generalized examples and next-step config guidance; added optional `--config-hint` for explicit experiment path output.
	  - `scripts/data/run_data_prep.py`: made slow-storage detection configurable with `MAXLLM_SLOW_STORAGE_PREFIXES`.
	  - `src/data/pipeline/{tokenize,extract_tokens,extract_text}.py`: generalized executable usage examples (removed machine-specific `/mnt` paths).
- Target path mapping:
	- `config/experiment.toml` → `config/milestones/p2_baseline.toml`
	- `config/experiment_tinystories.toml` → `config/milestones/p2_tinystories_baseline.toml`
	- `config/experiment_curriculum.toml` → `config/milestones/p2_curriculum.toml`
	- `config/experiment_p3_interleaved_10m_bpe.toml` → `config/milestones/p3_data_ramp_10m_bpe.toml`
	- `config/experiment_p3_optimized_2k.toml` → `config/milestones/p3_baseline.toml`
	- `config/experiment_p3_ddp_test.toml` → `config/tests/p3/p3_ddp.toml`
- Validation results:
	1) Legacy-path grep now returns only `SESSION_LOG.md` and migration notes in `MEMORY.md`.
	2) `runTests` on `tests/unit/test_config.py` + `tests/unit/test_inference_utils.py`: pass (48/48).
	3) `source /home/max/dev/max_llm/.venv/bin/activate && make test-quick`: pass (Python + C++ quick checks).
	4) Specificity re-scan: no hardcoded dataset/output logic remains in active refactored helper scripts.
	5) Strict grep sweep over `src/**/*.py` and `scripts/**/*.py` confirms previous machine-specific hardcoded paths were removed from executable workflow tools.

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

