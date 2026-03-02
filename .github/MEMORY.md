# MEMORY

**Working session memory**. Update frequently. For project progress, see [PLAN.md](../docs/PLAN.md). For history, see [SESSION_LOG.md](SESSION_LOG.md).

---

## Current Work

**UPDATE AFTER EVERY MEANINGFUL STATE CHANGE** (when user instruction points to a new direction, new task, or major checkpoint reached).

**Task**: Session management redesign COMPLETE
**Started**: 2026-03-02
**Checkpoint**: 100% — All files streamlined; CONTRIBUTING.md simplified to point to AGENTS.md
**Next**: Ready for next agent or next task from PLAN.md
**Blockers**: None

**Recent completions** (this session):
- ✅ 15:50: Simplified CONTRIBUTING.md (136→80 lines); now points to AGENTS.md for workflow
- ✅ 15:45: Moved MEMORY.md and SESSION_LOG.md to .github/ folder; updated all references
- ✅ 15:40: Merged SESSION_WORKFLOW.md into AGENTS.md; deleted redundant file
- ✅ 15:35: Clarified file purposes: MEMORY=working state, PLAN=project progress
- ✅ 15:25: Updated AGENTS.md and CONTRIBUTING.md to reference MEMORY.md
- ✅ 15:00: Consolidated session files → MEMORY.md + SESSION_LOG.md

---

## Thinking Notes

**UPDATE IMMEDIATELY** as you work. Capture: What are you doing RIGHT NOW? If we crash, what's essential to resume?

**Current immediate state**:
- Just restructured AGENTS.md ↔ SKILLS.md separation (AGENTS=general instructions, SKILLS=detailed workflows with specific commands)
- About to receive next user instruction on MEMORY.md update frequency

**Active reasoning chain**:
- Working on refining session management documentation
- User emphasis: "immediate" thinking notes, not historical context

---

## Quick Links

| Need | See |
|------|-----|
| **Project progress** | [PLAN.md](../docs/PLAN.md) ← mark items ✅ when complete |
| **Architecture** | [DESIGN.md](../docs/DESIGN.md) |
| **Past sessions** | [SESSION_LOG.md](SESSION_LOG.md) |

---

## Agent Checklist

**Session start**:
- [ ] Read "Thinking Notes" and "Current Work" sections first
- [ ] Read [PLAN.md](../docs/PLAN.md) — project status + pick task
- [ ] `git status && source .venv/bin/activate`

**After each meaningful state change** (new user instruction → new direction):
- [ ] Update "Current Work" to reflect new task/checkpoint
- [ ] Update "Thinking Notes" with: current action and active reasoning

**During work - UPDATE AT NEARLY EVERY THINKING STEP**:
- [ ] After major decision or direction change: update "Thinking Notes" (immediate state, active reasoning)
- [ ] After completing a logical unit of work: update "Current Work" checkpoint
- [ ] When discovering blockers or pivoting: update both sections immediately
- [ ] [See SKILLS.md](SKILLS.md#commit-workflow-required) for before-commit steps

**If crash/hang**:
- [ ] Read "Thinking Notes" (what was I doing?)
- [ ] Read "Current Work" checkpoint (where was I?)
- [ ] Resume from there

