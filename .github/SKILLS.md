---
name: max-llm-agent-skills
description: >
  Tool-agnostic workflows and patterns for agents working on max_llm.
  Builds on the universal AGENTS.md standard with multi-step reasoning
  strategies, context optimization, and session management. Any agent
  can use these workflows regardless of underlying model or platform.
version: "1.1"
updated: "2026-03-04"
---

# Agent Skills for max_llm

Platform-agnostic workflows and patterns for working efficiently on max_llm. This document assumes you have read [AGENTS.md](AGENTS.md) for the universal project standard.

> **Foundation**: This extends but does not replace [AGENTS.md](AGENTS.md). Always follow AGENTS.md first.
>
> **Platform-specific tool names**: Use your agent's native tools for each action described here.
> - Claude Code → see [`.claude/CLAUDE.md`](../.claude/CLAUDE.md)
> - GitHub Copilot → see [`.github/copilot-instructions.md`](copilot-instructions.md)

## Effective Work Patterns

### 1. Multi-Step Reasoning & Planning
- Use a task or todo manager to track complex, multi-step work autonomously
- Break work into logical checkpoints; mark tasks in-progress before execution
- For search-heavy tasks, batch codebase exploration to parallelize file discovery
- For focused research (docs, web), use your agent's fetch or targeted search tools
- Visible task tracking helps if work is handed off to another agent or human

### 2. Parallel Operations
- **Batch independent reads**: Reading multiple files, pattern searches, and content searches can run in parallel
- **Never parallelize terminal/shell commands**: Shell execution must be sequential (shell state is dependent)
- **Notebook cells**: Must run sequentially (kernel state is dependent)
- **Efficiency threshold**: If >5 independent file operations are needed, batch them via subagent or parallel search

### 3. Context Efficiency
- Context is a shared resource: balance reading, tool results, and reasoning carefully
- **For large codebases**: Prefer targeted or semantic search (returns filtered snippets) over reading entire files
- **For exploration**: When the starting point is uncertain, use exploratory/semantic search to discover relevant files efficiently before reading
- **For verification**: After edits, run tests via your agent's structured test tool where available — structured output is more concise than raw terminal logs

### 4. Documentation & Code Sync
- After multi-file implementation, review what changed before committing
- Apply multiple related edits atomically where possible (more efficient than sequential single-file edits)
- Always validate that documentation is current before committing (MEMORY.md, PLAN.md status)
- For complex diagrams, validate and preview before merging

### 5. Testing & Validation
- Run the full test suite before every commit — all must pass
- **Python**: `source .venv/bin/activate && make test-quick` (or your agent's native test runner)
- **C++**: Use your build system's test runner (`ctest` or equivalent)
- Track coverage progress when available
- Use `test_failure` structured output if a suite fails — more actionable than raw logs

### 6. Terminal & Background Process Management
- Shell commands must run sequentially — do not parallelize terminal execution
- For long-running processes (servers, training runs), use background execution and capture output
- Clean up background processes when no longer needed

### 7. Notebook Editing
- Use your agent's native notebook editing tool (not raw terminal Jupyter commands)
- Read the notebook structure/summary before editing to understand cell layout
- Execute cells after editing to validate changes

### 8. Build System & Project Setup
- **Python**: Always activate the venv first (`source .venv/bin/activate`) before any Python tooling
- **C++**: Use build system introspection to discover targets and tests before invoking terminal builds
- **Project scaffolding**: Use your platform's workspace or project creation tools where available

---

## Session Workflow (Required)

### Session Bootstrap

**Do this at the start of every session:**
1. Read [LESSONS.md](LESSONS.md) — past agent mistakes to avoid
2. Read [MEMORY.md](MEMORY.md) — working session context (see [MEMORY vs SESSION_LOG Pattern](MEMORY.md#memory-vs-session_log-pattern-must-understand))
3. Check [SESSION_LOG.md](SESSION_LOG.md) for historical context if needed
4. Read [docs/PLAN.md](../docs/PLAN.md) — pick task from current phase
5. Activate environment: `git status && source .venv/bin/activate`

**Context Compaction Detection**: If you see a `conversation-summary` block or references to work not in visible messages, context was compacted. RE-READ [MEMORY.md](MEMORY.md) and update "Thinking Notes" to synchronize with current state before proceeding.

### During Work
- **Update checkpoint frequently**: Add to [MEMORY.md](MEMORY.md) every 30–60 minutes with: current task, checkpoint location, recent completions
- **Track phase progress**: Mark items ✅ in [docs/PLAN.md](../docs/PLAN.md) when completing phase deliverables, update progress percentages
- **Follow TDD discipline**: Write failing test first → implement → verify test passes
- **🚨 Git restriction (L001 CRITICAL)**: Use **only** local `git` CLI. Never use GUI wrappers, integrated git extensions, or git server tools of any kind. See [LESSONS.md](LESSONS.md#-l001--never-use-gui-git-wrappers-critical).

### File Purposes
- **MEMORY.md** = working/thinking state (what you're doing now) — see [MEMORY vs SESSION_LOG Pattern](MEMORY.md#memory-vs-session_log-pattern-must-understand)
- **PLAN.md** = project progress (phase items, roadmap)
- **SESSION_LOG.md** = history (append-only log)

### If Crash/Hang
Read [MEMORY.md](MEMORY.md) checkpoint → resume from there

---

## Commit Workflow (Required)

Before every commit:

1. **Run verification**: `make test-quick` — verify nothing broke
2. **Update phase progress**: If applicable, mark ✅ in [docs/PLAN.md](../docs/PLAN.md) and update progress %
3. **Log session summary**: Add one row to [SESSION_LOG.md](SESSION_LOG.md) with detailed summary of work completed
4. **Stage and review**: `git status && git diff --staged` to confirm changes
5. **Commit atomically**: `git add . && git commit -m "Phase X.Y: clear message"`

**Commit message format**: `Phase X.Y: <imperative verb> <what changed>` (e.g., "Phase 3.2: add xformers attention implementation")

---

## Slash Commands (Claude Code)

Project-specific slash commands live in `.claude/commands/`. Invoke them with `/command-name [args]`.

| Command | What it does |
|---------|-------------|
| `/chat [run_name]` | Start an interactive chat session with a trained checkpoint. Finds the most recent checkpoint under `outputs/ephemeral/` and its matching config in `config/ephemeral/` automatically. Pass an optional run name (e.g. `/chat p4_norm_ab_rms`) to pick a specific run. Runs `python -m src.inference.chat`. |
| `/train [config_name]` | Launch a training run across both GPUs (RTX 4090 + RTX 3090 Ti) via `torchrun --nproc_per_node=2`. Resolves the config from `config/ephemeral/` or `config/milestones/` by name, or lists available configs if no argument given. Always uses DDP — never single-GPU. |
| `/dataprep [config_name]` | Run the data preparation pipeline (`python -m src.data.preparation`) to tokenize and stage a dataset. Resolves config from `config/ephemeral/` or `config/data_prep/` by name. Reports output artifact paths and token counts on completion. |

---

## Common Workflow Patterns

### Fast Iteration Loop
1. **Plan**: Use your task manager to break work into steps
2. **Search**: Parallel reads and searches for context gathering
3. **Implement**: Edit files (TDD: test first)
4. **Validate**: Run tests before moving on
5. **Document**: Verify doc sync, then commit
6. **Reflect**: Update task list, mark completed items

### Complex Multi-File Refactors
1. Map all affected files in one search pass
2. Read affected sections in parallel
3. Apply changes atomically across files where possible
4. Validate with targeted test run on affected area
5. Full `make test-quick` before final commit

### Codebase Exploration (Uncertain Starting Point)
1. Use semantic or exploratory search to map relevant files
2. Review returned file/snippet locations
3. Read promising candidates in parallel
4. Follow up with targeted searches as needed

### Git & Commit Management
- Review all diffs before committing (`git status && git diff --staged`)
- Use atomic commits across related files (avoid multiple commits per logical change)
- Commit format: `Phase X.Y: <imperative verb> <what changed>`

---

## Efficiency Guidelines

- **Context usage**: Each file read costs context (proportional to range); tool results and reasoning also consume budget
- **When context is tight**: Switch to semantic/targeted search for conciseness, batch searches, or reduce log verbosity
- **Multi-session work**: Always summarize progress in MEMORY.md before ending a session; start fresh with a clear checkpoint when continuing Phase work
