---
name: max-llm-copilot-instructions
description: GitHub Copilot bootstrap for max_llm. Inline rules for highest-risk violations; full standard in AGENTS.md, SKILLS.md, and LESSONS.md.
applyTo: "**"
---

# GitHub Copilot Instructions

**Before any work**, read these files in order:
1. [AGENTS.md](AGENTS.md) — complete development standard (non-negotiable)
2. [LESSONS.md](LESSONS.md) — past agent mistakes to avoid
3. [MEMORY.md](MEMORY.md) — current session context
4. [docs/PLAN.md](../docs/PLAN.md) — pick task from current phase

> Use `read_file` to open each of the above before starting work. The inline rules below apply immediately as a fallback when file access is unavailable.

---

## Critical Rules (Inline — Highest-Risk Violations)

### Git operations
Use **only** the local `git` CLI (`git status`, `git add`, `git commit`, `git diff`, `git log`, `git push`) via `run_in_terminal`. Never use GitKraken, GitLens, GitHub Desktop, or any other GUI wrapper or VS Code git extension. See L001 in [LESSONS.md](LESSONS.md).

### Virtual environment
Always activate before any Python command: `source /home/max/dev/max_llm/.venv/bin/activate`. Required for `pytest`, `ruff`, `mypy`, `black`, `pip`, and `make` targets. Do not trial-and-error — activate proactively.

### Hugging Face datasets
Before any new dataset fetch from Hugging Face, load auth first: `source setup.sh` or `export HF_TOKEN=$(cat .huggingface/.hf_token)`. Also set `HUGGING_FACE_HUB_TOKEN=$HF_TOKEN`. Do not start with anonymous Hugging Face downloads.

### Test gate
Run before every commit: `source .venv/bin/activate && make test-quick && ruff check src tests && mypy src && black --check src tests`. All must pass clean.

### Commit format
`Phase X.Y: <imperative verb> <what changed>` — example: `Phase 3.2: add xformers attention implementation`

---

## Everything Else

Full workflows, patterns, and tool guidance: [SKILLS.md](SKILLS.md)
