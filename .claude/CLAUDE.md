# ⚠️ AGENT BOOTSTRAP — READ FIRST

**CRITICAL**: Every agent must read these FIRST before any work:
1. [`.github/AGENTS.md`](.github/AGENTS.md) — Universal development standard
2. [`.github/SKILLS.md`](.github/SKILLS.md) — Practical workflows and tool patterns
3. [`.github/LESSONS.md`](.github/LESSONS.md) — Past mistakes to avoid

These are NON-NEGOTIABLE. Read them with file tools before starting work. The inline rules below apply immediately as a fallback when file access is unavailable.

## Critical Rules (always apply)

**Git**: Use only local `git` CLI via terminal. Never use GUI wrappers, GitLens, GitKraken, or any integrated git tool. See L001 in [`.github/LESSONS.md`](.github/LESSONS.md).

**Virtual environment**: Activate before any Python command: `source /home/max/dev/max_llm/.venv/bin/activate`. Applies to `pytest`, `ruff`, `mypy`, `black`, `pip`, and `make` targets.

**Hugging Face datasets**: Before fetching any new Hugging Face dataset, load auth via `source setup.sh` or `export HF_TOKEN=$(cat .huggingface/.hf_token)`, and mirror it to `HUGGING_FACE_HUB_TOKEN`. Do not start with anonymous dataset downloads.

**Test gate**: Before every commit: `source .venv/bin/activate && make test-quick && ruff check src tests && mypy src && black --check src tests`. All must pass.

**Commit format**: `Phase X.Y: <imperative verb> <what changed>`

---

## Working Memory Location

- Active session state belongs in [`.github/MEMORY.md`](.github/MEMORY.md)
- Historical session summaries belong in [`.github/SESSION_LOG.md`](.github/SESSION_LOG.md)

`CLAUDE.md` is bootstrap-only and must **not** contain session memory, experiment reports, or mutable run status.
