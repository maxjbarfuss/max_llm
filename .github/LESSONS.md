# Agent Lessons

> Maintained by [@maxjbarfuss](https://github.com/maxjbarfuss). Read this at the start of every session (see [SKILLS.md](SKILLS.md#working-rule)).
>
> Each lesson captures a specific mistake observed in an agent session and the correct behavior. Agents must internalize these before starting work.
>
> ⚠️ **START HERE**: Read L001 immediately. It is non-negotiable and applies to every session.

---

## 🚨 L001 — Never Use MCP Git Tools (CRITICAL)

**Observed behavior**: Agent used GitKraken, GitLens, or another MCP git server for repository operations instead of the local `git` CLI. This has been observed repeatedly despite being documented.

**Correct approach**: Use **only** local `git` commands for all git operations (`git status`, `git add`, `git commit`, `git diff`, `git log`, `git push`). Execute these via `run_in_terminal` with plain shell commands. Never invoke any MCP server, wrapper tool, or git integration tool.

**Why this matters**: MCP wrappers add latency, obscure error messages, and reduce control. Plain `git` CLI is faster, clearer, and more reliable. This is a non-negotiable rule that must be enforced across all sessions and agents.

---

## L002 — Always Clear MEMORY and Update SESSION_LOG at Session End

**Observed behavior**: Agent committed without (a) appending a summary row to [SESSION_LOG.md](SESSION_LOG.md), or (b) clearing the "Current Work" and "Thinking Notes" sections of [MEMORY.md](MEMORY.md). This leaves stale state that confuses the next agent session. Note: the old `docs/SESSION.md` was consolidated into MEMORY.md + SESSION_LOG.md — any reference to SESSION.md is outdated.

**Correct approach**: Before every commit: (1) append one row to SESSION_LOG.md with date | branch | detailed summary, (2) delete the contents of "Current Work" and "Thinking Notes" in MEMORY.md (but keep the headers), (3) then commit. The next agent sees empty MEMORY and reads SESSION_LOG to understand context.

---

## L003 — Push Only When Feature-Complete and Reviewed

**Observed behavior**: Agent pushed commits to the remote repository mid-task, before the work was complete or reviewed.

**Correct approach**: Commit locally and often. Push only when the work is feature-complete, tests pass, docs are updated, and a review (human preferred, agentic acceptable) has been done. See [CONTRIBUTING.md](../CONTRIBUTING.md#standard-workflow-every-session) step 9.

---

## L004 — Do Not Modify Governance Files Without an Explicit Request

**Observed behavior**: Agent modified `.github/SKILLS.md`, `.github/LESSONS.md`, `CONTRIBUTING.md`, or `.github/CODEOWNERS` as incidental cleanup or as part of an unrelated task.

**Correct approach**: Governance and agent instruction files require an explicit user request to change. Do not touch them speculatively or as a side effect of another task. Note: `.github/MEMORY.md`, `.github/SESSION_LOG.md`, and `docs/PLAN.md` are *not* governance files — updating them is expected as part of every session.

---

## L005 — Never Put Phase Progress Status in README.md

**Observed behavior**: Agent added a "Current Phase" section to `README.md` containing task checklists, test counts, coverage percentages, or other mutable status information.

**Correct approach**: `README.md` is a stable navigation document — it describes the project and links to resources. Mutable status belongs exclusively in `.github/MEMORY.md` (current focus), `.github/SESSION_LOG.md` (history), and `docs/PLAN.md` (task checklists, exit criteria). Never add current-phase progress tables, checklist items, or quality metrics to `README.md`.

---

## L006 — Run Complete Lint + Format Checks Before Commit

**Observed behavior**: Agent ran `ruff check` and `mypy` before committing, but skipped `black` formatting check. CI then failed on formatting violations that were local-checkable.

**Correct approach**: Before committing, always run the complete quality gate: `ruff check src tests`, `mypy src`, and `black --check src tests`. Fix any issues with `black --fix` and `ruff --fix` before staging. All three must pass clean before git commit. See [tests/README.md](../tests/README.md) for the full checklist.

---

## L007 — Always Activate Virtual Environment for Python Commands

**Observed behavior**: Agent ran Python commands (`python`, `pytest`, `ruff`, `mypy`, `black`) without first sourcing the virtual environment, causing "command not found" or "module not found" errors. Agent retried multiple times before realizing the venv needed to be activated.

**Correct approach**: This workspace uses a Python virtual environment at `/home/max/dev/max_llm/.venv`. Always prefix Python-related commands with `source /home/max/dev/max_llm/.venv/bin/activate &&`. For example: `source /home/max/dev/max_llm/.venv/bin/activate && python -m pytest tests/`. The venv must be activated for all Python tooling (pytest, ruff, mypy, black, pip) and script execution.

---

## L008 — Activate Venv Before Running Make Commands

**Observed behavior**: Agent ran `make test-quick` without first sourcing the venv, got a permission error, then re-ran it with `source .venv/bin/activate && make test-quick`. Wasted context and time on a predictable failure.

**Correct approach**: Before running ANY `make` command (`make test-quick`, `make check`, `make format`, etc.), always activate the venv first in the same command line: `source .venv/bin/activate && make test-quick`. Do not trial-and-error this — activate proactively on every new terminal session or command sequence.

---

## L009 — Use `config/ephemeral/` for Scratch Training Runs (No Config Bleed)

**Observed behavior**: Agent created or modified configs in `config/milestones/` or `config/templates/` to run a quick convergence experiment, polluting canonical configs with temporary hyperparameter tweaks.

**Correct approach**: All scratch/experimental/temporary training runs must use `config/ephemeral/` (gitignored). Never place temporary configs in `config/milestones/` (milestone artifacts), `config/tests/` (unit-test fixtures), or `config/templates/` (reusable templates). The workflow is: copy the closest canonical config to `config/ephemeral/<name>.toml`, tweak freely, then throw it away. The `config/ephemeral/` directory is the only place for "I'm just trying this" configs. If a run produces publishable results, promote the config to `config/milestones/` with a proper name.

---

## Adding a New Lesson

When @maxjbarfuss observes a repeated agent mistake, add a new entry with the next sequential number:

    ## L<NNN> — <Short descriptive title>

    **Observed behavior**: <What the agent did wrong — be specific enough that another agent recognizes the pattern.>

    **Correct approach**: <What to do instead — be prescriptive, not advisory.>
