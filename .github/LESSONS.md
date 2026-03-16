# Agent Lessons

> Maintained by [@maxjbarfuss](https://github.com/maxjbarfuss). Read this at the start of every session (see [SKILLS.md](SKILLS.md#working-rule)).
>
> Each lesson captures a specific mistake observed in an agent session and the correct behavior. Agents must internalize these before starting work.
>
> ⚠️ **START HERE**: Read L001 immediately. It is non-negotiable and applies to every session.

---

## 🚨 L001 — Never Use GUI Git Wrappers (CRITICAL)

**Observed behavior**: Agent used GitKraken, GitLens, GitHub Desktop, or another GUI wrapper or integrated git extension for repository operations instead of the local `git` CLI. This has been observed repeatedly despite being documented.

**Correct approach**: Use **only** local `git` commands for all git operations (`git status`, `git add`, `git commit`, `git diff`, `git log`, `git push`). Execute these via your agent's terminal tool with plain shell commands. Never invoke any GUI git client, VS Code git extension, integrated git tool, or git server wrapper of any kind.

**Why this matters**: Wrappers add latency, obscure error messages, and reduce control. Plain `git` CLI is faster, clearer, and more reliable. This is a non-negotiable rule that must be enforced across all sessions and agents.

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

## L010 — Keep `.claude/CLAUDE.md` Bootstrap-Only

**Observed behavior**: Agent wrote session memory, experiment summaries, and mutable status into `.claude/CLAUDE.md`, turning a bootstrap file into a stale state store.

**Correct approach**: `.claude/CLAUDE.md` must remain a short bootstrap file. It may contain static critical rules (git CLI, venv, test gate, commit format) as inline reminders so they apply in chat sessions too. It must not contain session memory, experiment reports, or mutable run status — those belong in `.github/MEMORY.md` (active state) and `.github/SESSION_LOG.md` (history).

---

## L011 — One-Off Scripts Belong in `/tmp/`, Not `scripts/`

**Observed behavior**: Agent created validation scripts, debugging tools, or ad-hoc analysis scripts directly in `scripts/`, treating them as permanent repository artifacts even though they were single-use for a specific request.

**Correct approach**: One-off scripts (validation, debugging, data inspection, analysis) that serve a single session or task belong in `/tmp/`, a local `.scratch/` directory, or another ephemeral location — **not** in the `scripts/` directory. Only commit permanent scripts to `scripts/` when they are: (1) reusable across multiple sessions/tasks, (2) part of the standard pipeline or workflow, and (3) documented with clear purpose. This keeps `scripts/` clean and signals intent: files here are canonical tools, not experimental work.

---

## L012 — Do Not Create New Data-Prep Scripts or Config Files (Without Instruction)

**Observed behavior**: Agent created a new `README.md` or new data-preparation config file (JSON, YAML, TOML) in `scripts/data/` or `config/` as "documentation" or "example", without being explicitly asked to do so.

**Correct approach**: Do not create new files for data preparation or documentation purposes without explicit instruction. Do not create config files outside of `config/ephemeral/` unless explicitly asked. Write observations and workflow notes to `.github/MEMORY.md` instead of creating README files. L009 covers ephemeral configs specifically.

---

## L013 — Never Promote a Training Config to `config/milestones/` Until the User Confirms

**Observed behavior**: Agent added a training config (e.g. `p3_unigram.toml`) to `config/milestones/` while the experiment was still in progress and results were unvalidated. This pollutes the milestone directory with premature artifacts and causes snapshot tests to break every time hyperparameters are tuned.

**Correct approach**: Use `config/ephemeral/` for all in-progress training configs. Only move a config to `config/milestones/` after the user explicitly confirms the results are good and the config represents a true milestone. Do not write snapshot tests that pin mutable hyperparameters like `max_steps` — these always break during tuning.

---

## L014 — Data-Prep Configs Are Ephemeral Until a Dataset Is Validated

**Observed behavior**: Agent committed a data-preparation config (JSON/YAML/TOML) to `config/data_prep/` for an experiment that was still in progress. The dataset had not been built, validated, or confirmed by the user.

**Correct approach**: Data-prep configs are experiments. They live in `config/ephemeral/` (gitignored) until the resulting dataset is built, verified, and the user explicitly says it is the canonical dataset for a phase. Only then does the config get promoted to `config/data_prep/`. This applies to mixing ratios, source paths, and tokenizer settings — all of which change during experimentation.

---

## L015 — Log Files and Runtime Outputs Belong in `outputs/ephemeral/`, Not Project Root

**Observed behavior**: Agent (or human during verification) created `.log` files, benchmark results, or other runtime outputs directly in the project root folder. These files clutter the workspace and are not gitignored by default at the root level.

**Correct approach**: All runtime outputs — logs, benchmark results, debug traces, profiling data, or any other ephemeral file produced during execution — must be placed in `outputs/ephemeral/` (which is gitignored). Never write logs to the project root. If you need to create a log file during debugging or verification, place it in `outputs/ephemeral/` from the start. The project root should contain only permanent artifacts: source code, docs, configs, build system files, and the repository's canonical structure.

---

## L016 — Point TensorBoard at the Parent Output Dir, Not a Single Experiment Dir

**Observed behavior**: Agent launched TensorBoard with `--logdir` pointing at a single experiment's subdirectory (e.g. `outputs/ephemeral/p3-some-run-20260313/tensorboard`). This works for that run in isolation but makes it impossible to compare curves across experiments without restarting TensorBoard.

**Correct approach**: Always launch TensorBoard one level up — either at `outputs/ephemeral/` or at the shared parent that contains multiple experiment folders. TensorBoard will discover all nested `tensorboard/` subdirectories automatically and label each run by its folder name. Example: `tensorboard --logdir outputs/ephemeral/ --host 0.0.0.0 --port 6006`. This allows side-by-side comparison of every run in that directory without any restarts. Never point `--logdir` at a single experiment subfolder unless the user explicitly asks for an isolated view.

---

## Adding a New Lesson

When @maxjbarfuss observes a repeated agent mistake, add a new entry with the next sequential number:

    ## L<NNN> — <Short descriptive title>

    **Observed behavior**: <What the agent did wrong — be specific enough that another agent recognizes the pattern.>

    **Correct approach**: <What to do instead — be prescriptive, not advisory.>
