# Contributing

This guide is for both human contributors and AI agents.

## Authorization

| Name | GitHub | Role |
|---|---|---|
| Max Barfuss | [@maxjbarfuss](https://github.com/maxjbarfuss) | Owner, final approver, sole maintainer of `main` |

AI agents may contribute only under direct instruction or explicit written approval from @maxjbarfuss. Agent requirements:
- **Start here**: Read [.github/AGENTS.md](.github/AGENTS.md) (universal standard) and [.github/SKILLS.md](.github/SKILLS.md) (practical workflows) for agent-specific instructions, working discipline, and project patterns
- Follow the workflow in this document (session setup, TDD, doc updates, session close)
- Self-identify in commits and handoff notes (`AI agent: <name>` or similar)

Any contributor not listed above requires written approval from @maxjbarfuss before contributing. Unauthorized changes may be closed without review.

## Source of Truth

- Code and tests are the source of truth. Docs must match the current behavior.
- [docs/SESSION.md](docs/SESSION.md): current focus and immediate next tasks
- [docs/PLAN.md](docs/PLAN.md): phased execution roadmap
- [docs/DESIGN.md](docs/DESIGN.md): architecture and engineering constraints
- [.github/CODEOWNERS](.github/CODEOWNERS): review ownership

If docs conflict, follow [docs/SESSION.md](docs/SESSION.md) and [docs/PLAN.md](docs/PLAN.md) for active execution and reconcile in the same change set.

## Workflow

This repo uses a lightweight feature-branch workflow with phase-based branches and release tags.

### Standard Workflow (Every Session)

> **AI agents**: Read [.github/AGENTS.md](.github/AGENTS.md) and [.github/LESSONS.md](.github/LESSONS.md) before starting.

1. Read [docs/SESSION.md](docs/SESSION.md#next-steps-priority-order) and choose a scoped task.
2. Skim [docs/PLAN.md](docs/PLAN.md) for the current phase and any relevant sections for your task.
3. Check constraints in [docs/DESIGN.md](docs/DESIGN.md#coding-standards).
4. Ensure you are on the correct branch (phase branch or feature sub-branch).
5. Implement in small, testable steps (TDD preferred).
6. Run minimal tests for the touched area; do not proceed with failing tests.
7. Update [docs/SESSION.md](docs/SESSION.md) and [docs/PLAN.md](docs/PLAN.md) to reflect progress.
8. Commit locally with a clear, reviewable message; commit often and keep each commit small.
9. Review the change (human preferred; agentic acceptable for well-scoped work), then push when feature-complete, tested, and documented.

### Branch Strategy

- **main**: Protected. Only the owner merges to `main`.
- **phase branches**: Each phase gets a dedicated feature branch.
  - Format: `phase<phase-number>` (example: `phase2`)
  - Minimum requirement: all Phase work happens on its phase branch.
- **feature branches**: Optional sub-branches for scoped work.
  - Format: `phase<phase-number>/<topic>` (example: `phase2/data-pipeline`)

### Release Tags

- Tag releases at phase milestones or major checkpoints.
- Format: `phase<phase-number>-v<major>.<minor>.<patch>` (example: `phase2-v0.1.0`).
- Tag only from the phase branch after tests and docs are current.

### Validation and Quality

**Before every commit**, run the complete pre-commit gate:

```bash
source .venv/bin/activate
make check    # Runs: black, ruff, mypy, test-py-quick
```

If any check fails, auto-fix and re-run:

```bash
make format   # Auto-fixes: black, isort, clang-format
make check    # Re-validate
```

Then commit. Only push when all checks pass.

**Individual checks** (if needed):

```bash
make lint          # black --check, ruff, mypy
make format-check  # Check formatting without changes
make type-check    # Run mypy type checker
```

See [tests/README.md](tests/README.md) for additional test commands and expectations.

### Change Checklist

Use the required commit workflow in [.github/SKILLS.md](.github/SKILLS.md#commit-workflow-required) as the single source of truth.

Before committing and pushing, confirm:

1. **Quality Gate**: All checks pass locally:
   ```bash
   make check    # Must pass: black, ruff, mypy, test-py-quick
   ```

2. **Tests**: Relevant tests are comprehensive and passing:
   - New code has unit tests (TDD: test-first approach preferred)
   - All tests pass locally
   - Coverage is maintained or improved

3. **Summary**: Change has a clear, atomic commit message:
   - What changed and why (not how)
   - Focused scope—one logical change per commit
   - Reviewable—easy to understand in isolation

4. **Documentation**: Docs reflect the change:
   - [docs/SESSION.md](docs/SESSION.md): Updated with log entry and current status
   - [docs/PLAN.md](docs/PLAN.md): Phase progress updated if applicable
   - Code comments for non-obvious logic
   - Docstrings for new functions/classes

5. **Risks & Follow-up**: Known limitations documented:
   - Any tech debt or TODOs noted in code
   - Follow-up tasks documented in SESSION.md or PLAN.md
   - Breaking changes flagged clearly

### CI/CD

Continuous integration runs on every push via GitHub Actions. See [scripts/ci/README.md](scripts/ci/README.md) for CI monitoring tools and working commands.

### Review and Approval

PRs are not used for routine work. If a PR is opened, it requires owner approval and evidence that docs and tests are current.

**AI agent tooling**:
- Use local Git commands only; never use MCP servers (GitKraken, GitLens, or any Git MCP tools)
