# Contributing

This guide is for both human contributors and AI agents.

## Authorization

| Name | GitHub | Role |
|---|---|---|
| Max Barfuss | [@maxjbarfuss](https://github.com/maxjbarfuss) | Owner, final approver, sole maintainer of `main` |

AI agents may contribute only under direct instruction or explicit written approval from @maxjbarfuss. Agent requirements:
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

Run the minimal checks relevant to your change. Testing commands and expectations live in [tests/README.md](tests/README.md).

Check (read-only):
```bash
source .venv/bin/activate
make lint
make format-check
make type-check
```

Fix (modifies files):
```bash
make format
make cpp-lint
make cpp-format-check
```

### Change Checklist

Before pushing or tagging a milestone, confirm:

1. **Summary**: what changed and why.
2. **Validation**: the relevant tests passed locally.
3. **Risks & Next**: known limitations and follow-up tasks are documented.
4. **Docs**: [docs/SESSION.md](docs/SESSION.md) and [docs/PLAN.md](docs/PLAN.md) reflect the change.

### CI/CD

Continuous integration runs on every push via GitHub Actions. See [scripts/ci/README.md](scripts/ci/README.md) for CI monitoring tools and working commands.

### Review and Approval

PRs are not used for routine work. If a PR is opened, it requires owner approval and evidence that docs and tests are current.

**AI agent tooling**:
- Use local Git commands only; never use MCP servers (GitKraken, GitLens, or any Git MCP tools)
