# Contributing

This guide is for both human contributors and AI agents.

## Authorization

| Name | GitHub | Role |
|---|---|---|
| Max Barfuss | [@maxjbarfuss](https://github.com/maxjbarfuss) | Owner, final approver, sole maintainer of `main` |

AI agents may contribute only under direct instruction or explicit written approval from @maxjbarfuss. Agent requirements:
- Follow the [Agent Workflow in docs/DESIGN.md](docs/DESIGN.md#agent-workflow) (session start/end, TDD, doc updates)
- Self-identify in commits and PR notes (`AI agent: <name>` or similar)

Any contributor not listed above requires written approval from @maxjbarfuss before opening a PR. Unauthorized PRs may be closed without review.

## Source of Truth

- [docs/SESSION.md](docs/SESSION.md): current focus and immediate next tasks
- [docs/PLAN.md](docs/PLAN.md): phased execution roadmap
- [docs/DESIGN.md](docs/DESIGN.md): architecture and engineering constraints
- [.github/CODEOWNERS](.github/CODEOWNERS): review ownership

If docs conflict, follow [docs/SESSION.md](docs/SESSION.md) and [docs/PLAN.md](docs/PLAN.md) for active execution and reconcile in the same PR.

## Standard Workflow

1. Read [docs/SESSION.md](docs/SESSION.md#immediate-next-steps) and choose a scoped task.
2. Check relevant constraints in [docs/DESIGN.md](docs/DESIGN.md#coding-standards).
3. Implement in small, testable steps (TDD preferred).
4. Run validation commands.
5. Update impacted docs (including [docs/SESSION.md](docs/SESSION.md) scratch pad and log).
6. Commit one logical change using local git commands.

**Tooling policy for AI agents**:
- **NEVER use MCP servers** (GitKraken, GitLens, or any git MCP tools).
- **ALWAYS use local Git commands** for all version control operations.
- **Commits are always local first**; push to remote only with explicit human confirmation.

## Validation Commands

Run before opening a PR:

```bash
source .venv/bin/activate
make lint
make format-check
make test
```

Optional but recommended:

```bash
make test-quick
make test-cov
```

Optional acceleration dependency probes (may fail if not installed):

```bash
python3 -c "import flash_attn; print(flash_attn.__version__)"
python3 -c "import torchao; print(torchao.__version__)"
python3 -c "import xformers; print(xformers.__version__)"
```

## PR Validation Checklist

Before opening a PR, confirm:

1. **Summary**: what changed and why.
2. **Validation**: `make lint`, `make format-check`, `make test` all pass locally.
3. **Risks & Next**: known limitations and follow-up tasks are documented.
4. **Docs**: behavior/architecture changes are reflected in [docs/SESSION.md](docs/SESSION.md) and affected docs.

## Commit and PR Rules

- One logical change per PR.
- Use clear commit messages: `feat|fix|refactor|test|docs|chore(scope): summary`.
- Keep changes minimal and reviewable.
- Include a short handoff note when there is follow-up work.
- **For AI agents**: Use `git commit` directly (never MCP/Kraken tools); push to remote only with human confirmation.
- Only [@maxjbarfuss](https://github.com/maxjbarfuss) may merge to `main`.
