# Contributing

This guide is for both human contributors and AI agents.

Contributor authorization is defined in [.github/CONTRIBUTORS.md](.github/CONTRIBUTORS.md).

## Source of Truth

- [design/PLAN.md](design/PLAN.md): current execution state and immediate next tasks
- [design/DESIGN.md](design/DESIGN.md): architecture and engineering constraints
- [.github/CONTRIBUTORS.md](.github/CONTRIBUTORS.md): authorization policy
- [.github/CODEOWNERS](.github/CODEOWNERS): review ownership

If docs conflict, follow [design/PLAN.md](design/PLAN.md) for active execution and reconcile docs in the same PR.

## Standard Workflow

1. Read [design/PLAN.md](design/PLAN.md#next-steps) and choose a scoped task.
2. Check relevant constraints in [design/DESIGN.md](design/DESIGN.md#coding-standards).
3. Implement in small, testable steps (TDD preferred).
4. Run validation commands.
5. Update impacted docs (including [design/PLAN.md](design/PLAN.md) session/log sections).
6. Commit one logical change using local git commands.

**Tooling policy for AI agents**:
- **NEVER use MCP servers** (GitKraken, GitLens, or any git MCP tools).
- **ALWAYS use local Git commands** for all version control operations.
- **NEVER push to remote** — commits are local only; human maintainer handles remote sync.

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
4. **Docs**: behavior/architecture changes are reflected in [design/PLAN.md](design/PLAN.md) and affected docs.

## Commit and PR Rules

- One logical change per PR.
- Use clear commit messages: `feat|fix|refactor|test|docs|chore(scope): summary`.
- Keep changes minimal and reviewable.
- Include a short handoff note when there is follow-up work.
- **For AI agents**: Use `git commit` directly (never MCP/Kraken tools); never push to remote.
- Only [@maxjbarfuss](https://github.com/maxjbarfuss) may merge to `main` and push to remote.
