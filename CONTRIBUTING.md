# Contributing

Authorized contributors are defined in [.github/CONTRIBUTORS.md](.github/CONTRIBUTORS.md).

## What Each Doc Is For

- [design/plan-checklist.md](design/plan-checklist.md): execution tracker for the current session (`Next Steps`, `Current Session Scratch Pad`, `Running Session Log`).
- [design/plan.md](design/plan.md): architecture baseline, precision/tokenizer policy, delivery order, and definition of done.
- [design/philosophy.md](design/philosophy.md): non-negotiable engineering rules (TDD, Big-O awareness, SOLID enforcement, reproducibility, API/doc discipline).
- [.github/CONTRIBUTORS.md](.github/CONTRIBUTORS.md): who is authorized to contribute and merge.
- [.github/CODEOWNERS](.github/CODEOWNERS): review ownership.
- [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md): required PR structure and validation checklist.

If documents conflict during implementation, follow [design/plan-checklist.md](design/plan-checklist.md) for immediate execution, then reconcile the other docs in the same PR.

## Workflow

1. Start with [design/plan-checklist.md](design/plan-checklist.md) and pick from `Next Steps`.
2. Confirm design constraints in [design/plan.md](design/plan.md).
3. Implement with TDD and typed interfaces per [design/philosophy.md](design/philosophy.md).
4. Run local validation.
5. Update [design/plan-checklist.md](design/plan-checklist.md) (`Current Session Scratch Pad`, `Running Session Log`).
6. Open PR using [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md).

## Local Validation

```bash
mypy src/
ruff check src/
black --check src/ tests/
pytest tests/ -v
```

## Commit and PR Rules (Summary)

- One logical change per PR
- Keep commits atomic and descriptive (`feat|fix|refactor|test|docs|chore(scope): summary`)
- Update affected docs when behavior/architecture changes
- Include a short `Next steps` handoff note
- Only [@maxjbarfuss](https://github.com/maxjbarfuss) may merge to `main`
