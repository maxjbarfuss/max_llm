# Contributing

Authorized contributors are defined in [.github/CONTRIBUTORS.md](.github/CONTRIBUTORS.md).

## What Each Doc Is For

- [design/PLAN_CHECKLIST.md](design/PLAN_CHECKLIST.md): execution tracker for the current session (`Next Steps`, `Current Session Scratch Pad`, `Running Session Log`).
- [design/DESIGN.md](design/DESIGN.md): architecture baseline, engineering rules, testing strategy, and delivery order.
- [.github/CONTRIBUTORS.md](.github/CONTRIBUTORS.md): authorization policy and PR validation checklist.
- [.github/CODEOWNERS](.github/CODEOWNERS): review ownership.

If documents conflict during implementation, follow [design/PLAN_CHECKLIST.md](design/PLAN_CHECKLIST.md) for immediate execution, then reconcile the other docs in the same PR.

## Workflow

1. Start with [design/PLAN_CHECKLIST.md](design/PLAN_CHECKLIST.md) and pick from `Next Steps`.
2. Confirm design constraints in [design/DESIGN.md](design/DESIGN.md).
3. Implement with TDD and typed interfaces per [design/DESIGN.md](design/DESIGN.md).
4. Run local validation.
5. Update [design/PLAN_CHECKLIST.md](design/PLAN_CHECKLIST.md) (`Current Session Scratch Pad`, `Running Session Log`).
6. Open PR: validate via [.github/CONTRIBUTORS.md](.github/CONTRIBUTORS.md) checklist (Summary, Validation, Risks & Next).

## Local Validation

```bash
make lint       # ruff + mypy + black check
make test       # Python + C++ tests
make format     # Auto-format Python + C++
```

## Commit and PR Rules (Summary)

- One logical change per PR
- Keep commits atomic and descriptive (`feat|fix|refactor|test|docs|chore(scope): summary`)
- Update affected docs when behavior/architecture changes
- Include a short `Next steps` handoff note
- Only [@maxjbarfuss](https://github.com/maxjbarfuss) may merge to `main`
