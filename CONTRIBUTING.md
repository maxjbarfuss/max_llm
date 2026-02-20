# Contributing

Authorized contributors are defined in [.github/CONTRIBUTORS.md](.github/CONTRIBUTORS.md).

## What Each Doc Is For

- [design/PLAN.md](design/PLAN.md#phase-progress): session execution plan (phase progress, `Next Steps`, `Current Session Scratch Pad`, `Running Session Log`).
- [design/DESIGN.md](design/DESIGN.md#architecture-overview): architecture baseline, engineering principles, testing strategy, phased roadmap.
- [.github/CONTRIBUTORS.md](.github/CONTRIBUTORS.md#pr-validation-checklist): authorization, PR validation checklist.
- [.github/CODEOWNERS](.github/CODEOWNERS): review ownership.

If documents conflict during implementation, follow [design/PLAN.md](design/PLAN.md) for immediate execution, then reconcile the other docs in the same PR.

## Workflow

1. Read [design/PLAN.md](design/PLAN.md#phase-progress): check Phase Progress and pick from `Next Steps`.
2. Read relevant section of [design/DESIGN.md](design/DESIGN.md#architecture-overview) for design constraints.
3. Check existing tests and interfaces (avoid duplication).
4. Implement with TDD and typed interfaces per [design/DESIGN.md](design/DESIGN.md#coding-standards).
5. Run local validation (see below).
6. Update [design/PLAN.md](design/PLAN.md#current-session-scratch-pad) (`Current Session Scratch Pad`, `Running Session Log`).
7. Open PR: validate via [.github/CONTRIBUTORS.md](.github/CONTRIBUTORS.md#pr-validation-checklist) checklist (Summary, Validation, Risks & Next).

## Local Validation

Run the [Quality Gates from design/DESIGN.md](design/DESIGN.md#quality-gates) before merge:

```bash
make lint           # ruff + mypy
make test           # Python + C++ tests
make format-check   # verify formatting (black, isort, clang-format)
make format         # auto-fix formatting
```

## Commit and PR Rules (Summary)

- One logical change per PR
- Keep commits atomic and descriptive (`feat|fix|refactor|test|docs|chore(scope): summary`)
- Update affected docs when behavior/architecture changes
- Include a short `Next steps` handoff note
- Only [@maxjbarfuss](https://github.com/maxjbarfuss) may merge to `main`
