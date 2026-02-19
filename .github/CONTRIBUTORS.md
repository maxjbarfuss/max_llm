# Contributors

Defines who may contribute and merge.

## Owner

| Name | GitHub | Role |
|---|---|---|
| Max Barfuss | [@maxjbarfuss](https://github.com/maxjbarfuss) | Owner, final approver, sole maintainer of `main` |

## Authorized AI Agents

AI agents may contribute only under direct instruction or explicit written approval from @maxjbarfuss.

Agent requirements:
- Follow [design/DESIGN.md](../design/DESIGN.md)
- Start/end each session with [design/PLAN_CHECKLIST.md](../design/PLAN_CHECKLIST.md) updates (`Current Session Scratch Pad` + `Running Session Log`)
- Self-identify in commits/PR notes when applicable

## PR Validation Checklist

Before opening a PR, ensure:

1. **Summary**: Clear description of what changed and why
2. **Validation**:
   - `make lint` (ruff, mypy, black)
   - `make test` (Python + C++ tests)
3. **Risks & Next**: Document any known limitations or follow-up work

## Authorization Policy

Any contributor not listed above requires written approval from @maxjbarfuss before opening a PR. Unauthorized PRs may be closed without review.
