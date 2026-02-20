# Contributors

Defines who may contribute and merge.

## Owner

| Name | GitHub | Role |
|---|---|---|
| Max Barfuss | [@maxjbarfuss](https://github.com/maxjbarfuss) | Owner, final approver, sole maintainer of `main` |

## Authorized AI Agents

AI agents may contribute only under direct instruction or explicit written approval from @maxjbarfuss.

Agent requirements:
- Follow the [Agent Workflow in design/DESIGN.md](../design/DESIGN.md#agent-workflow) (session start/end, TDD, doc updates)
- Self-identify in commits and PR notes (`AI agent: <name>` or similar)

## PR Validation Checklist

Before opening a PR, ensure:

1. **Summary**: Clear description of what changed and why (including phase/component reference)
2. **Validation**:
   - `make lint` (ruff, mypy)
   - `make format-check` (black, isort, clang-format)
   - `make test` (Python + C++ tests, appropriate to phase)
   - All tests pass locally
3. **Risks & Next**: Document any known limitations, incomplete components, and follow-up work
4. **Docs updated**: If behavior/architecture changed, update [design/PLAN.md](../design/PLAN.md) and impacted docs in same PR

## Authorization Policy

Any contributor not listed above requires written approval from @maxjbarfuss before opening a PR. Unauthorized PRs may be closed without review.
