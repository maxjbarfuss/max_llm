# Design Philosophy

This project is optimized for collaboration between technically competent humans and AI coding agents. The guiding goal is low cognitive load with high implementation reliability.

## Core Engineering Principles

- **KISS first:** choose the simplest correct design.
- **TDD-first:** write a failing test before implementation for new behavior.
- **Type-driven design:** typed public interfaces and validated configs.
- Treat time and space complexity as design constraints, not post-hoc concerns.
- Document expected complexity for non-trivial paths (especially training/data loops).
- Prefer asymptotically better algorithms before micro-optimizations.
- Set performance budgets for hot paths and validate with benchmarks/profiling.
- Avoid hidden quadratic behavior in token/sequence operations.
- Enforce SRP, OCP, LSP, ISP, and DIP in all new/modified code.
- Reject changes that violate SOLID unless there is a documented, benchmark-backed exception.
- **Composition over indirection:** prefer clear data flow to deep abstraction chains.
- **Reproducibility by default:** deterministic seeds, explicit config values, and checkpoint-compatible changes.
- **Fail fast on invalid config/state:** validate early with precise error messages.
- **API change discipline:** when behavior or interfaces change, update tests/docs in the same PR.
- **Measure before optimizing:** profile first; keep optimization decisions evidence-backed.

## Context-Efficient Code Structure

- Use vertical feature slices (keep related logic in one feature area).
- Keep files maintainable; split large files by responsibility.
- Expose clean public APIs through package `__init__.py` files.
- Avoid scattering feature logic across unrelated utility modules.

## Documentation Contract

- Docstrings are part of implementation quality.
- Update docs in the same PR when behavior or architecture changes.
- Keep roles clear:
  - `README.md`: fast orientation and links
  - `design/plan.md`: architecture and implementation targets
  - `design/plan-checklist.md`: session tracker (`Next Steps`, `Current Session Scratch Pad`, `Running Session Log`)
  - `CONTRIBUTING.md`: workflow and validation gates

## Quality Gates (Non-Negotiable)

Before proposing merge-ready work:

```bash
mypy src/
ruff check src/
black --check src/ tests/
pytest tests/ -v
```

## AI Agent Workflow Rules

### Start of session
1. Read `design/plan-checklist.md` (`Next Steps`, `Running Session Log`).
2. Read relevant section in `design/plan.md`.
3. Check existing tests and interfaces before modifying code.

### During work
- Keep edits focused and atomic.
- Preserve API stability unless change is intentional and documented.
- Add/adjust tests with each behavior change.
- Keep handoff notes concise and actionable.

### End of session
- Update `design/plan-checklist.md` (`Current Session Scratch Pad` + `Running Session Log`) with completed work, next step, and blockers.
- Ensure docs impacted by the change are updated in the same PR.

## Coding Standards

- Python 3.10+ idioms only.
- Type hints for public classes/functions.
- Minimal inline comments; prefer clear names and strong docstrings.
- Import order: stdlib -> third-party -> local.
- Avoid global state when deterministic behavior is required.

## Reliability and MLOps Baseline

New training-facing components should include:

- measurable metrics
- failure/alert conditions
- checkpoint compatibility considerations
- reproducibility hooks where relevant (seed/state preservation)

## Contributor Checklist

- [ ] Requirement is explicit and testable
- [ ] Test added/updated before or with implementation
- [ ] Public interfaces are typed
- [ ] Docs updated where behavior changed
- [ ] Validation commands pass locally
- [ ] `design/plan-checklist.md` handoff is updated
