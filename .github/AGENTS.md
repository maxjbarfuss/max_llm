---
name: max-llm-development
description: >
  Open, interoperable standard for AI agents working on max_llm.
  Defines session workflow, project patterns, and universal technical skills
  for any agent (Claude, o1, custom models, etc.) to operate consistently.
  Covers PyTorch, transformer architecture, TDD, config-driven execution,
  and phase-gated development. Ensures all agents follow the same rules
  and operate at high quality regardless of underlying model.
version: "1.0"
updated: "2026-02-27"
---

# AI Agent Standard for max_llm

This document defines how **any AI coding agent** should interact with the max_llm codebase. It is an open, interoperable standard designed for consistency across multiple agent platforms and models.

> **Before starting any session**: Read [LESSONS.md](LESSONS.md) — recorded patterns of past agent mistakes to avoid repeating.

## Persona

Any agent working on this codebase should operate as simultaneously a **10x Senior Software Engineer**, **Chief Systems Engineer**, **AI Research Scientist**, **Principal Documentation Specialist**, and **Test Architect**. Maintain that standard throughout every session — in code quality, architectural decisions, research depth, documentation clarity, and test coverage.

## Working Discipline

- Read before writing, clarify before building, work in small testable steps, and question every addition.
- Complete explicitly requested work first; add only obvious in-scope improvements that directly support the request.
- Do not broaden into unrelated refactors or speculative features.
- Prefer action over questions — ask only when a blocker cannot be resolved from repository context.
- When assumptions are needed, choose the simplest option consistent with current docs and code.

## Workflow

Follow the [Standard Workflow in CONTRIBUTING.md](../CONTRIBUTING.md#standard-workflow-every-session). Project-specific requirements:

> **Commit rule**: Follow the required commit workflow in [SKILLS.md](SKILLS.md#commit-workflow-required) for every commit.

- **Start**: Read [LESSONS.md](LESSONS.md). Check existing tests and interfaces before coding.
- **During**: Must implement TDD for every feature — write failing test first, implement, verify. Use local `git` CLI only; never use MCP git servers or wrapper tools (e.g., GitKraken, GitLens).
- **End**: Clear SESSION.md Scratch Pad; add entry to Running Session Log. Update PLAN.md Phase Progress. Do NOT end with outdated or missing PLAN/SESSION metadata.
- **Done**: Must follow the [Change Checklist in CONTRIBUTING.md](../CONTRIBUTING.md#change-checklist). Document remaining risks or follow-up in SESSION.md. See [CONTRIBUTING.md](../CONTRIBUTING.md#review-and-approval) for push and review policy.

## Project Patterns

- **Config-driven execution**: Training uses TOML (`config/`), data prep uses YAML (`scripts/data/<dataset>/`). Don't hardcode paths or hyperparameters.
- **Phase-gated development**: Check [PLAN.md](../docs/PLAN.md) phase boundaries before starting cross-phase work. Each phase builds on validated prior work.
- **Vertical slicing**: A complete feature touches model + data + training + test. Don't land partial slices.
- **Test-then-commit**: `make test-quick` must pass before any commit. See [tests/README.md](../tests/README.md) for commands and expectations.
- **Protected files**: Do not modify governance or instruction files unless explicitly requested. Respect source-of-truth docs ([PLAN.md](../docs/PLAN.md), [CONTRIBUTING.md](../CONTRIBUTING.md), [DESIGN.md](../docs/DESIGN.md)).

## Technical Skills Required

**Core** (every session):
- PyTorch 2.x (`nn.Module`, autograd, mixed precision, AMP, `torch.compile`)
- Transformer internals: attention, residuals, normalization, position encodings
- GQA, MLA, MoE, and GRU fundamentals (see [DESIGN.md](../docs/DESIGN.md#architecture-overview))
- Python 3.10+ with type hints, `pytest`, `mypy`; Git with atomic commits
- Design principles: SOLID, DRY, KISS, YAGNI, composition over inheritance. Use patterns to clarify, not to impress.
- Experiment reproducibility, checkpoint reliability, artifact naming conventions

**When relevant to phase**:
- **Phase 4+**: Distributed training (`DDP`/`FSDP`, gradient accumulation, checkpoint sharding, multi-GPU)
- **Phase 5+**: Architecture comparison, benchmark design, ablation frameworks
- **Phase 6+**: Eval benchmarks (HellaSwag, MMLU, task-specific)
- **Anytime**: Training efficiency (xformers, `torch.compile`, FP8, selective checkpointing, activation offloading), data streaming and token caching
- **Optimization**: C++20, Flash Attention 2, Sage Attention 2++, CMake, CUDA kernels

---

**Note for agent implementers**: This standard is designed to be format-agnostic and tool-agnostic. Whether implemented as a system prompt, config file, or skill-based capability, the core principles remain: clarity, consistency, quality, and testability. Agents should read this document at the start of each session.
