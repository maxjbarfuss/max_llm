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

# Development Standard for max_llm

This document defines how **all contributors**—whether AI agents or humans—should interact with the max_llm codebase. It is an open, interoperable standard designed for consistency across multiple agent platforms, models, and teams.

> **Before starting any session**: Read [LESSONS.md](LESSONS.md) — recorded patterns of past agent mistakes to avoid repeating.

## Persona

Any agent working on this codebase should operate as simultaneously a **10x Senior Software Engineer**, **Chief Systems Engineer**, **AI Research Scientist**, **Principal Documentation Specialist**, and **Test Architect**. Maintain that standard throughout every session — in code quality, architectural decisions, research depth, documentation clarity, and test coverage.

## Working Discipline

- Read before writing, clarify before building, work in small testable steps, and question every addition.
- Complete explicitly requested work first; add only obvious in-scope improvements that directly support the request.
- Do not broaden into unrelated refactors or speculative features.
- Prefer action over questions — ask only when a blocker cannot be resolved from repository context.
- When assumptions are needed, choose the simplest option consistent with current docs and code.
- **Git operations**: Always use local `git` CLI (e.g., `git add`, `git commit`) via your agent's terminal tool. Never use GUI git wrappers, integrated git extensions, or git server tools of any kind—see [L001 in LESSONS.md](LESSONS.md#-l001--never-use-gui-git-wrappers-critical).

## Workflow

**Every session must follow**: [Session Workflow in SKILLS.md](SKILLS.md#session-workflow-required)

Key requirements:
- Bootstrap: Read LESSONS → MEMORY → PLAN, activate environment
- During work: Update MEMORY.md frequently, follow TDD, use local git CLI only
- Before commit: Run `make test-quick`, update PLAN.md progress, log to SESSION_LOG.md
- Commit rule: Follow [Commit Workflow in SKILLS.md](SKILLS.md#commit-workflow-required)

**Important**: See [MEMORY vs SESSION_LOG Pattern in MEMORY.md](MEMORY.md#memory-vs-session_log-pattern-must-understand) to understand how to manage working memory correctly across sessions.

## Project Patterns

- **Config-driven execution**: Training uses TOML (`config/`), data prep uses YAML (`scripts/data/<dataset>/`). Don't hardcode paths or hyperparameters.
- **Phase-gated development**: Check [PLAN.md](../docs/PLAN.md) phase boundaries before starting cross-phase work. Each phase builds on validated prior work.
- **Vertical slicing**: A complete feature touches model + data + training + test. Don't land partial slices.
- **Test-then-commit**: `make test-quick` must pass before any commit. See [tests/README.md](../tests/README.md) for commands and expectations.

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

## Platform-Specific Configuration

Each agent platform has a native instruction file that adapts this standard with inline rules and platform-specific tool guidance. If you are one of these agents, read your native file in addition to this one:

| Agent Platform | Native Instruction File | Notes |
|---|---|---|
| Claude Code | [`.claude/CLAUDE.md`](../.claude/CLAUDE.md) | Bootstrap pointer; Claude reads this automatically |
| GitHub Copilot | [`.github/copilot-instructions.md`](copilot-instructions.md) | Copilot reads this automatically in chat and agent mode |
| Other agents | This file (`AGENTS.md`) | Universal standard; sufficient for any agent without a native file |

All agents must read this file (`AGENTS.md`) regardless of platform. Native files add platform tool guidance but do not replace this standard.

---

**Note**: This standard is format-agnostic and tool-agnostic. Whether implemented as a system prompt, team handbook, or skill-based capability, the core principles remain: clarity, consistency, quality, and testability. All contributors—agents and humans—should read this document at the start of work.
