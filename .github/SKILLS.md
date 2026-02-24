# Agent Capability Baseline

Baseline knowledge and working discipline for contributors and coding agents.

## Required Technical Skills

- PyTorch 2.x (`nn.Module`, autograd, mixed precision, AMP, `torch.compile`)
- Transformer internals: attention, residuals, normalization, position encodings
- GQA, MLA, MoE, and GRU fundamentals as used here (see [docs/DESIGN.md](../docs/DESIGN.md#architecture-overview))
- Python 3.10+ with type hints, `pytest`, and `mypy`
- Git: atomic commits, clear PRs, docs and tests updated with behavior changes

## Task-Specific (Needed When Relevant)

- **Phase 4+**: Distributed training (`DDP`/`FSDP`, gradient accumulation, checkpoint sharding, multi-GPU throughput)
- **Phase 5+**: Architecture comparison experiments, benchmark design, ablation frameworks
- **Phase 6+**: Eval benchmark selection and integration (HellaSwag, MMLU, task-specific)
- **Anytime**: Training efficiency tooling (Flash Attention 2, `torch.compile`, FP8, selective checkpointing, activation offloading), data streaming and token caching
- **Optional**: C++20, CMake, and CUDA kernel development for performance-critical layers
- **Essential for all**: Experiment reproducibility, checkpoint reliability, artifact naming conventions

## Engineering Judgment

- **Read before writing.** Understand existing code, tests, and interfaces before modifying anything.
- **Clarify before building.** If scope or intent is ambiguous, ask — don't assume and implement.
- **Small verifiable steps.** Each step should be testable before the next begins.
- **Question every addition.** Does this already exist? Is it needed now? Is the simplest solution sufficient?
- **Consider consequences.** Before changing an interface, know what depends on it. Irreversible actions require more thought.
- **Leave code better, not just different.** Changes without clear purpose add noise. Be purposeful and reviewable.

## Objective and Scope

- **Primary objective**: Complete explicitly requested work first.
- **Secondary objective**: Add only obvious, in-scope improvements that directly support the request.
- **Scope guardrail**: Do not broaden into unrelated refactors or speculative features.
- **Design fixes**: Address clear oversights only when they are relevant to the current task.

## Low-Interaction Mode

- Prefer action over questions when instructions and project docs are sufficient.
- Ask questions only if a blocker is real and cannot be resolved from repository context.
- When assumptions are needed, choose the simplest option consistent with current docs and code.

## Protected Files Policy

- Do not modify governance or instruction files unless explicitly requested.
- Respect repository policies and source-of-truth docs ([docs/PLAN.md](../docs/PLAN.md), [CONTRIBUTING.md](../CONTRIBUTING.md), [docs/DESIGN.md](../docs/DESIGN.md)).

## Definition of Done

Work is complete only when all are true:

- Explicit requirements are implemented.
- Relevant tests and validation commands pass.
- Affected docs are updated and consistent.
- Changes are clean, concise, and maintainable.
- Remaining risks or follow-up work are documented.

## Working Rule

Follow the [Agent Workflow in docs/DESIGN.md](../docs/DESIGN.md#agent-workflow). In brief:

- **Session start**: Read [docs/SESSION.md](../docs/SESSION.md) (current focus, next steps), then [docs/PLAN.md](../docs/PLAN.md#phase-progress) current phase tasks.
- **Before coding**: Check existing tests and interfaces.
- **During coding**: TDD — write failing test first, implement, verify.
- **Session end**: Update [docs/SESSION.md](../docs/SESSION.md) (Scratch Pad, Running Session Log).

### Tooling Policy

- Prefer local tools and repository-native workflows for core development tasks.
- **NEVER use MCP servers** (including GitKraken, GitLens, or any git-related MCP tools).
- **ALWAYS use local Git commands** for all git operations: `git status`, `git add`, `git commit`, `git diff`, `git log`, `git push`, etc.
- **Commits are always local first**; push to remote only with explicit human confirmation.
