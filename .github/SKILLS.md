# Agent Capability Baseline

Baseline knowledge and working discipline for contributors and coding agents.

## Required Technical Skills

- PyTorch 2.x (`nn.Module`, autograd, mixed precision, AMP, `torch.compile`)
- Transformer internals: attention, residuals, normalization, position encodings
- GQA, MLA, MoE, and GRU fundamentals as used here (see [design/DESIGN.md](../design/DESIGN.md#architecture-overview))
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

Speed is not the goal. Correctness, clarity, and minimal surface area are.

- **Read before writing.** Understand existing code, tests, and interfaces before modifying anything.
- **Clarify before building.** If scope or intent is ambiguous, ask — don't assume and implement.
- **Small verifiable steps.** Each step should be testable before the next begins.
- **Question every addition.** Does this already exist? Is it needed now? Is the simplest solution sufficient?
- **Consider consequences.** Before changing an interface, know what depends on it. Irreversible actions require more thought.
- **Leave code better, not just different.** Changes without clear purpose add noise. Be purposeful and reviewable.

## Working Rule

Follow the [Agent Workflow in design/DESIGN.md](../design/DESIGN.md#agent-workflow). In brief:

- **Session start**: Read [design/PLAN.md](../design/PLAN.md#phase-progress) (Phase Progress, Next Steps), then relevant [design/DESIGN.md](../design/DESIGN.md) section.
- **Before coding**: Check existing tests and interfaces.
- **During coding**: TDD — write failing test first, implement, verify.
- **Session end**: Update [design/PLAN.md](../design/PLAN.md#current-session-scratch-pad) (Scratch Pad, Running Session Log).
