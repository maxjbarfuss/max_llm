# Agent Capability Baseline

Baseline knowledge and working discipline for contributors and coding agents.

## Required Technical Skills

- PyTorch 2.x (`nn.Module`, autograd, mixed precision, AMP, `torch.compile`)
- Transformer internals: attention, residuals, normalization, position encodings
- MLA, MoE, and GRU fundamentals as used here (see `design/DESIGN.md`)
- Python 3.10+ with type hints, `pytest`, and `mypy`
- Git: atomic commits, clear PRs, docs and tests updated with behavior changes

## Task-Specific (Needed When Relevant)

- Distributed training (`DDP`/`FSDP`, gradient accumulation, checkpoint sharding)
- Training efficiency tooling (Flash Attention 2, `torch.compile`, FP8, selective checkpointing, activation offloading)
- Data streaming and token caching for large corpora
- C++20, CMake, and CUDA kernel development
- Monitoring, experiment reproducibility, and checkpoint reliability

## Engineering Judgment

Speed is not the goal. Correctness, clarity, and minimal surface area are.

- **Read before writing.** Understand existing code, tests, and interfaces before modifying anything.
- **Clarify before building.** If scope or intent is ambiguous, ask — don't assume and implement.
- **Small verifiable steps.** Each step should be testable before the next begins.
- **Question every addition.** Does this already exist? Is it needed now? Is the simplest solution sufficient?
- **Consider consequences.** Before changing an interface, know what depends on it. Irreversible actions require more thought.
- **Leave code better, not just different.** Changes without clear purpose add noise. Be purposeful and reviewable.

## Working Rule

At session start, read `design/PLAN_CHECKLIST.md` (`Next Steps`, `Running Session Log`), then the relevant section of `design/DESIGN.md`. Check existing tests and interfaces before modifying code. Write a failing test before implementing new behavior. Update `PLAN_CHECKLIST.md` at session end.
