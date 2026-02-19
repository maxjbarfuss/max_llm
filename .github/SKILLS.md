# Required Skills

Baseline capabilities expected for contributors and AI agents.

## Required Baseline

- PyTorch 2.x (`nn.Module`, autograd, mixed precision)
- Transformer internals (attention, residuals, normalization)
- MLA, MoE, and GRU fundamentals used in this repo
- Python 3.10+ with type hints and `pytest`
- Git workflow (atomic commits, clear PRs)

## Task-Specific Skills (Needed When Relevant)

- Distributed training (`DDP`/`FSDP`, gradient accumulation)
- Precision scheduling (FP4/FP8/BF16 transitions)
- Data streaming and caching for large datasets
- Performance tooling (`torch.compile`, Flash Attention)
- Monitoring and checkpoint reliability

## Working Rule

If you are missing context for a task, read [design/plan-checklist.md](../design/plan-checklist.md) (`Next Steps`, `Running Session Log`) first, then [design/plan.md](../design/plan.md) and [design/philosophy.md](../design/philosophy.md). Write a failing test before implementation.
