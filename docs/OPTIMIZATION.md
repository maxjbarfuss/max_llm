# Project Optimization Learnings

**Purpose**: Consolidate validated optimization learnings across Phases 2-4 and keep one compact baseline reference.

**Sources**:
- `docs/PHASE_2_CLOSEOUT.md`
- `docs/PHASE_3_CLOSEOUT.md`
- `docs/PHASE_4_CLOSEOUT.md`

## Validated Defaults

- Flash Attention is the primary training backend at current scales; it enabled high-throughput long runs in Phase 3 and remained stable in Phase 4.
- bf16 mixed precision improved speed and memory efficiency in Phase 2 and remained stable across long Phase 3 and 4 runs.
- Gradient accumulation was used to reach larger effective batch sizes without OOM.
- Data-loader parallelism (`num_workers`, `prefetch_factor`, `pin_memory`) removed CPU bottlenecks in longer runs.
- DDP is production-valid for 2-GPU runs; FSDP is integrated for larger-memory pressure cases.
- Reproducibility guardrails (seed control, deterministic loader behavior, checkpoint restore correctness) are foundational and were proven in Phase 2.
- AdamW fused + WSD schedule is a robust long-run baseline from Phase 3 onward.
- Selective weight decay and gradient clipping are part of the stable stack used in successful Phase 3/4 runs.
- Plateau rescue via coordinated weight soup + data seed reset + SGDR restart worked in Phase 4 (P4-DEC-2).
- Curriculum ordering affects general quality: factual anchors first, broad web later, mixed consolidation after (P4-DEC-3).
- Low-LR simplicity anneal on clean structured text at the end improved final P4 quality.
- NFKC normalization and unknown-token filtering improved data cleanliness and training stability in Phase 3.
- Spill-cache and mmap-backed preparation/reads reduced RAM pressure and supported larger corpora in Phase 4.
- Domain-diverse validation gave better quality signals than domain-only validation.

## Phase Notes

### Phase 2
- Mixed precision (bf16) gave meaningful speedup with minimal quality impact.
- Large effective batches via gradient accumulation improved training practicality on single GPU.
- Embedding-only runs tolerated higher LR than the initial baseline, while high weight decay could stall learning.
- Reproducibility guardrails came before optimization tuning.

### Phase 3
- DecoderLM architecture change was the main unlock for large-vocab training quality.
- Flash Attention + bf16 + grad accumulation + fused AdamW + WSD produced stable long runs.
- Supporting wins: fused QKV, scaled residual init, chunked CE loss, gradient norm logging, and epoch-based sequence offset variation.
- Result: stable long schedules and coherent-output gate pass.

### Phase 4
- Best-performing stack combined FlashNorm, MLA, xIELU, block_attn residuals, and RoPE.
- 5-stage curriculum plus final simplicity anneal produced the best project checkpoint.
- P4-DEC-2: multi-axis plateau escape (soup + seed reset + SGDR restart).
- P4-DEC-3: domain transfer can regress general coherence despite better domain PPL.
- General quality tracking beat domain-matched validation alone.

## Current Baseline

```toml
[training]
attention_backend = "flash"           # flash|sage|xformers|standard
use_torch_compile = true
batch_size = 12
gradient_accumulation_steps = 10
scheduler_type = "wsd"                # cosine|wsd
wsd_stable_fraction = 0.55
wsd_decay_fraction = 0.35
wsd_decay_shape = "sqrt"              # linear|sqrt|lowered_linear
precision_schedule = [[0, -1, "bf16"]]
use_distributed = true
distributed_backend = "ddp"           # ddp|fsdp

[data]
num_workers = 4                        # use 8 for dual-GPU
prefetch_factor = 4
pin_memory = true
persistent_workers = false
```

Commands:
- Single GPU: `python -m src.training.train --config config/your_config.toml`
- Multi GPU: `torchrun --standalone --nnodes=1 --nproc_per_node=2 -m src.training.train --config config/your_config.toml --distributed`

## Runbook Notes

### P4-DEC-2 Plateau Escape

1. Select two checkpoints from the same basin.
2. Weight-average them (soup).
3. Reset data seed.
4. Restart LR cycle schedule (SGDR restart).
5. Validate in a short early window before committing full continuation.

- Observed usage in Phase 4: rescue workflow rather than default cadence.
- Log soup pair, seed change, and scheduler restart parameters.

### P4-DEC-3 Curriculum Ordering

1. Start with factual/structured anchors (Wikipedia, then Cosmopedia-like educational corpus).
2. Transition into broader web domains.
3. Consolidate with a mixed-corpus stage.
4. Optionally run a low-LR clean-text anneal at the end.

- Domain-only validation can overstate progress.
- During Phase 4 transitions, domain-diverse validation was used to avoid misleading progress signals.

## Troubleshooting

- DDP init conflict: `export MASTER_PORT=29501`
- Slow startup/hangs: check GPU state (`nvidia-smi`) and data-loader workers
- DDP divergence: verify same seed/config across ranks
- OOM with workers: reduce `num_workers` or `batch_size`, increase `gradient_accumulation_steps`
- Missing flash-attn: `pip install flash-attn --no-build-isolation` (fallback is `standard`)
- NCCL checks: `python -c "import torch; print(torch.cuda.is_available())"`

