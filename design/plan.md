# Plan: Hybrid LLM with MLA + MoE + GRU

This is the execution plan for a local-first LLM (100–500M params) targeting dual 24GB GPUs for training, with inference designed to run on a single GPU (or CPU fallback), plus 8 CPU cores and streaming-scale data.

## Architecture (Current Baseline)

Flow: `Text -> Tokenizer -> Embeddings -> Input FFN -> Transformer (MLA + RoPE) -> MoE -> GRU -> Projection (tied)`

- Tokenizer baseline: GPT-2 BPE, vocab padded to 50304
- Tokenizer exploration: Unigram tokenizer as an alternative baseline candidate
- Max context target: 2048 (extensible)
- Embeddings: BF16, scaled by `sqrt(hidden_size)`, tied with output projection
- MLA: full-size Q, latent K/V (`latent_dim=512`), RoPE in latent space
- MoE: 16–32 experts, Top-2 routing, BF16 experts + gate
- Output: GRU stage for sequential continuity

Architecture is intentionally adjustable. Changes are allowed when supported by tests and benchmark evidence.

## Precision Policy (Current Baseline)

- Always BF16: attention, MoE, embeddings
- Progressive for FFN/RNN:
  - Phase 1 (0–30%): FP4 weights + FP8 activations
  - Phase 2 (30–80%): FP8 weights + FP8 activations
  - Phase 3 (80–100%): FP8/BF16 mixed runtime
- Inference KV cache: FP8
- Transition safety: automatic rollback on divergence

Precision policy is intentionally adjustable. Phase boundaries and runtime dtypes may be tuned based on stability and performance results.

## Tokenizer Exploration Track

- Compare GPT-2 BPE vs Unigram on the same data slices
- Evaluate: validation perplexity, throughput, sequence length efficiency, and tokenizer memory footprint
- Keep vocabulary alignment constraints explicit when swapping tokenizers
- Promote Unigram only if quality/performance is neutral or better

## Training Infrastructure

- Data: streaming-first for multi-TB corpus
- Cache: RAM LRU + SSD token cache
- Dataloader baseline: 6 workers, prefetch 2, pinned memory, persistent workers
- Distributed mode:
  - 100–300M: DDP
  - 300–500M: FSDP full sharding
- Scale mechanism: microbatching + gradient accumulation

## Training Program

1. Pre-train: large mixed corpus, long schedule, precision progression
2. Fine-tune: curated domain data, lower LR, earlier BF16 transition
3. Post-train: SFT, optional DPO/RLHF, calibration, export-ready weights

## Reliability and Observability

- Required metrics: loss, perplexity, throughput, memory, quantization error, expert utilization
- Required alerts: NaN/Inf, loss spikes, OOM, disk pressure, thermal issues
- Checkpoint policy: rolling recent + best, atomic writes, include RNG + precision phase

## Validation Gates

- Smoke overfit test (small model)
- Precision phase transition checks
- Multi-GPU consistency checks
- Attention backend equivalence checks
- Periodic generation regression samples

## Delivery Order

1. Base components: embeddings, RoPE, MLA, transformer block
2. MoE + GRU integration
3. Data pipeline and training loop
4. Distributed training + checkpointing
5. Monitoring + optimization (`torch.compile`, selective checkpointing)

## Definition of Done

The project is done when end-to-end pre-train/fine-tune/post-train is reproducible locally, checkpoint recovery is reliable, and quality/performance metrics remain stable across precision phases.
