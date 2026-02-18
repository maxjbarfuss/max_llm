# Plan: Hybrid LLM with MLA-Transformer-MoE-RNN (Expanded)

This plan builds a local, learning-focused LLM (100-500M params) with a hybrid architecture: input FFN for richer embeddings, MLA attention for efficient long context, MoE for capacity without full compute cost, and a GRU output stage for sequential continuity. It is designed for dual 24GB GPUs, 8 CPU cores, and a very large dataset (~5TB) using streaming and progressive precision training.

## 1) Architecture overview

Flow: Text -> Tokenizer -> Token Embeddings -> Input FFN -> Transformer Blocks (MLA + RoPE) -> MoE Layers -> Output GRU -> Projection (tied weights)

Why this design:
- MLA reduces KV cache via latent compression, enabling longer context at lower memory cost.
- MoE adds capacity without full compute per token.
- GRU output adds sequential continuity for generation.
- Input FFN enriches embeddings before attention.

## 2) Tokenization and embeddings

Tokenizer:
- GPT-2 BPE tokenizer, vocab padded to 50304.
- Left padding for batched generation.
- Max length 2048 (extend later if needed).

Token embeddings:
- BF16 only (never quantized).
- Scaled by sqrt(hidden_size).
- Tied with output projection weights for efficiency.

## 3) Attention and position encoding

MLA attention:
- Q stays full-size, K/V compressed to latent_dim=512.
- RoPE applied in latent space for position signals.
- Flash Attention 2/3 for speed and memory.
- All MLA weights in BF16.

Benefit:
- 75% KV cache reduction from MLA, plus FP8 KV cache for inference.

## 4) Transformer blocks and MoE

Transformer block:
- Pre-LN: x + MLA(LN(x)), x + FFN(LN(x)).
- FFN in FP8 (FP4 early phase), fused MLP kernels.

MoE layer:
- 16-32 experts, Top-2 routing.
- Experts and gating in BF16.
- Scheduled load-balance loss to avoid collapse.

## 5) Output GRU

- GRU weights FP8, hidden state BF16.
- Reset per batch during training, persistent state in inference.

## 6) Progressive precision training

Schedule:
- Phase 1 (0-30%): FP4 weights for non-MoE FFN and RNN, FP8 activations.
- Phase 2 (30-80%): FP8 weights for those layers.
- Phase 3 (80-100%): Mixed BF16 (attention/MoE/embeddings) + FP8 (FFN/RNN).

Policy:
- BF16 for attention, MoE, embeddings always.
- Auto-rollback on divergence during transitions.

## 7) Data pipeline (5TB, 8 cores)

Constraints:
- 5TB too large to fully pretokenize.
- 8 CPU cores limit preprocessing throughput.

Strategy:
- Streaming-first, shard 64-128 pieces.
- Pre-tokenize hot subsets, stream the rest.
- Two-tier cache: RAM LRU + SSD cache for tokenized shards.
- Dataloader: 6 workers, prefetch 2, pinned memory, persistent workers.
- Deterministic validation slice cached locally.

## 8) Distributed training setup

- 100-300M: DDP.
- 300-500M: FSDP full sharding.
- CPU offload optimizer states for large models.
- Microbatching + grad accumulation to reach effective batch sizes.

## 9) Training phases

A) Pre-train:
- 5TB mixed corpus, streaming.
- FP4 -> FP8 -> mixed BF16/FP8.
- Long schedule, large batch.

B) Fine-tune:
- 10-200GB curated domain data.
- FP8 then early BF16.
- Lower LR, higher MoE balance weight earlier.

C) Post-train:
- Instruction tuning (SFT).
- Optional DPO/RLHF for preferences.
- Calibration (temperature, penalties).
- Expert pruning/merging, export BF16.

## 10) Optimizations (early wins)

- Flash Attention for MLA.
- torch.compile for kernel fusion.
- Fused LayerNorm/MLP/optimizer.
- Selective checkpointing (attention only).
- Dynamic batching by length.
- Quantized KV cache for inference.

## 11) Monitoring and safety

Track:
- Loss, perplexity, throughput, memory, quant error.
- Expert utilization and balance loss.

Alerts:
- NaN/Inf, loss spikes, OOM, disk low, GPU temp.

Checkpointing:
- Rolling last 3 + best, atomic saves.
- Store precision state and RNG state.

## 12) Validation and testing

- Smoke test (10M params) to overfit.
- Verify precision transitions.
- Multi-GPU sync tests.
- Flash Attention equivalence checks.
- Regular generation samples for degeneration detection.

## Outcome

A local, trainable hybrid LLM with efficient long-context attention, conditional MoE capacity, progressive precision training, and a full pretrain -> finetune -> post-train pipeline with monitoring and safety guardrails.
