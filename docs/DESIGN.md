# Max LLM Design

Purpose: durable architecture, data, and system reference for the project. Use [PLAN.md](PLAN.md) for execution, [OPTIMIZATION.md](OPTIMIZATION.md) for validated training learnings, and the phase closeouts for historical evidence.

## Current State

- Phase 4 is complete: final anneal checkpoint reached ppl ~9.2, a 2.6x improvement over the best Phase 3 run.
- The validated Phase 4 stack is FlashNorm + MLA + xIELU + block-attn residuals + RoPE on a 12-layer, ~60M-parameter decoder.
- Phase 5 is the next execution surface: data cleanup, tokenizer decision, inference wins, Muon/Z-loss/uP, then SFT/grounding/alignment.

## Phase Map

| Phase | Purpose | Main deltas | Canonical reference |
|---|---|---|---|
| 1 | Foundation | Repo layout, config/build/test/CI baseline | [PHASE_1_CLOSEOUT.md](PHASE_1_CLOSEOUT.md) |
| 2 | Skeleton + reproducibility | Char baseline, seed control, checkpointing, overfit/inference gates | [PHASE_2_CLOSEOUT.md](PHASE_2_CLOSEOUT.md) |
| 3 | Capable decoder baseline | Unigram 8K, DecoderLM, Flash/bf16/DDP/WSD long-run stack | [PHASE_3_CLOSEOUT.md](PHASE_3_CLOSEOUT.md) |
| 4 | Llama-style scale-up | FlashNorm, RoPE family, MLA, xIELU, block-attn residuals, curriculum + anneal | [PHASE_4_CLOSEOUT.md](PHASE_4_CLOSEOUT.md) |
| 5 | Post-training | KV-cache, LoRA/SFT, grounding, DPO or PPO/GRPO, continual-learning eval | [PLAN.md](PLAN.md) |
| 6 | Sparse expert stack | MoE on the Phase 5 stack, routing diagnostics, dense-vs-sparse comparison | [PLAN.md](PLAN.md) |
| 7 | Dual-stream reasoning | GRU reasoning stream + gated combiner + STaR-style loop | [PLAN.md](PLAN.md) |

## Architecture

### Validated Stack (Phase 4)

- Tokenizer: Unigram 8K inherited from Phase 3.
- Model: 12 layers, hidden size 1024, 16 heads, 4 KV heads, context 1024, ~60M parameters.
- Core blocks: FlashNorm, MLA with decoupled RoPE, xIELU FFN, block-attn residuals.
- Inference: top-k / top-p / temperature sampler with repetition penalty.

### Phased Architecture Diagrams

```mermaid
---
title: Phase 2 - Skeleton
---
graph TD
    A[Text]:::io --> B[Char Tokenizer]:::p2 --> C[Token Emb]:::p2 --> D[GELU MLP]:::p2 --> E[LM Head]:::p2 --> F[Logits]:::io
    F -->|training| G[Cross-Entropy Loss]:::io
    F -->|inference| H[Greedy Sampling]:::p2 --> I[Text]:::io
    classDef io fill:#212121,stroke:#FFFFFF,color:#FFFFFF,stroke-width:2px
    classDef p2 fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
```

```mermaid
---
title: Phase 3 - Minimal Transformer + Tokenizer Upgrade
---
graph TD
    A[Text]:::io --> B[Unigram Tokenizer 8K]:::p3 --> C[Token Emb]:::p2 --> Block
    subgraph Block[Transformer Block x N]
        direction LR
        D[LayerNorm]:::p3 --> E[Multi-Head Attn<br/>Fused QKV]:::p3 --> F[+ Residual]:::p3 --> G[LayerNorm]:::p3 --> H[GELU FFN]:::p3 --> I[+ Residual]:::p3
    end
    Block --> J[LM Head]:::p3 --> K[Logits]:::io
    K -->|training| L[Chunked CE Loss]:::p3
    K -->|inference| M[Sampler<br/>temp/top-k/top-p]:::p3 --> N[Text]:::io
    classDef io fill:#212121,stroke:#FFFFFF,color:#FFFFFF,stroke-width:2px
    classDef p2 fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
    classDef p3 fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
```

```mermaid
---
title: Phase 4 - Llama-Style Upgrades
---
graph TD
    A[Text]:::io --> B[Tokenizer]:::p3 --> C[Token Emb]:::p2 --> Block
    subgraph Block[Transformer Block x N]
        direction LR
        D[FlashNorm]:::p4 --> E[MLA + decoupled RoPE]:::p4 --> F[block_attn Residual]:::p4 --> G[FlashNorm]:::p4 --> H[xIELU FFN]:::p4 --> I[block_attn Residual]:::p4
    end
    Block --> J[LM Head]:::p3 --> K[Logits]:::io
    K -->|training| L[Cross-Entropy Loss]:::io
    K -->|inference| M[Sampler<br/>temp/top-k/top-p]:::p3 --> N[Text]:::io
    classDef io fill:#212121,stroke:#FFFFFF,color:#FFFFFF,stroke-width:2px
    classDef p2 fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
    classDef p3 fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    classDef p4 fill:#FFE0B2,stroke:#E65100,color:#BF360C
```

```mermaid
---
title: Phase 5 - Post-Training
---
graph TD
    A[Text]:::io --> B[Tokenizer]:::p3 --> C[Token Emb]:::p2 --> Block
    subgraph Block[Transformer Block x N]
        direction LR
        D[FlashNorm]:::p4 --> E[MLA + decoupled RoPE + KV-cache]:::p4 --> F[block_attn Residual]:::p4 --> G[FlashNorm]:::p4 --> H[xIELU FFN]:::p4 --> I[block_attn Residual]:::p4
    end
    Block --> J[LM Head]:::p3 --> K[Logits]:::io
    K -->|training| L[CE Loss + DPO]:::p5
    K -->|inference| M[Sampler + KV-cache]:::p5 --> N[Text]:::io
    LoRA:::p5 -.-> Block
    RewardModel:::p5 -.-> Block
    classDef io fill:#212121,stroke:#FFFFFF,color:#FFFFFF,stroke-width:2px
    classDef p2 fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
    classDef p3 fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    classDef p4 fill:#FFE0B2,stroke:#E65100,color:#BF360C
    classDef p5 fill:#E1BEE7,stroke:#6A1B9A,color:#4A148C
```

```mermaid
---
title: Phase 6 - MoE + MLA
---
graph TD
    A[Text]:::io --> B[Tokenizer]:::p3 --> C[Token Emb]:::p2 --> Block
    subgraph Block[Transformer Block x N]
        direction LR
        D[FlashNorm]:::p4 --> E[MLA + decoupled RoPE]:::p6 --> F[block_attn Residual]:::p4 --> G[FlashNorm]:::p4 --> H[MoE Sparse xIELU]:::p6 --> I[block_attn Residual]:::p4
    end
    LoRA:::p5 -.-> Block
    RewardModel:::p5 -.-> Block
    Block --> J[LM Head]:::p3 --> K[Logits]:::io
    K -->|training| L[CE Loss + DPO]:::p5
    K -->|inference| M[Sampler + KV-cache]:::p5 --> N[Text]:::io
    classDef io fill:#212121,stroke:#FFFFFF,color:#FFFFFF,stroke-width:2px
    classDef p2 fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
    classDef p3 fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    classDef p4 fill:#FFE0B2,stroke:#E65100,color:#BF360C
    classDef p5 fill:#E1BEE7,stroke:#6A1B9A,color:#4A148C
    classDef p6 fill:#FFCDD2,stroke:#C62828,color:#B71C1C
```

```mermaid
---
title: Phase 7 - Dual-Stream Reasoning
---
graph TD
    A[Text]:::io --> B[Tokenizer]:::p3 --> C[Token Emb]:::p2
    C --> TStream
    C --> RGRU[GRU Reasoning Stream]:::p7

    subgraph TStream[Transformer Stream x N]
        direction LR
        D[FlashNorm]:::p4 --> E[MLA + decoupled RoPE]:::p6 --> F[block_attn Residual]:::p4 --> G[FlashNorm]:::p4 --> H[MoE Sparse xIELU]:::p6 --> I[block_attn Residual]:::p4
    end

    LoRA:::p5 -.-> TStream
    RewardModel:::p5 -.-> TStream
    TStream --> COMB[GRU Combiner]:::p7
    RGRU --> COMB
    COMB --> J[LM Head]:::p3 --> K[Logits]:::io
    K -->|training| L[CE Loss + DPO]:::p5
    K -->|inference| M[Sampler + KV-cache]:::p5 --> N[Text]:::io
    classDef io fill:#212121,stroke:#FFFFFF,color:#FFFFFF,stroke-width:2px
    classDef p2 fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
    classDef p3 fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    classDef p4 fill:#FFE0B2,stroke:#E65100,color:#BF360C
    classDef p5 fill:#E1BEE7,stroke:#6A1B9A,color:#4A148C
    classDef p6 fill:#FFCDD2,stroke:#C62828,color:#B71C1C
    classDef p7 fill:#FFF9C4,stroke:#F57F17,color:#F57F17
```

### Forward-Carried Additions

| Phase | Additions on top of prior stack |
|---|---|
| 5 | KV-cache, YaRN or rope-base standardization, uP, Muon, Z-loss, LoRA, reward model, MoD candidate |
| 6 | Sparse MoE FFN and routing diagnostics |
| 7 | Parallel GRU reasoning stream and gated combiner |

### Component Inventory

| Component family | Current/default | Alternatives retained |
|---|---|---|
| Normalization | `flash` | `rms`, `dyt`, `crms`, `layer` |
| Position encoding | `rope` | `add_rope`, `alibi`, `rel_pos` |
| Attention | `mla` | `mha`, `swa`, `rla` |
| FFN | `xielu` | `swiglu`, `relu2`, `gelu` |
| Residual routing | `block_attn` | `full_attn`, `standard` |
| Fine-tuning | LoRA planned | full finetune intentionally deferred |

Hardware target: dual 24 GB-class RTX GPUs for training, single GPU inference, CPU fallback for correctness/debugging.

## Training System

- Data access: streaming plus RAM LRU and SSD token cache.
- Loader path: parallel workers, prefetch, pinned memory, persistent-worker support.
- Distributed path: DDP validated; FSDP available when memory pressure dominates.
- Attention backends: Flash preferred; Sage/xFormers/standard available with fallback.
- Compilation: `torch.compile` enabled on the stable path where backend compatibility allows it.
- Optimization baseline: AdamW fused + WSD scheduler + gradient accumulation + gradient clipping + bf16.
- Planned stack upgrades: Muon on 2-D weights, uP for hyperparameter transfer, Z-loss for logit stabilization.

## Data Design

Principle: unrestricted pretraining in Phases 2-4, then post-training safety and alignment in Phases 5-7.

### Actual Phase 4 Corpus

| Source | Tokens | Share | Key note |
|---|---|---|---|
| OpenWebText | 8.87B | 33% | Very short docs; strong packing/filtering target |
| FineWeb | 6.72B | 25% | Overlaps with FineWeb-Edu |
| Cosmopedia v2 | 5.37B | 20% | Synthetic Mistral-generated educational corpus |
| FineWeb-Edu | 2.68B | 10% | Subset of FineWeb; current double exposure |
| Wikipedia | 1.88B | 7% | Best factual anchor |
| WikiText-103 | 1.07B | 4% | Redundant with Wikipedia |
| TinyStories | 0.13B | 0.5% | Holdover from early phases |
| stories_young_children | 0.13B | 0.5% | Holdover from early phases |

### Current Quality Path

- NFKC normalization.
- Cc control-character stripping.
- Minimum-length filtering.
- Two-stage UNK filtering.
- Stratified per-source validation split.
- Spill-cache tokenization with resume support.

### Next Data Corrections

- Remove WikiText-103 from future pretraining mixes.
- Choose a single FineWeb policy instead of FineWeb + FineWeb-Edu overlap.
- Add OWT heuristics, language filtering, and near-dedup.
- Add sequence packing for short-document efficiency.
- Retrain tokenizer on the cleaned Phase 4 distribution if more pretraining is planned.

### Phase 5+ Data Shapes

| Phase | Data products |
|---|---|
| 5 | Cleaned pretraining corpus, 32K tokenizer decision, SFT pairs, grounding corpus, preference pairs, reward labels |
| 6 | Partitioned SFT/preference data for expert specialization |
| 7 | Reasoning-trace triples and STaR-derived traces |

## Data Pipeline

```mermaid
flowchart LR
    A[Raw text] --> B[Normalize + filter]
    B --> C[Near-dedup]
    C --> D[Tokenize + cache]
    D --> E[Mix + sequence pack]
    E --> F[Training shards]
```

Design rule: expensive cleanup and dedup happen once per source; mixed packed shards are then reused across runs.

## Config Evolution

- Config classes are versioned so schema changes stay explicit.
- Required breaking fields increment version; optional defaulted fields do not.
- Checkpoints store config-version metadata to protect reproducibility.
- Runtime values remain canonical in [config/README.md](../config/README.md) and the milestone or ephemeral config files.
