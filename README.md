# Max LLM

Hands-on LLM research lab for building, training, and evaluating modern architectures on local hardware.

**Status:** Current phase and progress live in [docs/SESSION.md](docs/SESSION.md) and [docs/PLAN.md](docs/PLAN.md).

**Repository:**
[github.com/maxjbarfuss/max_llm](https://github.com/maxjbarfuss/max_llm)

## Why This Project?

**Learn by building.** Hands-on lab for modern sequence modeling where architecture and training decisions are tested, measured, and explained through incremental implementation.

**Experiment locally.** Optimized for consumer dual-GPU hardware and fast iteration. Validates tradeoffs in throughput, memory, and quality without cloud infrastructure.

**Reproducibility by design.** Configs, seeds, checkpoints, and data pipeline tracked for reliable re-runs. Test-driven development with measurable validation at each phase.

---

## Quick Start

Start with setup, then pick the entrypoint you need:

- [scripts/setup/README.md](scripts/setup/README.md): full environment setup steps and platform requirements
- [scripts/build/README.md](scripts/build/README.md): C++ build wrapper and common build modes
- [scripts/data/README.md](scripts/data/README.md): end-to-end data prep workflow with various configurations and datasets
- [src/training/README.md](src/training/README.md): how to run training with various configurations
- [src/inference/README.md](src/inference/README.md): how to run inference on multiple configurations
- [tests/README.md](tests/README.md): how to run tests from the repo root or the tests/ folder

Docs and policies:

- [docs/SESSION.md](docs/SESSION.md): current focus, next steps, and session log (start here for status)
- [docs/PLAN.md](docs/PLAN.md): phased execution roadmap and exit criteria (reference for current phase)
- [docs/DESIGN.md](docs/DESIGN.md): architecture, engineering constraints, and data strategy
- [CONTRIBUTING.md](CONTRIBUTING.md): repository workflow and contributor authorization
- [.github/AGENTS.md](.github/AGENTS.md): AI agent standard — open, interoperable specification for any agent (Claude, o1, custom models)
- [.github/SKILLS.md](.github/SKILLS.md): AI agent instructions and working discipline
- [.github/LESSONS.md](.github/LESSONS.md): recorded agent mistake patterns (read before each session)
- [.github/CODEOWNERS](.github/CODEOWNERS): code ownership and review responsibility

## Design

**7-phase progression** (100–500M params on dual-GPU hardware): minimal working model → incremental architectural upgrades → full pretraining → post-training alignment. Each phase produces a working text-in → text-out LLM with measurable validation.

**Final architecture (Phase 7):** Transformer with modern optimizations (RMSNorm, RoPE, MLA, sparse MoE) + dual-stream reasoning (GRU reasoning + transformer streams → GRU combiner). Solid rectangles show Phase 7 components; rounded pills show replaced predecessors color-coded by introduction phase.

```mermaid
graph TD
    In[Text Input]:::io --> Tok[BPE/Unigram Tokenizer]:::p3 --> Emb[Token Embedding]:::p2 --> N1
    Emb --> RGRU[GRU Reasoning Stream]:::p7

    subgraph Block[Transformer Stream × N]
        N1[RMSNorm]:::p4 --> ATT[MLA]:::p6 --> R1[+ Residual]:::p3
        RoPE[RoPE]:::p4 -.-> ATT
        R1 --> N2[RMSNorm]:::p4 --> MOE[MoE Sparse SwiGLU]:::p6 --> R2[+ Residual]:::p3
    end

    LoRA:::p5 -.-> Block
    RewardModel:::p5 -.-> Block

    R2 --> COMB[GRU Combiner]:::p7
    RGRU --> COMB
    COMB --> Head[LM Head]:::p3 --> Logits[Logits]:::io
    Logits -->|training| Loss[CE Loss + DPO]:::p5
    Logits -->|inference| Samp["Sampler<br/>(top-p/temp/top-k)<br/>+ KV-cache"]:::p5 --> GenOut[Generated Text]:::io

    subgraph Replaced[Replaced Predecessors]
        direction LR
        CT([Char Tokenizer]):::p2
        LPE([Learned Pos Emb]):::p2
        LF([GELU MLP]):::p2
        LN([LayerNorm]):::p3
        MHA([Multi-Head Attn]):::p3
        GF([GELU FFN]):::p3
        GQA2([GQA]):::p4
        SW([Dense SwiGLU]):::p4
    end

    CT -.-> Tok
    LPE -.-> RoPE
    LF -.-> GF
    LN -.-> N1
    MHA -.-> GQA2
    GF -.-> SW
    GQA2 -.-> ATT
    SW -.-> MOE

    classDef io fill:#212121,stroke:#FFFFFF,color:#FFFFFF,stroke-width:2px
    classDef p2 fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
    classDef p3 fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    classDef p4 fill:#FFE0B2,stroke:#E65100,color:#BF360C
    classDef p5 fill:#E1BEE7,stroke:#6A1B9A,color:#4A148C
    classDef p6 fill:#FFCDD2,stroke:#C62828,color:#B71C1C
    classDef p7 fill:#FFF9C4,stroke:#F57F17,color:#F57F17
```

| Color | Phase | Component |
|-------|-------|-----------|
| ⬛ Black | — | I/O: Text Input, Logits, Loss, Generated Text |
| 🟢 Green | 2 | Token Embedding |
| 🔵 Blue | 3 | Residual connections, LM Head, BPE / Unigram Tokenizer |
| 🟠 Orange | 4 | RMSNorm, RoPE, GQA (Llama-Style Upgrades; GQA replaced in Phase 7) |
| 🟣 Purple | 5 | **LoRA** adapters (parameter-efficient fine-tuning), **Reward model** (for DPO alignment), **Sampler** (top-p/temperature/top-k), **KV-cache** optimization, DPO/RLHF, SFT, grounding |
| 🔴 Red | 6 | MLA (replaces GQA), MoE Sparse SwiGLU (replaces dense GQA FFN) |
| 🟡 Yellow | 7 | GRU Reasoning Stream (parallel to transformer), GRU Combiner (gated fusion) |
| Rounded pill | — | Replaced predecessors (colored by introducing phase) |

### Data Strategy

**1–500M token progression** across 7 phases: toy datasets (Phase 2–3) → unrestricted pretraining with curriculum learning (Phase 4: 75% neutral/technical, 20% adult/controversial, 5% harmful) → post-training alignment (Phase 5–7: SFT, grounding, preference data, reasoning traces). Safety guardrails applied via post-training after establishing comprehensive generalization.

| Phase | Tokens | Data Sources & Purpose | Training Configuration |
|-------|--------|------------------------|------------------------|
| **2** | 1–10M | **TinyStories + WikiText-103** — establish reproducibility, overfit tests, seed hardening | Single-GPU, char tokenizer, learning loop validation |
| **3** | 10–50M | **WikiText BPE (442K tokens, 4.54 chars/token)** — single-GPU training stability, baseline transformer | Single-GPU, BPE tokenizer, attention + FFN, LR scheduling |
| **4** | 10–500M | **OpenWebText (10–50M) → FineWeb (50–100M) → Curriculum (100–500M)** — staged introduction: 75% neutral/technical, 20% adult/controversial, 5% harmful; curriculum learning based on validation loss | Multi-GPU (DDP/FSDP), distributed training, torch.compile, chunked token caching |
| **5** | 1–5M SFT<br/>50K–500K grounding<br/>10K–100K preference | **SFT pairs** (OpenAssistant, ShareGPT) + **grounding** (GSM8K, MATH, ARC) + **preference data** (HH-RLHF, UltraFeedback) + 5–10% harmful for robustness | LoRA fine-tuning, KV-cache inference, DPO alignment, reward modeling |
| **6** | 1–5M pairs | **Partitioned SFT + preference data by topic/domain** — drive expert specialization; curriculum scheduling for expert drift monitoring | MoE routing diagnostics, expert utilization entropy tracking |
| **7** | 50K–500K triples | **Reasoning traces** (GSM8K, MATH, ARC-Challenge, OpenOrca/Orca-2) + **STaR self-generated** — 60% reasoned / 40% direct mix | Dual-stream training with teacher forcing, reasoning accuracy validation |

**Full details:** Architecture decisions, component rationale, training efficiency (BF16/FP8), data sourcing principles, and engineering constraints in [docs/DESIGN.md](docs/DESIGN.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
