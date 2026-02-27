# Max LLM

Hands-on LLM research lab for building, training, and evaluating modern architectures on local hardware.

**Status:** Current phase and progress live in [docs/SESSION.md](docs/SESSION.md) and [docs/PLAN.md](docs/PLAN.md).

**Repository:**
[github.com/maxjbarfuss/max_llm](https://github.com/maxjbarfuss/max_llm)

## Why This Project?

**Learn by building.** Max LLM is a hands-on lab for modern sequence modeling. Each phase adds one core capability so architecture and training decisions can be tested, measured, and explained.

**Experiment locally.** The project is optimized for consumer GPUs and fast iteration. The goal is to validate tradeoffs in throughput, memory, and quality without relying on cloud-scale infrastructure.

**Progressive roadmap.** The 8 phases move from minimal tokenization and linear models to full transformers and hybrid architectures. The emphasis is on clear, verifiable improvements rather than paper-chasing.

**Data strategy with intent.** Data selection, preprocessing, and evaluation are treated as first-class engineering work.

**Reproducibility by design.** Configs, seeds, and checkpoints are tracked so experiments can be re-run and compared reliably.

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
- [.github/SKILLS.md](.github/SKILLS.md): AI agent instructions and working discipline
- [.github/LESSONS.md](.github/LESSONS.md): recorded agent mistake patterns (read before each session)
- [.github/CODEOWNERS](.github/CODEOWNERS): code ownership and review responsibility

## Architecture

Phase 8 combines GQA/MLA attention, MoE feedforward blocks, and a GRU output stage into a single local-first stack. The diagram below is a Phase 8 snapshot of the full text → output pipeline. Solid rectangles are current components in Phase 8. Rounded pills show predecessors replaced in earlier phases, colored by the phase they were introduced. For deeper design context, see [docs/DESIGN.md](docs/DESIGN.md).

```mermaid
graph TD
    In[Text Input]:::io --> Tok[BPE/Unigram Tokenizer]:::p3 --> Emb[Token Embedding]:::p2 --> N1

    subgraph Block[Transformer Block x N]
        N1[RMSNorm]:::p4 --> ATT[MLA]:::p7 --> R1[+ Residual]:::p3
        RoPE[RoPE]:::p4 -.-> ATT
        R1 --> N2[RMSNorm]:::p4 --> MOE[MoE Sparse SwiGLU]:::p7 --> R2[+ Residual]:::p3
    end

    R2 --> GRU[GRU Block]:::p8 --> Head[LM Head]:::p3 --> Logits[Logits]:::io
    Logits -->|training| Loss[Cross-Entropy Loss]:::io
    Logits -->|inference| Samp[Sampler<br/>temp/top-k/top-p]:::p3 --> GenOut[Generated Text]:::io

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
    Emb -.-> LPE
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
    classDef p7 fill:#FFCDD2,stroke:#C62828,color:#B71C1C
    classDef p8 fill:#FFF9C4,stroke:#F57F17,color:#F57F17
```

| Color | Phase | Component |
|-------|-------|-----------|
| ⬛ Black | — | I/O: Text Input, Logits, Loss, Generated Text |
| 🟢 Green | 2 | Token Embedding |
| 🔵 Blue | 3 | Residual connections, LM Head, BPE / Unigram Tokenizer, Sampler (temp/top-k/top-p) |
| 🟠 Orange | 4 | RMSNorm, RoPE, GQA (Llama-Style Upgrades) |
| 🔴 Red | 7 | MLA (replaces GQA), MoE (replaces dense SwiGLU) |
| 🟡 Yellow | 8 | GRU hybrid blocks |
| Rounded pill | — | Replaced predecessors (colored by introducing phase) |

## License

Apache License 2.0. See [LICENSE](LICENSE).
