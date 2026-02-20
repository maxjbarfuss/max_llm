# Max LLM

Hybrid LLM research project (100–500M params): GQA/MLA + MoE + GRU output, optimized for local training on consumer hardware.

**Status:** Phase 1 (foundation ⏳ in progress) → Phase 2 (skeleton) next
**Repository:** [github.com/maxjbarfuss/max_llm](https://github.com/maxjbarfuss/max_llm)

## Why This Project?

**Learn by building.** Max LLM is a hands-on laboratory for understanding modern LLM architectures: how attention works, why mixture-of-experts scales differently, how to trade off memory for speed, and what makes one tokenizer better than another. Each phase builds incrementally—no black boxes, no mystery.

**Experiment locally.** Train and iterate on consumer hardware (dual 24GB GPUs). No cloud costs, no waiting for expensive cluster time. Benchmark new ideas cheaply, measure trade-offs precisely, and keep reproducible records.

**9-phase roadmap (phases 1–9) from scratch to hybrid models.** Start with character-level tokenization and linear layers (Phase 2), progress through standard transformers (Phase 3–4 with modern upgrades), then explore exotic architectures: mixture-of-experts (Phase 8) and GRU-transformer hybrids (Phase 9). **See what actually works**, not what papers claim.

**Comprehensive data strategy.** Build an unrestricted, diverse world model across Phases 2–5 (100–500M tokens including adult, controversial, and specialized content) for robust generalization. Layer safety guardrails through SFT and DPO in Phases 6–9. See [design/DESIGN.md#data-strategy-summary](design/DESIGN.md#data-strategy-summary) for sourcing guidelines and phase-by-phase data tasks.

**Reproducibility as a first principle.** Deterministic seeds, explicit configs, atomic commits linked to results. Every experiment is repeatable; every result is explainable.

---

## Quick Start

```bash
source setup.sh
```

See [SETUP.md](SETUP.md) for full instructions.

## Architecture Evolution

Final architecture (Phase 9) showing the complete pipeline from text to output.
Solid color = current component, rounded pill = replaced predecessor (colored by introducing phase).

```mermaid
graph TD
    In[Text Input]:::io --> Tok[BPE/Unigram Tokenizer]:::p4 --> Emb[Token Embedding]:::p2 --> N1

    subgraph Block[Transformer Block x N]
        N1[RMSNorm]:::p5 --> ATT[MLA]:::p8 --> R1[+ Residual]:::p3
        RoPE[RoPE]:::p5 -.-> ATT
        R1 --> N2[RMSNorm]:::p5 --> MOE[MoE Sparse SwiGLU]:::p8 --> R2[+ Residual]:::p3
    end

    R2 --> GRU[GRU Block]:::p9 --> Head[LM Head]:::p3 --> Logits[Logits]:::io
    Logits -->|training| Loss[Cross-Entropy Loss]:::io
    Logits -->|inference| Samp[Sampling]:::io --> GenOut[Generated Text]:::io

    subgraph Replaced[Replaced Predecessors]
        direction LR
        CT([Char Tokenizer]):::p2
        LPE([Learned Pos Emb]):::p2
        LF([Linear FFN]):::p2
        LN([LayerNorm]):::p3
        MHA([Multi-Head Attn]):::p3
        GF([GELU FFN]):::p3
        GQA2([GQA]):::p5
        SW([Dense SwiGLU]):::p5
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
    classDef p5 fill:#E1BEE7,stroke:#7B1FA2,color:#4A148C
    classDef p8 fill:#FFCDD2,stroke:#C62828,color:#B71C1C
    classDef p9 fill:#FFF9C4,stroke:#F57F17,color:#F57F17
```

| Color | Phase | Component |
|-------|-------|-----------|
| ⬛ Black | — | I/O: Text Input, Logits, Loss, Sampling, Generated Text |
| 🟢 Green | 2 | Token Embedding |
| 🔵 Blue | 3 | Residual connections, LM Head |
| 🟠 Orange | 4 | BPE / Unigram Tokenizer |
| 🟣 Purple | 5 | RMSNorm, RoPE |
| 🔴 Red | 8 | MLA (replaces GQA), MoE (replaces dense SwiGLU) |
| 🟡 Yellow | 9 | GRU hybrid blocks |
| Rounded pill | — | Replaced predecessors (colored by introducing phase) |

## Docs Summary

- [SETUP.md](SETUP.md): environment setup (WSL2 on Windows; Linux/macOS/WSL1 not supported)
- [CONTRIBUTING.md](CONTRIBUTING.md): workflow and PR rules
- [design/DESIGN.md](design/DESIGN.md): architecture, engineering principles, phased roadmap, testing strategy
- [design/PLAN.md](design/PLAN.md): phase progress, next steps, execution tracking
- [.github/CONTRIBUTORS.md](.github/CONTRIBUTORS.md): contributor authorization and PR validation checklist
- [.github/SKILLS.md](.github/SKILLS.md): required technical skills and working discipline
- [.github/CODEOWNERS](.github/CODEOWNERS): code review ownership

## Contributor Entry Points

- Human contributors: [CONTRIBUTING.md](CONTRIBUTING.md)
- AI agents: [design/PLAN.md](design/PLAN.md) -> [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
