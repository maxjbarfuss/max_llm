# Max LLM

Hybrid LLM research project (100–500M params): MLA + MoE + GRU output, optimized for local training on consumer hardware.

**Status:** Phase 1 (foundation)
**Repository:** [github.com/maxjbarfuss/max_llm](https://github.com/maxjbarfuss/max_llm)

## Quick Start

```bash
source setup.sh
```

See [SETUP.md](SETUP.md) for full instructions.

## Architecture (Summary)

```
Text -> Tokenizer (GPT-2 BPE baseline, Unigram under evaluation) -> Embeddings -> Input FFN -> Transformer (MLA + RoPE)
    -> MoE -> GRU output -> Projection (tied) -> Logits
```

- Architecture and tokenizer are treated as evolving baselines, not frozen decisions.
- Attention, MoE, embeddings: BF16
- FFN and RNN: progressive precision (BF16 AMP → FP8 → mixed), tunable by experiment
- Inference KV cache: FP8 with MLA latent compression

Full design and implementation plan: [design/DESIGN.md](design/DESIGN.md)

## Docs Summary

- [SETUP.md](SETUP.md): environment setup for WSL2 on Windows (native Linux/macOS and WSL1 are not supported)
- [CONTRIBUTING.md](CONTRIBUTING.md): workflow + PR rules
- [design/DESIGN.md](design/DESIGN.md): architecture, engineering standards, testing strategy
- [design/PLAN_CHECKLIST.md](design/PLAN_CHECKLIST.md): session tracker
- [.github/CONTRIBUTORS.md](.github/CONTRIBUTORS.md): authorization policy + PR checklist
- [.github/SKILLS.md](.github/SKILLS.md): capability baseline
- [.github/CODEOWNERS](.github/CODEOWNERS): review ownership

## Contributor Entry Points

- Human contributors: [CONTRIBUTING.md](CONTRIBUTING.md)
- AI agents: [design/PLAN_CHECKLIST.md](design/PLAN_CHECKLIST.md) -> [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
