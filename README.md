# Max LLM

Hybrid LLM research project (100–500M params): MLA + MoE + GRU output, optimized for local training on consumer hardware.

**Status:** Phase 1 (foundation)  
**Repository:** [github.com/maxjbarfuss/max_llm](https://github.com/maxjbarfuss/max_llm)

## Quick Start

```bash
# Check system requirements
python3 check_system.py

# Run setup (creates venv, installs all dependencies with uv)
source setup.sh

# (Optional) Install VSCode extensions
bash install_vscode_extensions.sh
```

**Requirements (target):** Python 3.10+, CUDA 12.1, 32GB+ RAM  
**Detailed setup:** [SETUP.md](SETUP.md) | Testing: `make test`

## Architecture (Summary)

```
Text -> Tokenizer (GPT-2 BPE baseline, Unigram under evaluation) -> Embeddings -> Input FFN -> Transformer (MLA + RoPE)
    -> MoE -> GRU output -> Projection (tied) -> Logits
```

- Architecture and tokenizer are treated as evolving baselines, not frozen decisions.
- Attention, MoE, embeddings: BF16
- FFN and RNN: progressive precision baseline (FP4 -> FP8 -> BF16), tunable by experiment
- Inference KV cache: FP8 with MLA latent compression

Full design and implementation plan: [design/plan.md](design/plan.md)

## Docs Summary

- [SETUP.md](SETUP.md): environment setup for all platforms (WSL2, Linux, macOS, Windows)
- [GETTING_STARTED.md](GETTING_STARTED.md): setup + verification
- [CONTRIBUTING.md](CONTRIBUTING.md): workflow + PR rules
- [design/plan-checklist.md](design/plan-checklist.md): session tracker
- [design/plan.md](design/plan.md): architecture baseline + delivery order
- [design/philosophy.md](design/philosophy.md): engineering rules
- [.github/CONTRIBUTORS.md](.github/CONTRIBUTORS.md): authorization policy
- [.github/SKILLS.md](.github/SKILLS.md): capability baseline
- [.github/CODEOWNERS](.github/CODEOWNERS): review ownership
- [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md): PR checklist

## Contributor Entry Points

- Human contributors: [CONTRIBUTING.md](CONTRIBUTING.md)
- AI agents: [design/plan-checklist.md](design/plan-checklist.md) -> [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
