# Getting Started

## Requirements

| Resource | Baseline | Recommended |
|---|---|---|
| GPU (inference) | Single CUDA-capable GPU (or CPU fallback) | 1× 24GB NVIDIA |
| GPU (training) | 1× CUDA-capable GPU | 2× 24GB NVIDIA (CUDA 12.1) |
| CPU | 8 cores | 8+ cores |
| RAM | 32GB | 64GB |
| Storage | SSD | 1TB+ SSD for large-data workflows |

- Python 3.10+
- CUDA 12.1
- PyTorch 2.3+

## Setup (Fast Path)

```bash
git clone https://github.com/maxjbarfuss/max_llm.git
cd max_llm
source setup.sh
```

## Setup (Manual)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e ".[cuda121,dev,training]"
uv pip install deepspeed
python check_deps.py
```

## Verify

```bash
source .venv/bin/activate
python check_deps.py
pytest tests/ -v
mypy src/
ruff check src/
black --check src/ tests/
```

## Workspace Landmarks

- `src/config/`: model/training/data config dataclasses
- `src/models/`: attention, MoE, RNN modules
- `tests/unit/`: fast unit tests
- `design/`: architecture plan and contributor philosophy

## Next Read

- Human contributors: [CONTRIBUTING.md](CONTRIBUTING.md)
- AI agents: [design/plan-checklist.md](design/plan-checklist.md) (`Next Steps` first)
