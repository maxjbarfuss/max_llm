# Setup Guide

## Requirements

**WSL2 on Windows is required.** This project does not support native Linux, macOS, or legacy WSL1.

- Install WSL2: https://docs.microsoft.com/en-us/windows/wsl/install
- Required: Ubuntu 22.04+ LTS distro (24.04 tested)

## Quick Start

From the project root:

```bash
source setup.sh
```

The script detects your environment and auto-installs required components (GPU/CUDA support included).
It is safe to re-run any time after dependency changes; existing components are reused when already installed.
User interaction is minimal (typically only a sudo password prompt when system packages are installed).

Python dependencies are installed from `requirements.txt` during setup.

### CI / Fast Mode

For GitHub Actions or CPU-only environments, use fast mode:

```bash
FAST_SETUP=1 source setup.sh
```

Fast mode skips WSL-only and GPU-heavy steps (CUDA preflight, GPU package installs, VS Code extension setup) and installs a CPU-only PyTorch build for lint/type-check/test workflows.

SageAttention is installed by default in full setup mode (source build), pinned to `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` and built for `8.9` by default.

Disable it:

```bash
INSTALL_SAGEATTN=0 source setup.sh
```

Use a different ref:

```bash
SAGE_ATTN_GIT_REF="main" source setup.sh
```

Build parallelism uses a formula with default target `4 × 2` (~8 compile threads), then scales down automatically if RAM/CPU is tighter.

Common overrides:

```bash
MAX_JOBS_OVERRIDE=4 NVCC_THREADS_OVERRIDE=2 source setup.sh   # 4x2
MAX_JOBS_OVERRIDE=3 NVCC_THREADS_OVERRIDE=2 source setup.sh   # 3x2
```

Override SageAttention arch:

```bash
SAGE_ARCH_LIST="8.9" source setup.sh
```

---

## WSL2 Performance Tuning

For optimal WSL2 performance, copy the configuration template to Windows:

```
config/.wslconfig  →  C:\Users\<YourUsername>\.wslconfig
```

Then restart: `wsl --shutdown`

See [config/.wslconfig](../../config/.wslconfig) for full documentation.

---

## After Setup

Verify your environment works:

```bash
make test       # Python + C++ tests
```

Then read [CONTRIBUTING.md](../../CONTRIBUTING.md), [.github/AGENTS.md](../../.github/AGENTS.md), and check [.github/MEMORY.md](../../.github/MEMORY.md) for current session state to begin.

---

## Dataset Tooling

Dataset preparation is first-class in `src/data/preparation/`. Use
`python -m src.data.preparation --config <config.json|config.toml>` and see
[src/data/README.md](../../src/data/README.md) for data tooling details.

For repeated Hugging Face dataset downloads, use the rerunnable helper below.
It defaults to `D:\dev\data` via `/mnt/d/dev/data/hf` so raw corpora stay off the local
fast artifact area:

```bash
/home/max/dev/max_llm/.venv/bin/python scripts/setup/download_hf_dataset.py \
	wikimedia/wikipedia \
	--revision 20231101.en \
	--list-files \
	--allow '*.parquet'
```

Then download the subset you actually want:

```bash
/home/max/dev/max_llm/.venv/bin/python scripts/setup/download_hf_dataset.py \
	wikimedia/wikipedia \
	--revision 20231101.en \
	--allow '*.parquet'
```

For shard-limited downloads, prefer explicit brace ranges over broad wildcards so you do not
accidentally match an entire shard family:

```bash
/home/max/dev/max_llm/.venv/bin/python scripts/setup/download_hf_dataset.py \
	HuggingFaceFW/fineweb \
	--allow 'data/CC-MAIN-2023-14/000_000{00..07}.parquet'
```

The helper now prints the matched file count before download so an overly broad pattern is
obvious immediately.

The default raw download root is `/mnt/d/dev/data/hf` and the default HF cache dir
is `/mnt/d/dev/data/hf_cache`.

For fast local source discovery (without broad filesystem scans), use:

```bash
/home/max/dev/max_llm/.venv/bin/python scripts/setup/locate_dataset.py --dataset tinystories-gpt4-clean
```

To inspect all ranked candidates under specific roots:

```bash
/home/max/dev/max_llm/.venv/bin/python scripts/setup/locate_dataset.py \
	--dataset tinystories-gpt4-clean \
	--root /mnt/d/dev/data \
	--all
```

For higher download rate limits, place a Hugging Face token in `.huggingface/.hf_token` (ignored by git) or export `HF_TOKEN`.

---

## Troubleshooting

Useful commands:

```bash
make test       # Run tests
make build      # Build C++ components
make format     # Format code
make lint       # Check code quality
deactivate      # Exit virtual environment
```

For architecture and development workflow, see [CONTRIBUTING.md](../../CONTRIBUTING.md) and [docs/DESIGN.md](../../docs/DESIGN.md).

Read: [NVIDIA WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)

## Next Read

- [README.md](../../README.md): Project overview and status
- [CONTRIBUTING.md](../../CONTRIBUTING.md): Workflow and contribution rules
- [.github/MEMORY.md](../../.github/MEMORY.md): Current session working state
- [.github/SESSION_LOG.md](../../.github/SESSION_LOG.md): Recent history of completed work
- [.github/SESSION_LOG_ARCHIVE.md](../../.github/SESSION_LOG_ARCHIVE.md): Archived older completed-work history
