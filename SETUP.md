# Setup Guide

## Requirements

**WSL2 on Windows is required.** This project does not support native Linux, macOS, or legacy WSL1.

- Install WSL2: https://docs.microsoft.com/en-us/windows/wsl/install
- Required: Ubuntu 22.04 LTS distro

## Quick Start

```bash
source setup.sh
```

The script detects your environment and auto-installs required components (GPU/CUDA support included).
It is safe to re-run any time after dependency changes; existing components are reused when already installed.
User interaction is minimal (typically only a sudo password prompt when system packages are installed).

Python dependencies are installed from `requirements.txt` during setup.

---

## WSL2 Performance Tuning

For optimal WSL2 performance, copy the configuration template to Windows:

```
config/.wslconfig  →  C:\Users\<YourUsername>\.wslconfig
```

Then restart: `wsl --shutdown`

See [config/.wslconfig](config/.wslconfig) for full documentation.

---

## After Setup

```bash
make test       # Run tests
make build      # Build C++ components
make format     # Format code
make lint       # Check code quality
deactivate      # Exit virtual environment
```

For architecture and development workflow, see [CONTRIBUTING.md](CONTRIBUTING.md) and [design/DESIGN.md](design/DESIGN.md).

Read: [NVIDIA WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)

## Next Read

- [README.md](README.md): Project overview
- [CONTRIBUTING.md](CONTRIBUTING.md): Development workflow
- [design/PLAN_CHECKLIST.md](design/PLAN_CHECKLIST.md): Session execution tracker
