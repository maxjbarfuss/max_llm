# Setup Guide for Max LLM Development

This document provides comprehensive setup instructions for developing max_llm in a fresh environment (including WSL2, Linux, macOS, and Windows).

## Quick Start (Experienced Users)

```bash
# Check system requirements
python3 check_system.py

# Fix any critical issues (see output)

# Run full setup
source setup.sh

# Install VSCode extensions (optional)
bash install_vscode_extensions.sh
```

## Detailed Setup

### 1. System Requirements Check

Start by checking if your system meets all requirements:

```bash
python3 check_system.py
```

This script validates:
- ✓ Python version (3.10+)
- ✓ System dependencies (git, gcc, curl)
- ✓ Python dev tools (will be installed automatically)
- ✓ GPU/CUDA availability (optional)
- ✓ VSCode extensions (optional)

**Note:** The script will provide specific installation guidance if any critical issues are found.

### 2. Install System Dependencies

#### For WSL2 (Recommended for Windows)

```bash
# Update WSL2
wsl --update

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install build tools
sudo apt install -y build-essential git curl python3.10 python3.10-venv python3.10-dev

# Install uv (fast package manager)
python3 -m pip install --upgrade pip uv
```

#### For Ubuntu/Debian Linux

```bash
sudo apt update && sudo apt install -y build-essential git curl

# If Python 3.10+ not available:
sudo apt install -y python3.10 python3.10-venv python3.10-dev
python3.10 -m pip install --upgrade pip uv
```

#### For macOS

```bash
# Install Homebrew (if not already)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install build tools
brew install git gcc

# Install Python 3.10+ (if needed)
brew install python@3.10
python3.10 -m pip install --upgrade pip uv
```

#### For Windows (WSL2 Recommended)

See WSL2 instructions above, or:

1. Install Python 3.10+ from [python.org](https://www.python.org/)
2. Install Git from [git-scm.com](https://git-scm.com/)
3. Install NVIDIA CUDA Toolkit 12.1 (if you have NVIDIA GPU)
4. Open PowerShell and run:
   ```powershell
   pip install --upgrade uv
   cd path\to\max_llm
   .\setup.sh  # Note: may need to run in bash/WSL2 for best results
   ```

### 3. GPU/CUDA Setup (Optional but Recommended)

#### Check CUDA Availability

The system check will report CUDA status. If CUDA is not available and you have an NVIDIA GPU:

**For WSL2 with NVIDIA GPU:**

1. Install NVIDIA WSL2 support: https://docs.nvidia.com/cuda/wsl-user-guide/
2. Install CUDA Toolkit 12.1 in WSL2
3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

**For Linux with NVIDIA GPU:**

1. Install NVIDIA driver
2. Install CUDA Toolkit 12.1 from https://developer.nvidia.com/cuda-12-1-0
3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

**For macOS:**

GPU support on macOS is limited. Consider using CPU mode or cloud GPUs.

### 4. Run Setup

Once system requirements are met, run the setup script:

```bash
# From the max_llm project directory
source setup.sh
```

This will:
1. ✓ Create a Python virtual environment (.venv)
2. ✓ Install UV package manager
3. ✓ Install PyTorch with CUDA 12.1 support
4. ✓ Install all project dependencies
5. ✓ Install optional development tools (DeepSpeed, etc.)
6. ✓ Verify the installation

The virtual environment will remain active in your current shell session after the script completes.

### 5. VSCode Setup (Optional)

#### Install VSCode Extensions

Automatically install recommended extensions:

```bash
bash install_vscode_extensions.sh
```

Or manually from VSCode:
1. Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on macOS)
2. Search for each extension ID and install:

**Required:**
- `ms-python.python` - Python language support
- `ms-python.vscode-pylance` - Type checking and diagnostics
- `GitHub.copilot` - AI-powered code completion

**Recommended:**
- `ms-python.debugpy` - Python debugger
- `ms-python.black-formatter` - Code formatter
- `charliermarsh.ruff` - Fast Python linter
- `ms-python.mypy-type-checker` - Type checker
- `ms-vscode.makefile-tools` - Makefile support
- `GitHub.copilot-chat` - Copilot chat interface
- `eamodio.gitlens` - Git insights

### 6. Verify Installation

```bash
# Check dependencies
python3 check_deps.py

# Run tests
make test

# Type check code
make type-check

# Format check
make format-check
```

## Development Workflow

### Common Commands

```bash
# Check/update system requirements
make check

# Run setup or update environment
make setup

# Code quality checks
make lint                # Ruff + MyPy
make format-check        # Check formatting
make format              # Auto-format code

# Testing
make test                # Run all tests
make test-cov            # Tests with coverage report

# Maintenance
make clean               # Remove build artifacts
make pre-commit-run      # Run pre-commit hooks
```

### Using UV Package Manager

UV is pre-installed and used throughout the setup. Common commands:

```bash
# Add a new package
uv pip install package_name

# Install from requirements file
uv pip install -r requirements.txt

# Sync exact versions (lock file)
uv sync

# Run Python in isolation (no venv needed)
uv run python script.py
```

See [astral.sh/uv](https://astral.sh/uv) for full documentation.

## Troubleshooting

### Problem: "gcc: command not found"

**Solution:**
```bash
# Ubuntu/Debian
sudo apt install -y build-essential

# macOS
brew install gcc

# WSL2
sudo apt install -y build-essential
```

### Problem: "uv: command not found"

**Solution:**
```bash
pip install --upgrade uv
```

### Problem: PyTorch/CUDA installation fails

**Solution:**
Try CPU-only mode temporarily:
```bash
uv pip install torch  # CPU version
```

Then install GPU support when available:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Problem: Virtual environment not activated after setup.sh

**Solution:**
Make sure you're sourcing (not executing) the script:
```bash
source setup.sh    # Correct
bash setup.sh      # Will not persist
./setup.sh         # Will not persist
```

### Problem: VSCode extensions not appearing

**Solution:**
1. Close and reopen VSCode (`Ctrl+R` or `Cmd+R` to reload window)
2. Check that VSCode has access to `code` command:
   ```bash
   which code
   ```
3. Reinstall extension:
   ```bash
   code --install-extension <extension-id> --force
   ```

### Problem: Pre-commit hooks failing

**Solution:**
```bash
# Install pre-commit
uv pip install pre-commit

# Install hooks
pre-commit install

# Run all hooks
make pre-commit-run
```

## Environment Details

After setup, verify your environment:

```bash
python --version               # Should be 3.10+
pip --version                  # Should show pip version
uv --version                   # Should show uv version
python -c "import torch; print(torch.cuda.is_available())"  # GPU check
```

## Platform-Specific Notes

### WSL2 Performance Tips

- Place project files in WSL filesystem (`/home/username/...`) for best performance
- Avoid Windows filesystem (`/mnt/c/...`) for development
- Use `wsl.exe --shutdown` to restart WSL if issues occur

### macOS Notes

- GPU acceleration not supported natively
- Consider using remote/cloud GPU for training
- Use CPU mode for development and testing

### Linux Notes

- Most straightforward setup
- Full GPU/CUDA support available
- Use system Python 3.10+ or alternatives

## Next Steps

After successful setup:

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for workflow guidelines
2. Review [design/philosophy.md](design/philosophy.md) for engineering principles
3. Check [design/plan-checklist.md](design/plan-checklist.md) for current tasks
4. Start developing: `make lint test format` before committing

## Getting Help

If you encounter issues:

1. Run `python3 check_system.py` to identify the problem
2. Check this guide's Troubleshooting section
3. Review the error messages carefully (they often contain solutions)
4. Check [CONTRIBUTING.md](CONTRIBUTING.md) for workflow questions
