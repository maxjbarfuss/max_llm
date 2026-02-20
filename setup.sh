#!/bin/bash
# Unified setup for max_llm: dependencies, venv, and project packages
# Usage: source setup.sh

# Detect if sourced (needed for venv activation to persist)
# NOTE: Do NOT use set -e here — it propagates to the calling shell when sourced
SOURCED=0
[[ -n "$ZSH_VERSION" && $ZSH_EVAL_CONTEXT =~ :file$ ]] && SOURCED=1
[[ -n "$BASH_VERSION" && "${BASH_SOURCE[0]}" != "${0}" ]] && SOURCED=1

exit_script() { [ $SOURCED -eq 1 ] && return "$1" || exit "$1"; }

# Colors
GREEN='\033[0;32m' BLUE='\033[0;34m' YELLOW='\033[1;33m' RED='\033[0;31m' CYAN='\033[0;36m' NC='\033[0m'

echo -e "${BLUE}=== Max LLM Setup ===${NC}\n"

# Sanity checks
[ -f pyproject.toml ] || { echo -e "${RED}Error: Run from max_llm root${NC}"; exit_script 1; }
grep -qi microsoft /proc/version 2>/dev/null || { echo -e "${RED}Error: WSL2 required${NC}"; exit_script 1; }
[ $SOURCED -eq 0 ] && echo -e "${YELLOW}Note: source setup.sh to keep venv active\n${NC}"

# Pre-step: Find Python 3.10+ with venv support; install 3.13 only if none found
echo -e "${BLUE}Pre-Step: Python${NC}"
PYTHON_CMD=""
for cmd in python3.13 python3.12 python3.11 python3.10; do
    if command -v "$cmd" &>/dev/null && "$cmd" -c "import ensurepip" 2>/dev/null; then
        PYTHON_CMD="$cmd"
        PYTHON_VERSION=$("$cmd" --version 2>&1 | awk '{print $2}')
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "  No Python 3.10+ with venv support found. Installing Python 3.13..."
    sudo apt update && sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt update
    sudo apt install -y python3.13 python3.13-venv python3.13-dev || { echo -e "${RED}Error: Python install failed${NC}"; exit_script 1; }
    PYTHON_CMD="python3.13"
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
fi
echo -e "${GREEN}✓${NC} Python ${PYTHON_VERSION}\n"

# Step 0: System dependencies (check + auto-install + idempotent)
echo -e "${BLUE}Step 0: System Dependencies${NC}"
SETUP_DEPS_SCRIPT="scripts/setup/setup_dependencies.py"
[ -f "$SETUP_DEPS_SCRIPT" ] || { echo -e "${RED}Error: Missing $SETUP_DEPS_SCRIPT${NC}"; exit_script 1; }
# Pre-cache sudo credentials (passwordless sudo on WSL2, prompts once if password required)
sudo -v 2>/dev/null || true
$PYTHON_CMD "$SETUP_DEPS_SCRIPT" || { echo -e "${RED}Error: System dependency setup failed${NC}"; exit_script 1; }

# Add CUDA to PATH if installed (setup_dependencies.py may have just installed it;
# the Python subprocess cannot propagate PATH changes back to this shell)
for cuda_dir in /usr/local/cuda /usr/local/cuda-12.*; do
    if [ -d "$cuda_dir/bin" ]; then
        export PATH="$cuda_dir/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_dir/lib64:${LD_LIBRARY_PATH:-}"
        break
    fi
done

# Step 1: Virtual environment (reuse or create)
echo -e "\n${BLUE}Step 1: Virtual Environment${NC}"
if [ -z "$VIRTUAL_ENV" ]; then
    VENV_VALID=0
    if [ -d .venv ] && [ -f .venv/bin/activate ]; then
        echo "  Activating existing venv..."
        if source .venv/bin/activate; then
            VENV_VALID=1
        else
            echo -e "${YELLOW}⚠${NC} Existing venv corrupt, recreating..."
        fi
    fi

    if [ $VENV_VALID -eq 0 ]; then
        echo "  Creating new venv..."
        rm -rf .venv
        if ! $PYTHON_CMD -m venv .venv; then
            echo -e "${RED}Error: venv creation failed${NC}"
            exit_script 1
        fi
        if ! source .venv/bin/activate; then
            echo -e "${RED}Error: venv activation failed${NC}"
            exit_script 1
        fi
    fi
fi
echo -e "${GREEN}✓${NC} venv active\n"

# Step 2: UV package manager
echo -e "${BLUE}Step 2: UV${NC}"
if ! command -v uv &>/dev/null; then
    pip install --upgrade --quiet pip uv || { echo -e "${RED}Error: uv install failed${NC}"; exit_script 1; }
    echo -e "${GREEN}✓${NC} UV installed"
else
    echo -e "${GREEN}✓${NC} UV ready ($(uv --version 2>/dev/null || echo 'unknown'))"
fi
echo ""

# Step 3: Python dependencies
echo -e "${BLUE}Step 3: Python Dependencies${NC}"
REQ_FILE="requirements.txt"
[ -f "$REQ_FILE" ] || { echo -e "${RED}Error: Missing $REQ_FILE${NC}"; exit_script 1; }
uv pip install --quiet -r "$REQ_FILE" || { echo -e "${RED}Error: requirements install failed${NC}"; exit_script 1; }
uv pip install --quiet -e . || { echo -e "${RED}Error: package install failed${NC}"; exit_script 1; }
echo -e "${GREEN}✓${NC} Python dependencies installed\n"

# Step 4: PyTorch + GPU packages (CUDA 12.1)
echo -e "${BLUE}Step 4: PyTorch & GPU Dependencies${NC}"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
uv pip install --quiet "torch>=2.3.0" --index-url "$PYTORCH_INDEX" || \
  pip install --quiet "torch>=2.3.0" --index-url "$PYTORCH_INDEX" || \
  { echo -e "${RED}Error: PyTorch install failed${NC}"; exit_script 1; }
uv pip install --quiet "xformers>=0.0.22" "deepspeed>=0.11.0" "bitsandbytes>=0.41.0" || \
  pip install --quiet "xformers>=0.0.22" "deepspeed>=0.11.0" "bitsandbytes>=0.41.0" || \
  echo -e "${YELLOW}⚠${NC} Some GPU packages failed — check manually"
echo -e "${GREEN}✓${NC} PyTorch and GPU packages installed\n"

# Step 5: Verify installation
echo -e "${BLUE}Step 5: Verification${NC}"
$PYTHON_CMD -m pip check || echo -e "${YELLOW}⚠${NC} pip check reported warnings"
$PYTHON_CMD -c "import torch; avail = torch.cuda.is_available(); print(f'  CUDA available: {avail}'); print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || \
  echo -e "${YELLOW}⚠${NC} Could not verify torch/CUDA (torch may not be importable yet)"

# Step 6: VSCode extensions (optional)
echo ""
if command -v code &>/dev/null && [ -f scripts/setup/install_vscode_extensions.py ]; then
    echo -e "${BLUE}Step 6: VSCode Extensions${NC}"
    $PYTHON_CMD scripts/setup/install_vscode_extensions.py
fi

# Summary
echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}\n"
echo -e "${CYAN}Environment:${NC}"
echo "  Python: $($PYTHON_CMD --version 2>&1)"
echo "  venv: $VIRTUAL_ENV"
echo "  UV: $(uv --version 2>/dev/null || echo 'N/A')"
echo ""

if [ $SOURCED -eq 1 ]; then
    echo -e "${GREEN}✓ venv active. Commands:${NC}"
    echo "  make test          # Run tests"
    echo "  make format        # Format code"
    echo "  make type-check    # Type check"
    echo "  deactivate         # Exit venv"
else
    echo -e "${YELLOW}To activate venv: source .venv/bin/activate${NC}"
fi
echo ""
