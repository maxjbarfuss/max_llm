#!/bin/bash
# Unified setup for max_llm: dependencies, venv, and project packages
# Usage: source setup.sh

# Detect if sourced (needed for venv activation to persist)
# NOTE: Do NOT use set -e here — it propagates to the calling shell when sourced
SOURCED=0
[[ -n "$ZSH_VERSION" && $ZSH_EVAL_CONTEXT =~ :file$ ]] && SOURCED=1
[[ -n "$BASH_VERSION" && "${BASH_SOURCE[0]}" != "${0}" ]] && SOURCED=1

exit_script() { [ $SOURCED -eq 1 ] && return "$1" || exit "$1"; }

run_uv_pip() {
    if [ -n "${UV_BIN:-}" ] && [ -x "$UV_BIN" ]; then
        "$UV_BIN" pip install "$@"
    else
        uv pip install "$@"
    fi
}

install_required() {
    local desc="$1"
    shift
    run_uv_pip "$@" || {
        echo -e "${RED}Error: ${desc} install failed${NC}"
        exit_script 1
    }
}

install_optional() {
    local desc="$1"
    shift
    run_uv_pip "$@" || {
        echo -e "${YELLOW}⚠${NC} ${desc} install failed — check manually"
        return 1
    }
}

# Colors
GREEN='\033[0;32m' BLUE='\033[0;34m' YELLOW='\033[1;33m' RED='\033[0;31m' CYAN='\033[0;36m' NC='\033[0m'

echo -e "${BLUE}=== Max LLM Setup ===${NC}\n"

FAST_SETUP="${FAST_SETUP:-0}"

# Sanity checks
[ -f pyproject.toml ] || { echo -e "${RED}Error: Run from max_llm root${NC}"; exit_script 1; }
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    if [ "$FAST_SETUP" = "1" ]; then
        echo -e "${YELLOW}⚠${NC} Non-WSL environment detected; continuing because FAST_SETUP=1"
    else
        echo -e "${RED}Error: WSL2 required (or set FAST_SETUP=1 for CI/core-only setup)${NC}"
        exit_script 1
    fi
fi
[ $SOURCED -eq 0 ] && echo -e "${YELLOW}Note: source setup.sh to keep venv active\n${NC}"

# Pre-step: GitHub Token (for API access in CI debugging)
echo -e "${BLUE}Pre-Step: GitHub Token${NC}"
if [ -z "$GITHUB_TOKEN" ]; then
    if [ -f ".github/.github_token" ]; then
        export GITHUB_TOKEN=$(cat ".github/.github_token")
        echo -e "${GREEN}✓${NC} GitHub token loaded from .github/.github_token"
    else
        echo -e "${YELLOW}⚠${NC} No GitHub token found. To enable API access:"
        echo -e "  1. Create a personal access token at https://github.com/settings/tokens"
        echo -e "  2. Save it to .github/.github_token (will be in .gitignore)"
        echo -e "  3. Run: export GITHUB_TOKEN=\$(cat .github/.github_token)"
    fi
else
    echo -e "${GREEN}✓${NC} GitHub token set via environment variable"
fi
echo ""

# Pre-step: Require Python 3.12 with venv support (matches pinned cp312 wheels)
echo -e "${BLUE}Pre-Step: Python${NC}"
PYTHON_CMD=""
for cmd in python3.12; do
    if command -v "$cmd" &>/dev/null && "$cmd" -c "import ensurepip" 2>/dev/null; then
        PYTHON_CMD="$cmd"
        PYTHON_VERSION=$("$cmd" --version 2>&1 | awk '{print $2}')
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "  Python 3.12 not found. Installing Python 3.12..."
    sudo apt update && sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt update
    sudo apt install -y python3.12 python3.12-venv python3.12-dev || { echo -e "${RED}Error: Python install failed${NC}"; exit_script 1; }
    PYTHON_CMD="python3.12"
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
fi
echo -e "${GREEN}✓${NC} Python ${PYTHON_VERSION}\n"

# Step 0: System dependencies (check + auto-install + idempotent)
echo -e "${BLUE}Step 0: System Dependencies${NC}"
SETUP_DEPS_SCRIPT="scripts/setup/setup_dependencies.py"
[ -f "$SETUP_DEPS_SCRIPT" ] || { echo -e "${RED}Error: Missing $SETUP_DEPS_SCRIPT${NC}"; exit_script 1; }
if [ "$FAST_SETUP" = "1" ]; then
    echo -e "${YELLOW}↷ Skipping system dependency bootstrap in FAST_SETUP mode${NC}"
else
    # Pre-cache sudo credentials (passwordless sudo on WSL2, prompts once if password required)
    sudo -v 2>/dev/null || true
    $PYTHON_CMD "$SETUP_DEPS_SCRIPT" || { echo -e "${RED}Error: System dependency setup failed${NC}"; exit_script 1; }
fi

# Pre-flight: Verify CUDA toolkit is installed (required for torch, flash-attn, etc.)
echo -e "\n${BLUE}Pre-Step: CUDA Toolkit${NC}"
CUDA_DIR=""
if [ "$FAST_SETUP" = "1" ]; then
    echo -e "${YELLOW}↷ Skipping CUDA preflight in FAST_SETUP mode${NC}\n"
else
    for cuda_dir in /usr/local/cuda /usr/local/cuda-12.*; do
        if [ -d "$cuda_dir/bin" ] && [ -x "$cuda_dir/bin/nvcc" ]; then
            CUDA_DIR="$cuda_dir"
            break
        fi
    done
    if [ -z "$CUDA_DIR" ]; then
        echo -e "${RED}Error: CUDA toolkit not found in /usr/local/cuda*${NC}"
        echo -e "  Install CUDA toolkit first: https://developer.nvidia.com/cuda-downloads"
        exit_script 1
    fi
    CUDA_VERSION=$($CUDA_DIR/bin/nvcc --version | grep -oP 'release \K[0-9.]+')
    echo -e "${GREEN}✓${NC} CUDA ${CUDA_VERSION} at ${CUDA_DIR}\n"
fi

# Step 1: Virtual environment (reuse or create)
echo -e "\n${BLUE}Step 1: Virtual Environment${NC}"
if [ -n "${VIRTUAL_ENV:-}" ] && [ ! -x "${VIRTUAL_ENV}/bin/python" ]; then
    echo -e "${YELLOW}⚠${NC} Stale VIRTUAL_ENV detected (${VIRTUAL_ENV}); reinitializing .venv"
    unset VIRTUAL_ENV
fi

if [ -z "$VIRTUAL_ENV" ] || [ "$VIRTUAL_ENV" != "$(pwd)/.venv" ]; then
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
echo -e "${GREEN}✓${NC} venv active"
VENV_BIN="$VIRTUAL_ENV/bin"
if [ -x "$VENV_BIN/python" ]; then
    VENV_PYTHON="$VENV_BIN/python"
elif [ -x "$VENV_BIN/python3" ]; then
    VENV_PYTHON="$VENV_BIN/python3"
elif [ -x "$VENV_BIN/python3.12" ]; then
    VENV_PYTHON="$VENV_BIN/python3.12"
else
    echo -e "${RED}Error: No Python executable found in venv ($VENV_BIN)${NC}"
    exit_script 1
fi

# Add CUDA to PATH after venv activation (venv overwrites PATH)
# This ensures nvcc is available for packages that compile CUDA kernels
if [ "$FAST_SETUP" != "1" ]; then
    export PATH="$CUDA_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
fi

# Configure CUDA compilation parallelism.
# Formula targets ~8 compile threads by default (4 jobs × 2 nvcc threads),
# then scales down based on available CPU/RAM.
TOTAL_CORES=$(nproc)
TOTAL_RAM_GB=$(free -g | awk '/Mem:/{print $7}')  # available (not total)
SLOTS_BY_RAM=$(( TOTAL_RAM_GB / 4 ))              # ~4GB per slot (conservative)
SLOTS_BY_CPU=$(( TOTAL_CORES > 4 ? TOTAL_CORES - 2 : 2 ))  # leave 2 cores for OS
RAW_JOBS=$(( SLOTS_BY_RAM < SLOTS_BY_CPU ? SLOTS_BY_RAM : SLOTS_BY_CPU ))

export NVCC_THREADS="${NVCC_THREADS_OVERRIDE:-2}"
MAX_JOBS_CALC=$(( RAW_JOBS / NVCC_THREADS ))
TARGET_BUILD_THREADS="${TARGET_BUILD_THREADS_OVERRIDE:-8}"
TARGET_MAX_JOBS=$(( TARGET_BUILD_THREADS / NVCC_THREADS ))

if [ "$TARGET_MAX_JOBS" -lt 1 ]; then
    TARGET_MAX_JOBS=1
fi
if [ "$MAX_JOBS_CALC" -lt 1 ]; then
    MAX_JOBS_CALC=1
fi

# Keep formula-based default unless explicitly overridden.
if [ -n "${MAX_JOBS_OVERRIDE:-}" ]; then
    export MAX_JOBS="$MAX_JOBS_OVERRIDE"
else
    if [ "$MAX_JOBS_CALC" -lt "$TARGET_MAX_JOBS" ]; then
        export MAX_JOBS="$MAX_JOBS_CALC"
    else
        export MAX_JOBS="$TARGET_MAX_JOBS"
    fi
fi
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"

if [ "$FAST_SETUP" != "1" ]; then
    echo -e "${GREEN}✓${NC} CUDA paths configured (nvcc: $(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+'))"
    echo -e "${GREEN}✓${NC} CUDA build parallelism: ${MAX_JOBS} jobs × ${NVCC_THREADS} threads (${TOTAL_CORES} cores, ${TOTAL_RAM_GB}GB avail)\n"
else
    echo -e "${YELLOW}↷ Skipping CUDA build parallelism configuration in FAST_SETUP mode\n${NC}"
fi

# Step 2: UV package manager
echo -e "${BLUE}Step 2: UV${NC}"
UV_BIN="${VENV_BIN}/uv"
if [ ! -x "$UV_BIN" ]; then
    $VENV_PYTHON -m pip install --upgrade --quiet pip uv || { echo -e "${RED}Error: uv install failed${NC}"; exit_script 1; }
fi
if [ ! -x "$UV_BIN" ] && ! command -v uv &>/dev/null; then
    echo -e "${RED}Error: uv install failed${NC}"
    exit_script 1
fi
if [ -x "$UV_BIN" ]; then
    echo -e "${GREEN}✓${NC} UV installed"
else
    echo -e "${GREEN}✓${NC} UV ready ($(uv --version 2>/dev/null || echo 'unknown'))"
fi
echo ""

# Step 3: Python dependencies
echo -e "${BLUE}Step 3: Python Dependencies${NC}"
REQ_FILE="requirements.txt"
[ -f "$REQ_FILE" ] || { echo -e "${RED}Error: Missing $REQ_FILE${NC}"; exit_script 1; }
echo "  Installing requirements (uv)..."
run_uv_pip -r "$REQ_FILE" || {
    echo -e "${YELLOW}⚠${NC} uv requirements install failed, retrying with pip"
    $VENV_PYTHON -m pip install -r "$REQ_FILE" || { echo -e "${RED}Error: requirements install failed${NC}"; exit_script 1; }
}
echo "  Installing editable package..."
run_uv_pip -e . || {
    echo -e "${YELLOW}⚠${NC} uv editable install failed, retrying with pip"
    $VENV_PYTHON -m pip install -e . || { echo -e "${RED}Error: package install failed${NC}"; exit_script 1; }
}
echo -e "${GREEN}✓${NC} Python dependencies installed\n"

# Step 4: PyTorch + GPU packages (CUDA 12.8)
echo -e "${BLUE}Step 4: PyTorch & GPU Dependencies${NC}"
TORCH_WHEEL_URL="https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
FLASH_ATTN_WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl"
SAGE_ATTN_GIT_REF="${SAGE_ATTN_GIT_REF:-main}"

if [ "$FAST_SETUP" = "1" ]; then
    echo -e "${YELLOW}↷ FAST_SETUP=1: installing CPU-only torch for CI${NC}"
    install_required "PyTorch CPU" "torch" "--index-url" "https://download.pytorch.org/whl/cpu"
else
    PY_MM=$($VENV_PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [ "$PY_MM" != "3.12" ]; then
        echo -e "${RED}Error: This setup requires Python 3.12 for pinned cp312 wheels (found ${PY_MM})${NC}"
        exit_script 1
    fi

    echo "  Installing pinned PyTorch wheel..."
    install_required "PyTorch wheel" "$TORCH_WHEEL_URL"
    echo "  Installing xformers/deepspeed/bitsandbytes/torchao..."
    install_optional "GPU packages" "xformers==0.0.35" "deepspeed>=0.11.0" "bitsandbytes>=0.41.0" "torchao>=0.2.0"

    # Install flash-attn after torch using the pre-built wheel
    echo "  Installing flash-attn prebuilt wheel..."
    install_optional "flash-attn wheel" "$FLASH_ATTN_WHEEL_URL"

    # Install SageAttention by default (can compile CUDA/Triton kernels and take longer)
    if [ "${INSTALL_SAGEATTN:-1}" = "1" ]; then
        SAGE_ARCH_LIST="${SAGE_ARCH_LIST:-8.9}"
        SAGE_EXT_PARALLEL="${SAGE_EXT_PARALLEL:-$MAX_JOBS}"
        SAGE_NVCC_THREADS="${SAGE_NVCC_THREADS:-$NVCC_THREADS}"
        SAGE_MAX_JOBS="${SAGE_MAX_JOBS:-$MAX_JOBS}"
        export EXT_PARALLEL="$SAGE_EXT_PARALLEL"
        export MAX_JOBS="$SAGE_MAX_JOBS"
        export TORCH_CUDA_ARCH_LIST="$SAGE_ARCH_LIST"
        export NVCC_APPEND_FLAGS="--threads ${SAGE_NVCC_THREADS}"
        echo "  SageAttention build config: ARCH=${TORCH_CUDA_ARCH_LIST}, EXT_PARALLEL=${EXT_PARALLEL}, MAX_JOBS=${MAX_JOBS}, NVCC_APPEND_FLAGS='${NVCC_APPEND_FLAGS}'"
        echo -e "${YELLOW}⏳ Installing SageAttention from source (git@${SAGE_ATTN_GIT_REF}) — this may compile CUDA/Triton kernels.${NC}"
        run_uv_pip "git+https://github.com/thu-ml/SageAttention.git@${SAGE_ATTN_GIT_REF}" --no-build-isolation 2>&1 || \
            pip install "git+https://github.com/thu-ml/SageAttention.git@${SAGE_ATTN_GIT_REF}" --no-build-isolation 2>&1 || \
            echo -e "${YELLOW}⚠${NC} SageAttention install failed — check CUDA/PyTorch toolchain"
    else
        echo -e "${YELLOW}↷ Skipping SageAttention (set INSTALL_SAGEATTN=0 to disable default install)${NC}"
    fi
fi

echo -e "${GREEN}✓${NC} PyTorch and GPU packages installed\n"

# Step 4b: Verify PyTorch can see the GPU
echo -e "${BLUE}Step 4b: GPU Verification${NC}"
if [ "$FAST_SETUP" = "1" ]; then
    echo -e "${YELLOW}↷ Skipping GPU verification in FAST_SETUP mode${NC}"
else
    if ! $VENV_PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'  ✓ CUDA available: {torch.cuda.get_device_name(0)}'); print(f'  ✓ Compute capability: {torch.cuda.get_device_capability(0)}')" 2>/dev/null; then
      echo -e "${RED}Error: PyTorch cannot access CUDA.${NC}"
      echo -e "  CUDA toolkit is installed (${CUDA_DIR}), but PyTorch can't use it."
      echo -e "  Check: NVIDIA drivers, PyTorch CUDA version match, WSL2 GPU passthrough."
      exit_script 1
    fi
fi
echo ""

# Step 5: Verify installation
echo -e "${BLUE}Step 5: Verification${NC}"
$VENV_PYTHON -m pip check || echo -e "${YELLOW}⚠${NC} pip check reported warnings"
$VENV_PYTHON -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || \
  echo -e "${YELLOW}⚠${NC} Could not verify torch (torch may not be importable yet)"

# Step 6: VSCode extensions (optional)
echo ""
if command -v code &>/dev/null && [ -f scripts/setup/install_vscode_extensions.py ]; then
    if [ "$FAST_SETUP" = "1" ]; then
        echo -e "${YELLOW}↷ Skipping VSCode extension install in FAST_SETUP mode${NC}"
    else
        echo -e "${BLUE}Step 6: VSCode Extensions${NC}"
        $VENV_PYTHON scripts/setup/install_vscode_extensions.py
    fi
fi

# Summary
echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}\n"
echo -e "${CYAN}Environment:${NC}"
echo "  Python: $($PYTHON_CMD --version 2>&1)"
echo "  venv: $VIRTUAL_ENV"
echo "  UV: $([ -x "${UV_BIN:-}" ] && "$UV_BIN" --version 2>/dev/null || uv --version 2>/dev/null || echo 'N/A')"
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
