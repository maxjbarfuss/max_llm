#!/bin/bash
# Setup script for max_llm project
# Creates virtual environment and installs all dependencies using uv
# Usage: source setup.sh

# Detect if script is being sourced or executed
SOURCED=0
if [ -n "$ZSH_VERSION" ]; then
    [[ $ZSH_EVAL_CONTEXT =~ :file$ ]] && SOURCED=1
elif [ -n "$BASH_VERSION" ]; then
    [[ "${BASH_SOURCE[0]}" != "${0}" ]] && SOURCED=1
fi

# Function to exit/return appropriately
exit_script() {
    if [ $SOURCED -eq 1 ]; then
        return "$1"
    else
        exit "$1"
    fi
}

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Max LLM Setup ===${NC}"
echo ""

# Check if running from correct directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Run from max_llm directory${NC}"
    exit_script 1
fi

# Notify if not sourced
if [ $SOURCED -eq 0 ]; then
    echo -e "${YELLOW}Note: Script is being executed. Virtual environment will not persist after completion.${NC}"
    echo -e "${YELLOW}To keep the virtual environment active, run: source setup.sh${NC}"
    echo ""
fi

# Step 0: Run system requirements check
echo -e "${BLUE}Step 0: System Requirements Check${NC}"
if [ -f "check_system.py" ]; then
    python3 check_system.py
    CHECK_STATUS=$?
    if [ $CHECK_STATUS -ne 0 ]; then
        echo ""
        echo -e "${YELLOW}Please install missing dependencies before continuing.${NC}"
        exit_script 1
    fi
else
    echo -e "${YELLOW}⚠${NC} check_system.py not found, skipping system check"
fi
echo ""

set -e

# Step 1: Create/activate virtual environment
echo -e "${BLUE}Step 1: Virtual Environment${NC}"
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        echo -e "${GREEN}✓${NC} Found existing virtual environment"
        echo "  Activating..."
        source .venv/bin/activate
    else
        echo "  Creating new virtual environment..."
        python3 -m venv .venv
        source .venv/bin/activate
        echo -e "${GREEN}✓${NC} Virtual environment created"
    fi
else
    echo -e "${GREEN}✓${NC} Virtual environment already active: $VIRTUAL_ENV"
fi
echo ""

# Step 2: Install/upgrade UV package manager
echo -e "${BLUE}Step 2: Setting up UV package manager${NC}"
if ! command -v uv &>/dev/null; then
    echo "  Installing UV..."
    pip install --upgrade --quiet pip uv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install UV${NC}"
        exit_script 1
    fi
    echo -e "${GREEN}✓${NC} UV installed successfully"
else
    UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓${NC} UV already available: $UV_VERSION"
    # Upgrade UV to latest
    echo "  Upgrading UV to latest..."
    uv self update --quiet 2>/dev/null || true
fi
echo ""

# Step 3: Install PyTorch with CUDA 12.1 using UV
echo -e "${BLUE}Step 3: Installing PyTorch with CUDA 12.1${NC}"
echo "  This may take several minutes..."
echo "  Using UV for fast installation..."
uv pip install --quiet "torch>=2.3.0" --index-url https://download.pytorch.org/whl/cu121
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠${NC} PyTorch installation encountered issues"
    echo "  Trying again with pip..."
    pip install --quiet "torch>=2.3.0" --index-url https://download.pytorch.org/whl/cu121
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: PyTorch installation failed${NC}"
        exit_script 1
    fi
fi
echo -e "${GREEN}✓${NC} PyTorch installed"
echo ""

# Step 4: Install project dependencies using UV
echo -e "${BLUE}Step 4: Installing project dependencies${NC}"
echo "  Installing in editable mode with cuda121, dev, and training extras..."
uv pip install --quiet -e ".[cuda121,dev,training]"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Dependency installation failed${NC}"
    exit_script 1
fi
echo -e "${GREEN}✓${NC} Project dependencies installed"
echo ""

# Step 5: Compile any build dependencies
echo -e "${BLUE}Step 5: Building optional components${NC}"
echo "  Installing DeepSpeed for distributed training..."
uv pip install --quiet deepspeed 2>/dev/null || {
    echo -e "${YELLOW}⚠${NC} DeepSpeed installation optional - CPU-only training possible"
}
echo -e "${GREEN}✓${NC} Optional components installed"
echo ""

# Step 6: Verify complete installation
echo -e "${BLUE}Step 6: Verifying installation${NC}"
if [ -f "check_deps.py" ]; then
    python3 check_deps.py
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠${NC} Some dependencies have warnings"
    else
        echo -e "${GREEN}✓${NC} All dependencies verified"
    fi
else
    echo -e "${YELLOW}⚠${NC} check_deps.py not found, skipping detailed verification"
fi
echo ""

# Summary
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo -e "${CYAN}Environment Summary:${NC}"
echo "  Virtual environment: $(python3 -c 'import sys; print(sys.prefix)')"
echo "  Python version: $(python3 --version)"
echo "  UV version: $(uv --version 2>/dev/null || echo 'N/A')"
echo ""

if [ $SOURCED -eq 1 ]; then
    echo -e "${GREEN}✓${NC} Virtual environment is active in your current shell"
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo "  • Run tests: ${BLUE}make test${NC}"
    echo "  • Format code: ${BLUE}make format${NC}"
    echo "  • Type check: ${BLUE}make type-check${NC}"
    echo "  • Deactivate env: ${BLUE}deactivate${NC}"
else
    echo -e "${YELLOW}Note: To activate the virtual environment, run:${NC}"
    echo "  ${BLUE}source .venv/bin/activate${NC}"
fi
echo ""

exit_script 0
