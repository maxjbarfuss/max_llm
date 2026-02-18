#!/bin/bash
# Setup script for max_llm project
# Creates virtual environment and installs all dependencies
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

set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

# Step 2: Upgrade pip
echo -e "${BLUE}Step 2: Upgrading pip${NC}"
pip install --upgrade pip setuptools wheel -q
echo -e "${GREEN}✓${NC} pip upgraded to $(pip --version | cut -d' ' -f2)"
echo ""

# Step 3: Install uv
echo -e "${BLUE}Step 3: Installing uv package manager${NC}"
if ! command -v uv &>/dev/null; then
    echo "  Installing uv..."
    pip install -q uv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install uv${NC}"
        exit_script 1
    fi
    echo -e "${GREEN}✓${NC} uv installed successfully"
else
    echo -e "${GREEN}✓${NC} uv already installed"
fi
echo ""

# Step 4: Install PyTorch with CUDA 12.1
echo -e "${BLUE}Step 4: Installing PyTorch with CUDA 12.1${NC}"
echo "  This may take several minutes..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: PyTorch installation failed${NC}"
    exit_script 1
fi
echo -e "${GREEN}✓${NC} PyTorch installed"
echo ""

# Step 5: Install project dependencies
echo -e "${BLUE}Step 5: Installing project dependencies${NC}"
echo "  Installing in editable mode with dev and training extras..."
uv pip install -e ".[cuda121,dev,training]"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Dependency installation failed${NC}"
    exit_script 1
fi
echo -e "${GREEN}✓${NC} Dependencies installed"
echo ""

# Step 6: Install DeepSpeed
echo -e "${BLUE}Step 6: Installing DeepSpeed${NC}"
echo "  Installing DeepSpeed for distributed training..."
uv pip install deepspeed
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: DeepSpeed installation failed${NC}"
    exit_script 1
fi
echo -e "${GREEN}✓${NC} DeepSpeed installed"
echo ""

# Step 7: Verify installation
echo -e "${BLUE}Step 7: Verifying installation${NC}"
if [ -f "check_deps.py" ]; then
    python3 check_deps.py
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠${NC} Some dependencies may have issues"
    else
        echo -e "${GREEN}✓${NC} All dependencies verified"
    fi
else
    echo -e "${YELLOW}⚠${NC} check_deps.py not found, skipping verification"
fi
echo ""

# Summary
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Virtual environment: $(python3 -c 'import sys; print(sys.prefix)')"
echo "Python version: $(python3 --version)"
echo "pip version: $(pip --version | cut -d' ' -f2)"
echo "uv version: $(uv --version 2>/dev/null || echo 'N/A')"
echo ""

if [ $SOURCED -eq 1 ]; then
    echo -e "${GREEN}✓${NC} Virtual environment is active in your current shell"
    echo "  To deactivate: ${BLUE}deactivate${NC}"
else
    echo -e "${YELLOW}Note: To activate the virtual environment, run:${NC}"
    echo "  ${BLUE}source .venv/bin/activate${NC}"
fi
echo ""

exit_script 0
