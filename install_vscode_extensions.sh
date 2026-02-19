#!/bin/bash
# Install VSCode extensions for max_llm development
# Usage: ./install_vscode_extensions.sh

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if VSCode is installed
if ! command -v code &>/dev/null; then
    echo -e "${RED}Error: VSCode is not installed or not in PATH${NC}"
    echo "Install VSCode from: https://code.visualstudio.com/"
    exit 1
fi

echo -e "${BLUE}=== Installing VSCode Extensions ===${NC}"
echo ""

# Required extensions
REQUIRED=(
    "ms-python.python"
    "ms-python.vscode-pylance"
    "GitHub.copilot"
)

# Recommended extensions
RECOMMENDED=(
    "ms-python.debugpy"
    "ms-python.black-formatter"
    "charliermarsh.ruff"
    "ms-python.mypy-type-checker"
    "ms-vscode.makefile-tools"
    "ms-vscode.cmake-tools"
    "GitHub.copilot-chat"
    "eamodio.gitlens"
    "me-dutour-mathieu.vscode-json-pretty-printer"
)

INSTALLED=0
FAILED=0

install_extension() {
    local ext_id=$1
    local ext_type=$2  # "required" or "recommended"
    
    echo -n "  Installing $ext_id... "
    
    if code --install-extension "$ext_id" --force >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        ((INSTALLED++))
    else
        echo -e "${YELLOW}⚠${NC} (may already be installed)"
        ((INSTALLED++))
    fi
}

echo -e "${BLUE}Required Extensions:${NC}"
for ext in "${REQUIRED[@]}"; do
    install_extension "$ext" "required"
done

echo ""
echo -e "${BLUE}Recommended Extensions:${NC}"
for ext in "${RECOMMENDED[@]}"; do
    install_extension "$ext" "recommended"
done

echo ""
echo -e "${GREEN}=== Installation Complete ===${NC}"
echo "Installed: $INSTALLED extensions"
echo ""
echo -e "${BLUE}Note:${NC} You may need to reload VSCode for extensions to activate"
echo "  • Use Ctrl+R (or Cmd+R on macOS) to reload the window, or"
echo "  • Close and reopen VSCode"
echo ""
