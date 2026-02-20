#!/usr/bin/env python3
"""
Install VSCode extensions for max_llm development.
"""

import shutil
import subprocess

# Color codes
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
NC = "\033[0m"

# Required extensions
REQUIRED = [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "anthropic.claude-code",
    "GitHub.copilot",
]

# Recommended extensions
RECOMMENDED = [
    "ms-python.debugpy",
    "ms-python.black-formatter",
    "charliermarsh.ruff",
    "ms-python.mypy-type-checker",
    "ms-python.vscode-python-envs",
    "ms-vscode.makefile-tools",
    "ms-vscode.cmake-tools",
    "ms-vscode.cpp-devtools",
    "GitHub.copilot-chat",
    "andrepimenta.claude-code-chat",
    "eamodio.gitlens",
    "bierner.markdown-mermaid",
    "yzhang.markdown-all-in-one",
]


def log(msg: str) -> None:
    print(f"{BLUE}{msg}{NC}")


def ok(msg: str) -> None:
    print(f"{GREEN}✓{NC} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}⚠{NC} {msg}")


def err(msg: str) -> None:
    print(f"{RED}✗{NC} {msg}")


def code_installed() -> bool:
    """Check if VSCode is installed."""
    return shutil.which("code") is not None


def install_extension(ext_id: str) -> bool:
    """Install a single VSCode extension."""
    print(f"  Installing {ext_id}... ", end="", flush=True)
    result = subprocess.run(
        ["code", "--install-extension", ext_id, "--force"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        ok("")
        return True
    else:
        warn("failed")
        return False


def main() -> None:
    """Main function."""
    # Check if VSCode is installed
    if not code_installed():
        warn("VSCode not found in PATH")
        print("VSCode extensions require VSCode to be installed and available as 'code' command.")
        print("Install from: https://code.visualstudio.com/")
        return

    log("Installing VSCode Extensions")
    print()

    installed = 0

    log("Required:")
    for ext in REQUIRED:
        if install_extension(ext):
            installed += 1

    print()
    log("Recommended:")
    for ext in RECOMMENDED:
        if install_extension(ext):
            installed += 1

    print()
    ok(f"Installed {installed} extensions")
    warn("Note: Reload VSCode (Ctrl+R or Cmd+R) for changes to take effect")


if __name__ == "__main__":
    main()
