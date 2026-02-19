#!/usr/bin/env python3
"""
System requirements checker for max_llm.
Validates Python version, system dependencies, CUDA, and VSCode extensions.
Provides clear guidance on what needs to be installed.
"""

import json
import platform
import shutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

# Color codes
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
NC = "\033[0m"  # No Color

# Python version requirement
MIN_PYTHON_VERSION = (3, 10)
MAX_PYTHON_VERSION = (3, 13)

# System dependencies
SYSTEM_DEPS_UNIX = {
    "git": "Version control",
    "gcc": "C compiler (for building packages)",
}

SYSTEM_DEPS_WSL2 = {
    "git": "Version control",
    "gcc": "C compiler (for building packages)",
    "curl": "Download utility",
}

# VSCode extensions
REQUIRED_EXTENSIONS = [
    "ms-python.python",  # Python
    "ms-python.vscode-pylance",  # Pylance for type checking
    "GitHub.copilot",  # GitHub Copilot
]

RECOMMENDED_EXTENSIONS = [
    "ms-python.debugpy",  # Python debugger
    "ms-python.black-formatter",  # Black formatter
    "charliermarsh.ruff",  # Ruff linter
    "ms-python.mypy-type-checker",  # MyPy type checker
    "ms-vscode.makefile-tools",  # Makefile tools
    "ms-vscode.cmake-tools",  # CMake tools
    "GitHub.copilot-chat",  # GitHub Copilot Chat
    "eamodio.gitlens",  # GitLens
    "me-dutour-mathieu.vscode-json-pretty-printer",  # JSON formatter
]


class CheckResult(NamedTuple):
    """Result of a system check."""
    name: str
    passed: bool
    version: str = ""
    message: str = ""
    severity: str = "error"  # error, warning, info


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 70}{NC}")
    print(f"{BLUE}{text:^70}{NC}")
    print(f"{BLUE}{'=' * 70}{NC}\n")


def print_section(text: str) -> None:
    """Print a formatted section."""
    print(f"\n{CYAN}{text}{NC}")
    print(f"{CYAN}{'-' * len(text)}{NC}")


def print_result(result: CheckResult) -> None:
    """Print a check result with color coding."""
    if result.passed:
        icon = f"{GREEN}✓{NC}"
        status = f"{GREEN}OK{NC}"
    else:
        if result.severity == "warning":
            icon = f"{YELLOW}⚠{NC}"
            status = f"{YELLOW}WARNING{NC}"
        else:
            icon = f"{RED}✗{NC}"
            status = f"{RED}FAILED{NC}"

    version_str = f" ({result.version})" if result.version else ""
    print(f"  {icon} {result.name:<35} {status}{version_str}")
    if result.message:
        print(f"     {YELLOW}{result.message}{NC}")


def check_python_version() -> CheckResult:
    """Check Python version."""
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"

    if (major, minor) < MIN_PYTHON_VERSION:
        return CheckResult(
            "Python version",
            False,
            version_str,
            f"Need Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+, have {major}.{minor}",
            "error",
        )
    elif (major, minor) > MAX_PYTHON_VERSION:
        return CheckResult(
            "Python version",
            True,
            version_str,
            f"Using {major}.{minor} (newer than max target {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]})",
            "warning",
        )
    else:
        return CheckResult("Python version", True, version_str)


def check_system_command(cmd: str, display_name: str) -> CheckResult:
    """Check if a system command is available."""
    path = shutil.which(cmd)
    if not path:
        return CheckResult(
            display_name,
            False,
            "",
            f"Install with: apt-get install {cmd}",
            "error",
        )

    try:
        result = subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = result.stdout.split("\n")[0] if result.stdout else ""
        if version:
            version = version.split()[-1] if version.split() else ""
        return CheckResult(display_name, True, version)
    except Exception as e:
        return CheckResult(display_name, True, "installed", f"(could not get version: {e})")


def detect_wsl2() -> bool:
    """Detect if running in WSL2."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def check_system_dependencies() -> List[CheckResult]:
    """Check all required system dependencies."""
    deps = SYSTEM_DEPS_WSL2 if detect_wsl2() else SYSTEM_DEPS_UNIX

    results = []
    for cmd, description in deps.items():
        results.append(check_system_command(cmd, f"{cmd} ({description})"))
    return results


def check_cuda() -> CheckResult:
    """Check CUDA availability and version."""
    try:
        import torch  # pyright: ignore[reportMissingImports]

        if not torch.cuda.is_available():
            return CheckResult(
                "CUDA",
                False,
                "",
                "CUDA not available. Install NVIDIA GPU drivers and CUDA Toolkit 12.1+",
                "warning",
            )

        cuda_version = torch.version.cuda or "unknown"
        gpu_count = torch.cuda.device_count()
        gpu_info = f"{gpu_count} GPU(s)"

        if gpu_count > 0:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            gpu_info += f": {', '.join(set(gpu_names))}"

        # Test GPU compute
        try:
            x = torch.randn(512, 512, device="cuda", dtype=torch.float32)
            y = torch.randn(512, 512, device="cuda", dtype=torch.float32)
            _ = torch.matmul(x, y)
            compute_status = "compute works"
        except Exception as e:
            compute_status = f"compute failed: {str(e)[:30]}"

        return CheckResult(
            "CUDA",
            True,
            f"{cuda_version}, {gpu_info}, {compute_status}",
        )
    except ImportError:
        return CheckResult(
            "CUDA",
            False,
            "",
            "PyTorch not installed yet. Will be installed during setup.",
            "info",
        )


def check_vscode_extensions() -> Tuple[List[CheckResult], List[CheckResult]]:
    """Check VSCode extensions (required and recommended)."""
    required_results = []
    recommended_results = []

    # Try to detect VSCode and installed extensions
    vscode_extensions = []
    try:
        # Look in common VSCode extension directories
        extensions_dirs = [
            Path.home() / ".vscode" / "extensions",
            Path.home() / ".vscode-server" / "extensions",  # For remote/WSL
        ]

        for ext_dir in extensions_dirs:
            if ext_dir.exists():
                vscode_extensions = [d.name for d in ext_dir.iterdir() if d.is_dir()]
                break
    except Exception:
        pass

    def is_extension_installed(ext_id: str) -> bool:
        """Check if extension is installed by ID."""
        # Extension folders are named like "publisher.extension-version"
        return any(ext_id.lower() in ext.lower() for ext in vscode_extensions)

    for ext_id in REQUIRED_EXTENSIONS:
        installed = is_extension_installed(ext_id)
        result = CheckResult(
            ext_id,
            installed,
            "installed" if installed else "",
        )
        required_results.append(result)

    for ext_id in RECOMMENDED_EXTENSIONS:
        installed = is_extension_installed(ext_id)
        result = CheckResult(
            ext_id,
            installed,
            "installed" if installed else "",
        )
        recommended_results.append(result)

    return required_results, recommended_results


def check_python_modules() -> List[CheckResult]:
    """Check Python development tools."""
    results = []
    tools = [
        ("uv", "UV package manager"),
        ("black", "Black formatter"),
        ("ruff", "Ruff linter"),
        ("mypy", "MyPy type checker"),
        ("pytest", "PyTest"),
    ]

    for module_name, display_name in tools:
        try:
            mod = import_module(module_name)
            version = getattr(mod, "__version__", "")
            results.append(CheckResult(display_name, True, version))
        except ImportError:
            results.append(
                CheckResult(
                    display_name,
                    False,
                    "",
                    "Will be installed with setup.sh",
                    "info",
                )
            )

    return results


def generate_install_guidance() -> str:
    """Generate installation guidance based on checks."""
    os_type = platform.system()
    guidance = ""

    if os_type == "Linux":
        if detect_wsl2():
            guidance += f"""
{YELLOW}WSL2 Setup Guidance:{NC}
  1. Update WSL2 and system packages:
     wsl --update
     sudo apt update && sudo apt upgrade -y

  2. Install system dependencies:
     sudo apt install -y build-essential git curl

  3. Install Python 3.10+ (if not already available):
     sudo apt install -y python3.10 python3.10-venv python3.10-dev
     python3.10 -m pip install --upgrade pip uv

  4. Install NVIDIA GPU support (if you have NVIDIA GPU):
     # WSL2 CUDA setup: https://docs.nvidia.com/cuda/wsl-user-guide/
     # Install CUDA Toolkit 12.1 in WSL2

  5. Install VSCode (if not already):
     # Download from https://code.visualstudio.com/download

  6. Install required VSCode extensions:
     # Run: code --install-extension <extension-id>
     # Or install from extensions panel in VSCode
"""
        else:
            guidance += f"""
{YELLOW}Linux Setup Guidance:{NC}
  1. Install system dependencies:
     sudo apt update && sudo apt install -y build-essential git curl

  2. Install Python 3.10+ (if not already available):
     python3 --version  # Check current version
     # If needed, install from deadsnakes PPA or source

  3. Install UV package manager:
     pip install --upgrade uv

  4. Install NVIDIA GPU support (for CUDA compute):
     # Install NVIDIA drivers: https://developer.nvidia.com/cuda-12-1-0
     # Install CUDA Toolkit 12.1

  5. Install VSCode (if not already):
     # Download from https://code.visualstudio.com/download

  6. Install required VSCode extensions (see below)
"""

    elif os_type == "Darwin":
        guidance += f"""
{YELLOW}macOS Setup Guidance:{NC}
  1. Install Homebrew (if not already):
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  2. Install system dependencies:
     brew install git gcc

  3. Install Python 3.10+ (if not already):
     brew install python@3.10
     python3.10 -m pip install --upgrade pip uv

  4. Install VSCode (if not already):
     brew install visual-studio-code

  5. Install required VSCode extensions (see below)

  Note: GPU support (CUDA) requires NVIDIA GraphNeuralNetworks environment
        Consider using CPU or cloud GPU for now
"""

    elif os_type == "Windows":
        guidance += f"""
{YELLOW}Windows Setup Guidance:{NC}
  Recommended: Use WSL2 (Windows Subsystem for Linux 2) for best compatibility

  1. Enable WSL2:
     # In PowerShell (as Administrator):
     wsl --install
     wsl --set-default-version 2

  2. Install Ubuntu 22.04 LTS from Microsoft Store

  3. Then follow the WSL2 setup guidance above

  Alternatively, for native Windows:
  1. Install Python 3.10+ from https://www.python.org/
  2. Install Git from https://git-scm.com/
  3. Install NVIDIA CUDA Toolkit 12.1
  4. Install VSCode from https://code.visualstudio.com/
  5. Install Python extension in VSCode
"""

    guidance += f"""

{YELLOW}VSCode Extensions Installation:{NC}
  Required extensions (install these):
"""

    for ext_id in REQUIRED_EXTENSIONS:
        guidance += f"    • {ext_id}\n"

    guidance += f"""
  Recommended extensions (optional but helpful):
"""

    for ext_id in RECOMMENDED_EXTENSIONS:
        guidance += f"    • {ext_id}\n"

    guidance += f"""
  Install extensions in VSCode:
    - Press Ctrl+Shift+X (or Cmd+Shift+X on macOS)
    - Search for extension ID
    - Click Install
    
  Or install from command line:
    code --install-extension <extension-id>
    code --install-extension <extension-id> --force  (to reinstall)

{BLUE}Next Steps:{NC}
  1. Fix any critical issues above (marked {RED}FAILED{NC})
  2. Run: source setup.sh
     (or: bash setup.sh if not using bash)
"""

    return guidance


def main() -> int:
    """Run all system checks."""
    print_header("MAX LLM SYSTEM REQUIREMENTS CHECK")

    # Python version (critical)
    print_section("Core Requirements")
    py_result = check_python_version()
    print_result(py_result)

    # System dependencies
    print_section("System Dependencies")
    sys_results = check_system_dependencies()
    errors = 0
    for result in sys_results:
        print_result(result)
        if not result.passed and result.severity == "error":
            errors += 1

    # Python development tools
    print_section("Python Development Tools")
    py_tools = check_python_modules()
    for result in py_tools:
        print_result(result)

    # CUDA (warning if not available)
    print_section("GPU & CUDA")
    cuda_result = check_cuda()
    print_result(cuda_result)

    # VSCode extensions
    print_section("VSCode Extensions")
    req_ext, rec_ext = check_vscode_extensions()

    print(f"{CYAN}Required:{NC}")
    missing_required = 0
    for result in req_ext:
        print_result(result)
        if not result.passed:
            missing_required += 1

    print(f"{CYAN}Recommended:{NC}")
    for result in rec_ext:
        print_result(result)

    # Summary and guidance
    total_failures = 0 if py_result.passed else 1
    total_failures += errors

    print_header("SUMMARY")

    if total_failures > 0:
        print(f"{RED}⚠ Critical issues found:{NC}")
        if not py_result.passed:
            print(f"  • {py_result.message}")
        for result in sys_results:
            if not result.passed and result.severity == "error":
                print(f"  • {result.name}: {result.message}")

        print(generate_install_guidance())
        return 1

    if missing_required > 0:
        print(
            f"{YELLOW}⚠ Missing {missing_required} required VSCode extension(s).{NC}"
        )
        print("  Install from VSCode Extensions panel (Ctrl+Shift+X).")
        print()

    print(f"{GREEN}✓ System check passed!{NC}")
    print(f"{GREEN}You can now run: source setup.sh{NC}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
