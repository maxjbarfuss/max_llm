#!/usr/bin/env python3
"""
System dependency setup for max_llm.
Single-pass check and auto-install on WSL2. Idempotent and re-runnable.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# Colors
RED, GREEN, YELLOW, BLUE, CYAN, NC = (
    "\033[0;31m",
    "\033[0;32m",
    "\033[1;33m",
    "\033[0;34m",
    "\033[0;36m",
    "\033[0m",
)

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 13)

# All system dependencies in one place: name -> apt package
CHECKS = {
    "git": ("system", "git"),
    "curl": ("system", "curl"),
    "wget": ("system", "wget"),
    "cmake": ("build", "cmake"),
    "ninja": ("build", "ninja-build"),
}

PYTHON_DEV_PKGS = ["python3-dev", "python3-numpy"]

COMPILERS = {
    "C": ["gcc", "clang"],
    "C++": ["g++", "clang++"],
}

CUDA_LIBS = ["libnccl2", "libnccl-dev"]
CUDA_DEB_TEMPLATE = "https://developer.download.nvidia.com/compute/cuda/repos/{distro}/x86_64/cuda-keyring_1.1-1_all.deb"


class Result(NamedTuple):
    name: str
    passed: bool
    version: str = ""
    msg: str = ""


def log(s: str) -> None:
    print(f"{BLUE}[setup]{NC} {s}")


def ok(s: str) -> None:
    print(f"{GREEN}✓{NC} {s}")


def warn(s: str) -> None:
    print(f"{YELLOW}⚠{NC} {s}")


def err(s: str) -> None:
    print(f"{RED}✗{NC} {s}")


def header(s: str) -> None:
    print(f"\n{BLUE}{'='*70}\n{s:^70}\n{'='*70}{NC}\n")


def cmd_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(
    cmd: list[str], check: bool = True, quiet: bool = False, timeout: int = 300
) -> tuple[bool, str]:
    """Run a command, return (success, output)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        if check and r.returncode != 0:
            if not quiet:
                print(r.stderr or r.stdout)
            return False, r.stderr or r.stdout
        return r.returncode == 0, r.stdout
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


# Cached sudo state — check once per process, not per install call
_sudo_ok: bool | None = None


def sudo_check() -> bool:
    global _sudo_ok
    if _sudo_ok is not None:
        return _sudo_ok
    if run(["sudo", "-n", "true"], check=False, quiet=True)[0]:
        _sudo_ok = True
        return True
    print("Enter sudo password (cached for 15 min):")
    _sudo_ok = run(["sudo", "-v"], quiet=True)[0]
    return _sudo_ok


def apt_update() -> bool:
    """Run apt-get update once. Call this before any batch of installs."""
    if not sudo_check():
        return False
    return run(["sudo", "apt-get", "update", "-qq"], quiet=True)[0]


def dpkg_repair() -> None:
    """Fix broken dpkg state (e.g. interrupted installs, missing directories).

    CUDA toolkit config packages use update-alternatives with paths like
    /usr/local/cuda-12.x that may not exist yet, causing dpkg --configure to
    fail. We create the expected directories before retrying.
    """
    # Check if there are unconfigured packages
    ok, out = run(["dpkg", "--audit"], check=False, quiet=True, timeout=15)
    if ok and not out.strip():
        return  # nothing broken

    log("Repairing broken dpkg state...")
    # Pre-create /usr/local/cuda-* dirs that config packages expect
    _, pending = run(["dpkg", "--yet-to-unpack"], check=False, quiet=True, timeout=15)
    _, audit = run(["dpkg", "--audit"], check=False, quiet=True, timeout=15)
    for text in [pending or "", audit or ""]:
        for token in text.split():
            if "cuda-toolkit" in token and "config" in token:
                # Extract version: cuda-toolkit-12-9-config-common → 12.9
                parts = token.split("-")
                for i, p in enumerate(parts):
                    if p.isdigit() and i + 1 < len(parts) and parts[i + 1].isdigit():
                        cuda_dir = Path(f"/usr/local/cuda-{p}.{parts[i+1]}")
                        if not cuda_dir.exists():
                            run(["sudo", "mkdir", "-p", str(cuda_dir)], quiet=True)
                        break

    run(["sudo", "dpkg", "--configure", "-a"], quiet=True, timeout=120)


def apt_install(pkgs: list[str]) -> bool:
    """Install packages via apt. Repairs broken dpkg state first for resilience."""
    if not sudo_check():
        return False
    dpkg_repair()
    return run(["sudo", "apt-get", "install", "-y", "-qq"] + pkgs)[0]


def package_installed(pkg: str) -> bool:
    """Check whether a Debian package is installed."""
    success, output = run(["dpkg-query", "-W", "-f=${Status}", pkg], check=False, quiet=True)
    return success and "install ok installed" in output


def is_wsl2() -> bool:
    try:
        content = Path("/proc/version").read_text().lower()
        return "microsoft" in content and ("wsl2" in content or "hyper-v" in content)
    except Exception:
        return False


def check_cmd(name: str, apt_pkg: str | None = None) -> Result:
    """Check if command exists; install via apt if missing."""
    if cmd_exists(name):
        success, out = run([name, "--version"], check=False, quiet=True)
        version = out.split("\n")[0].split()[-1] if success and out else ""
        return Result(name, True, version)

    if apt_pkg:
        log(f"Installing {name}...")
        if apt_install([apt_pkg]) and cmd_exists(name):
            return Result(name, True, "installed")

    return Result(name, False, "", f"apt install {apt_pkg or name}")


def check_compiler(lang: str, cmds: list[str]) -> tuple[bool, str | None]:
    """Check for at least one compiler of the given type."""
    for cmd in cmds:
        if cmd_exists(cmd):
            return True, cmd
    log(f"Installing {lang} compiler ({cmds[0]})...")
    if apt_install([cmds[0]]) and cmd_exists(cmds[0]):
        return True, cmds[0]
    return False, None


def detect_ubuntu_version() -> str:
    """Detect Ubuntu version for CUDA repo URL (e.g. 'ubuntu2204')."""
    try:
        info: dict[str, str] = {}
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k] = v.strip('"')
        return "ubuntu" + info.get("VERSION_ID", "22.04").replace(".", "")
    except Exception:
        return "ubuntu2204"


def find_best_cuda_toolkit() -> str | None:
    """Find the best available cuda-toolkit-12-x package in apt cache."""
    _, out = run(["apt-cache", "search", "^cuda-toolkit-12-"], check=False, quiet=True, timeout=15)
    if not out:
        return None
    # Parse package names like "cuda-toolkit-12-5 - CUDA Toolkit 12.5 meta-package"
    # Pick the latest 12.x version
    candidates = []
    for line in out.strip().splitlines():
        pkg = line.split()[0]  # e.g. "cuda-toolkit-12-5"
        parts = pkg.split("-")
        if len(parts) >= 4 and parts[2] == "12" and parts[3].isdigit():
            candidates.append((int(parts[3]), pkg))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def check_cuda() -> list[Result]:
    results: list[Result] = []

    # NVIDIA driver — must be installed on the Windows side; not apt-installable from Linux
    if cmd_exists("nvidia-smi"):
        _, out = run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            quiet=True,
        )
        version = out.strip().split("\n")[0] if out.strip() else "detected"
        results.append(Result("nvidia-smi", True, version))
    else:
        results.append(
            Result(
                "nvidia-smi",
                False,
                "",
                "Install NVIDIA drivers on Windows; WSL2 exposes them automatically",
            )
        )

    # nvcc — installable from Linux side via CUDA toolkit
    nvcc_installed = cmd_exists("nvcc") or any(
        (Path(p) / "nvcc").exists() for p in ["/usr/local/cuda/bin", "/usr/local/cuda-12.1/bin"]
    )

    if not nvcc_installed:
        log("Installing CUDA toolkit...")
        if not sudo_check():
            return results + [Result("CUDA", False, "", "Requires sudo")]

        # Repair any broken dpkg state before attempting install
        dpkg_repair()

        distro = detect_ubuntu_version()
        cuda_deb_url = CUDA_DEB_TEMPLATE.format(distro=distro)
        deb_file = Path("/tmp/cuda-keyring.deb")
        success, _ = run(["wget", "-q", cuda_deb_url, "-O", str(deb_file)], quiet=True, timeout=60)

        if success and deb_file.exists():
            run(["sudo", "dpkg", "-i", str(deb_file)], quiet=True)
            deb_file.unlink(missing_ok=True)
            apt_update()
            # Find best available CUDA 12.x toolkit
            best_pkg = find_best_cuda_toolkit()
            if best_pkg:
                log(f"Installing {best_pkg}...")
                apt_install([best_pkg])
            else:
                log("Warning: No cuda-toolkit-12-x package found in repo")

    if cmd_exists("nvcc"):
        results.append(check_cmd("nvcc"))
    else:
        # Check common install paths
        for cuda_dir in ["/usr/local/cuda/bin", "/usr/local/cuda-12.1/bin"]:
            if (Path(cuda_dir) / "nvcc").exists():
                results.append(Result("nvcc", True, f"installed ({cuda_dir})"))
                break
        else:
            results.append(Result("nvcc", False, "", "CUDA toolkit install failed"))

    # NCCL
    for pkg in CUDA_LIBS:
        if not package_installed(pkg):
            apt_install([pkg])
        installed = package_installed(pkg)
        results.append(
            Result(
                pkg, installed, "installed" if installed else "", "" if installed else "apt install"
            )
        )

    # cuDNN — install via apt (cudnn9-cuda-12 or libcudnn9-cuda-12)
    cudnn_found = any(
        Path(p).exists()
        for p in [
            "/usr/local/cuda/include/cudnn.h",
            "/usr/include/cudnn.h",
            "/usr/include/x86_64-linux-gnu/cudnn_v9.h",
        ]
    ) or package_installed("libcudnn9-cuda-12")

    if not cudnn_found:
        log("Installing cuDNN...")
        # Try meta-package first, then lib package
        if apt_install(["cudnn9-cuda-12"]) or apt_install(
            ["libcudnn9-cuda-12", "libcudnn9-dev-cuda-12"]
        ):
            cudnn_found = True

    results.append(
        Result(
            "cuDNN",
            cudnn_found,
            "installed" if cudnn_found else "",
            "" if cudnn_found else "apt install cudnn9-cuda-12",
        )
    )

    return results


def main() -> int:
    header("SYSTEM DEPENDENCY SETUP")

    if not is_wsl2():
        err("WSL2 required. See: https://docs.microsoft.com/en-us/windows/wsl/install")
        return 1
    ok("WSL2 detected")

    if not cmd_exists("apt-get"):
        err("apt-get not found. Requires an apt-based WSL2 distribution (Ubuntu/Debian).")
        return 1

    major, minor = sys.version_info[:2]
    if not (MIN_PYTHON <= (major, minor) <= MAX_PYTHON):
        err(
            f"Python {major}.{minor} outside supported range {MIN_PYTHON[0]}.{MIN_PYTHON[1]}–{MAX_PYTHON[0]}.{MAX_PYTHON[1]}"
        )
        return 1
    ok(f"Python {major}.{minor}.{sys.version_info[2]}")

    # Determine which system packages are missing before touching apt
    print(f"\n{CYAN}System Dependencies{NC}")
    missing_system: list[str] = []
    results_system: list[Result] = []
    for name, (_, pkg) in CHECKS.items():
        if not cmd_exists(name):
            missing_system.append(pkg)

    # One apt-get update if anything is missing
    if missing_system:
        log(f"Installing: {', '.join(missing_system)}")
        apt_update()
        apt_install(missing_system)

    errors = 0
    for name, (_, pkg) in CHECKS.items():
        result = check_cmd(name)  # re-check; already installed above if it was missing
        icon = f"{GREEN}✓{NC}" if result.passed else f"{RED}✗{NC}"
        print(f"  {icon} {result.name:<20} {result.version or result.msg}")
        if not result.passed:
            errors += 1

    # Python dev headers / NumPy for CMake
    print(f"\n{CYAN}Python Dev{NC}")
    missing_py: list[str] = [pkg for pkg in PYTHON_DEV_PKGS if not package_installed(pkg)]
    if missing_py:
        if not missing_system:
            apt_update()
        apt_install(missing_py)

    for pkg in PYTHON_DEV_PKGS:
        installed = package_installed(pkg)
        icon = f"{GREEN}✓{NC}" if installed else f"{RED}✗{NC}"
        print(f"  {icon} {pkg:<20} {'installed' if installed else 'missing'}")
        if not installed:
            errors += 1

    # Compilers
    print(f"\n{CYAN}Compilers{NC}")
    compiler_ok = True
    missing_compilers: list[str] = []
    for lang, cmds in COMPILERS.items():
        if not any(cmd_exists(c) for c in cmds):
            missing_compilers.append(cmds[0])  # install first option (gcc / g++)

    if missing_compilers:
        if not missing_system:  # apt_update already ran if system deps were missing
            apt_update()
        apt_install(missing_compilers)

    for lang, cmds in COMPILERS.items():
        found, cmd = next(((True, c) for c in cmds if cmd_exists(c)), (False, None))
        icon = f"{GREEN}✓{NC}" if found else f"{RED}✗{NC}"
        print(f"  {icon} {lang:<20} {cmd or 'NOT FOUND'}")
        if not found:
            compiler_ok = False

    # CUDA
    print(f"\n{CYAN}CUDA & GPU (Optional){NC}")
    cuda_errors = 0
    for result in check_cuda():
        icon = f"{GREEN}✓{NC}" if result.passed else f"{RED}✗{NC}"
        print(f"  {icon} {result.name:<20} {result.version or result.msg}")
        # Only count driver (nvidia-smi) as critical; nvcc/cuDNN are optional
        if not result.passed and result.name == "nvidia-smi":
            cuda_errors += 1

    total = errors + (0 if compiler_ok else 1) + cuda_errors
    header("SUMMARY")
    if total > 0:
        err(f"{total} critical issue(s) found")
        return 1

    if cuda_errors == 0 and errors == 0 and compiler_ok:
        ok("All system dependencies ready!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
