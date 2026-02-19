#!/usr/bin/env python3
import sys
from importlib import import_module

DEPENDENCIES = {
    "Core Deep Learning": [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
    ],
    "Performance": [
        ("xformers", "xFormers"),
        ("deepspeed", "DeepSpeed"),
        ("bitsandbytes", "bitsandbytes"),
    ],
    "Data & Utils": [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("yaml", "PyYAML"),
        ("pydantic", "Pydantic"),
        ("omegaconf", "OmegaConf"),
        ("tqdm", "tqdm"),
    ],
    "Tokenization": [
        ("sentencepiece", "SentencePiece"),
        ("tokenizers", "Tokenizers"),
    ],
    "Monitoring": [
        ("tensorboard", "TensorBoard"),
    ],
}

DEV_DEPENDENCIES = [
    ("black", "Black"),
    ("ruff", "Ruff"),
    ("mypy", "MyPy"),
    ("pytest", "Pytest"),
]


def check_module(module_name: str, display_name: str) -> bool:
    """Check if module exists and return version."""
    try:
        mod = import_module(module_name)
        version = getattr(mod, "__version__", "?")
        print(f"  ✅ {display_name:<25} {version}")
        return True
    except (ImportError, Exception):
        print(f"  ❌ {display_name:<25} NOT INSTALLED")
        return False


def check_cuda() -> bool:
    """Check CUDA availability."""
    try:
        import torch  # pyright: ignore[reportMissingImports]

        print("\nCUDA & GPU:")
        if not torch.cuda.is_available():
            print("  ⚠️  CUDA not available")
            return False

        print(f"  ✅ CUDA: {torch.version.cuda}")
        print(f"  ✅ GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"      [{i}] {name} ({mem:.0f}GB)")

        # GPU compute test
        x = torch.randn(1024, 1024, device="cuda")
        y = torch.randn(1024, 1024, device="cuda")
        z = torch.matmul(x, y)
        del x, y, z
        print("  ✅ GPU compute: PASS")
        return True
    except Exception as e:
        print(f"  ❌ CUDA check failed: {e}")
        return False


def main() -> int:
    """Check all dependencies."""
    print("=" * 60)
    print("max_llm Dependencies Check")
    print("=" * 60)

    all_ok = True
    for category, deps in DEPENDENCIES.items():
        print(f"\n{category}:")
        for module_name, display_name in deps:
            if not check_module(module_name, display_name):
                all_ok = False

    print("\nDev Tools (Optional):")
    for module_name, display_name in DEV_DEPENDENCIES:
        check_module(module_name, display_name)

    if not check_cuda():
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All dependencies OK!")
        return 0
    else:
        print("❌ Some deps missing. Install with:")
        print("  uv pip install -e '.[cuda121,dev,training]'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
