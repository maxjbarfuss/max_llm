"""GPU acceleration stack smoke tests."""

import pytest
import torch
import torch.nn as nn


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _make_qkv(batch=2, seq=8, heads=4, head_dim=32, dtype=torch.bfloat16):
    q = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=dtype)
    return q, k, v


def test_bf16_matmul_smoke():
    _require_cuda()

    # BF16 matmul should work on the target stack.
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    z = x @ y
    assert z.shape == (64, 64)
    assert z.dtype == torch.bfloat16


def test_bf16_flash_attn_smoke():
    _require_cuda()

    flash_attn = pytest.importorskip("flash_attn")
    q, k, v = _make_qkv()

    out_flash = flash_attn.flash_attn_func(q, k, v)
    assert out_flash.shape == q.shape
    assert out_flash.dtype == torch.bfloat16


def test_bf16_xformers_smoke():
    _require_cuda()

    xformers_ops = pytest.importorskip("xformers.ops")
    q, k, v = _make_qkv()

    out_xformers = xformers_ops.memory_efficient_attention(q, k, v)
    assert out_xformers.shape == q.shape
    assert out_xformers.dtype == torch.bfloat16


def test_bf16_sageattention_smoke():
    _require_cuda()

    sageattention = pytest.importorskip("sageattention")

    q, k, v = _make_qkv()

    out = sageattention.sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16


def test_torchao_int8_dynamic_quantization_smoke():
    torchao_quant = pytest.importorskip("torchao.quantization")

    quantize_ = getattr(torchao_quant, "quantize_", None)
    int8_cfg_cls = getattr(torchao_quant, "Int8DynamicActivationInt8WeightConfig", None)
    if quantize_ is None or int8_cfg_cls is None:
        pytest.skip("torchao dynamic int8 quantization API not available")

    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    ).eval()

    quantize_(model, int8_cfg_cls())
    x = torch.randn(4, 32)
    y = model(x)

    assert y.shape == (4, 16)
    assert torch.isfinite(y).all()
