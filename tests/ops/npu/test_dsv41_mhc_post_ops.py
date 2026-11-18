"""Unit tests for the CANN mhc_post op and its torchturbo provider.

Covers, against the HcPost broadcast reference chain:
- Forward: fused op vs reference at the production shape.
- Backward: fused mhc_post_backward vs reference autograd for all four
  gradients (x, h_res, h_out, h_post).
- Provider wiring: ops.mhc_post dispatch and the dim/dtype contract checks.

Run:

    python3 -m pytest tests/ops/npu/test_dsv41_mhc_post_ops.py -v

Environment (per the ops-transformer build doc): source the Ascend toolkit
set_env.sh before running.
"""

from __future__ import annotations

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")  # noqa: F401

if not torch.npu.is_available():
    pytest.skip("NPU required", allow_module_level=True)

from torchturbo.ops.providers.npu.mhc_post import (  # noqa: E402
    MhcPostFn,
    has_mhc_post,
)

B, S, N, D = 1, 8192, 4, 5120  # production shape (D multiple of 128)

# bf16 outputs and bf16-bound gradients: bounds sit well above bf16
# accumulation noise while still catching functional errors.
TOL = (2e-2, 2e-2)

requires_mhc_post = pytest.mark.skipif(
    not has_mhc_post(),
    reason="mhc_post(_backward) not registered (install cann_ops_transformer)",
)


def _reference_forward(x, residual, post, comb):
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return y.type_as(x)


def _make_inputs(seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    residual = torch.randn(B, S, N, D, generator=gen, dtype=torch.bfloat16).to("npu")
    x = torch.randn(B, S, D, generator=gen, dtype=torch.bfloat16).to("npu")
    post = (torch.rand(B, S, N, generator=gen) * 2).to("npu")
    comb = torch.rand(B, S, N, N, generator=gen).to("npu")
    comb = comb / comb.sum(-2, keepdim=True)
    return residual, x, post, comb


@requires_mhc_post
def test_forward_matches_reference():
    residual, x, post, comb = _make_inputs(42)

    actual = MhcPostFn.apply(residual, comb, x, post)
    expected = _reference_forward(x, residual, post, comb)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=TOL[0], atol=TOL[1])


@requires_mhc_post
def test_backward_matches_reference():
    grad_out = torch.randn(B, S, N, D, dtype=torch.bfloat16, device="npu")

    fused = tuple(t.clone().requires_grad_(True) for t in _make_inputs(43))
    MhcPostFn.apply(fused[0], fused[3], fused[1], fused[2]).backward(grad_out)
    reference = tuple(t.clone().requires_grad_(True) for t in _make_inputs(43))
    _reference_forward(reference[1], reference[0], reference[2], reference[3]).backward(
        grad_out
    )

    for fused_tensor, reference_tensor in zip(fused, reference):
        torch.testing.assert_close(
            fused_tensor.grad,
            reference_tensor.grad,
            rtol=TOL[0],
            atol=TOL[1],
        )


@requires_mhc_post
def test_provider_dispatch(monkeypatch):
    import importlib

    from torchturbo.ops import get_or_create_ops
    from torchturbo.runtime.npu import NpuRuntime

    # Full-suite runs leave a CPU dispatcher in the process; reset the
    # singleton to build the NPU one (monkeypatch restores it afterwards).
    # import_module reaches the real module: the package attribute "ops"
    # resolves to the dispatcher proxy, not the module.
    monkeypatch.setattr(
        importlib.import_module("torchturbo.ops.ops"), "_OPS", None
    )
    dispatcher = get_or_create_ops(device_runtime=NpuRuntime())
    residual, x, post, comb = _make_inputs(44)

    actual = dispatcher.mhc_post(residual, comb, x, post)
    expected = _reference_forward(x, residual, post, comb)

    torch.testing.assert_close(actual, expected, rtol=TOL[0], atol=TOL[1])


@requires_mhc_post
def test_provider_rejects_invalid_layout():
    from torchturbo.ops.providers.npu.mhc_post import NpuMhcPost

    residual, x, post, comb = _make_inputs(45)
    impl = NpuMhcPost()
    assert impl.available()

    tnd = residual.reshape(-1, N, D)
    with pytest.raises(ValueError, match="BSND"):
        impl(tnd, comb, x, post)
    with pytest.raises(ValueError, match="bf16/fp16"):
        impl(residual.float(), comb, x, post)
