"""BSND-layout unit tests for the DeepSeek V4.1 CANN mhc_pre_sinkhorn_premix ops.

Covers, against numpy goldens ported from the official ST (which itself
ports tests/assets/golden.py from the ops-transformer master branch):

- Forward: premix passed/omitted x out_flag True/False x bf16/fp16, BSND
  (arch22; TND is Ascend950 forward-only, and the model path is BSND).
- A3 norm_out contract: the kernel only writes even steps (row-normalized
  states); odd steps hold undefined memory, so only even steps compare,
  with a probe warning when odd steps regain golden-matching content.
- Backward: kernel vs golden with the merged-point three-state contract
  (premix omitted: d_pre = sum(grad_hin*x) + gradPre; premix passed:
  d_pre = gradPre, grad_pre_mix = sum(grad_hin*x)) and the
  zero_odd_norm probe that feeds the raw forward output back into the
  backward (the production path must not read the undefined odd steps).
- Golden backward self-check: host-side central-difference checks guarding
  the golden port itself (no kernel involved).
- Provider autograd bridge end to end: MhcPreSinkhornPremixFn under
  torch.autograd.grad vs golden_backward, covering the gradPre zero
  materialization (missing h_pre grad) and the empty grad_pre_mix when
  premix is omitted.
- aclgraph capture/replay, with and without premix.

Run:

    python3 -m pytest tests/ops/npu/test_dsv41_mhc_sinkhorn_ops.py -v

Environment (per the ops-transformer build doc):

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=\\
        ${ASCEND_HOME_PATH}/opp/vendors/dsv41exp_transformer/op_api/lib:\\
        ${LD_LIBRARY_PATH}
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")  # noqa: F401

if not torch.npu.is_available():
    pytest.skip("NPU required", allow_module_level=True)

from torchturbo.ops.providers.npu.mhc_pre_sinkhorn_premix import (  # noqa: E402
    MhcPreSinkhornPremixFn,
    _load_ops,
    has_mhc_pre_sinkhorn_premix,
)

DEVICE = "npu:0"
HC_MULT = 4
NUM_ITERS = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6
D_MODEL = 4096  # hidden_size: D must be a multiple of 128 (op contract)

requires_mhc = pytest.mark.skipif(
    not has_mhc_pre_sinkhorn_premix(),
    reason="mhc_pre_sinkhorn_premix(_backward) not registered "
    "(install the dsv41exp vendor package: cann_ops_transformer_dsv41exp)",
)

# Comparison tolerances (rtol, atol), measured against the goldens: random
# large-K inputs accumulate fp32 ordering differences up to the ~1e-3
# scale; strict ST tolerances need fixed small inputs instead. These
# bounds catch functional errors with margin to spare.
TOL_BF16 = (2e-2, 2e-2)
TOL_FP16 = (1e-2, 1e-3)
TOL_FP32 = (5e-3, 1e-3)


def _is_ascend_950() -> bool:
    """Conservative platform probe: only trust the ASCEND_MODEL env; the
    default (A3) matches the first SFT deployment."""
    model = os.environ.get("ASCEND_MODEL", "").lower()
    return "950" in model or "pr" in model or "dt" in model


IS_ASCEND_950 = _is_ascend_950()


# ---------------------------------------------------------------------------
# numpy goldens (ported from the official ST; port guarded by the
# self-check test below).
# ---------------------------------------------------------------------------


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -80.0, 80.0)))


def golden(
    x_np,
    phi,
    alpha,
    bias,
    premix=None,
    hc_mult=HC_MULT,
    num_iters=NUM_ITERS,
    hc_eps=HC_EPS,
    norm_eps=NORM_EPS,
    need_backward=True,
    dtype=torch.bfloat16,
):
    """numpy reference forward. With premix passed, hin collapses using
    premix as the weights; all other outputs are unchanged."""
    # The kernel reads x rounded to the input dtype; mirror that here.
    x_f32 = torch.from_numpy(x_np.astype(np.float32)).to(dtype).float().numpy()
    b, s, n, d = x_f32.shape
    flat_x = x_f32.reshape(b, s, n * d)

    inv_rms = 1.0 / np.sqrt(np.mean(flat_x * flat_x, axis=-1, keepdims=True) + norm_eps)
    hc_before_norm = np.matmul(flat_x, phi.T)
    normalized = hc_before_norm * inv_rms

    h_pre = sigmoid(normalized[..., :hc_mult] * alpha[0] + bias[:hc_mult]) + hc_eps
    h_post = 2.0 * sigmoid(
        normalized[..., hc_mult : 2 * hc_mult] * alpha[1] + bias[hc_mult : 2 * hc_mult]
    )

    residual_logits = (
        normalized[..., 2 * hc_mult :] * alpha[2] + bias[2 * hc_mult :]
    ).reshape(b, s, hc_mult, hc_mult)
    sum_out = np.empty((2 * num_iters, b, s, hc_mult), dtype=np.float32)
    norm_out = np.empty((2 * num_iters, b, s, hc_mult, hc_mult), dtype=np.float32)
    row_max = np.max(residual_logits, axis=-1, keepdims=True)
    row_exp = np.exp(residual_logits - row_max)
    row_sum = np.sum(row_exp, axis=-1)
    sum_out[0] = row_sum + hc_eps
    current = row_exp / row_sum[..., None] + hc_eps
    norm_out[0] = current
    column_sum = np.sum(current, axis=-2)
    sum_out[1] = column_sum + hc_eps
    current = current / (column_sum[..., None, :] + hc_eps)
    norm_out[1] = current
    for it in range(1, num_iters):
        row_sum = np.sum(current, axis=-1)
        sum_out[2 * it] = row_sum + hc_eps
        current = current / (row_sum[..., :, None] + hc_eps)
        norm_out[2 * it] = current
        column_sum = np.sum(current, axis=-2)
        sum_out[2 * it + 1] = column_sum + hc_eps
        current = current / (column_sum[..., None, :] + hc_eps)
        norm_out[2 * it + 1] = current
    h_res = current.reshape(b, s, hc_mult * hc_mult)

    # With premix passed, hin collapses with premix directly (no hc_eps on
    # the weights); otherwise with the internal h_pre.
    mix_for_y = premix.astype(np.float32) if premix is not None else h_pre
    hin = np.sum(x_f32 * mix_for_y[..., :, None], axis=-2)
    # hin accumulates fp32 inside the kernel and rounds to the input dtype.
    hin = torch.from_numpy(hin).to(dtype).float().numpy()

    if not need_backward:
        empty = np.empty((0,), dtype=np.float32)
        h_pre, hc_before_norm, inv_rms, sum_out, norm_out = (
            empty,
            empty,
            empty,
            empty,
            empty,
        )
    return (
        hin.astype(np.float32),
        h_post.astype(np.float32),
        h_res.astype(np.float32),
        h_pre.astype(np.float32),
        hc_before_norm.astype(np.float32),
        inv_rms.astype(np.float32),
        sum_out,
        norm_out,
    )


def golden_backward(
    grad_hin,
    grad_h_post,
    grad_h_res,
    x,
    phi,
    alpha,
    bias,
    h_pre,
    hc_before_norm,
    inv_rms,
    sum_out,
    norm_out,
    premix=None,
    grad_pre=None,
    hc_eps=HC_EPS,
    dtype=torch.bfloat16,
):
    """numpy reference backward (BSND, arch22).

    Ported from the official golden (ops-transformer master
    tests/assets/golden.py, mhc_pre_sinkhorn_premix_backward_golden):
    - Merged-point three states: premix omitted -> d_pre =
      sum_d(grad_hin*x) (+gradPre); premix passed -> d_pre = gradPre (the
      official autograd packaging materializes zeros), and grad_pre_mix =
      sum_d(grad_hin*x).
    - The Sinkhorn chain backward only consumes even steps of norm_out
      (row-normalized states); odd steps are never read - self-consistent
      with the A3 forward that only writes even steps.
    - The step-0 row-normalization denominator carries no hc_eps while the
      stored values do; both are compensated.
    - grad_x rounds to the input dtype; grad_phi/alpha/bias/premix stay
      fp32. Inputs are host numpy (x and grad_hin must be pre-rounded to
      the kernel-read dtype and cast to fp32).
    """
    assert x.ndim == 4, "golden_backward supports BSND (4-dim) only"
    b, s, n, c = x.shape
    h_dim = phi.shape[0]  # 2n + n*n
    grad_h_res = grad_h_res.reshape(b, s, n, n)

    # hin fold: the mix is premix when passed, the internal h_pre otherwise.
    mix_weight = premix if premix is not None else h_pre
    grad_x_from_hin = grad_hin[..., None, :] * mix_weight[..., :, None]  # (b,s,n,c)
    grad_mix = np.sum(grad_hin[..., None, :] * x, axis=-1)  # (b,s,n)
    if premix is None:
        d_pre = grad_mix if grad_pre is None else grad_mix + grad_pre
    else:
        d_pre = grad_pre if grad_pre is not None else np.zeros_like(grad_mix)
    grad_pre_mix = grad_mix if premix is not None else None

    normalized = hc_before_norm * inv_rms
    z_pre = normalized[..., :n] * alpha[0] + bias[:n]
    z_post = normalized[..., n : 2 * n] * alpha[1] + bias[n : 2 * n]
    z_res = (normalized[..., 2 * n :] * alpha[2] + bias[2 * n :]).reshape(b, s, n, n)

    sg_pre = sigmoid(z_pre)
    grad_z_pre = d_pre * sg_pre * (1.0 - sg_pre)
    sg_post = sigmoid(z_post)
    grad_z_post = 2.0 * grad_h_post * sg_post * (1.0 - sg_post)

    # Sinkhorn backward: walk grad_h_res back iteration by iteration to the
    # initial softmax (only even norm_out steps are read).
    g = grad_h_res.copy()
    for i in range(sum_out.shape[0] // 2 - 1, -1, -1):
        col_sum = sum_out[2 * i + 1][..., None, :]  # (b,s,1,n): column sum + eps
        row_sum = sum_out[2 * i][..., :, None]  # (b,s,n,1): row sum + eps
        xrn = norm_out[2 * i]  # (b,s,n,n) row-normalized state
        # Column-normalization backward: out = xrn / col_sum.
        gxrn = g / col_sum - np.sum(g * xrn / col_sum**2, axis=-2, keepdims=True)
        if i == 0:
            # Step-0 row denominators carry no eps while the stored values
            # include it; subtract both.
            row_sum = row_sum - hc_eps
            xrn = xrn - hc_eps
        # Row-normalization backward.
        g = gxrn / row_sum - np.sum(gxrn * xrn / row_sum, axis=-1, keepdims=True)

    # exp with row-max subtraction backward (argmax takes the max grad).
    row_max_idx = np.argmax(z_res, axis=-1, keepdims=True)
    row_max = np.max(z_res, axis=-1, keepdims=True)
    y = np.exp(z_res - row_max)
    is_max = np.arange(n).reshape((1,) * (z_res.ndim - 1) + (n,)) == row_max_idx
    total = np.sum(y * g, axis=-1, keepdims=True)
    grad_z_res = (y * g - is_max * total).reshape(b, s, n * n)

    grad_bias = np.concatenate(
        [
            grad_z_pre.sum(axis=(0, 1)),
            grad_z_post.sum(axis=(0, 1)),
            grad_z_res.sum(axis=(0, 1)),
        ]
    )
    grad_alpha = np.array(
        [
            np.sum(normalized[..., :n] * grad_z_pre),
            np.sum(normalized[..., n : 2 * n] * grad_z_post),
            np.sum(normalized[..., 2 * n :] * grad_z_res),
        ],
        dtype=np.float32,
    )

    # normalized = hc_before_norm * inv_rms backward.
    grad_norm = np.concatenate(
        [grad_z_pre * alpha[0], grad_z_post * alpha[1], grad_z_res * alpha[2]],
        axis=-1,
    )
    grad_hc = grad_norm * inv_rms
    grad_inv_rms = np.sum(hc_before_norm * grad_norm, axis=-1, keepdims=True)

    # x has three grad paths: the hin fold, the RMS normalization, and the
    # linear projection; phi grads come from the transposed projection.
    nc = n * c
    x_flat = x.reshape(b, s, nc)
    grad_x_rms = -(inv_rms**3) * x_flat * grad_inv_rms / nc
    grad_x_mm = grad_hc @ phi
    grad_phi = grad_hc.reshape(b * s, h_dim).T @ x_flat.reshape(b * s, nc)

    grad_x = grad_x_from_hin + grad_x_rms.reshape(b, s, n, c) + grad_x_mm.reshape(
        b, s, n, c
    )
    grad_x = torch.from_numpy(np.ascontiguousarray(grad_x)).to(dtype).float().numpy()
    return grad_x, grad_phi, grad_alpha, grad_bias, grad_pre_mix


# ---------------------------------------------------------------------------
# Fixtures and op callers.
# ---------------------------------------------------------------------------


def make_inputs(bs, s_len=2, d=D_MODEL, seed=0, with_premix=False):
    """BSND inputs; premix is drawn from a different distribution than the
    internal h_pre on purpose, to prove hin really uses it."""
    rng = np.random.default_rng(seed)
    n = HC_MULT
    hc_mix = n * n + 2 * n
    x = (rng.standard_normal((bs, s_len, n, d)) * 0.5).astype(np.float32)
    phi = (rng.standard_normal((hc_mix, n * d)) / np.sqrt(n * d)).astype(np.float32)
    alpha = (rng.standard_normal(3) * 0.5 + 1.0).astype(np.float32)
    bias = (rng.standard_normal(hc_mix) * 0.5).astype(np.float32)
    premix = rng.random((bs, s_len, n)).astype(np.float32) * 2.0 if with_premix else None
    return x, phi, alpha, bias, premix


def make_grads(bs, s_len, n, d, seed=0, with_grad_pre=False):
    """Upstream grads for the backward tests. grad_h_res uses the flat
    (bs, s, n*n) view (official autograd packaging; the kernel contract
    accepts both flat and (n, n))."""
    rng = np.random.default_rng(seed)
    grad_hin = rng.standard_normal((bs, s_len, d)).astype(np.float32)
    grad_h_post = rng.standard_normal((bs, s_len, n)).astype(np.float32)
    grad_h_res = rng.standard_normal((bs, s_len, n * n)).astype(np.float32)
    grad_pre = (
        rng.standard_normal((bs, s_len, n)).astype(np.float32) if with_grad_pre else None
    )
    return grad_hin, grad_h_post, grad_h_res, grad_pre


def to_npu(x, phi, alpha, bias, premix, dtype=torch.bfloat16):
    # x supports bf16/fp16 only; phi/alpha/bias/premix stay fp32.
    x_t = torch.from_numpy(x).to(dtype).npu()
    phi_t = torch.from_numpy(phi).npu()
    alpha_t = torch.from_numpy(alpha).npu()
    bias_t = torch.from_numpy(bias).npu()
    premix_t = torch.from_numpy(premix).npu() if premix is not None else None
    return x_t, phi_t, alpha_t, bias_t, premix_t


def run_op(x_t, phi_t, alpha_t, bias_t, premix_t, out_flag):
    return _load_ops().mhc_pre_sinkhorn_premix(
        x_t,
        phi_t,
        alpha_t,
        bias_t,
        premix_t,
        HC_MULT,
        NUM_ITERS,
        HC_EPS,
        NORM_EPS,
        out_flag,
    )


def run_backward(
    grad_hin_t,
    grad_h_post_t,
    grad_h_res_t,
    x_t,
    phi_t,
    alpha_t,
    bias_t,
    caches,
    premix_t,
    grad_pre_t,
):
    """Backward op, master torch_extension signature: (gradHin, gradHPost,
    gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut,
    normOut, preMix?, gradPre?, hcEps) -> (grad_x, grad_phi, grad_alpha,
    grad_bias, grad_pre_mix)."""
    h_pre_t, hcbn_t, invrms_t, sum_t, norm_t = caches
    return _load_ops().mhc_pre_sinkhorn_premix_backward(
        grad_hin_t,
        grad_h_post_t,
        grad_h_res_t,
        x_t,
        phi_t,
        alpha_t,
        bias_t,
        h_pre_t,
        hcbn_t,
        invrms_t,
        sum_t,
        norm_t,
        premix_t,
        grad_pre_t,
        HC_EPS,
    )


# ---------------------------------------------------------------------------
# Tolerance handling and comparison helpers.
# ---------------------------------------------------------------------------


def get_tol(name, dtype=torch.bfloat16):
    """Forward tolerances: hin rounds to the input dtype; the rest are
    fp32 reductions."""
    if name == "hin":
        return TOL_BF16 if dtype == torch.bfloat16 else TOL_FP16
    return TOL_FP32


def get_bwd_tol(name, dtype=torch.bfloat16):
    """Backward tolerances: grad_x rounds to the x dtype; the rest are fp32
    reductions (arch22 backward is non-deterministic by default; the
    measured tolerances match the forward fp32 outputs)."""
    if name == "grad_x":
        return TOL_BF16 if dtype == torch.bfloat16 else TOL_FP16
    return TOL_FP32


def diagnose_norm_out(name, act, exp):
    """Failure-mode diagnosis for norm_out mismatches (prints only).

    Separates three root causes: kernel write misses (some steps or (b,s)
    blocks fully zero), semantic/scale differences (ratio ~0.5 or ~2), and
    genuine numeric errors (irregular ratios with large diffs).
    """
    k_num = exp.shape[0]
    flat_mid = int(np.prod(exp.shape[1:-2])) if exp.ndim > 2 else 1
    a2 = act.reshape(k_num, flat_mid, -1)
    e2 = exp.reshape(k_num, flat_mid, -1)
    print(
        f"[DIAG] {name}: shape={act.shape}, decomposed into {k_num} steps "
        f"x {flat_mid} (b,s) blocks"
    )
    zero_steps = []
    for k in range(k_num):
        a, e = a2[k], e2[k]
        zero_frac = float((a == 0).mean())
        max_diff = float(np.abs(a - e).max())
        big = e > 0.05  # ratios only on sufficiently large elements
        ratio = float(np.median(a[big] / e[big])) if big.any() else float("nan")
        if zero_frac == 1.0:
            zero_steps.append(k)
        print(
            f"[DIAG]   step={k:2d} zero_frac={zero_frac:.4f} "
            f"max_abs_diff={max_diff:.6f} ratio={ratio:.4f}"
        )
    block_zero = (a2 == 0).all(axis=(0, 2))
    zero_blocks = np.nonzero(block_zero)[0]
    if zero_blocks.size:
        print(
            f"[DIAG] {zero_blocks.size}/{flat_mid} (b,s) blocks zero across "
            f"all steps, indices [{int(zero_blocks.min())}, "
            f"{int(zero_blocks.max())}], first 10: {zero_blocks[:10].tolist()}"
        )
    else:
        print("[DIAG] no (b,s) block is zero across all steps")
    if zero_steps:
        print(f"[DIAG] fully-zero steps: {zero_steps}")
    big_all = e2 > 0.05
    global_ratio = (
        float(np.median(a2[big_all] / e2[big_all])) if big_all.any() else float("nan")
    )
    if (a2 == 0).all():
        verdict = "norm_out all zero: the op wrote nothing on this platform"
    elif zero_steps:
        verdict = f"some steps fully zero ({zero_steps}): step-level write misses"
    elif zero_blocks.size:
        verdict = f"{zero_blocks.size} (b,s) blocks zero: block-level write misses"
    elif 1.9 <= global_ratio <= 2.1 or 0.4 <= global_ratio <= 0.6:
        verdict = f"systematic ratio ~{global_ratio:.3f}: recording-scale drift"
    else:
        verdict = "no systematic zero/scale pattern: numeric issue in this output"
    print(f"[DIAG] verdict: {verdict}")


def assert_close(name, actual, expected, rtol, atol):
    actual_np = actual.cpu().float().numpy()
    assert actual_np.shape == expected.shape, (
        f"{name} shape {actual_np.shape} != {expected.shape}"
    )
    if "norm_out" in name and not IS_ASCEND_950:
        # A3 measured contract: norm_out carries only even steps
        # (row-normalized states, matching golden); odd steps hold
        # undefined residual memory (both all-zero and dirty reused blocks
        # have been observed). The kernel's internal column normalization
        # still runs (odd sum_out steps and h_res are correct). Compare
        # even steps only; warn when odd steps match golden again so the
        # branch can be dropped. Ascend950 compares everything.
        odd_act = actual_np[1::2]
        odd_exp = expected[1::2]
        if (odd_act != 0).any():
            odd_diff = np.abs(odd_act - odd_exp)
            odd_thresh = atol + rtol * np.abs(odd_exp)
            if (odd_diff <= odd_thresh).all():
                print(
                    f"[WARN] {name}: odd steps match the golden column-"
                    "normalized states - the op may now write them; drop "
                    "this branch and compare all steps"
                )
        actual_np = actual_np[0::2]
        expected = expected[0::2]
        name = f"{name}(even-steps)"
    diff = np.abs(actual_np - expected)
    thresh = atol + rtol * np.abs(expected)
    if (diff > thresh).any():
        if "norm_out" in name:
            diagnose_norm_out(name, actual_np, expected)
        raise AssertionError(
            f"{name} mismatch: max_abs={diff.max()}, "
            f"max_rel={(diff / np.maximum(np.abs(expected), 1e-6)).max()}"
        )


# ---------------------------------------------------------------------------
# Forward cases.
# ---------------------------------------------------------------------------


def check_case(bs, with_premix, out_flag, seed=0, d=D_MODEL, dtype=torch.bfloat16):
    x, phi, alpha, bias, premix = make_inputs(bs, d=d, seed=seed, with_premix=with_premix)
    x_t, phi_t, alpha_t, bias_t, premix_t = to_npu(x, phi, alpha, bias, premix, dtype)
    outs = run_op(x_t, phi_t, alpha_t, bias_t, premix_t, out_flag)
    # A3 measured contract: with premix passed and out_flag=False, h_pre
    # still carries the internal value (matches golden); compare it
    # strictly here, and drop the branch if the op ever returns it empty
    # again. The premix=None path matches Ascend950: all backward outputs
    # stay empty when out_flag=False.
    a3_premix_hpre = (not out_flag) and (not IS_ASCEND_950) and with_premix
    expected = golden(
        x,
        phi,
        alpha,
        bias,
        premix,
        need_backward=out_flag or a3_premix_hpre,
        dtype=dtype,
    )
    names = [
        "hin",
        "h_post",
        "h_res",
        "h_pre",
        "hc_before_norm",
        "inv_rms",
        "sum_out",
        "norm_out",
    ]
    check_names = names if out_flag else names[:3]
    for name, act, exp in zip(names, outs, expected):
        if name not in check_names:
            if name == "h_pre" and a3_premix_hpre:
                assert_close("h_pre(A3-premix)", act, exp, *TOL_FP32)
                continue
            assert act.numel() == 0, f"{name} should be empty when out_flag=False"
            continue
        rtol, atol = get_tol(name, dtype)
        assert_close(name, act, exp, rtol, atol)


@requires_mhc
@pytest.mark.parametrize("bs", [16, 256, 8192])
@pytest.mark.parametrize("with_premix", [False, True])
@pytest.mark.parametrize("out_flag", [False, True])
def test_single_op(bs, with_premix, out_flag):
    check_case(
        bs,
        with_premix,
        out_flag,
        # Deterministic per-case seed (stable across runs; Python's hash()
        # is salted per process and would make failures unreproducible).
        seed=bs % 1000 * 2 + int(with_premix) + int(out_flag) * 3,
    )


@requires_mhc
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_single_op_dtypes(dtype):
    # Both input dtypes share the same contract; fp16 only needs spot
    # coverage (tolerances are measured separately).
    check_case(16, True, True, seed=77, dtype=dtype)


@requires_mhc
def test_premix_changes_hin():
    # Same inputs, premix passed vs omitted: hin must differ (proves premix
    # is effective) while the internal h_pre output stays unchanged.
    x, phi, alpha, bias, premix = make_inputs(32, seed=7, with_premix=True)
    x_t, phi_t, alpha_t, bias_t, premix_t = to_npu(x, phi, alpha, bias, premix)
    hin_no, _, _, h_pre_no, *_ = run_op(x_t, phi_t, alpha_t, bias_t, None, True)
    hin_pm, _, _, h_pre_pm, *_ = run_op(x_t, phi_t, alpha_t, bias_t, premix_t, True)
    diff = (hin_no.float() - hin_pm.float()).abs().max().item()
    assert diff > 1e-3, f"premix not effective, hin diff={diff}"
    assert_close("h_pre", h_pre_pm, h_pre_no.cpu().float().numpy(), 0.0, 0.0)


@requires_mhc
def test_aclgraph():
    bs = 32
    x, phi, alpha, bias, premix = make_inputs(bs, seed=11, with_premix=True)
    x_t, phi_t, alpha_t, bias_t, premix_t = to_npu(x, phi, alpha, bias, premix)

    # Eager reference.
    ref = [
        o.cpu().float().numpy() for o in run_op(x_t, phi_t, alpha_t, bias_t, premix_t, True)
    ]

    # aclgraph capture/replay.
    torch.npu.synchronize()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(3):  # warmup
            run_op(x_t, phi_t, alpha_t, bias_t, premix_t, True)
    torch.npu.current_stream().wait_stream(stream)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_outs = run_op(x_t, phi_t, alpha_t, bias_t, premix_t, True)
    graph.replay()
    torch.npu.synchronize()

    names = [
        "hin",
        "h_post",
        "h_res",
        "h_pre",
        "hc_before_norm",
        "inv_rms",
        "sum_out",
        "norm_out",
    ]
    for name, act, exp in zip(names, graph_outs, ref):
        rtol, atol = get_tol(name)
        assert_close(f"aclgraph:{name}", act, exp, rtol, atol)

    # Replay with fresh inputs: outputs must track the new values.
    x2, phi2, alpha2, bias2, premix2 = make_inputs(bs, seed=22, with_premix=True)
    x_t.copy_(torch.from_numpy(x2).to(torch.bfloat16))
    phi_t.copy_(torch.from_numpy(phi2))
    alpha_t.copy_(torch.from_numpy(alpha2))
    bias_t.copy_(torch.from_numpy(bias2))
    premix_t.copy_(torch.from_numpy(premix2))
    graph.replay()
    torch.npu.synchronize()
    expected2 = golden(x2, phi2, alpha2, bias2, premix2, need_backward=True)
    for name, act, exp in zip(names, graph_outs, expected2):
        rtol, atol = get_tol(name)
        assert_close(f"aclgraph-replay:{name}", act, exp, rtol, atol)


@requires_mhc
def test_aclgraph_no_premix():
    bs = 16
    x, phi, alpha, bias, _ = make_inputs(bs, seed=33, with_premix=False)
    x_t, phi_t, alpha_t, bias_t, _ = to_npu(x, phi, alpha, bias, None)
    ref = [
        o.cpu().float().numpy() for o in run_op(x_t, phi_t, alpha_t, bias_t, None, False)
    ]

    torch.npu.synchronize()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(3):
            run_op(x_t, phi_t, alpha_t, bias_t, None, False)
    torch.npu.current_stream().wait_stream(stream)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_outs = run_op(x_t, phi_t, alpha_t, bias_t, None, False)
    graph.replay()
    torch.npu.synchronize()
    for name, act, exp in zip(["hin", "h_post", "h_res"], graph_outs[:3], ref[:3]):
        rtol, atol = get_tol(name)
        assert_close(f"aclgraph-nopremix:{name}", act, exp, rtol, atol)


# ---------------------------------------------------------------------------
# Backward cases.
# ---------------------------------------------------------------------------


def check_backward_case(
    bs, with_premix, with_grad_pre, seed=0, d=D_MODEL, dtype=torch.bfloat16,
    zero_odd_norm=True,
):
    # arch22 constraints: BSND, N=4, D % 128 == 0 (D_MODEL and the bridge
    # fixture below both satisfy them).
    x, phi, alpha, bias, premix = make_inputs(bs, d=d, seed=seed, with_premix=with_premix)
    x_t, phi_t, alpha_t, bias_t, premix_t = to_npu(x, phi, alpha, bias, premix, dtype)
    # Real forward (out_flag=True) supplies the backward caches: the
    # end-to-end composition path.
    outs = run_op(x_t, phi_t, alpha_t, bias_t, premix_t, True)
    hin_t, h_post_t, h_res_t, h_pre_t, hcbn_t, invrms_t, sum_t, norm_t = outs
    # Upstream grads: grad_hin shares the x dtype; grad_h_post/grad_h_res
    # are fp32 (flat view).
    grad_hin, grad_h_post, grad_h_res, grad_pre = make_grads(
        bs, 2, HC_MULT, d, seed + 1, with_grad_pre
    )
    grad_hin_t = torch.from_numpy(grad_hin).to(dtype).npu()
    grad_h_post_t = torch.from_numpy(grad_h_post).npu()
    grad_h_res_t = torch.from_numpy(grad_h_res).npu()
    # aclnn binding contract: premix and gradPre pass together or stay
    # omitted together; premix passed without an external h_pre grad is
    # materialized as zeros (official autograd packaging).
    if with_premix and grad_pre is None:
        grad_pre = np.zeros_like(premix)
    grad_pre_t = torch.from_numpy(grad_pre).npu() if grad_pre is not None else None
    # A3 forward leaves norm_out odd steps unwritten (undefined residual
    # memory). Zero them for deterministic comparison (the backward math
    # only reads even steps); zero_odd_norm=False feeds the raw forward
    # output back in, probing that the production path is insensitive to
    # the undefined odd steps.
    norm_t_used = norm_t.clone()
    if zero_odd_norm:
        norm_t_used[1::2].zero_()
    grads_out = run_backward(
        grad_hin_t,
        grad_h_post_t,
        grad_h_res_t,
        x_t,
        phi_t,
        alpha_t,
        bias_t,
        (h_pre_t, hcbn_t, invrms_t, sum_t, norm_t_used),
        premix_t,
        grad_pre_t,
    )
    # Host golden with exactly the values the kernel consumed (caches from
    # the real forward).
    x_np = torch.from_numpy(x).to(dtype).float().numpy()
    grad_hin_np = torch.from_numpy(grad_hin).to(dtype).float().numpy()
    ref = golden_backward(
        grad_hin_np,
        grad_h_post,
        grad_h_res,
        x_np,
        phi,
        alpha,
        bias,
        h_pre_t.cpu().float().numpy(),
        hcbn_t.cpu().float().numpy(),
        invrms_t.cpu().float().numpy(),
        sum_t.cpu().float().numpy(),
        norm_t_used.cpu().float().numpy(),
        premix,
        grad_pre,
        HC_EPS,
        dtype,
    )
    names = ["grad_x", "grad_phi", "grad_alpha", "grad_bias", "grad_pre_mix"]
    for name, act, exp in zip(names, grads_out, ref):
        if exp is None:
            # premix omitted: the official Meta marks the 5th output empty.
            assert act.numel() == 0, f"{name} should be empty when premix is None"
            continue
        rtol, atol = get_bwd_tol(name, dtype)
        assert_close(f"backward:{name}", act, exp, rtol, atol)


@requires_mhc
@pytest.mark.parametrize(
    "bs,with_premix,with_grad_pre",
    [
        # Premix omitted: gradPre must stay omitted - the aclnn binding
        # layer rejects premix=None with a gradPre tensor (EZ1001).
        (16, False, False),
        (256, False, False),
        (8192, False, False),
        # Premix passed: gradPre omitted (materialized as zeros by the
        # autograd packaging) or an external h_pre grad.
        (16, True, False),
        (16, True, True),
        (256, True, False),
        (256, True, True),
        (8192, True, False),
        (8192, True, True),
    ],
)
def test_backward(bs, with_premix, with_grad_pre):
    # Covers the merged-point states the op accepts: premix omitted (d_pre
    # from the internal fold chain), premix + zero-materialized gradPre,
    # premix + external gradPre; small bs (K split) and bs=8192 (M split /
    # non-deterministic accumulation path).
    check_backward_case(
        bs,
        with_premix,
        with_grad_pre,
        # Deterministic per-case seed (stable across runs).
        seed=300 + bs % 90 + int(with_premix) + 2 * int(with_grad_pre),
    )


@requires_mhc
def test_backward_tolerates_undefined_odd_norm_steps():
    # zero_odd_norm=False feeds the raw forward output (with undefined
    # odd steps on A3) into the backward: the production composition path
    # must be insensitive to them.
    check_backward_case(16, True, True, seed=321, zero_odd_norm=False)


def test_backward_golden_selfcheck():
    """Host-side numeric self-check of golden_backward against golden via
    central differences (no kernel involved). Runs in fp32 to avoid bf16
    quantization, directly validating the backward math (merged-point
    three states, the Sinkhorn chain, exp/max, RMS/projection paths);
    guards the port and any future golden edit."""
    f32 = torch.float32
    h = 1e-2
    for with_premix in (False, True):
        bs, d = 2, 128
        x, phi, alpha, bias, premix = make_inputs(
            bs, d=d, seed=90 + with_premix, with_premix=with_premix
        )
        grad_hin, grad_h_post, grad_h_res, grad_pre = make_grads(
            bs, 2, HC_MULT, d, seed=91 + with_premix, with_grad_pre=with_premix
        )
        outs = golden(x, phi, alpha, bias, premix, need_backward=True, dtype=f32)
        ref = golden_backward(
            grad_hin,
            grad_h_post,
            grad_h_res,
            x,
            phi,
            alpha,
            bias,
            outs[3],
            outs[4],
            outs[5],
            outs[6],
            outs[7],
            premix,
            grad_pre,
            HC_EPS,
            dtype=f32,
        )

        def loss(name, new_p):
            xx, pphi, palpha, pbias, ppremix = x, phi, alpha, bias, premix
            if name == "x":
                xx = new_p
            elif name == "phi":
                pphi = new_p
            elif name == "alpha":
                palpha = new_p
            elif name == "bias":
                pbias = new_p
            elif name == "premix":
                ppremix = new_p
            o = golden(xx, pphi, palpha, pbias, ppremix, need_backward=True, dtype=f32)
            total = (
                np.sum(grad_hin * o[0])
                + np.sum(grad_h_post * o[1])
                + np.sum(grad_h_res * o[2])
            )
            if grad_pre is not None:
                total = total + np.sum(grad_pre * o[3])
            return total

        rng = np.random.default_rng(95)
        params = [
            ("x", x, ref[0]),
            ("phi", phi, ref[1]),
            ("alpha", alpha, ref[2]),
            ("bias", bias, ref[3]),
        ]
        if premix is not None:
            params.append(("premix", premix, ref[4]))
        for name, p, g in params:
            gmax = float(np.abs(g).max())
            for _ in range(6):
                coord = tuple(int(v) for v in rng.integers(0, p.shape))
                pp, pm = p.copy(), p.copy()
                pp[coord] += h
                pm[coord] -= h
                num = (loss(name, pp) - loss(name, pm)) / (2 * h)
                ana = float(g[coord])
                tol = 5e-3 * abs(ana) + 1e-3 * h * gmax + 1e-6
                assert abs(num - ana) <= tol, (
                    f"selfcheck[{name}{coord}] num={num}, ana={ana}, tol={tol}"
                )


# ---------------------------------------------------------------------------
# Provider autograd bridge end to end.
# ---------------------------------------------------------------------------


@requires_mhc
@pytest.mark.parametrize(
    "with_premix,use_hpre_loss",
    [(False, False), (True, True), (True, False)],
)
def test_provider_autograd_bridge(with_premix, use_hpre_loss):
    """End-to-end over the provider's autograd bridge: weighted outputs ->
    torch.autograd.grad vs golden_backward.

    premix omitted: the loss must not include h_pre (the aclnn binding
    rejects premix=None with a gradPre tensor), so grad_h_pre arrives as
    None. premix passed without an h_pre loss checks the zero gradPre
    materialization; with it, gradPre is the external h_pre grad.
    """
    bs, d = 16, 128
    x, phi, alpha, bias, premix = make_inputs(bs, d=d, seed=51, with_premix=with_premix)
    x_t, phi_t, alpha_t, bias_t, premix_t = to_npu(x, phi, alpha, bias, premix)

    # Backward caches from a real forward (out_flag=True); odd steps zeroed
    # the same way the backward cases do.
    outs = run_op(x_t, phi_t, alpha_t, bias_t, premix_t, True)
    h_pre_t, hcbn_t, invrms_t, sum_t, norm_t = outs[3:]
    norm_t_used = norm_t.clone()
    norm_t_used[1::2].zero_()

    rng = np.random.default_rng(52)
    gin = rng.standard_normal((bs, 2, d)).astype(np.float32)
    gpost = rng.standard_normal((bs, 2, HC_MULT)).astype(np.float32)
    gres = rng.standard_normal((bs, 2, HC_MULT * HC_MULT)).astype(np.float32)
    gpre = rng.standard_normal((bs, 2, HC_MULT)).astype(np.float32)

    leaves = [
        torch.from_numpy(x).to(torch.bfloat16).npu().detach().requires_grad_(True),
        torch.from_numpy(phi).npu().detach().requires_grad_(True),
        torch.from_numpy(alpha).npu().detach().requires_grad_(True),
        torch.from_numpy(bias).npu().detach().requires_grad_(True),
    ]
    if with_premix:
        leaves.append(torch.from_numpy(premix).npu().detach().requires_grad_(True))

    hin, h_post, h_res, h_pre = MhcPreSinkhornPremixFn.apply(
        leaves[0],
        leaves[1],
        leaves[2],
        leaves[3],
        leaves[4] if with_premix else None,
        HC_MULT,
        NUM_ITERS,
        HC_EPS,
        NORM_EPS,
    )
    loss = (
        (hin.float() * torch.from_numpy(gin).npu()).sum()
        + (h_post * torch.from_numpy(gpost).npu()).sum()
        + (h_res * torch.from_numpy(gres).npu()).sum()
    )
    if use_hpre_loss:
        loss = loss + (h_pre * torch.from_numpy(gpre).npu()).sum()
    grads = torch.autograd.grad(loss, leaves)
    grad_x, grad_phi, grad_alpha, grad_bias = grads[0], grads[1], grads[2], grads[3]

    # Host golden with exactly the values the bridge consumed.
    x_np = torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    gin_np = torch.from_numpy(gin).to(torch.bfloat16).float().numpy()
    grad_pre_np = None
    if with_premix:
        grad_pre_np = gpre if use_hpre_loss else np.zeros_like(premix)
    ref = golden_backward(
        gin_np,
        gpost,
        gres,
        x_np,
        phi,
        alpha,
        bias,
        h_pre_t.cpu().float().numpy(),
        hcbn_t.cpu().float().numpy(),
        invrms_t.cpu().float().numpy(),
        sum_t.cpu().float().numpy(),
        norm_t_used.cpu().float().numpy(),
        premix,
        grad_pre_np,
        HC_EPS,
        torch.bfloat16,
    )
    assert_close("bridge:grad_x", grad_x, ref[0], *get_bwd_tol("grad_x"))
    assert_close("bridge:grad_phi", grad_phi, ref[1], *TOL_FP32)
    assert_close("bridge:grad_alpha", grad_alpha, ref[2], *TOL_FP32)
    assert_close("bridge:grad_bias", grad_bias, ref[3], *TOL_FP32)
    if with_premix:
        assert_close("bridge:grad_premix", grads[4], ref[4], *TOL_FP32)
