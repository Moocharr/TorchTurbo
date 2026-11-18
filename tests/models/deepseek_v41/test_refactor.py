"""CPU regression tests for forward-local metadata and model-local adapters."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from torchturbo.models.deepseek_v4.npu import (
    _get_compressed_kv_tnd_data,
    _get_li_query_tnd_data,
    _get_sfa_query_kv_tnd_data,
)
from torchturbo.models.deepseek_v41 import npu
from torchturbo.models.deepseek_v41.model import TurboDeepseekV41ForCausalLM
from torchturbo.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41DecoderLayer
from torchturbo.models.deepseek_v41.packed_metadata import PackedAttentionMetadata
from torchturbo.models.deepseek_v41.parallel_plan import DeepseekV41MoELayer
from torchturbo.models.modules.moe.moe_layer import MoELayer


@pytest.mark.parametrize("ratio", [1, 2, 4])
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("padding_only", [False, True])
def test_packed_metadata_matches_existing_adapters(ratio, rank, padding_only):
    mask = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0],
                         [0, 3, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0]])
    if padding_only:
        mask.zero_()
    local_length = 3
    packed = PackedAttentionMetadata(mask, local_length, rank, ratio)
    q = torch.randn(2, local_length, 2, 8)
    kv = torch.randn(2, 12, 8)
    qi, ki, cuq, cuk, local_mask = _get_sfa_query_kv_tnd_data(q, kv, mask, rank)
    torch.testing.assert_close(packed.query_indices, qi)
    torch.testing.assert_close(packed.kv_indices, ki)
    torch.testing.assert_close(packed.cu_query, cuq)
    torch.testing.assert_close(packed.local_mask, local_mask)
    assert packed.query_lengths == (cuq[1:] - cuq[:-1]).tolist()
    assert packed.kv_lengths == (cuk[1:] - cuk[:-1]).tolist()
    li_indices, li_cu = _get_li_query_tnd_data(local_mask)
    torch.testing.assert_close(packed.query_indices, li_indices)
    torch.testing.assert_close(packed.cu_query, li_cu)
    keys = torch.randn(2, 12 // ratio, 8, requires_grad=True)
    actual, cu_actual, residual_actual = packed.compressed_keys(keys)
    expected, cu_expected, block_expected, residual_expected = _get_compressed_kv_tnd_data(
        keys, local_mask, mask, rank * local_length, ratio
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(packed.block_indices, block_expected)
    torch.testing.assert_close(cu_actual, cu_expected)
    torch.testing.assert_close(residual_actual, residual_expected)
    if expected.requires_grad:
        weights = torch.randn_like(expected)
        ga = torch.autograd.grad(actual, keys, weights)[0]
        ge = torch.autograd.grad(expected, keys, weights)[0]
        torch.testing.assert_close(ga, ge)


@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_global_view_packed_metadata_matches_fused_indexer_packing(ratio):
    """The fused-indexer path packs the full mask as one shard; that global
    view must stay identical to the v4 TND helpers it replaced."""
    mask = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0],
                         [0, 3, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0]])
    packed = PackedAttentionMetadata(mask, mask.size(1), 0, ratio)
    li_indices, li_cu = _get_li_query_tnd_data(mask)
    torch.testing.assert_close(packed.query_indices, li_indices)
    torch.testing.assert_close(packed.cu_query, li_cu)
    keys = torch.randn(2, 12 // ratio, 8, requires_grad=True)
    actual, cu_actual, residual_actual = packed.compressed_keys(keys)
    expected, cu_expected, tnd_to_block, residual_expected = _get_compressed_kv_tnd_data(
        keys, mask, mask, 0, ratio
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(cu_actual, cu_expected)
    torch.testing.assert_close(packed.block_indices, tnd_to_block)
    torch.testing.assert_close(residual_actual, residual_expected)


def test_metadata_parses_global_rows_once(monkeypatch):
    from torchturbo.models.deepseek_v41 import packed_metadata

    original = packed_metadata._segment_spans
    calls = []

    def counted(row):
        calls.append(True)
        return original(row)

    monkeypatch.setattr(packed_metadata, "_segment_spans", counted)
    packed = PackedAttentionMetadata(torch.tensor([[1, 1, 2, 2]]), 2, 1, 2)
    _ = packed.query_indices, packed.kv_indices, packed.cu_query
    packed.compressed_keys(torch.randn(1, 2, 8))
    assert len(calls) == 1
    with pytest.raises(ValueError, match="unique"):
        PackedAttentionMetadata(torch.tensor([[1, 2, 1]]), 3, 0, 1).block_infos


def test_v41_initialization_does_not_patch_shared_moe():
    original = MoELayer.forward
    for _ in range(2):
        TurboDeepseekV41ForCausalLM._pre_init_patches(SimpleNamespace())
        assert MoELayer.forward is original


def test_v41_moe_adapter_preserves_keys_and_gradients(monkeypatch):
    def shared_forward(self, hidden_states, input_ids=None):
        return hidden_states * self.weight, input_ids

    monkeypatch.setattr(MoELayer, "forward", shared_forward)
    layer = DeepseekV41MoELayer.__new__(DeepseekV41MoELayer)
    torch.nn.Module.__init__(layer)
    layer.weight = torch.nn.Parameter(torch.tensor(2.0))
    x = torch.tensor(3.0, requires_grad=True)
    output = layer(x)
    assert list(layer.state_dict()) == ["weight"]
    assert MoELayer.forward is shared_forward
    assert output.item() == 6
    gx, gw = torch.autograd.grad(output, (x, layer.weight))
    assert gx.item() == 2 and gw.item() == 3


@pytest.mark.parametrize("checkpointed", [False, True])
def test_mhc_helper_routes_no_grad_to_fused_op(monkeypatch, checkpointed):
    layer = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
    torch.nn.Module.__init__(layer)

    class Mixes:
        def mixes(self, x, fn, scale, base):
            pre = x.sum(-1) * fn + base
            return pre, pre * scale, pre.unsqueeze(-1) * pre.unsqueeze(-2)

    layer.hc = Mixes()
    layer.hc_mult = 2
    layer.hc.hc_split_sinkhorn = SimpleNamespace(hc_sinkhorn_iters=20, hc_eps=1e-6)
    layer.hc.norm_eps = 1e-5
    calls = []

    def fused(x, fn, scale, base, pre, hc_mult, iterations, eps, norm_eps):
        assert (hc_mult, iterations, eps, norm_eps) == (2, 20, 1e-6, 1e-5)
        calls.append(True)
        hin, post, comb, new_pre = DeepseekV41DecoderLayer._hc_pre(
            layer, x, fn, scale, base, pre
        )
        return hin, post, comb.flatten(-2), new_pre

    monkeypatch.setattr(npu, "ops", SimpleNamespace(mhc_pre_sinkhorn_premix=fused))
    layer._hc_pre = MethodType(npu._npu_hc_pre, layer)
    assert layer.forward.__func__ is DeepseekV41DecoderLayer.forward
    inputs = [torch.randn(1, 3, 2, 4, requires_grad=True),
              torch.randn((), requires_grad=True), torch.randn((), requires_grad=True),
              torch.randn((), requires_grad=True), torch.randn(1, 3, 2, requires_grad=True)]
    expected = DeepseekV41DecoderLayer._hc_pre(layer, *inputs)
    actual = (checkpoint(layer._hc_pre, *inputs, use_reentrant=False)
              if checkpointed else layer._hc_pre(*inputs))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)
    ga = torch.autograd.grad(sum(x.sum() for x in actual), inputs)
    ge = torch.autograd.grad(sum(x.sum() for x in expected), inputs)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)
    assert len(calls) == (2 if checkpointed else 1)
    with torch.no_grad():
        actual = layer._hc_pre(*inputs)
    # The no-grad forward (reentrant AC initial pass, eval) also runs fused;
    # only the reference implementation stays available for direct callers.
    assert len(calls) == (3 if checkpointed else 2)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


def test_kl_preserves_slot_order_and_projection_gradients(monkeypatch):
    indexer = SimpleNamespace(
        _sp_size=4, _sp_rank=1, compress_ratio=2,
        n_heads=2, head_dim=4, rope_head_dim=0, softmax_scale=0.5,
        wq_b=torch.nn.Linear(3, 8, bias=False),
        weights_proj=torch.nn.Linear(5, 2, bias=False),
    )
    from types import MethodType
    from torchturbo.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Indexer

    indexer.project_queries = MethodType(DeepseekV41Indexer.project_queries, indexer)
    attn = SimpleNamespace(indexer=indexer, q_norm=torch.nn.Identity(),
                           wq_a=torch.nn.Linear(5, 3, bias=False))
    mask = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0]])
    packed = PackedAttentionMetadata(mask, 3, 1, 2)
    x = torch.randn(1, 3, 5, requires_grad=True)
    keys = torch.randn(1, 3, 4, requires_grad=True)
    selected = torch.tensor([[[-1, -1], [-1, 1], [1, -1]]])
    targets = torch.rand(1, 3, 2, requires_grad=True)
    positions = torch.tensor([[0, 1, 2]])
    seen = []

    def loss(q, k, weights, sparse, probabilities, **kwargs):
        seen.append((sparse.clone(), probabilities.clone()))
        assert not probabilities.requires_grad
        return q.float().square().sum() + k.float().square().sum() + weights.square().sum()

    monkeypatch.setattr(npu, "ops", SimpleNamespace(
        sli_kl_loss_grad_metadata=lambda *a, **kw: None, sli_kl_loss=loss,
    ))
    actual = npu._indexer_kl_loss(attn, x, keys, selected, targets,
                                  torch.empty(0), mask, positions, packed=packed)
    torch.testing.assert_close(seen[0][0][:, 0],
                               torch.tensor([[-1, -1], [-1, 0], [0, -1]], dtype=torch.int32))
    torch.testing.assert_close(seen[0][1][:, 0], targets[0].detach())
    projected_q = indexer.wq_b(attn.wq_a(x)).to(torch.bfloat16).float()
    projected_w = indexer.weights_proj(x) * (0.5 * 2 ** -0.5)
    expected = (projected_q.square().sum() + projected_w.square().sum()
                + keys[0, 1:2].to(torch.bfloat16).float().square().sum())
    variables = (x, keys, attn.wq_a.weight, indexer.wq_b.weight, indexer.weights_proj.weight)
    ga = torch.autograd.grad(actual, variables)
    ge = torch.autograd.grad(expected, variables)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)
    invalid = selected.clone()
    invalid[0, 0, 0] = 1
    with pytest.raises(ValueError, match="non-causal"):
        npu._indexer_kl_loss(attn, x, keys, invalid, targets, torch.empty(0),
                            mask, positions, packed=packed)


@pytest.mark.parametrize("ratio", [1, 2, 4])
@pytest.mark.parametrize("rank", range(4))
def test_shared_selection_contract_against_token_positions(ratio, rank):
    mask = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0]])
    packed = PackedAttentionMetadata(mask, 3, rank, ratio)
    for segment in packed.segments:
        for token in range(segment.start, segment.end):
            for block in range(-2, 10):
                indices = torch.full((1, 3, 1), -1)
                indices[0, token, 0] = block
                position = rank * 3 + token - segment.sample_start
                relative = block - segment.block_start
                legal = block == -1 or 0 <= relative < min(
                    segment.block_count, (position + 1) // ratio
                )
                if legal:
                    result, visible = packed.local_indices(indices, segment)
                    assert result[token - segment.start, 0] == (relative if block >= 0 else -1)
                    assert visible[token - segment.start] == (position + 1) // ratio
                else:
                    with pytest.raises(ValueError, match="cross-sample|causal"):
                        packed.local_indices(indices, segment)


def test_shared_selection_rejects_padding_and_wrong_positions():
    packed = PackedAttentionMetadata(torch.tensor([[1, 1, 1, 0]]), 4, 0, 1)
    indices = torch.tensor([[[0], [1], [2], [-1]]])
    packed.check_padding(indices)
    packed.local_indices(indices, packed.segments[0], torch.tensor([[0, 1, 2, 0]]))
    with pytest.raises(ValueError, match="positions"):
        packed.local_indices(indices, packed.segments[0], torch.tensor([[1, 2, 3, 0]]))
    indices[0, -1, 0] = 0
    with pytest.raises(ValueError, match="Padding"):
        packed.check_padding(indices)
