"""DeepSeek V4.1 contract for backend-owned chunked LM loss.

FSDP2Backend._forward_for_lm_loss shadows ``head`` with nn.Identity and
expects the model to return the normed fp32 hidden states through the
logits output. The wrapper's _compute_logits override detects the shadow
(attribute view vs _modules registration) and passes the hidden states to
the no-op head in a single argument. These tests pin that contract.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.models.deepseek_v41.model import TurboDeepseekV41ForCausalLM
from torchturbo.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41ForCausalLM,
    DeepseekV41Head,
    DeepseekV41RMSNorm,
)


class _PassThroughLayer(nn.Module):
    """Decoder-layer stub: returns the streams unchanged, refreshes no slot."""

    is_kv_source = False
    engram = None

    def forward(self, hidden_states, pre_mix, *slots, **kwargs):
        return hidden_states, pre_mix, None


class _FailingLayer(nn.Module):
    is_kv_source = False
    engram = None

    def forward(self, *args, **kwargs):
        raise RuntimeError("model forward failed")


def _make_model(layer=None):
    torch.manual_seed(0)
    hidden_size = 4
    hc_mult = 2
    vocab_size = 7

    model = object.__new__(TurboDeepseekV41ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        use_return_dict=True,
        use_cache=False,
        text_config=SimpleNamespace(hc_mult=hc_mult),
    )
    model.embed = nn.Embedding(11, hidden_size)
    model.layers = nn.ModuleList([layer if layer is not None else _PassThroughLayer()])
    model.norm = DeepseekV41RMSNorm(hidden_size)
    model.head = DeepseekV41Head(SimpleNamespace(vocab_size=vocab_size, hidden_size=hidden_size))
    # DeepseekV41Head leaves the weight as torch.empty; init it so the
    # value assertions never read allocator garbage (NaNs).
    model.head.init_weights(0.02)
    model.engram_hash = None
    model.gradient_checkpointing = False
    model.train()
    return model


def test_chunk_loss_returns_normed_hidden_and_skips_real_head():
    model = _make_model()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    real_lm_head = model.head
    real_head_calls = []

    hook = real_lm_head.register_forward_hook(
        lambda _module, _args, _output: real_head_calls.append(True)
    )
    regular_outputs = model(input_ids=input_ids, return_dict=True)
    hook.remove()

    backend = FSDP2Backend.__new__(FSDP2Backend)
    chunk_outputs, hidden_states = backend._forward_for_lm_loss(
        model,
        {"input_ids": input_ids},
        use_chunked_loss=True,
    )
    hidden_state = hidden_states[0]

    # The real head ran once (regular forward) and never under the shadow.
    assert real_head_calls == [True]
    assert chunk_outputs.logits is None
    assert hidden_state.shape == (1, 4, 4)
    assert hidden_state.dtype == real_lm_head.weight.dtype

    # v4.1 has no head-forward wrapper: the adaptation lives in
    # _compute_logits, and forward itself is the base implementation.
    assert TurboDeepseekV41ForCausalLM.forward is DeepseekV41ForCausalLM.forward

    # Chunked projection over the returned hidden states must reproduce the
    # regular logits bit-for-bit (norm + fp32 cast already applied).
    chunk_logits = nn.functional.linear(hidden_state, real_lm_head.weight)
    torch.testing.assert_close(regular_outputs.logits, chunk_logits, rtol=0, atol=0)

    parameters = (model.embed.weight, model.norm.weight, model.head.weight)
    regular_grads = torch.autograd.grad(regular_outputs.logits.sum(), parameters)
    chunk_grads = torch.autograd.grad(chunk_logits.sum(), parameters)
    for regular_grad, chunk_grad in zip(regular_grads, chunk_grads):
        torch.testing.assert_close(regular_grad, chunk_grad)

    # The shadow never touched the module registration or the state dict.
    assert model.head is real_lm_head
    assert model.get_lm_head() is real_lm_head
    assert model.get_output_embeddings() is real_lm_head
    assert "head" not in vars(model)
    assert "head.weight" in model.state_dict()


def test_chunk_loss_restores_deepseek_head_after_forward_error():
    model = _make_model(layer=_FailingLayer())
    real_lm_head = model.head
    backend = FSDP2Backend.__new__(FSDP2Backend)

    with pytest.raises(RuntimeError, match="model forward failed"):
        backend._forward_for_lm_loss(
            model,
            {"input_ids": torch.tensor([[1, 2, 3, 4]])},
            use_chunked_loss=True,
        )

    assert "head" not in vars(model)
    assert model.head is real_lm_head
