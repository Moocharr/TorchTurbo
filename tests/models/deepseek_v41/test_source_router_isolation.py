"""Keep internal V4.1 routing controls out of the public router path."""

import pytest
import torch
import torch.nn.functional as F

from torchturbo.models.deepseek_v41.moe_bridge import SourceRouter
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.router import TopKRouter


def _expected(logits, score_function, normalize):
    scores = logits.softmax(-1) if score_function == "softmax" else logits.sigmoid()
    selected, indices = scores.topk(2, dim=-1)
    if normalize:
        selected = selected / (selected.sum(-1, keepdim=True) + 1e-20)
    return torch.zeros_like(scores).scatter(-1, indices, selected * 1.7)


@pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
@pytest.mark.parametrize("normalize", [False, True])
def test_source_controls_preserve_public_routing(score_function, normalize):
    config = MoEConfig(
        hidden_size=3, num_moe_experts=4, moe_router_topk=2,
        params_dtype=torch.float32, perform_initialization=False,
        moe_router_pre_softmax=True, moe_router_score_function=score_function,
        moe_router_temperature=2.0, moe_router_norm_topk_prob=normalize,
        moe_router_topk_scaling_factor=1.7,
    )
    weights = torch.tensor([[1., 0., -1.], [0., 2., 1.], [-1., 0., 1.], [1., -1., 0.]])
    inputs = torch.tensor([[[1., 2., 3.], [-1., 1., 2.]]])
    source = SourceRouter(config)
    public = TopKRouter(config)
    with torch.no_grad():
        source.weight.copy_(weights)
        public.weight.copy_(weights)

    source_probs, _, _, source_logits = source(inputs)
    public_probs, _, _, public_logits = public(inputs)
    raw_logits = F.linear(inputs, weights)
    torch.testing.assert_close(source_logits, raw_logits / 2, rtol=0, atol=0)
    torch.testing.assert_close(public_logits, raw_logits, rtol=0, atol=0)
    torch.testing.assert_close(
        source_probs, _expected((raw_logits / 2).reshape(-1, 4), score_function, normalize),
    )
    # Public softmax keeps selected full-softmax scores; sigmoid normalizes top-k.
    torch.testing.assert_close(
        public_probs, _expected(raw_logits.reshape(-1, 4), score_function, score_function == "sigmoid"),
    )
