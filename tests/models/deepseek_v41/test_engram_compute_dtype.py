"""Engram compute dtype must not depend on host-offloaded master storage."""
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from torchturbo.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Engram


@pytest.mark.parametrize('activation_dtype',[torch.bfloat16,torch.float32])
def test_engram_preserves_master_parameter_and_matches_inference(activation_dtype):
    torch.manual_seed(42)
    config=SimpleNamespace(hidden_size=16,hc_mult=4,rms_norm_eps=1e-20,
        engram_num_embeddings=[23],engram_max_ngram_size=3,engram_n_heads=2,engram_head_dim=8)
    model=DeepseekV41Engram(config,1,0)
    x=torch.randn(1,7,4,16,dtype=activation_dtype,requires_grad=True)
    ids=torch.randint(23,(1,7,4))
    output=model(x,ids)
    memory=F.embedding(ids,model.embed.weight).flatten(-2).to(activation_dtype)
    inference_weight=model.wkv.weight.detach().to(activation_dtype).clone().requires_grad_()
    projected=F.linear(memory,inference_weight)
    key,value=projected.split([64,16],-1)
    key=key.float().unflatten(-1,(4,16));h=x.float()
    rstd=torch.rsqrt(h.square().mean(-1)+1e-20)*torch.rsqrt(key.square().mean(-1)+1e-20)
    dot=(h*(model.q_weight.float()*model.k_weight.float())*key).sum(-1)*rstd*16**-0.5
    gate=torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(),dot))
    expected=(h+gate.unsqueeze(-1)*value.float().unsqueeze(-2)).to(activation_dtype)
    torch.testing.assert_close(output,expected,rtol=0,atol=0)
    upstream=torch.randn_like(output)
    actual_grad=torch.autograd.grad(output,model.wkv.weight,upstream,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected,inference_weight,upstream)[0].float()
    torch.testing.assert_close(actual_grad,expected_grad,rtol=0,atol=0)
    assert model.wkv.weight.dtype==torch.float32
    assert actual_grad.dtype==torch.float32
    assert torch.isfinite(actual_grad).all()
    assert actual_grad.count_nonzero()>0
