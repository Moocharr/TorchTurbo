"""CPU contracts at the boundary of the optional MegaMoE runtime."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchturbo.engine.config import EngineConfig
from torchturbo.optimizations.mega_moe.integration import (
    RoutedExpertsBackend,
    configure_token_capacity,
    install,
)
from torchturbo.parallel import FSDPParallelPlan
from torchturbo.runtime import TurboDevice


def test_internal_capacity_accounts_for_sp_padding():
    config = EngineConfig.from_dict({"mega_moe": True})
    config.parallel_plan = FSDPParallelPlan(sp_size=4)
    configure_token_capacity(config, 513, 2)
    assert config.extra["mega_moe_local_tokens"] == 384
    configure_token_capacity(config, 8192, 4)
    assert config.extra["mega_moe_local_tokens"] == 8192


def test_route_enters_expert_hooks_and_keeps_zero_weight_selections():
    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))
            self.indices = None
            self.config = SimpleNamespace(moe_router_topk=2)

        def forward(self, hidden, indices, probabilities):
            self.indices = indices
            return hidden * self.weight + probabilities.sum().to(hidden.dtype), None

    experts = Experts()
    hooks = []
    experts.register_forward_pre_hook(lambda *_: hooks.append("forward"))
    hidden = torch.randn(1, 2, 3, requires_grad=True)
    probabilities = torch.tensor([[0.0, 0.2, 0.0], [0.4, 0.0, 0.6]], requires_grad=True)
    indices = torch.tensor([[True, True, False], [True, False, True]])
    output, bias = RoutedExpertsBackend(None, 128).route(
        experts, hidden, probabilities, indices,
    )
    output.sum().backward()
    assert hooks == ["forward"]
    torch.testing.assert_close(experts.indices, torch.tensor([[0, 1], [0, 2]]))
    assert bias is None
    assert probabilities.grad[0, 0] != 0
    assert probabilities.grad[0, 2] == 0
    assert hidden.grad is not None
    assert experts.weight.grad is not None


def test_megamoe_requires_shared_experts():
    config = EngineConfig.from_dict({
        "mega_moe": True, "extra": {"mega_moe_local_tokens": 128},
    })
    with pytest.raises(ValueError, match="MoELayer"):
        install(nn.Linear(2, 2), config, None)


def test_internal_bootstrap_uses_yaml_boolean(tmp_path, monkeypatch):
    entry = Path(__file__).resolve().parents[2] / "src/torchturbo/optimizations/mega_moe/bootstrap.py"
    spec = spec_from_file_location("mega_moe_bootstrap", entry)
    bootstrap = module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    calls = []
    monkeypatch.setenv("TORCH_TURBO_MOE_TOKEN_DISPATCHER", "mega_moe")
    monkeypatch.setattr(
        bootstrap.importlib.util, "find_spec",
        lambda _: SimpleNamespace(origin=str(tmp_path / "__init__.py")),
    )
    monkeypatch.setattr(
        bootstrap.runpy, "run_path",
        lambda _: {"activate": lambda: calls.append("activated") or True},
    )
    config = tmp_path / "engine.yaml"
    config.write_text("engine_config:\n  mega_moe: false\n")
    assert not bootstrap.activate(["--tt_config", str(config)])
    config.write_text("engine_config:\n  mega_moe: true\n")
    assert bootstrap.activate(["--tt_config", str(config)])
    assert calls == ["activated"]
