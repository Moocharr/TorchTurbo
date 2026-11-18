"""Shared import setup and isolated operator state for v4.1 tests."""

import importlib
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


@pytest.fixture
def reference_npu_ops(monkeypatch):
    """Use reference compression when testing fused MLA gradients."""
    from torchturbo.ops import OpDispatcher
    from torchturbo.ops.registry import registry
    from torchturbo.runtime.npu import NpuRuntime

    dispatcher = OpDispatcher(
        device_runtime=NpuRuntime(), registry=registry,
        overrides={"compressor": "torch_compressor"},
    )
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", dispatcher)
    return dispatcher


@pytest.fixture
def unit_kl_scale(monkeypatch):
    """Restore the auxiliary-loss scale after each test."""
    from torchturbo.models.deepseek_v41.npu import V41IndexerLossAutoScaler

    monkeypatch.setattr(V41IndexerLossAutoScaler, "main_loss_backward_scale", 1.0)
