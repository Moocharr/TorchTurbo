"""Public engine lifecycle adapter checks for the internal V4.1 implementation."""
import json
import sys
import pytest
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from torchturbo import EngineConfig, TorchTurboEngine
from torchturbo.engine.args import TorchTurboArgs, set_torchturbo_args
from torchturbo.models.deepseek_v41 import DeepseekV41Config, TurboDeepseekV41ForCausalLM

from pathlib import Path

@pytest.fixture(autouse=True)
def cpu_stage_context(monkeypatch):
    set_torchturbo_args(TorchTurboArgs())
    path = Path(__file__).resolve().parents[3] / "src/torchturbo/models/deepseek_v41/host_offload_embedding.py"
    if path.exists():
        from torchturbo.models.deepseek_v41.host_offload_embedding import TorchHostOffloadEmbedding
        original = TorchHostOffloadEmbedding.__init__
        def initialize(self, *args, **kwargs):
            kwargs["pin_memory"] = False
            original(self, *args, **kwargs)
        monkeypatch.setattr(TorchHostOffloadEmbedding, "__init__", initialize)
    yield
    set_torchturbo_args(TorchTurboArgs())

@pytest.fixture
def checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "pretrained"
    checkpoint_dir.mkdir()
    vocab = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2}
    vocab.update({f"word{i}": i for i in range(3, 32)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]",
    )
    tokenizer.save_pretrained(checkpoint_dir)
    config = DeepseekV41Config(
        vision_config={"num_hidden_layers": 0},
        text_config={
            "vocab_size": len(vocab),
            "hidden_size": 16,
            "moe_intermediate_size": 24,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "head_dim": 8,
            "qk_rope_head_dim": 4,
            "q_lora_rank": 8,
            "o_lora_rank": 4,
            "o_groups": 1,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "gate_temp": 1.4,
            "sliding_window": 2,
            "compress_ratios": [2, 2, 2],
            "kv_source_layer_ids": [0],
            "index_source_layer_ids": [0, 1],
            "index_n_heads": 2,
            "index_head_dim": 8,
            "index_topk": 2,
            "candidate_source_layer_id": 0,
            "candidate_topk_blocks": 1,
            "candidate_block_size": 1,
            "hc_mult": 2,
            "hc_sinkhorn_iters": 2,
            "rope_scaling": {"factor": 2, "original_max_position_embeddings": 8},
            "engram_layer_ids": [0],
            "engram_num_embeddings": [32],
            "engram_max_ngram_size": 2,
            "engram_vocab_size": 7,
            "engram_n_heads": 1,
            "engram_head_dim": 4,
            "engram_compressed_vocab_size": len(vocab),
        },
    )
    torch.manual_seed(41)
    model = AutoModelForCausalLM.from_config(config)
    assert isinstance(model, TurboDeepseekV41ForCausalLM)
    config.save_pretrained(checkpoint_dir)
    weights = model.state_dict()
    filename = "model-00001-of-00001.safetensors"
    save_file(weights, checkpoint_dir / filename, metadata={"format": "pt"})
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict.fromkeys(weights, filename)})
    )
    return checkpoint_dir


def test_stage_training_smoke(checkpoint, tmp_path, monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform=="darwin" else "lo")
    dist.init_process_group("gloo",init_method=f"file://{tmp_path}/dist",rank=0,world_size=1)
    try:
        engine=TorchTurboEngine(EngineConfig.from_dict({"mixed_precision":"fp32"}))
        model=engine.prepare_model(str(checkpoint),torch_dtype=torch.float32)
        optimizer=engine.prepare_optimizer(torch.optim.AdamW,lr=0.01,weight_decay=0.0)
        ids=torch.tensor([[3,4,5,6,7]])
        loss=engine.compute_loss(model,{"input_ids":ids,"attention_mask":torch.ones_like(ids),"labels":ids})
        assert torch.isfinite(loss)
        loss.backward()
        assert model.layers[0].engram.q_weight.dtype==torch.float32
        optimizer.step()
        optimizer.zero_grad()
    finally:
        dist.destroy_process_group()
