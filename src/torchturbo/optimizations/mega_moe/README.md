# MegaMoE for DeepSeek V4.1

Enable dynamic expert placement and fused MoE execution with:

```yaml
engine_config:
  mega_moe: true
  activation_checkpointing:
    enabled: true
    use_reentrant: true
```

The DS4.1/LlamaFactory adapter preserves parameter ownership and pre-activation
SwiGLU clipping. Reentrant checkpointing retains the shared-weight fast path.
With checkpointing disabled or non-reentrant, forward releases shared weights;
backward fetches them again using the saved expert placement and FSDP-restored
parameters. This adds weight communication, not a persistent borrowed-weight
copy per layer. Disabling checkpointing still retains ordinary activations and
may exceed device memory at long sequence lengths. Keep activation checkpointing
settings consistent between YAML, adapter arguments and environment overrides.

## Runtime dependency

Provision `tt-megamoe==0.1.3+author.2094e90` and `cann-shmem==1.6.0` in the
container image or its startup script before launching training or verification.
TorchTurbo does not bundle wheels or install dependencies at runtime.

The matching environment is Linux aarch64, Python 3.11, Torch 2.10.0,
torch_npu 2.10.0.post4, CANN 9.1 and CXX11 ABI=1. The author 2094e905 routing
library must be built and installed with the runtime; the older allocator ABI
is incompatible. The source-bearing component wheel requires
`python -m tt_megamoe.build_allocator` with `bisheng` after installation.

## Representative verification

On one allocated 16-NPU worker with the training dependencies and pytest:

```bash
bash tests/npu/run_mega_moe_16card.sh
```

This runs two checks:

1. `test_mega_moe_planner.py`: EP64/384-expert summary, planner tables and virtual
   routing on one NPU, compared with an independent integer reference.
2. `mega_moe_precision.py`: two layers sharing buffers, DP2 x EP8 FSDP2 with
   resharding, and 384-expert hot routing. Compare output and input/router/expert
   gradients against ordinary operations with checkpointing disabled, non-reentrant
   and reentrant. A 16 KiB planner budget forces paging. The test requires actual
   expert gradient returns, so bypassing borrowed experts cannot produce a pass.

`OUTPUT_DIR` selects the log directory. Success ends with `VERIFY_PASS` and
16 passing `PRECISION_RESULT` entries for each checkpoint mode. Failures stop later stages. This validates
representative MoE behavior; full SFT, EP64 cross-node transport and performance
still require the full-model experiment.
