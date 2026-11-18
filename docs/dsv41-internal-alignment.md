# DeepSeek V4.1 staged internal-source migration

Source: `Language_Compiler/torchturbo`, branch `dsv41_sft_rl`, commit
`e49ff83aa5667099a79e9f5aa5f6be8db958bfc9`. Public base: `8a29e23`.

| Commit | Scope |
| --- | --- |
| 1 | Native model/config/weights, FP32 cross-layer input policy, MFU; no fusion/offload. |
| 2 | Internal BSND mHC/RMSNorm patches, vendor providers and disable switches. |
| 3 | Internal SP/packed masks, Sparse MLA, indexer/KL, CANN/PyPTO compressor. |
| 4 | Internal expert-output weighting, MegaMoE/MoonEP and YAML bootstrap. |
| 5 | Frozen EP BF16 host tables, row I/O, Engram checkpoint/no-shard, original activation offload, chunked computation and NPU memory limit. |
| 6 | Original table autograd routing, EDP synchronization, ActiveRowAdam and EP checkpoint files. |

The first four snapshots use device-resident, read-only Engram tables to
isolate the requested stages. Dense gate/projection training follows the
internal default; host table training starts in commit 6. Staging removes
later features without inventing replacements for their algorithms.

## Required public infrastructure adapters

See [the itemized interface audit](dsv41-public-interface-adapters.md) for
source/public differences, necessity, semantic effects and the explicit
Engram freeze-precedence correction in commit 6.

The public model lifecycle, optimizer plan and checkpoint-manager APIs differ:

- `deepseek_v41/model.py` connects the internal wrapper to `TurboModel`, nested
  dtype context and input/runtime hooks. Public cumulative packed boundaries
  become the internal global sequence-id mask.
- FSDP carries internal `cast_forward_inputs=False` to decoder units and
  finalizes no-shard synchronization after final HostOffloadEngram parameters
  exist. Sparse weights are excluded from dense gradient synchronization.
- `deepseek_v41/experts.py` retains public parameter construction and the
  unchanged internal GroupedMLP forward body. The dispatcher marks this
  source expert's output-weighting contract; other public models retain theirs.
- `moe_source_*` retain internal router/permutation and auxiliary-loss code;
  only import paths and tuple/dispatcher interface bridges differ. The source
  auxiliary-loss scaler shares the identical public scaler state used by the
  engine, so gradient accumulation updates the same scale.
- `source_packing_utils.py` retains the internal dynamic complete-block shape
  and cache semantics. V4.1 imports this source copy rather than the public
  V4 helper's changed static-capacity implementation; other families are intact.
- `source_hyper_connections.py` retains the internal FP32 bmm HcPost helper,
  rather than the public V4 helper's elementwise reduction and dtype behavior.
- `source_functools.py` retains the internal tensor-mutation-version cache
  checks, including in-place updates of packed masks and positions.
- Optimizer plans exclude `_engram_host_offload` parameters; the unmodified
  internal EngramOptimizer owns updates and EP files.
- Checkpoint adapters retain original row streaming, EP keys/layout and
  checkpoint-root `engram_optimizer_ep<N>.pt` files.
- The NPU gated-activation provider accepts the public dispatcher's extra
  clamp-after-activation keyword. The V4.1 false path is unchanged; true
  retains the existing non-V4.1 native contract.

`internal_source_manifest.json` records source/staged SHA-256 values and
unchanged source function bodies in files requiring API adapters. EOF newline
differences are identified separately from source-content differences.
`public-baseline-restored` entries pin the shared router and utilities to
the public base; V4.1 controls remain in the internal source router/helpers.

## Validation limits

Internal model, Engram, MFU and chunked-loss tests are retained. Added CPU
checks cover lifecycle, original EP tables, sparse optimizer, freezing and
checkpoint continuation, including two EP owners.

Four internal `test_shared_pre_attention_matches_full_sequence_gradients`
cases fail their strict zero-tolerance comparison by about 1e-7 on this macOS
CPU BLAS. The same cases fail in the unmodified internal checkout under the
same environment. Source and tolerances remain unchanged. CPU lifecycle tests
disable pinning through test instrumentation because this host selects an
unavailable MPS pinned allocator; production source pinning is unchanged.

NPU validation remains required for fused gradients, BF16, SP/EP/packing,
KL/GA, borrowed-expert gradients, long-sequence host/device/pinned memory,
asynchronous copies, multi-node performance and fresh-process adapter resume.
Source-baseline limitations are retained rather than changed during migration.
