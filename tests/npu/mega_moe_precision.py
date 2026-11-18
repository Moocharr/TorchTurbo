"""Compare standalone MegaMoE forward/backward with ordinary NPU experts."""
import json
import argparse
import os

# Bootstrap must run before Torch or torch_npu loads any native library.
import importlib.util
from pathlib import Path
import runpy

_component = Path(importlib.util.find_spec("tt_megamoe").origin).parent
runpy.run_path(str(_component / "bootstrap.py"))["activate"]()

import torch
import torch.distributed as dist
import torch_npu



def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a, b = actual.detach().float(), expected.detach().float()
    return {
        "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-12)).item(),
        "max_abs": (a - b).abs().max().item(),
        "finite": bool(torch.isfinite(a).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["none", "pre_activation"], default="none")
    parser.add_argument("--limit", type=float, default=0.)
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--routing", choices=["uniform", "hot"], default="uniform")
    checkpoint_args = parser.add_mutually_exclusive_group()
    checkpoint_args.add_argument("--checkpoint", dest="checkpoint_mode", action="store_const", const="reentrant")
    checkpoint_args.add_argument("--checkpoint-mode", choices=["none", "reentrant", "nonreentrant"], default="none")
    parser.set_defaults(checkpoint_mode="none")
    parser.add_argument("--fsdp", action="store_true", help="Use two data-parallel shards with FSDP2")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--planner-ub-bytes", type=int, default=0,
                        help="Test-only planner UB limit; zero uses the actual device budget")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    world = dist.get_world_size()
    ep_group = dist.group.WORLD
    dp_mesh = None
    if args.fsdp:
        from torch.distributed.device_mesh import init_device_mesh
        if world % 2:
            raise ValueError("FSDP verification requires an even world size")
        mesh = init_device_mesh("npu", (2, world // 2), mesh_dim_names=("dp", "ep"))
        dp_mesh = mesh["dp"]
        ep_group = mesh["ep"].get_group()
    ep_size = dist.get_world_size(ep_group)
    ep_rank = dist.get_rank(ep_group)
    t, h, intermediate, experts, k = args.tokens, 256, 256, args.experts, args.top_k
    if t <= 0 or t % 128 or experts <= 0 or experts % ep_size or not 0 < k <= experts:
        raise ValueError("Require 128-aligned tokens, experts divisible by ranks, and valid top-k")
    planner_calls = 0
    import tt_megamoe.moonep as moonep
    if args.planner_ub_bytes:
        if args.planner_ub_bytes < 16384:
            raise ValueError("Test planner UB budget must be at least 16384 bytes")
        moonep._device_ub_bytes = lambda device_index: args.planner_ub_bytes
    allocator = moonep.load_allocator()
    native_allocate = allocator.allocate

    def allocate(counts, capacity):
        nonlocal planner_calls
        planner_calls += 1
        return native_allocate(counts, capacity)

    allocator.allocate = allocate
    limit = args.limit
    mode = args.mode
    identity = args.identity
    if identity:
        h = 2 * intermediate
    local_experts = experts // ep_size
    device = torch.device("npu", rank)
    torch.manual_seed(8181)
    x_cpu = (torch.randn(ep_size * t, h) * 0.5).bfloat16()
    ids_cpu = torch.rand(ep_size * t, experts).topk(k, dim=-1).indices.sort(-1).values
    if args.routing == "hot":
        ids_cpu[:] = torch.arange(k)
    probs_cpu = torch.randn(ep_size * t, k).softmax(-1)
    w1_cpu = (torch.randn(experts, h, 2 * intermediate) * 0.03).bfloat16()
    w2_cpu = (torch.randn(experts, intermediate, h) * 0.03).bfloat16()
    if identity:
        w1_cpu = torch.eye(h).bfloat16().repeat(experts, 1, 1)
        w2_cpu = torch.eye(h).bfloat16()[:intermediate].repeat(experts, 1, 1)
    if identity:
        values = torch.tensor([-4.03125, -4., -0.251953125, -0.25, -0.248046875,
                               0., 0.248046875, 0.25, 0.251953125,
                               0.4140625, 0.416015625, 0.41796875, 4., 4.03125]).bfloat16()
        x_cpu[:14, :intermediate] = values[:, None]
        x_cpu[:14, intermediate:] = values.flip(0)[:, None]
    dy_cpu = (torch.randn(ep_size * t, h) * 0.1).bfloat16()
    sl = slice(ep_rank * t, (ep_rank + 1) * t)
    es = slice(ep_rank * local_experts, (ep_rank + 1) * local_experts)
    x = x_cpu[sl].to(device).requires_grad_()
    probs = probs_cpu[sl].to(device).requires_grad_()
    from torchturbo.models.modules.moe.moe_config import MoEConfig
    from torchturbo.optimizations.mega_moe.moonep.backend import MoonEPMegaMoeBackend

    config = MoEConfig(hidden_size=h, moe_ffn_hidden_size=intermediate,
                      num_moe_experts=experts, moe_router_topk=k,
                      gated_linear_unit=True, activation_func=torch.nn.functional.silu)
    backend = MoonEPMegaMoeBackend(
        ep_group, t, ((mode, limit),),
        refetch_in_backward=args.checkpoint_mode != "reentrant",
    )

    class ExpertLayer(torch.nn.Module):
        def __init__(self, weight1, weight2):
            super().__init__()
            self.config = config
            self.swiglu_limit = limit
            self.weight1 = torch.nn.Parameter(weight1[es].to(device))
            self.weight2 = torch.nn.Parameter(weight2[es].to(device))

        def forward(self, hidden, weights):
            return backend.execute(self, hidden, ids_cpu[sl].to(device), weights)

    # Distinct layers must survive two forwards before either backward runs.
    cpu_weights = [
        (w1_cpu, w2_cpu),
        ((torch.randn_like(w1_cpu.float()) * 0.03).bfloat16(),
         (torch.randn_like(w2_cpu.float()) * 0.03).bfloat16()),
    ]
    layers = [ExpertLayer(a, b) for a, b in cpu_weights]
    if args.fsdp:
        from torch.distributed.fsdp import fully_shard
        for layer in layers:
            fully_shard(layer, mesh=dp_mesh, reshard_after_forward=True)
    close = backend.close
    print(f"rank={rank} native_forward_start", flush=True)
    output = x
    for layer in layers:
        if args.checkpoint_mode != "none":
            from torch.utils.checkpoint import checkpoint
            output = checkpoint(layer, output, probs, use_reentrant=args.checkpoint_mode == "reentrant")
        else:
            output = layer(output, probs)
    torch.npu.synchronize()
    print(f"rank={rank} native_backward_start", flush=True)
    output.backward(dy_cpu[sl].to(device))
    torch.npu.synchronize()
    print(f"rank={rank} reference_start", flush=True)
    xr = x_cpu.to(device).requires_grad_()
    pr = probs_cpu.to(device).requires_grad_()
    reference_weights = [(a.to(device).requires_grad_(), b.to(device).requires_grad_()) for a, b in cpu_weights]

    def reference(hidden, wr1, wr2):
        ref = torch.zeros_like(hidden)
        for expert in range(experts):
            rows, slots = (ids_cpu == expert).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            rows, slots = rows.to(device), slots.to(device)
            up = hidden[rows] @ wr1[expert]
            if limit:
                gate, up_branch = up.float().chunk(2, dim=-1)
                if mode == "pre_activation":
                    act = torch.nn.functional.silu(gate.clamp(max=limit)) * up_branch.clamp(-limit, limit)
                else:
                    act = torch.nn.functional.silu(gate).clamp(max=limit) * up_branch.clamp(-limit, limit)
            else:
                act = torch_npu.npu_swiglu(up, dim=-1)
            act = act.to(up.dtype)
            y = act @ wr2[expert]
            ref = ref.index_add(0, rows, (y.float() * pr[rows, slots, None]).to(y.dtype))
        return ref

    ref = xr
    for wr1, wr2 in reference_weights:
        ref = reference(ref, wr1, wr2)
    ref.backward(dy_cpu.to(device))
    result = {
        "rank": rank, "mode": mode, "identity": identity, "limit": limit,
        "output": compare(output, ref[sl]),
        "dx": compare(x.grad, xr.grad[sl]),
        "dprobs": compare(probs.grad, pr.grad[sl]),
    }
    for index, (layer, (wr1, wr2)) in enumerate(zip(layers, reference_weights)):
        for name, expected in (("weight1", wr1), ("weight2", wr2)):
            actual = layer.get_parameter(name).grad
            if args.fsdp:
                actual = actual.full_tensor()
            result[f"layer{index}_{name}_grad"] = compare(actual, expected.grad[es])
    if identity:
        result["dx_gate"] = compare(x.grad[:, :intermediate], xr.grad[sl, :intermediate])
        result["dx_up"] = compare(x.grad[:, intermediate:], xr.grad[sl, intermediate:])
    passed = all(value["finite"] and value["relative_l2"] < 0.02
                 for value in result.values() if isinstance(value, dict))
    if args.planner_ub_bytes and args.routing == "hot":
        passed = passed and planner_calls > 0
    result.update(status="PASS" if passed else "FAIL", routing=args.routing,
                  checkpoint=args.checkpoint_mode, fsdp=args.fsdp, layers=len(layers),
                  world_size=world, ep_size=ep_size, experts=experts,
                  tokens=t, top_k=k, planner_ub_bytes=args.planner_ub_bytes,
                  planner_calls=planner_calls)
    # Hot routing must exercise actual weight transfers, not just the planner.
    if ep_size > 1 and args.routing == "hot" and args.planner_ub_bytes:
        passed = passed and backend.resources.statistics["gradient_returns"] > 0
    result.update(status="PASS" if passed else "FAIL", transfers=dict(backend.resources.statistics) if ep_size > 1 else {})
    collective_pass = torch.tensor(int(passed), device=device, dtype=torch.int32)
    dist.all_reduce(collective_pass, op=dist.ReduceOp.MIN)
    all_passed = bool(collective_pass.item())
    os.write(1, ("PRECISION_RESULT " + json.dumps(result) + "\n").encode())
    close()
    dist.destroy_process_group()
    assert all_passed, result


if __name__ == "__main__":
    main()
