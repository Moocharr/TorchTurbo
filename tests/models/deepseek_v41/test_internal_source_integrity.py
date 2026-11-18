"""Protect the pinned source import and each staged feature boundary."""
import ast
import hashlib
import json
from pathlib import Path


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _ast_digest(tree):
    for node in ast.walk(tree):
        if hasattr(node, "type_params") and not node.type_params:
            delattr(node, "type_params")
    return _digest(ast.dump(tree, include_attributes=False).encode())


def test_internal_source_manifest():
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "docs/internal_source_manifest.json").read_text())
    assert manifest["source_commit"] == "e49ff83aa5667099a79e9f5aa5f6be8db958bfc9"
    assert 1 <= manifest["stage"] <= 6
    for entry in manifest["files"]:
        data = (root / entry["target"]).read_bytes()
        assert _digest(data) == entry["target_sha256"], entry["target"]
        if entry["comparison"] == "byte-identical":
            assert entry["target_sha256"] == entry["source_sha256"]
        if "source_non_import_ast_sha256" in entry:
            tree = ast.parse(data.decode())
            tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
            assert _ast_digest(tree) == entry["source_non_import_ast_sha256"], entry["target"]
        if "unchanged_definitions" in entry:
            found = {}
            def walk(node, prefix=""):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        name = prefix + child.name
                        found[name] = _ast_digest(child)
                        walk(child, name + ".")
                    else:
                        walk(child, prefix)
            walk(ast.parse(data.decode()))
            for name, digest in entry["unchanged_definitions"].items():
                assert found[name] == digest, (entry["target"], name)


def test_no_replacement_host_architecture():
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "docs/internal_source_manifest.json").read_text())
    if manifest["stage"] >= 5:
        host = (root / "src/torchturbo/models/deepseek_v41/host_offload_embedding.py").read_text()
        assert "NodeSharedCPUResidentParameter" not in host
        assert "all_to_all_single" in host
        assert "class TorchHostOffloadEmbedding" in host
    if manifest["stage"] >= 6:
        optimizer = (root / "src/torchturbo/engine/backends/fsdp2/optimizers/engram.py").read_text()
        assert "EngramSwapOptimizer" not in optimizer
        assert "table.ep_size * self.replica_size" in optimizer


def test_stage_feature_boundaries():
    root = Path(__file__).resolve().parents[3]
    stage = json.loads((root / "docs/internal_source_manifest.json").read_text())["stage"]
    package = root / "src/torchturbo"
    features = {
        "models/deepseek_v41/npu.py": 2,
        "models/deepseek_v41/packed_metadata.py": 3,
        "ops/providers/npu/cann_compressor.py": 3,
        "ops/providers/npu/compressor.py": 3,
        "optimizations/mega_moe/integration.py": 4,
        "models/deepseek_v41/moe_source_token_dispatcher.py": 4,
        "models/deepseek_v41/host_offload_embedding.py": 5,
        "models/modules/moe/chunked_mlp.py": 5,
        "engine/backends/fsdp2/optimizers/engram.py": 6,
    }
    for path, first_stage in features.items():
        assert (package / path).exists() == (stage >= first_stage), path
    args = (package / "engine/args.py").read_text()
    assert ("--tt_train_engram" in args) == (stage == 6)
    assert ("--tt_indexer_kl_loss_coeff" in args) == (stage >= 3)
    assert ("--tt_ac_offload_input_to_cpu" in args) == (stage >= 5)
    if stage >= 2:
        npu = (package / "models/deepseek_v41/npu.py").read_text()
        assert ("def patch_deepseek_v41_npu_sfa(" in npu) == (stage >= 3)
    if stage == 5:
        host = (package / "models/deepseek_v41/host_offload_embedding.py").read_text()
        assert "class _TrainableHostFetch" not in host
        model = (package / "models/deepseek_v41/model.py").read_text()
        assert "train_engram = False" in model
