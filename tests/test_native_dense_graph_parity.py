from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from neuralfn.graph import NeuronGraph
from neuralfn.native_dense_checkpoint import inspect_native_dense_checkpoint
from neuralfn.native_ir import migrate_graph_to_native
from neuralfn.native_moa_checkpoint import (
    NATIVE_MOA_CANDIDATE_ACTIVATIONS,
    NATIVE_MOA_INFERENCE_SCHEMA,
)
from neuralfn.native_moa_graph_runtime import (
    NATIVE_MOA_GRAPH_RUNTIME_PROFILE,
    NativeMoaGraphRuntimeError,
    load_native_moa_graph_runtime,
)
from neuralfn.torch_backend import CompiledTorchGraph
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config
from tests.test_native_resident_binding import (
    _python_dense_logits,
    _write_tiny_dense_v5,
    resident_binding,
)


GEOMETRY = {
    "max_seq_len": 8,
    "vocab_size": 11,
    "padded_vocab_size": 16,
    "num_layers": 2,
    "num_heads": 2,
    "channels": 8,
}


def _write_moa_metadata(
    checkpoint: Path,
    graph_path: Path,
    *,
    selected_activation: str,
    interval: int = 50,
) -> Path:
    info = inspect_native_dense_checkpoint(checkpoint)
    step = "00000001"
    expected_checkpoint = checkpoint.with_name(f"model_{step}.bin")
    if checkpoint != expected_checkpoint:
        checkpoint.rename(expected_checkpoint)
        checkpoint = expected_checkpoint
    done = checkpoint.with_name(f"DONE_{step}")
    done.write_bytes(b"")
    metadata = checkpoint.with_name(f"model_{step}.moa.json")
    metadata.write_text(
        json.dumps(
            {
                "schema": NATIVE_MOA_INFERENCE_SCHEMA,
                "version": 1,
                "preset": "gpt2_moa",
                "checkpoint_kind": "trained_dense_v5",
                "model": {
                    "path": checkpoint.name,
                    "format": "neuralfn.native_dense_gpt.v5",
                    "nbytes": checkpoint.stat().st_size,
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                },
                "done_marker": done.name,
                "source_graph": {
                    "filename": graph_path.name,
                    "sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
                    "byte_identity_verified": True,
                },
                "selection": {
                    "activation": selected_activation,
                    "candidates": list(NATIVE_MOA_CANDIDATE_ACTIVATIONS),
                    "interval": interval,
                },
                "geometry": {
                    "max_seq_len": info.max_seq_len,
                    "vocab_size": info.vocab_size,
                    "padded_vocab_size": info.padded_vocab_size,
                    "num_layers": info.num_layers,
                    "model_dim": info.channels,
                    "num_heads": info.num_heads,
                    "head_dim": info.channels // info.num_heads,
                    "mlp_hidden_dim": 4 * info.channels,
                },
            }
        ),
        encoding="utf-8",
    )
    return metadata


def _dense_weights(checkpoint: Path) -> tuple[object, dict[str, torch.Tensor]]:
    info = inspect_native_dense_checkpoint(checkpoint)
    payload = checkpoint.read_bytes()
    weights: dict[str, torch.Tensor] = {}
    for tensor in info.tensors:
        raw = bytearray(payload[tensor.offset : tensor.offset + tensor.nbytes])
        weights[tensor.name] = torch.frombuffer(raw, dtype=torch.bfloat16).float().reshape(
            tensor.shape
        )
    return info, weights


def _compiled_dense_with_native_weights(
    graph: NeuronGraph,
    checkpoint: Path,
) -> CompiledTorchGraph:
    """Import every dense-v5 ABI tensor into its graph-authored Torch stage."""

    compiled = CompiledTorchGraph(graph, kernel_backend="torch")
    info, weights = _dense_weights(checkpoint)
    consumed: set[str] = set()

    def copy_parameter(
        parameter: torch.nn.Parameter,
        source: torch.Tensor,
        native_name: str,
    ) -> None:
        parameter.copy_(source.to(dtype=parameter.dtype).reshape(parameter.shape))
        consumed.add(native_name)

    model = compiled.node_modules["model"]
    channels = info.channels
    vocab_size = info.vocab_size
    with torch.no_grad():
        copy_parameter(
            model.node_modules["token_embed"].embedding.weight,
            weights["transformer.wte.weight"][:vocab_size],
            "transformer.wte.weight",
        )

        # The authoring graph retains its default maximum position table while
        # dense-v5 supplies the exact runtime context rows.  Clear the unused
        # tail so this import is deterministic even if a future assertion
        # inspects it; only the first max_seq_len rows are executable here.
        position_embedding = model.node_modules["pos_embed"].embedding.weight
        position_embedding.zero_()
        position_embedding[: info.max_seq_len].copy_(weights["transformer.wpe.weight"])
        consumed.add("transformer.wpe.weight")

        copy_parameter(
            model.node_modules["final_norm"].norm.weight,
            weights["transformer.ln_f.weight"],
            "transformer.ln_f.weight",
        )
        copy_parameter(
            model.node_modules["final_norm"].norm.bias,
            weights["transformer.ln_f.bias"],
            "transformer.ln_f.bias",
        )

        for layer in range(info.num_layers):
            prefix = f"transformer.h.{layer}"
            block = model.node_modules[f"block_{layer}"]
            attention = block.node_modules["attention"]
            mlp = block.node_modules["mlp"]

            for suffix, parameter in (
                ("ln_1.weight", block.node_modules["attn_norm"].norm.weight),
                ("ln_1.bias", block.node_modules["attn_norm"].norm.bias),
                ("attn.c_proj.weight", attention.node_modules["out_proj"].proj.weight),
                ("attn.c_proj.bias", attention.node_modules["out_proj"].proj.bias),
                ("ln_2.weight", block.node_modules["mlp_norm"].norm.weight),
                ("ln_2.bias", block.node_modules["mlp_norm"].norm.bias),
                ("mlp.c_fc.weight", mlp.node_modules["fc1"].proj.weight),
                ("mlp.c_fc.bias", mlp.node_modules["fc1"].proj.bias),
                ("mlp.c_proj.weight", mlp.node_modules["fc2"].proj.weight),
                ("mlp.c_proj.bias", mlp.node_modules["fc2"].proj.bias),
            ):
                native_name = f"{prefix}.{suffix}"
                copy_parameter(parameter, weights[native_name], native_name)

            packed_weight_name = f"{prefix}.attn.c_attn.weight"
            packed_bias_name = f"{prefix}.attn.c_attn.bias"
            packed_weight = weights[packed_weight_name]
            packed_bias = weights[packed_bias_name]
            for projection, index in (("q_proj", 0), ("k_proj", 1), ("v_proj", 2)):
                stage = attention.node_modules[projection].proj
                copy_parameter(
                    stage.weight,
                    packed_weight[index * channels : (index + 1) * channels],
                    packed_weight_name,
                )
                copy_parameter(
                    stage.bias,
                    packed_bias[index * channels : (index + 1) * channels],
                    packed_bias_name,
                )

    assert consumed == set(weights), "the graph import must account for every dense-v5 tensor"
    compiled.eval()
    return compiled


def _resident_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    z_loss_coef: float,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    if z_loss_coef <= 0.0:
        return F.cross_entropy(flat_logits, flat_targets)
    log_z = torch.logsumexp(flat_logits, dim=-1)
    target_logits = flat_logits.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
    return (log_z - target_logits).mean() + z_loss_coef * log_z.square().mean()


@pytest.mark.parametrize(
    ("preset", "use_qk_norm", "logit_softcap", "z_loss_coef"),
    (
        ("gpt2", False, 0.0, 0.0),
        ("gpt2_megakernel", False, 0.0, 0.0),
        ("gpt2_zloss", False, 0.0, 1.0e-4),
        ("gpt2_qknorm", True, 0.0, 0.0),
        ("gpt2_stable", True, 0.0, 1.0e-4),
        ("gpt2_softcap", False, 30.0, 0.0),
    ),
)
def test_migrated_dense_graph_matches_resident_logits_loss_and_formula(
    resident_binding: ModuleType,
    tmp_path: Path,
    preset: str,
    use_qk_norm: bool,
    logit_softcap: float,
    z_loss_coef: float,
) -> None:
    """Compare the same migrated bytes through graph, resident, and formula paths."""

    checkpoint = tmp_path / f"{preset}.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True, **GEOMETRY)
    spec = build_model_spec_from_config(
        {
            "preset": preset,
            "num_layers": GEOMETRY["num_layers"],
            "model_dim": GEOMETRY["channels"],
            "num_heads": GEOMETRY["num_heads"],
            "num_kv_heads": GEOMETRY["num_heads"],
            "multiple_of": 1,
            "vocab_size": GEOMETRY["vocab_size"],
        },
        preview_defaults=True,
    )
    graph_path = tmp_path / f"{preset}.json"
    graph_path.write_text(
        json.dumps(build_gpt_root_graph(name=preset, model_spec=spec).to_dict()),
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact"
    migration = migrate_graph_to_native(
        graph_path,
        weights_path=checkpoint,
        output_dir=artifact,
    )
    assert migration.report.compatible is True
    assert migration.manifest is not None
    manifest = migration.manifest.to_dict()
    artifact_checkpoint = artifact / "model.bin"
    assert manifest["source_graph"]["sha256"] == hashlib.sha256(
        graph_path.read_bytes()
    ).hexdigest()
    assert manifest["checkpoint"]["target_sha256"] == hashlib.sha256(
        artifact_checkpoint.read_bytes()
    ).hexdigest()
    assert artifact_checkpoint.read_bytes() == checkpoint.read_bytes()
    assert manifest["model"]["template_spec"]["z_loss_coef"] == pytest.approx(
        z_loss_coef
    )

    migrated_graph = NeuronGraph.from_dict(json.loads(graph_path.read_bytes()))
    compiled = _compiled_dense_with_native_weights(migrated_graph, artifact_checkpoint)
    prompt = [1, 7, 3, 9]
    targets = torch.tensor([[7, 3, 9, 2]], dtype=torch.long)
    with torch.no_grad():
        (graph_loss,), trace = compiled.trace(
            torch.tensor([prompt], dtype=torch.long),
            targets,
        )
    logits_node = "model/softcap" if logit_softcap > 0.0 else "model/tied_lm_head"
    graph_logits = trace[logits_node][0].squeeze(0)

    model = resident_binding.load_model(str(artifact), manifest)
    session = resident_binding.create_session(
        model, {"seed": 41, "kv_cache": {"effective_mode": "off"}}
    )
    try:
        resident_rows: list[list[float]] = []
        for position, token_id in enumerate(prompt):
            resident_binding.prefill(model, session, [token_id], position)
            resident_rows.append(resident_binding.current_logits(model, session))
    finally:
        resident_binding.close_session(model, session)
        resident_binding.close_model(model)

    resident_logits = torch.tensor(resident_rows, dtype=torch.float32)
    oracle_logits = torch.tensor(
        [
            _python_dense_logits(
                artifact_checkpoint,
                prompt[: position + 1],
                use_qk_norm=use_qk_norm,
                logit_softcap=logit_softcap,
            )
            for position in range(len(prompt))
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(resident_logits, oracle_logits, rtol=0.0, atol=4.0e-6)
    torch.testing.assert_close(graph_logits, resident_logits, rtol=1.0e-5, atol=4.0e-6)

    resident_loss = _resident_objective(
        resident_logits,
        targets,
        z_loss_coef=z_loss_coef,
    )
    torch.testing.assert_close(graph_loss, resident_loss, rtol=1.0e-6, atol=2.0e-6)


@pytest.mark.parametrize(
    "selected_activation",
    NATIVE_MOA_CANDIDATE_ACTIVATIONS,
)
def test_source_bound_moa_graph_runtime_matches_resident_logits_loss_and_formula(
    resident_binding: ModuleType,
    tmp_path: Path,
    selected_activation: str,
) -> None:
    """Apply committed MoA semantics without rewriting the authoring graph."""

    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "model_00000001.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True, **GEOMETRY)
    spec = build_model_spec_from_config(
        {
            "preset": "gpt2_moa",
            "num_layers": GEOMETRY["num_layers"],
            "model_dim": GEOMETRY["channels"],
            "num_heads": GEOMETRY["num_heads"],
            "num_kv_heads": GEOMETRY["num_heads"],
            "multiple_of": 1,
            "vocab_size": GEOMETRY["vocab_size"],
        },
        preview_defaults=True,
    )
    graph_path = source / "gpt2-moa.json"
    graph_path.write_text(
        json.dumps(build_gpt_root_graph(name="gpt2_moa", model_spec=spec).to_dict()),
        encoding="utf-8",
    )
    source_graph_bytes = graph_path.read_bytes()
    metadata = _write_moa_metadata(
        checkpoint,
        graph_path,
        selected_activation=selected_activation,
    )
    artifact = tmp_path / "artifact"
    migration = migrate_graph_to_native(
        graph_path,
        weights_path=metadata,
        output_dir=artifact,
    )
    assert migration.report.compatible is True
    assert migration.manifest is not None
    manifest = migration.manifest.to_dict()
    artifact_checkpoint = artifact / "model.bin"
    assert graph_path.read_bytes() == source_graph_bytes
    assert manifest["source_graph"]["sha256"] == hashlib.sha256(
        source_graph_bytes
    ).hexdigest()
    assert manifest["checkpoint"]["moa"]["selected_activation"] == selected_activation
    assert manifest["checkpoint"]["moa"]["metadata_artifact_path"] == "model.moa.json"
    assert manifest["checkpoint"]["moa"]["metadata_nbytes"] == metadata.stat().st_size
    assert artifact_checkpoint.read_bytes() == checkpoint.read_bytes()
    assert (artifact / "model.moa.json").read_bytes() == metadata.read_bytes()

    runtime = load_native_moa_graph_runtime(graph_path, artifact)
    assert graph_path.read_bytes() == source_graph_bytes
    assert runtime.binding.profile == NATIVE_MOA_GRAPH_RUNTIME_PROFILE
    assert runtime.binding.selected_activation == selected_activation
    assert runtime.binding.source_graph_sha256 == hashlib.sha256(
        source_graph_bytes
    ).hexdigest()
    assert runtime.binding.checkpoint_sha256 == hashlib.sha256(
        artifact_checkpoint.read_bytes()
    ).hexdigest()
    assert runtime.binding.activation_node_paths == (
        "model/block_0/mlp/gelu",
        "model/block_1/mlp/gelu",
    )
    # The source graph remains literal GELU; only the compiled checkpoint-bound
    # runtime carries the selected activation overlay.
    source_model = runtime.graph.nodes["model"].neuron_def.subgraph
    assert source_model is not None
    source_block = source_model.nodes["block_0"].neuron_def.subgraph
    assert source_block is not None
    source_mlp = source_block.nodes["mlp"].neuron_def.subgraph
    assert source_mlp is not None
    assert source_mlp.nodes["gelu"].neuron_def.module_type == "gelu"

    prompt = [1, 7, 3, 9]
    targets = torch.tensor([[7, 3, 9, 2]], dtype=torch.long)
    with torch.no_grad():
        (graph_loss,), trace = runtime.trace(
            torch.tensor([prompt], dtype=torch.long),
            targets,
        )
    graph_logits = trace["model/tied_lm_head"][0].squeeze(0)

    model = resident_binding.load_model(str(artifact), manifest)
    session = resident_binding.create_session(
        model, {"seed": 41, "kv_cache": {"effective_mode": "off"}}
    )
    try:
        resident_rows: list[list[float]] = []
        for position, token_id in enumerate(prompt):
            resident_binding.prefill(model, session, [token_id], position)
            resident_rows.append(resident_binding.current_logits(model, session))
    finally:
        resident_binding.close_session(model, session)
        resident_binding.close_model(model)

    resident_logits = torch.tensor(resident_rows, dtype=torch.float32)
    oracle_logits = torch.tensor(
        [
            _python_dense_logits(
                artifact_checkpoint,
                prompt[: position + 1],
                use_qk_norm=False,
                logit_softcap=0.0,
                mlp_activation=selected_activation,
            )
            for position in range(len(prompt))
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(resident_logits, oracle_logits, rtol=0.0, atol=4.0e-6)
    torch.testing.assert_close(graph_logits, resident_logits, rtol=1.0e-5, atol=4.0e-6)
    resident_loss = _resident_objective(
        resident_logits,
        targets,
        z_loss_coef=0.0,
    )
    torch.testing.assert_close(graph_loss, resident_loss, rtol=1.0e-6, atol=2.0e-6)


def test_source_bound_moa_graph_runtime_rejects_graph_and_manifest_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "model_00000001.bin"
    _write_tiny_dense_v5(checkpoint, nontrivial=True, **GEOMETRY)
    spec = build_model_spec_from_config(
        {
            "preset": "gpt2_moa",
            "num_layers": GEOMETRY["num_layers"],
            "model_dim": GEOMETRY["channels"],
            "num_heads": GEOMETRY["num_heads"],
            "num_kv_heads": GEOMETRY["num_heads"],
            "multiple_of": 1,
            "vocab_size": GEOMETRY["vocab_size"],
        },
        preview_defaults=True,
    )
    graph_path = source / "gpt2-moa.json"
    graph_path.write_text(
        json.dumps(build_gpt_root_graph(name="gpt2_moa", model_spec=spec).to_dict()),
        encoding="utf-8",
    )
    metadata = _write_moa_metadata(
        checkpoint,
        graph_path,
        selected_activation="relu2",
    )
    artifact = tmp_path / "artifact"
    migrate_graph_to_native(
        graph_path,
        weights_path=metadata,
        output_dir=artifact,
    )

    original_graph = graph_path.read_bytes()
    graph_path.write_bytes(original_graph + b"\n")
    with pytest.raises(NativeMoaGraphRuntimeError, match="source graph SHA-256"):
        load_native_moa_graph_runtime(graph_path, artifact)
    graph_path.write_bytes(original_graph)

    runtime = load_native_moa_graph_runtime(graph_path, artifact)
    too_long = torch.zeros((1, GEOMETRY["max_seq_len"] + 1), dtype=torch.long)
    with pytest.raises(NativeMoaGraphRuntimeError, match="context limit"):
        runtime.forward(too_long, too_long)

    manifest_path = artifact / "native-execution-manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest["checkpoint"]["moa"]["selected_activation"] = "relu"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(NativeMoaGraphRuntimeError, match="bound metadata artifact"):
        load_native_moa_graph_runtime(graph_path, artifact)

    manifest_path.write_bytes(original_manifest)
    manifest = json.loads(original_manifest)
    manifest["checkpoint"]["target_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(NativeMoaGraphRuntimeError, match="checkpoint bytes"):
        load_native_moa_graph_runtime(graph_path, artifact)

    manifest_path.write_bytes(original_manifest)
    manifest = json.loads(original_manifest)
    manifest["tensors"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(NativeMoaGraphRuntimeError, match="tensor table entry 0"):
        load_native_moa_graph_runtime(graph_path, artifact)

    manifest_path.write_bytes(
        b'{"schema":"duplicate",' + original_manifest[1:]
    )
    with pytest.raises(NativeMoaGraphRuntimeError, match="duplicate object key 'schema'"):
        load_native_moa_graph_runtime(graph_path, artifact)
