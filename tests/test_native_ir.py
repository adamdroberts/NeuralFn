from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from neuralfn.graph import Edge, NeuronGraph, NeuronInstance
from neuralfn.config import SHIPPED_GPT_TEMPLATE_PRESETS
from neuralfn.native_ir import (
    NATIVE_EXECUTION_MANIFEST_SCHEMA,
    NATIVE_EXECUTION_MANIFEST_VERSION,
    NATIVE_TENSOR_BUNDLE_FORMAT,
    NativeExecutionManifest,
    NativeTensorSpec,
    _capabilities,
    _kernel_abi,
    compile_graph_to_native_manifest,
    compile_native_graph_payload,
    migrate_graph_to_native,
)
from neuralfn.native_registry import capability_proof_for
from neuralfn.neuron import neuron_from_source
from neuralfn.port import Port
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


def _terminal(name: str):
    port = Port("x", range=(-1_000.0, 1_000.0), precision=0.001, dtype="float")
    return neuron_from_source(
        f"def {name}(x):\n    return x\n",
        name,
        [port],
        [port],
    )


def test_constrained_responses_capabilities_require_exact_artifact_profiles() -> None:
    proof = SimpleNamespace(
        resident_inference_proven=True,
        model_family="gpt2",
        evidence=(),
        turboquant_cache_proven=False,
        native_ir_lowering=True,
        trainer_registered=True,
        architecture_persistence_proven=True,
        native_forward_proven=True,
        lossless_cache_proven=True,
        serving_proven=True,
    )
    checkpoint = {
        "format": "neuralfn.native_dense_gpt.v5",
        "artifact_path": "model.bin",
    }
    tokenizer = {
        "constrained_decoding": {
            "version": 1,
            "profile": "json-schema-ascii-byte-greedy-v1",
            "token_selection": "current_logits_exact_prefill",
        }
    }
    chat_template = {
        "tool_template": {
            "version": 1,
            "profile": "responses-forced-function-call-v1",
        }
    }

    capabilities = _capabilities(
        proof,
        compatible=True,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
    abi = _kernel_abi(
        proof,
        checkpoint,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
    assert capabilities["structured_output"] is True
    assert capabilities["function_tools"] is True
    assert capabilities["session_prefix_cow"] is True
    assert capabilities["session_prefix_cow_cpu_turboquant"] is False
    assert abi["session_prefix_cow"] == {
        "version": 1,
        "status": "ready",
        "profile": "dense-full-cache-kv-final-hidden-v1",
        "operation": "fork_session",
    }
    assert abi["session_prefix_cow_cpu_turboquant"] == {
        "version": None,
        "status": "not_implemented",
        "profile": None,
        "operation": None,
        "backend": None,
    }
    assert abi["structured_output"] == {
        "version": 1,
        "status": "ready",
        "profile": "json-schema-ascii-byte-greedy-v1",
        "token_selection": "current_logits_exact_prefill",
    }
    assert abi["function_tools"]["status"] == "ready"

    turboquant_proof = SimpleNamespace(**vars(proof))
    turboquant_proof.turboquant_cache_proven = True
    assert _capabilities(
        turboquant_proof,
        compatible=True,
        checkpoint=checkpoint,
    )["session_prefix_cow_cpu_turboquant"] is True
    assert _kernel_abi(
        turboquant_proof,
        checkpoint,
    )["session_prefix_cow_cpu_turboquant"] == {
        "version": 1,
        "status": "ready",
        "profile": "dense-cpu-turboquant-mse-qjl-packed-kv-final-hidden-v1",
        "operation": "fork_session",
        "backend": "cpu-reference-packed",
    }

    for non_dense_format in (
        "neuralfn.native_family_llama.f32.v1",
        "neuralfn.native_family_standard_moe.f32.v1",
    ):
        non_dense_checkpoint = {
            "format": non_dense_format,
            "artifact_path": "model.bin",
        }
        assert _capabilities(
            proof,
            compatible=True,
            checkpoint=non_dense_checkpoint,
        )["session_prefix_cow"] is False
        assert _kernel_abi(proof, non_dense_checkpoint)["session_prefix_cow"] == {
            "version": None,
            "status": "not_implemented",
            "profile": None,
            "operation": None,
        }
        assert _capabilities(
            turboquant_proof,
            compatible=True,
            checkpoint=non_dense_checkpoint,
        )["session_prefix_cow_cpu_turboquant"] is False

    for model_family, checkpoint_format, expected_profile in (
        (
            "llama",
            "neuralfn.native_family_llama.f32.v1",
            "llama-full-cache-gqa-kv-final-hidden-v1",
        ),
        (
            "mixllama",
            "neuralfn.native_family_standard_moe.f32.v1",
            "standard-moe-full-cache-gqa-kv-final-hidden-v1",
        ),
    ):
        family_proof = SimpleNamespace(**vars(proof))
        family_proof.model_family = model_family
        family_checkpoint = {
            "format": checkpoint_format,
            "artifact_path": "model.f32",
        }
        assert _capabilities(
            family_proof,
            compatible=True,
            checkpoint=family_checkpoint,
        )["session_prefix_cow"] is True
        assert _kernel_abi(family_proof, family_checkpoint)["session_prefix_cow"] == {
            "version": 1,
            "status": "ready",
            "profile": expected_profile,
            "operation": "fork_session",
        }

        wrong_family = SimpleNamespace(**vars(family_proof))
        wrong_family.model_family = "mixllama" if model_family == "llama" else "llama"
        assert _capabilities(
            wrong_family,
            compatible=True,
            checkpoint=family_checkpoint,
        )["session_prefix_cow"] is False

    mismatched = deepcopy(tokenizer)
    mismatched["constrained_decoding"]["profile"] = "future-profile"
    closed = _capabilities(
        proof,
        compatible=True,
        checkpoint=checkpoint,
        tokenizer=mismatched,
        chat_template=chat_template,
    )
    assert closed["structured_output"] is False
    assert closed["function_tools"] is False

    for invalid_version in (True, 1.0):
        malformed_tokenizer = deepcopy(tokenizer)
        malformed_tokenizer["constrained_decoding"]["version"] = invalid_version
        closed = _capabilities(
            proof,
            compatible=True,
            checkpoint=checkpoint,
            tokenizer=malformed_tokenizer,
            chat_template=chat_template,
        )
        assert closed["structured_output"] is False
        assert closed["function_tools"] is False

    extra_tokenizer = deepcopy(tokenizer)
    extra_tokenizer["constrained_decoding"]["future"] = True
    assert _capabilities(
        proof,
        compatible=True,
        checkpoint=checkpoint,
        tokenizer=extra_tokenizer,
        chat_template=chat_template,
    )["structured_output"] is False

    for invalid_tool_template in (
        {"version": True, "profile": "responses-forced-function-call-v1"},
        {
            "version": 1,
            "profile": "responses-forced-function-call-v1",
            "future": True,
        },
    ):
        closed = _capabilities(
            proof,
            compatible=True,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            chat_template={"tool_template": invalid_tool_template},
        )
        assert closed["structured_output"] is True
        assert closed["function_tools"] is False


def _passthrough_graph(*, include_custom_function: bool = False) -> NeuronGraph:
    graph = NeuronGraph(name="native_ir_passthrough")
    graph.add_node(NeuronInstance(_terminal("graph_input"), instance_id="input"))
    graph.add_node(NeuronInstance(_terminal("graph_output"), instance_id="output"))
    graph.input_node_ids = ["input"]
    graph.output_node_ids = ["output"]

    if include_custom_function:
        graph.add_node(NeuronInstance(_terminal("custom_step"), instance_id="custom"))
        graph.add_edge(
            Edge(
                id="input_to_custom",
                src_node="input",
                src_port=0,
                dst_node="custom",
                dst_port=0,
            )
        )
        graph.add_edge(
            Edge(
                id="custom_to_output",
                src_node="custom",
                src_port=0,
                dst_node="output",
                dst_port=0,
            )
        )
    else:
        graph.add_edge(
            Edge(
                id="input_to_output",
                src_node="input",
                src_port=0,
                dst_node="output",
                dst_port=0,
            )
        )
    graph.validate()
    return graph


def _tiny_gpt2_graph() -> NeuronGraph:
    spec = build_model_spec_from_config(
        {
            "preset": "gpt2",
            "num_layers": 1,
            "model_dim": 16,
            "num_heads": 4,
            "num_kv_heads": 4,
            "multiple_of": 8,
            "vocab_size": 32,
        },
        preview_defaults=True,
    )
    return build_gpt_root_graph(name="native_ir_gpt2", model_spec=spec)


def _tiny_preset_graph(preset: str) -> NeuronGraph:
    spec = build_model_spec_from_config(
        {
            "preset": preset,
            "num_layers": 1,
            "model_dim": 16,
            "num_heads": 4,
            "num_kv_heads": 4,
            "multiple_of": 8,
            "vocab_size": 32,
        },
        preview_defaults=True,
    )
    return build_gpt_root_graph(name=f"native_ir_{preset}", model_spec=spec)


def _write_graph(path: Path, graph: NeuronGraph) -> bytes:
    # Deliberately retain non-canonical formatting so byte-for-byte source
    # preservation is stronger than an object-level equality check.
    raw = (json.dumps(graph.to_dict(), indent=1, sort_keys=False) + "\n  \n").encode()
    path.write_bytes(raw)
    return raw


def test_native_tensor_spec_emits_layout_and_reads_legacy_default() -> None:
    legacy = {
        "name": "weight",
        "source_name": "weight",
        "dtype": "float32",
        "shape": [2, 3],
        "offset": 0,
        "nbytes": 24,
        "sha256": "a" * 64,
        "role": "parameter",
        "byte_order": "little",
    }

    tensor = NativeTensorSpec.from_dict(legacy)

    assert tensor.layout == "row_major"
    assert tensor.to_dict()["layout"] == "row_major"
    column_major = NativeTensorSpec.from_dict({**legacy, "layout": "column_major"})
    assert column_major.layout == "column_major"
    assert column_major.to_dict()["layout"] == "column_major"


def test_manifest_is_deterministic_round_trips_and_preserves_source_bytes(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    original_bytes = _write_graph(graph_path, _tiny_gpt2_graph())

    first = migrate_graph_to_native(
        graph_path,
        output_dir=tmp_path / "dry-run-one",
        dry_run=True,
    )
    second = migrate_graph_to_native(
        graph_path,
        output_dir=tmp_path / "dry-run-two",
        dry_run=True,
    )

    assert first.report.compatible
    assert first.manifest is not None
    assert second.manifest is not None
    assert first.output_dir is None
    assert not (tmp_path / "dry-run-one").exists()
    assert not (tmp_path / "dry-run-two").exists()
    assert graph_path.read_bytes() == original_bytes
    first_payload = first.manifest.to_dict()
    second_payload = second.manifest.to_dict()
    assert first_payload == second_payload
    assert first_payload["schema"] == NATIVE_EXECUTION_MANIFEST_SCHEMA
    assert first_payload["version"] == NATIVE_EXECUTION_MANIFEST_VERSION
    assert first_payload["topology"]["resolved"] is True
    assert first_payload["topology"]["variant_graphs"]
    assert first_payload["source_graph"]["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert first_payload["source_graph"]["serialization_changed"] is False

    serialized = json.dumps(first_payload, sort_keys=True)
    assert NativeExecutionManifest.from_dict(json.loads(serialized)) == first.manifest

    payload_manifest = compile_native_graph_payload(json.loads(original_bytes))
    assert payload_manifest.topology == first.manifest.topology
    assert payload_manifest.model == first.manifest.model
    assert payload_manifest.capabilities == first.manifest.capabilities

    output_dir = tmp_path / "artifact"
    materialized = migrate_graph_to_native(graph_path, output_dir=output_dir)
    assert materialized.output_dir == output_dir.resolve()
    assert NativeExecutionManifest.load(output_dir / "native-execution-manifest.json") == first.manifest
    assert graph_path.read_bytes() == original_bytes


def test_dense_resident_proof_is_topology_specific_and_unbound_manifests_stay_closed() -> None:
    plain = compile_native_graph_payload(_tiny_gpt2_graph().to_dict())
    proof = capability_proof_for(plain.model, plain.topology)
    assert proof.resident_inference_proven is True
    assert proof.lossless_cache_proven is True
    assert "resident-dense-in-process-abi-v1" in proof.evidence
    assert plain.capabilities["resident_inference"] is False
    assert plain.capabilities["lossless_kv_cache"] is False
    assert plain.kernel_abi["resident_inference"] == {
        "version": None,
        "status": "not_implemented",
    }

    differential = compile_native_graph_payload(_tiny_preset_graph("gpt2_diff").to_dict())
    differential_proof = capability_proof_for(
        differential.model, differential.topology
    )
    assert differential.capabilities["native_ir"] is True
    assert differential.capabilities["native_training"] is False
    assert differential.capabilities["architecture_persistence"] is False
    assert differential.capabilities["native_inference"] is False
    assert differential_proof.native_ir_lowering is True
    assert differential_proof.architecture_persistence_proven is False
    assert differential_proof.native_forward_proven is False
    assert differential_proof.resident_inference_proven is False
    assert differential_proof.lossless_cache_proven is False
    assert (
        "architecture_parameter_persistence_proof"
        in differential_proof.missing_gates
    )
    assert "real_native_architecture_forward" in differential_proof.missing_gates
    assert "dense-native-checkpoint-and-forward-v5" not in differential_proof.evidence
    assert (
        "gpt2-diff-low-level-learned-lambda-training-bundle-v2-proven"
        in differential_proof.evidence
    )
    assert (
        "gpt2-diff-materialized-graph-training-proof-v1"
        in differential_proof.evidence
    )
    assert (
        "gpt2-diff-native-ir-migration-resident-bundle-consumer-unimplemented"
        in differential_proof.evidence
    )
    assert (
        "gpt2-diff-resident-differential-forward-unimplemented"
        in differential_proof.evidence
    )

    moa = compile_native_graph_payload(_tiny_preset_graph("gpt2_moa").to_dict())
    moa_proof = capability_proof_for(moa.model, moa.topology)
    assert moa_proof.resident_inference_proven is True
    assert moa_proof.lossless_cache_proven is True
    assert moa_proof.turboquant_cache_proven is True
    assert "native-dense-moa-source-bound-checkpoint-v1" in moa_proof.evidence
    assert "resident-dense-moa-selected-activation-v1" in moa_proof.evidence

    for preset in ("gpt2_qknorm", "gpt2_softcap", "gpt2_stable"):
        manifest = compile_native_graph_payload(_tiny_preset_graph(preset).to_dict())
        variant_proof = capability_proof_for(manifest.model, manifest.topology)
        assert variant_proof.resident_inference_proven is True, preset
        assert variant_proof.lossless_cache_proven is True, preset
        assert variant_proof.turboquant_cache_proven is True, preset


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("moa_activations", ["gelu", "relu"]),
        ("moa_activations", ["relu", "gelu", "silu", "relu2"]),
        ("moa_interval", 0),
        ("moa_interval", True),
    ],
)
def test_moa_resident_proof_requires_exact_selection_contract(
    field: str,
    value: object,
) -> None:
    manifest = compile_native_graph_payload(_tiny_preset_graph("gpt2_moa").to_dict())
    model = deepcopy(manifest.model)
    model["template_spec"]["block_spec"][field] = value
    proof = capability_proof_for(model, manifest.topology)
    assert proof.resident_inference_proven is False
    assert proof.lossless_cache_proven is False
    assert proof.turboquant_cache_proven is False


@pytest.mark.parametrize(
    ("preset", "operation", "field", "bad_value"),
    [
        ("gpt2_qknorm", "qk_norm", "eps", 1.0e-5),
        ("gpt2_softcap", "logit_softcap", "softcap", 29.0),
    ],
)
def test_dense_parameter_free_resident_variants_require_exact_active_topology(
    preset: str,
    operation: str,
    field: str,
    bad_value: float,
) -> None:
    manifest = compile_native_graph_payload(_tiny_preset_graph(preset).to_dict())
    topology = deepcopy(manifest.topology)
    mutated = False
    for graph in topology["graphs"]:
        path = str(graph.get("path") or "")
        if path != "root" and not path.startswith("root/"):
            continue
        for node in graph.get("nodes", []):
            if node.get("operation") == operation:
                node["module_config"][field] = bad_value
                mutated = True
                break
        if mutated:
            break
    assert mutated
    proof = capability_proof_for(manifest.model, topology)
    assert proof.resident_inference_proven is False
    assert proof.lossless_cache_proven is False
    assert proof.turboquant_cache_proven is False


@pytest.mark.parametrize("preset", ["gpt2_qknorm", "gpt2_softcap"])
def test_dense_parameter_free_resident_proof_rejects_dataflow_bypass(
    preset: str,
) -> None:
    payload = _tiny_preset_graph(preset).to_dict()
    rewired_edges = 0

    def rewrite_nested_graphs(value) -> None:
        nonlocal rewired_edges
        if isinstance(value, dict):
            edges = value.get("edges")
            if isinstance(edges, dict):
                for edge in edges.values():
                    if preset == "gpt2_qknorm":
                        if edge.get("src_node") != "qk_norm" or edge.get("dst_node") != "sdpa":
                            continue
                        edge["src_node"] = "q_heads" if edge.get("src_port") == 0 else "k_heads"
                        edge["src_port"] = 0
                        rewired_edges += 1
                    elif (
                        edge.get("src_node") == "softcap"
                        and edge.get("dst_node") == "ce"
                        and edge.get("dst_port") == 0
                    ):
                        edge["src_node"] = "tied_lm_head"
                        edge["src_port"] = 0
                        rewired_edges += 1
            for child in value.values():
                rewrite_nested_graphs(child)
        elif isinstance(value, list):
            for child in value:
                rewrite_nested_graphs(child)

    rewrite_nested_graphs(payload)
    assert rewired_edges > 0

    manifest = compile_native_graph_payload(payload)
    proof = capability_proof_for(manifest.model, manifest.topology)

    assert proof.resident_inference_proven is False
    assert proof.lossless_cache_proven is False
    assert proof.turboquant_cache_proven is False


@pytest.mark.parametrize(
    ("container", "field"),
    [
        ("block_spec", "use_qk_norm"),
        ("template_spec", "logit_softcap"),
        ("template_spec", "tie_embeddings"),
        ("block_spec", "linear_bias"),
        ("block_spec", "dropout_p"),
        ("block_spec", "activation_mode"),
    ],
)
def test_dense_resident_proof_requires_explicit_contract_fields(
    container: str,
    field: str,
) -> None:
    payload = _tiny_gpt2_graph().to_dict()
    template_spec = payload["torch_config"]["template_spec"]
    target = template_spec if container == "template_spec" else template_spec["block_spec"]
    del target[field]

    manifest = compile_native_graph_payload(payload)
    proof = capability_proof_for(manifest.model, manifest.topology)

    assert proof.resident_inference_proven is False
    assert proof.lossless_cache_proven is False
    assert proof.turboquant_cache_proven is False


def test_arbitrary_serialized_function_source_is_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "source-executed"
    graph = _passthrough_graph(include_custom_function=True)
    payload = graph.to_dict()
    payload["nodes"]["custom"]["neuron_def"]["source_code"] = (
        f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')\n"
        "def custom_step(x):\n"
        "    return x\n"
    )
    graph_path = tmp_path / "malicious-graph.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    result = migrate_graph_to_native(
        graph_path,
        output_dir=tmp_path / "must-not-exist",
        dry_run=True,
    )

    assert not marker.exists()
    assert not result.report.compatible
    assert any(issue.code == "unsupported_function" for issue in result.report.issues)
    assert "root/nodes/custom" in result.report.unsupported_node_paths
    assert not (tmp_path / "must-not-exist").exists()


def test_compatibility_rejection_happens_before_optional_checkpoint_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import neuralfn.inference as inference

    graph_path = tmp_path / "unsupported-graph.json"
    _write_graph(graph_path, _passthrough_graph(include_custom_function=True))
    checkpoint_path = tmp_path / "weights.pt"
    checkpoint_path.write_bytes(b"deliberately not a torch checkpoint")
    load_calls: list[Path] = []

    def fail_if_loaded(path, **_kwargs):
        load_calls.append(Path(path))
        raise AssertionError("checkpoint was opened before graph compatibility completed")

    monkeypatch.setattr(inference, "load_pt_checkpoint", fail_if_loaded)

    result = migrate_graph_to_native(
        graph_path,
        weights_path=checkpoint_path,
        output_dir=tmp_path / "artifact",
    )

    assert not result.report.compatible
    assert load_calls == []
    assert not (tmp_path / "artifact").exists()


def test_dense_moa_migration_rejects_unbound_raw_checkpoint(tmp_path: Path) -> None:
    from tests.test_native_moa_checkpoint import _write_bundle

    metadata, graph_path, _model, _payload = _write_bundle(tmp_path / "bundle")
    checkpoint = metadata.with_suffix("").with_suffix(".bin")
    destination = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="source-bound.*moa\\.json"):
        migrate_graph_to_native(
            graph_path,
            weights_path=checkpoint,
            output_dir=destination,
        )

    assert not destination.exists()


@pytest.mark.parametrize("suffix", (".bin", ".json", ".pt"))
def test_gpt2_diff_migration_rejects_every_unconsumed_checkpoint_shape(
    tmp_path: Path,
    suffix: str,
) -> None:
    graph_path = tmp_path / "gpt2-diff.json"
    _write_graph(graph_path, _tiny_preset_graph("gpt2_diff"))
    checkpoint_path = tmp_path / f"model_00000004{suffix}"
    checkpoint_path.write_bytes(b"unconsumed differential checkpoint")
    destination = tmp_path / "must-not-exist"

    with pytest.raises(
        ValueError,
        match=(
            "Native gpt2_diff migration does not yet consume.*"
            "training_checkpoint version 2"
        ),
    ):
        migrate_graph_to_native(
            graph_path,
            weights_path=checkpoint_path,
            output_dir=destination,
        )

    assert not destination.exists()


@pytest.mark.parametrize(
    ("max_seq_len", "channels", "tile_ready"),
    ((16_384, 2, True), (16_385, 2, False), (8, 258, False)),
)
def test_tile_turboquant_feature_abi_has_separate_cuda_geometry_gate(
    tmp_path: Path,
    max_seq_len: int,
    channels: int,
    tile_ready: bool,
) -> None:
    from tests.test_native_resident_binding import _write_tiny_dense_v5

    spec = build_model_spec_from_config(
        {
            "preset": "gpt2",
            "num_layers": 1,
            "model_dim": channels,
            "num_heads": 1,
            "num_kv_heads": 1,
            "multiple_of": 1,
            "vocab_size": 4,
        },
        preview_defaults=True,
    )
    graph = build_gpt_root_graph(name="tile_turboquant_geometry", model_spec=spec)
    graph_path = tmp_path / f"graph-{max_seq_len}-{channels}.json"
    graph_path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    checkpoint = tmp_path / f"model-{max_seq_len}-{channels}.bin"
    _write_tiny_dense_v5(
        checkpoint,
        max_seq_len=max_seq_len,
        channels=channels,
    )

    result = migrate_graph_to_native(
        graph_path,
        weights_path=checkpoint,
        output_dir=tmp_path / f"artifact-{max_seq_len}-{channels}",
        dry_run=True,
    )

    assert result.manifest is not None
    assert result.manifest.capabilities["turboquant_kv_cache"] is True
    assert result.manifest.capabilities["turboquant_tile_attention"] is tile_ready
    feature = result.manifest.kernel_abi["turboquant_tile_attention"]
    assert feature["symbol"] == "nfn_native_tile_turboquant_attention_forward_v1"
    assert feature["version"] == (1 if tile_ready else None)
    assert feature["status"] == ("ready" if tile_ready else "not_implemented")


def test_existing_output_directory_is_rejected_even_for_dry_run(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    _write_graph(graph_path, _passthrough_graph())
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        migrate_graph_to_native(graph_path, output_dir=output_dir, dry_run=True)


def test_pt_tensors_convert_to_aligned_hashed_native_bundle(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    graph_path = tmp_path / "graph.json"
    _write_graph(graph_path, _passthrough_graph())
    checkpoint_path = tmp_path / "weights.pt"
    state_dict = {
        "z.weight": torch.tensor([[1.0, -2.0], [3.5, 4.25]], dtype=torch.float32),
        "a.bias": torch.tensor([7, -8, 9], dtype=torch.int16),
    }
    torch.save(
        {
            "state_dict": state_dict,
            "checkpoint_metadata": {"fixture": "native-ir"},
        },
        checkpoint_path,
    )

    output_dir = tmp_path / "artifact"
    result = migrate_graph_to_native(
        graph_path,
        weights_path=checkpoint_path,
        output_dir=output_dir,
    )

    assert result.report.compatible
    assert result.manifest is not None
    assert result.manifest.checkpoint is not None
    assert result.manifest.checkpoint["target_format"] == NATIVE_TENSOR_BUNDLE_FORMAT
    assert result.manifest.checkpoint["metadata"] == {"fixture": "native-ir"}
    assert [tensor.source_name for tensor in result.manifest.tensors] == ["a.bias", "z.weight"]

    weights = (output_dir / "weights.bin").read_bytes()
    for tensor in result.manifest.tensors:
        assert tensor.offset % 64 == 0
        expected = (
            state_dict[tensor.source_name]
            .detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes(order="C")
        )
        actual = weights[tensor.offset : tensor.offset + tensor.nbytes]
        assert actual == expected
        assert tensor.sha256 == hashlib.sha256(expected).hexdigest()

    assert result.manifest.checkpoint["target_nbytes"] == len(weights)
    assert result.manifest.checkpoint["target_sha256"] == hashlib.sha256(weights).hexdigest()
    assert NativeExecutionManifest.load(output_dir / "native-execution-manifest.json") == result.manifest
    report_payload = json.loads((output_dir / "compatibility-report.json").read_text())
    assert report_payload["compatible"] is True
    assert [item["source_name"] for item in report_payload["tensor_mappings"]] == [
        "a.bias",
        "z.weight",
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda payload: payload["edges"]["input_to_output"].update(src_node="missing"), "missing_edge_source"),
        (lambda payload: payload["edges"]["input_to_output"].update(dst_port=99), "invalid_edge_port"),
        (lambda payload: payload["nodes"]["input"].update(instance_id="different"), "invalid_instance_id"),
    ],
)
def test_raw_structural_validation_reports_stable_edge_and_node_paths(
    tmp_path: Path,
    mutation,
    expected_code: str,
) -> None:
    payload = _passthrough_graph().to_dict()
    mutation(payload)
    graph_path = tmp_path / "malformed.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    result = migrate_graph_to_native(
        graph_path,
        output_dir=tmp_path / "must-not-exist",
        dry_run=True,
    )

    assert not result.report.graph_valid
    assert not result.report.compatible
    matching = [issue for issue in result.report.issues if issue.code == expected_code]
    assert matching
    assert matching[0].path.startswith("root/")
    assert not (tmp_path / "must-not-exist").exists()


def test_compile_does_not_mutate_programmatic_authoring_graph() -> None:
    graph = _tiny_gpt2_graph()
    before = json.dumps(graph.to_dict(), sort_keys=True)

    manifest = compile_graph_to_native_manifest(graph)

    assert manifest.topology["resolved"] is True
    assert json.dumps(graph.to_dict(), sort_keys=True) == before


def test_existing_broken_symlink_destination_is_rejected(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    _write_graph(graph_path, _passthrough_graph())
    destination = tmp_path / "broken-output-link"
    destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        migrate_graph_to_native(graph_path, output_dir=destination, dry_run=True)


def test_importing_native_ir_keeps_heavy_optional_stacks_unloaded() -> None:
    script = (
        "import sys; import neuralfn.native_ir; "
        "print(','.join(name for name in ('torch','numpy','networkx') if name in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == ""


def test_every_shipped_preset_lowers_all_root_variants_without_unclassified_nodes(
    tmp_path: Path,
) -> None:
    seen_operations: set[str] = set()
    for preset in SHIPPED_GPT_TEMPLATE_PRESETS:
        spec = build_model_spec_from_config({"preset": preset}, preview_defaults=True)
        graph = build_gpt_root_graph(name=f"{preset}_native_ir", model_spec=spec)
        graph_path = tmp_path / f"{preset}.json"
        graph_path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")

        result = migrate_graph_to_native(
            graph_path,
            output_dir=tmp_path / f"{preset}.native",
            dry_run=True,
        )

        assert result.report.graph_valid, (preset, result.report.to_dict())
        assert result.report.compatible, (preset, result.report.to_dict())
        assert result.manifest is not None
        assert result.report.capability_proof["unsupported_modules"] == []
        assert result.report.capability_proof["trainer_registered"] is True
        topology = result.manifest.topology
        expected_variants = {
            f"{family}@{version}"
            for family, versions in graph.variant_library.items()
            for version in versions
        }
        assert set(topology["variant_graphs"]) == expected_variants
        for lowered_graph in topology["graphs"]:
            for node in lowered_graph["nodes"]:
                assert node["operation"]
                seen_operations.add(node["operation"])
        # Torch template capability flags must not make unimplemented native
        # session/cache/serving features appear available.
        assert result.manifest.capabilities["resident_inference"] is False
        assert result.manifest.capabilities["lossless_kv_cache"] is False
        assert result.manifest.capabilities["turboquant_kv_cache"] is False
        assert result.manifest.capabilities["serve"] is False

    # Guard against a vacuous catalog loop or topology that only retained the
    # common boundary nodes.
    assert len(seen_operations) >= 60


def test_new_module_types_fail_closed_until_explicitly_registered(tmp_path: Path) -> None:
    graph = _tiny_gpt2_graph()
    payload = graph.to_dict()
    payload["nodes"]["model"]["neuron_def"]["subgraph"]["nodes"]["token_embed"][
        "neuron_def"
    ]["module_type"] = "future_unreviewed_operator"
    graph_path = tmp_path / "future-module.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    result = migrate_graph_to_native(
        graph_path,
        output_dir=tmp_path / "must-not-exist",
        dry_run=True,
    )

    assert not result.report.compatible
    issues = [issue for issue in result.report.issues if issue.code == "unsupported_module"]
    assert issues
    assert issues[0].operation == "future_unreviewed_operator"
    assert issues[0].path.endswith("/nodes/token_embed")


def test_missing_authoritative_serving_metadata_remains_explicit(tmp_path: Path) -> None:
    graph_path = tmp_path / "metadata-missing.json"
    _write_graph(graph_path, _passthrough_graph())

    result = migrate_graph_to_native(
        graph_path,
        output_dir=tmp_path / "dry-run",
        dry_run=True,
    )

    assert result.manifest is not None
    assert result.manifest.model["family"] == "unknown"
    assert result.manifest.tokenizer == {}
    assert result.manifest.chat_template["source"] == "missing"
    assert result.manifest.context_limits["max_context_tokens"] is None
    assert any("recognized model family" in warning for warning in result.report.warnings)
    assert any("tokenizer metadata" in warning for warning in result.report.warnings)
    assert any("chat template" in warning for warning in result.report.warnings)
    assert any("context limit" in warning for warning in result.report.warnings)


def test_raw_lowering_preserves_inline_dense_fallback_after_moe_variant_overwrite() -> None:
    dense = build_gpt_root_graph(
        model_spec=build_model_spec_from_config({"preset": "gpt2"}, preview_defaults=True)
    ).to_dict()
    moe = build_gpt_root_graph(
        model_spec=build_model_spec_from_config({"preset": "mixllama"}, preview_defaults=True)
    ).to_dict()
    dense["variant_library"].update(deepcopy(moe["variant_library"]))

    manifest = compile_native_graph_payload(dense)

    active_operations = {
        node["operation"]
        for graph in manifest.topology["graphs"]
        if graph["path"] == "root" or graph["path"].startswith("root/")
        for node in graph["nodes"]
    }
    assert "gelu" in active_operations
    assert "expert_dispatch" not in active_operations


def test_raw_variant_resolution_reports_missing_incomplete_and_recursive_refs(
    tmp_path: Path,
) -> None:
    cases: list[tuple[str, dict, str]] = []

    missing = _tiny_gpt2_graph().to_dict()
    del missing["variant_library"]["transformer_block"]
    del missing["variant_library"]["attn_block"]
    cases.append(("missing", missing, "missing_variant"))

    incomplete = _tiny_gpt2_graph().to_dict()
    incomplete["nodes"]["model"]["neuron_def"]["subgraph"]["nodes"]["block_0"][
        "neuron_def"
    ]["variant_ref"]["version"] = ""
    cases.append(("incomplete", incomplete, "incomplete_variant_ref"))

    recursive = _tiny_gpt2_graph().to_dict()
    recursive["variant_library"]["transformer_block"]["default"]["nodes"]["attention"][
        "neuron_def"
    ]["variant_ref"] = {"family": "transformer_block", "version": "default"}
    cases.append(("recursive", recursive, "recursive_variant_ref"))

    for name, payload, expected_code in cases:
        graph_path = tmp_path / f"{name}.json"
        graph_path.write_text(json.dumps(payload), encoding="utf-8")
        result = migrate_graph_to_native(
            graph_path,
            output_dir=tmp_path / f"{name}.native",
            dry_run=True,
        )
        assert not result.report.compatible
        assert any(issue.code == expected_code for issue in result.report.issues), (
            name,
            result.report.to_dict(),
        )
        assert not (tmp_path / f"{name}.native").exists()
