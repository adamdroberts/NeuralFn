from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from neuralfn.config import FineTuneSpec, SHIPPED_GPT_TEMPLATE_PRESETS
from neuralfn.graph import Edge, NeuronGraph, NeuronInstance
from neuralfn.native_graph_train import (
    NativeGraphTrainPlan,
    plan_native_graph_training,
    preflight_native_graph_training,
)
from neuralfn.native_registry import native_trainer_specs
from neuralfn.native_train import build_native_train_run_config
from neuralfn.neuron import neuron_from_source
from neuralfn.port import Port
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


def _preset_graph(
    preset: str,
    *,
    vocab_size: int = 64,
    model_dim: int = 32,
    num_heads: int = 4,
    num_kv_heads: int = 4,
    multiple_of: int | None = 16,
    mlp_multiplier: float | None = None,
    experts: int | None = None,
    top_k: int | None = None,
    router_aux_loss_coef: float | None = None,
) -> NeuronGraph:
    config: dict[str, object] = {
        "preset": preset,
        "num_layers": 1,
        "model_dim": model_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "multiple_of": multiple_of,
        "vocab_size": vocab_size,
    }
    if mlp_multiplier is not None:
        config["mlp_multiplier"] = mlp_multiplier
    if experts is not None:
        config["experts"] = experts
    if top_k is not None:
        config["top_k"] = top_k
    if router_aux_loss_coef is not None:
        config["router_aux_loss_coef"] = router_aux_loss_coef
    spec = build_model_spec_from_config(
        config,
        preview_defaults=True,
    )
    return build_gpt_root_graph(name=f"{preset}_native_train_plan", model_spec=spec)


def _write_graph(path: Path, graph: NeuronGraph) -> None:
    path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")


def _terminal(name: str):
    port = Port("x", range=(-100.0, 100.0), precision=0.001, dtype="float")
    return neuron_from_source(
        f"def {name}(x):\n    return x\n",
        name,
        [port],
        [port],
    )


def _custom_function_graph() -> NeuronGraph:
    graph = NeuronGraph(name="custom_native_train_plan")
    graph.add_node(NeuronInstance(_terminal("graph_input"), instance_id="input"))
    graph.add_node(NeuronInstance(_terminal("custom_step"), instance_id="custom"))
    graph.add_node(NeuronInstance(_terminal("graph_output"), instance_id="output"))
    graph.add_edge(Edge(id="in", src_node="input", src_port=0, dst_node="custom", dst_port=0))
    graph.add_edge(Edge(id="out", src_node="custom", src_port=0, dst_node="output", dst_port=0))
    graph.input_node_ids = ["input"]
    graph.output_node_ids = ["output"]
    graph.validate()
    return graph


def test_dense_plan_routes_to_registered_target_and_materializes_ir(tmp_path: Path) -> None:
    graph_path = tmp_path / "gpt2.json"
    _write_graph(graph_path, _preset_graph("gpt2", vocab_size=50257))
    source_bytes = graph_path.read_bytes()
    artifact_dir = tmp_path / "artifact"

    plan = plan_native_graph_training(
        graph_path,
        artifact_dir=artifact_dir,
        materialize=True,
    )

    assert isinstance(plan, NativeGraphTrainPlan)
    assert plan.trainer_family == "gpt2"
    assert plan.native_target == "nfn_gpt_native_train"
    assert plan.trainer_registered is True
    assert plan.architecture_persistence_proven is True
    assert plan.execution_ready is True
    assert plan.trainer_consumes_native_ir is False
    assert plan.graph_preflight_enforced is True
    assert plan.training_selector == "gpt2"
    assert plan.adapter_mode == "validated-dense-graph-file-v1"
    assert plan.trainer_arguments == (
        "--template-name",
        "gpt2",
        "--num-layers",
        "1",
        "--train-seq-len",
        "1024",
        "--native-cuda-activation",
        "gelu",
    )
    assert plan.compatibility_report.compatible
    assert plan.training_compatible
    assert plan.training_issues == ()
    assert plan.manifest is not None
    assert plan.blockers == ()
    assert plan.artifact_metadata["kind"] == "native-graph-training-preflight"
    assert plan.artifact_metadata["native_ir_schema"] == plan.manifest.schema
    assert plan.artifact_metadata["graph_fingerprint"] == plan.compatibility_report.graph_fingerprint
    assert plan.artifact_metadata["materialized"] is True
    assert plan.artifact_metadata["source_graph_serialization_changed"] is False
    assert plan.artifact_metadata["training_selector"] == "gpt2"
    assert plan.artifact_metadata["execution_ready"] is True
    assert plan.artifact_metadata["trainer_consumes_native_ir"] is False
    assert plan.launch_graph == artifact_dir.resolve() / "source-graph.json"
    assert (artifact_dir / "native-execution-manifest.json").is_file()
    assert (artifact_dir / "compatibility-report.json").is_file()
    assert (artifact_dir / "native-training-plan.json").is_file()
    assert (artifact_dir / "source-graph.json").read_bytes() == source_bytes
    assert not (artifact_dir / "weights.bin").exists()
    persisted_plan = json.loads((artifact_dir / "native-training-plan.json").read_text(encoding="utf-8"))
    assert persisted_plan["execution_ready"] is True
    assert persisted_plan["launch_graph"] == str(artifact_dir.resolve() / "source-graph.json")
    assert plan.to_dict()["training_compatibility"]["compatible"] is True

    with pytest.raises(FrozenInstanceError):
        plan.execution_ready = False  # type: ignore[misc]


def test_public_graph_training_adapter_registry_is_stable() -> None:
    from neuralfn import (
        NativeGraphTrainingAdapter,
        classify_native_graph_training_selector,
        native_graph_training_adapters,
    )

    adapters = native_graph_training_adapters()

    assert all(isinstance(adapter, NativeGraphTrainingAdapter) for adapter in adapters)
    assert tuple(adapter.selector for adapter in adapters) == (
        "gpt2",
        "gpt2_diff",
        "gpt2_megakernel",
        "gpt2_moa",
        "gpt2_qknorm",
        "gpt2_softcap",
        "gpt2_stable",
        "gpt2_zloss",
        "llama",
        "llama_fast",
        "moe",
        "mixllama",
        "mixllama_fast",
        "muse_glimmer",
    )
    assert callable(classify_native_graph_training_selector)
    assert all(adapter.trainer_consumes_native_ir is False for adapter in adapters)
    differential = next(
        adapter for adapter in adapters if adapter.selector == "gpt2_diff"
    )
    assert differential.architecture_persistence_proven is False
    assert (
        "gpt2-diff-low-level-learned-lambda-training-bundle-v2-proven"
        in differential.evidence
    )
    assert "gpt2-diff-materialized-graph-training-proof-v1" in differential.evidence
    assert (
        "gpt2-diff-native-ir-migration-resident-bundle-consumer-unimplemented"
        in differential.evidence
    )
    assert (
        "gpt2-diff-resident-differential-forward-unimplemented"
        in differential.evidence
    )
    assert all(
        adapter.architecture_persistence_proven is True
        for adapter in adapters
        if adapter.selector != "gpt2_diff"
    )


def test_production_muse_glimmer_graph_routes_only_to_exact_native_target(
    tmp_path: Path,
) -> None:
    production = build_model_spec_from_config({"preset": "muse_glimmer"})
    graph = build_gpt_root_graph(name="muse_glimmer_native_train", model_spec=production)
    path = tmp_path / "muse-glimmer.json"
    _write_graph(path, graph)

    plan = plan_native_graph_training(path)

    assert plan.training_selector == "muse_glimmer"
    assert plan.native_target == "nfn_muse_glimmer_native_train"
    assert plan.execution_ready is True
    assert plan.architecture_persistence_proven is True

    preview = _preset_graph("muse_glimmer")
    preview_path = tmp_path / "muse-glimmer-preview.json"
    _write_graph(preview_path, preview)
    preview_plan = plan_native_graph_training(preview_path)
    assert preview_plan.execution_ready is False
    assert preview_plan.training_selector == ""


def test_production_muse_glimmer_full_sft_routes_with_lineage_arguments(
    tmp_path: Path,
) -> None:
    spec = build_model_spec_from_config({"preset": "muse_glimmer"})
    spec.template.objective = "sft"
    spec.finetune = FineTuneSpec(
        objective="sft",
        tokenizer_sha256="a" * 64,
        chat_template_sha256="b" * 64,
    )
    graph = build_gpt_root_graph(name="muse_glimmer_native_sft", model_spec=spec)
    path = tmp_path / "muse-glimmer-sft.json"
    _write_graph(path, graph)

    plan = plan_native_graph_training(path)

    assert plan.execution_ready is True
    assert plan.training_selector == "muse_glimmer"
    assert "--objective" in plan.trainer_arguments
    objective = plan.trainer_arguments.index("--objective")
    assert plan.trainer_arguments[objective + 1] == "sft"
    template = plan.trainer_arguments.index("--chat-template-sha256")
    assert plan.trainer_arguments[template + 1] == "b" * 64


def test_production_muse_glimmer_native_lora_routes_exact_adapter_contract(
    tmp_path: Path,
) -> None:
    spec = build_model_spec_from_config(
        {
            "preset": "muse_glimmer",
            "adapter_type": "lora",
            "lora_rank": 4,
            "lora_alpha": 8.0,
            "lora_dropout": 0.1,
        }
    )
    spec.template.objective = "sft"
    spec.finetune = FineTuneSpec(
        objective="sft",
        tokenizer_sha256="a" * 64,
        chat_template_sha256="b" * 64,
        adapter_only_save=True,
    )
    graph = build_gpt_root_graph(name="muse_glimmer_native_lora", model_spec=spec)
    path = tmp_path / "muse-glimmer-lora.json"
    _write_graph(path, graph)

    plan = plan_native_graph_training(path)

    assert plan.execution_ready is True
    assert plan.training_selector == "muse_glimmer"
    assert plan.compatibility_report.compatible is True
    assert plan.training_issues == ()
    arguments = plan.trainer_arguments
    assert arguments[arguments.index("--adapter") + 1] == "lora"
    assert arguments[arguments.index("--lora-rank") + 1] == "4"
    assert arguments[arguments.index("--lora-alpha") + 1] == "8"
    assert float(arguments[arguments.index("--lora-dropout") + 1]) == pytest.approx(0.1)
    assert arguments[arguments.index("--lora-targets") + 1].split(",") == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "attn_gate_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    spec.finetune.adapter_only_save = False
    rejected_path = tmp_path / "muse-glimmer-lora-nonadapter-save.json"
    _write_graph(
        rejected_path,
        build_gpt_root_graph(name="muse_glimmer_native_lora_rejected", model_spec=spec),
    )
    rejected = plan_native_graph_training(rejected_path)
    assert rejected.execution_ready is False
    assert rejected.training_selector == ""


def test_production_muse_glimmer_native_qlora_routes_exact_nf4_contract(
    tmp_path: Path,
) -> None:
    spec = build_model_spec_from_config(
        {
            "preset": "muse_glimmer",
            "adapter_type": "qlora",
            "lora_targets": ["q_proj", "v_proj", "down_proj"],
            "lora_rank": 4,
            "lora_alpha": 8.0,
            "qlora_group_size": 64,
            "qlora_compute_dtype": "bf16",
        }
    )
    spec.template.objective = "sft"
    spec.finetune = FineTuneSpec(
        objective="sft",
        tokenizer_sha256="a" * 64,
        chat_template_sha256="b" * 64,
        adapter_only_save=True,
    )
    path = tmp_path / "muse-glimmer-qlora.json"
    _write_graph(
        path,
        build_gpt_root_graph(name="muse_glimmer_native_qlora", model_spec=spec),
    )

    plan = plan_native_graph_training(path)

    assert plan.execution_ready is True
    assert plan.training_selector == "muse_glimmer"
    assert plan.training_issues == ()
    arguments = plan.trainer_arguments
    assert arguments[arguments.index("--adapter") + 1] == "qlora"
    assert arguments[arguments.index("--qlora-group-size") + 1] == "64"
    assert arguments[arguments.index("--lora-targets") + 1] == (
        "q_proj,v_proj,down_proj"
    )

    spec.block_spec.qlora_group_size = 32
    rejected_path = tmp_path / "muse-glimmer-qlora-invalid.json"
    _write_graph(
        rejected_path,
        build_gpt_root_graph(
            name="muse_glimmer_native_qlora_invalid", model_spec=spec
        ),
    )
    rejected = plan_native_graph_training(rejected_path)
    assert rejected.execution_ready is False
    assert rejected.training_selector == ""


def test_gpt2_diff_is_ready_only_in_the_exact_graph_training_planner(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "gpt2-diff.json"
    _write_graph(graph_path, _preset_graph("gpt2_diff", vocab_size=50257))

    plan = preflight_native_graph_training(graph_path)

    assert plan.compatibility_report.compatible
    assert plan.manifest is not None
    assert plan.manifest.capabilities["native_ir"] is True
    # Generic Native IR classification remains fail-closed because it runs
    # before the exact graph-training validator and materialized proof.
    assert plan.manifest.capabilities["native_training"] is False
    assert plan.manifest.capabilities["architecture_persistence"] is False
    assert plan.manifest.capabilities["native_inference"] is False
    assert plan.trainer_registered is True
    assert plan.training_selector == "gpt2_diff"
    assert plan.adapter_mode == "validated-dense-graph-file-v1"
    assert plan.architecture_persistence_proven is True
    assert plan.execution_ready is True
    assert plan.training_issues == ()
    assert plan.blockers == ()
    assert plan.graph_preflight_proof is None
    assert "--graph-preflight-proof" not in plan.trainer_arguments
    assert plan.artifact_metadata["execution_ready"] is True
    assert (
        "gpt2-diff-low-level-learned-lambda-training-bundle-v2-proven"
        in plan.artifact_metadata["adapter_evidence"]
    )
    assert (
        "gpt2-diff-materialized-graph-training-proof-v1"
        in plan.artifact_metadata["adapter_evidence"]
    )
    assert (
        "gpt2-diff-native-ir-migration-resident-bundle-consumer-unimplemented"
        in plan.artifact_metadata["adapter_evidence"]
    )
    assert (
        "gpt2-diff-resident-differential-forward-unimplemented"
        in plan.artifact_metadata["adapter_evidence"]
    )


@pytest.mark.parametrize("root_runtime", ("torch", "native-cuda"))
def test_gpt2_diff_root_runtime_is_reviewed_orchestration_metadata_only(
    tmp_path: Path,
    root_runtime: str,
) -> None:
    payload = _preset_graph("gpt2_diff", vocab_size=50257).to_dict()
    payload["runtime"] = root_runtime
    assert (
        payload["nodes"]["model"]["neuron_def"]["subgraph"]["runtime"]
        == "torch"
    )
    graph_path = tmp_path / f"gpt2-diff-{root_runtime}.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = plan_native_graph_training(graph_path)

    assert plan.execution_ready is True
    assert plan.training_selector == "gpt2_diff"
    assert plan.training_issues == ()


def test_materialized_gpt2_diff_proof_is_canonical_deterministic_and_source_bound(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "authored-name.json"
    _write_graph(graph_path, _preset_graph("gpt2_diff", vocab_size=50257))
    source_bytes = graph_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    first_artifact = tmp_path / "first"

    first = plan_native_graph_training(
        graph_path,
        artifact_dir=first_artifact,
        materialize=True,
    )

    proof_path = first_artifact / "native-training-proof.json"
    assert first.execution_ready is True
    assert first.graph_preflight_proof == proof_path.resolve()
    assert first.artifact_metadata["graph_preflight_proof_path"] == str(
        proof_path.resolve()
    )
    assert first.artifact_metadata["graph_preflight_proof_schema"] == (
        "neuralfn.native_graph_training_proof"
    )
    assert first.artifact_metadata["graph_preflight_proof_version"] == 1
    assert proof_path.stat().st_mode & 0o777 == 0o600
    assert first.trainer_arguments[-2:] == (
        "--graph-preflight-proof",
        str(proof_path.resolve()),
    )

    contract = {
        "adapter_mode": "validated-dense-graph-file-v1",
        "attention_shape_sha256": (
            "13d8cd97dd07cbc6808839acab609ead157402fe10da5d9de053622d69958937"
        ),
        "block_shape_sha256": (
            "e722edf71b3a6c37f85a1c63d68df586d7c79f9c9730d1f948ee673102951bad"
        ),
        "configuration_contract": "dense-gpt2-graph-configuration-v1",
        "geometry": {
            "head_dim": 8,
            "max_seq_len": 1024,
            "mlp_hidden_dim": 128,
            "model_dim": 32,
            "num_heads": 4,
            "num_kv_heads": 4,
            "num_layers": 1,
            "padded_vocab_size": 50304,
            "vocab_size": 50257,
        },
        "mlp_shape_sha256": (
            "58a0daeb34c0b19ced46d6bc27f18935201077638c17d42b4f3ac290d3c4a71f"
        ),
        "passed": True,
        "root_shape_sha256": (
            "228b19b94790989400bc770185370fd42761a1c1634451b3bcc3b770788b2b93"
        ),
        "schema": "neuralfn.native_graph_training_proof",
        "source_graph_sha256": source_sha256,
        "topology_contract": "dense-gpt2-active-topology-v1",
        "training_selector": "gpt2_diff",
        "validator_contract": "dense-gpt2-exact-graph-validator-v1",
        "version": 1,
    }
    contract_bytes = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    expected_proof = (
        b'{"contract":'
        + contract_bytes
        + b',"contract_sha256":"'
        + contract_sha256.encode("ascii")
        + b'"}\n'
    )
    assert proof_path.read_bytes() == expected_proof
    assert first.artifact_metadata["graph_preflight_proof_contract_sha256"] == (
        contract_sha256
    )

    # Neither authored filename nor artifact path participates in the contract.
    renamed_graph = tmp_path / "renamed-but-identical.json"
    renamed_graph.write_bytes(source_bytes)
    second = plan_native_graph_training(
        renamed_graph,
        artifact_dir=tmp_path / "second",
        materialize=True,
    )
    assert second.graph_preflight_proof is not None
    assert second.graph_preflight_proof.read_bytes() == expected_proof

    # A serialization-only byte change is source identity, even when the
    # resolved active graph remains semantically identical.
    changed_graph = tmp_path / "changed-bytes.json"
    changed_graph.write_bytes(source_bytes + b"\n")
    changed = plan_native_graph_training(
        changed_graph,
        artifact_dir=tmp_path / "changed",
        materialize=True,
    )
    assert changed.graph_preflight_proof is not None
    changed_payload = json.loads(changed.graph_preflight_proof.read_text("utf-8"))
    assert changed_payload["contract"]["source_graph_sha256"] != source_sha256
    assert changed.graph_preflight_proof.read_bytes() != expected_proof


@pytest.mark.parametrize(
    "mutation",
    (
        "tokenization",
        "ar_loss",
        "finetune",
        "backend_capabilities",
        "tie_embeddings",
        "norm_type",
        "active_string_spelling",
        "dropout",
        "dropout_spec_integer",
        "diff_lambda",
        "inactive_jepa",
        "inactive_ema",
        "inactive_window",
        "inactive_route_evo",
        "inactive_rope",
        "inactive_sparse",
        "inactive_adapter",
        "unknown_spec_field",
        "unknown_block_field",
        "unknown_template_field",
        "root_training_method",
        "nested_training_method",
        "unreviewed_runtime",
        "unknown_graph_field",
        "unknown_node_field",
        "unknown_neuron_field",
        "unknown_edge_field",
        "unknown_port_field",
        "noncanonical_port_range",
        "noncanonical_source_code",
        "nested_subgraph_alias",
        "module_source_code",
        "subgraph_module_config",
        "function_module_type",
        "function_module_config",
        "subgraph_variant_ref",
        "module_name",
        "huge_ar_loss",
        "huge_mlp_multiplier",
        "context_alias_conflict",
        "active_edge_transform",
        "is_causal_integer",
        "dropout_boolean",
        "dropout_integer",
        "reshape_heads_float",
        "edge_integer_transforms",
        "edge_numeric_strings",
    ),
)
def test_gpt2_diff_semantic_or_active_topology_decoys_never_issue_proof(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _preset_graph("gpt2_diff", vocab_size=50257).to_dict()
    spec = payload["torch_config"]["template_spec"]
    block = spec["block_spec"]
    template = spec["template"]
    model_graph = payload["nodes"]["model"]["neuron_def"]["subgraph"]
    attention_graph = model_graph["nodes"]["block_0"]["neuron_def"]["subgraph"][
        "nodes"
    ]["attention"]["neuron_def"]["subgraph"]
    if mutation == "tokenization":
        template["tokenization"] = "byte_hnet"
    elif mutation == "ar_loss":
        spec["ar_loss_coef"] = 2.0
    elif mutation == "finetune":
        spec["finetune"] = {"objective": "dpo"}
    elif mutation == "backend_capabilities":
        template["backend_capabilities"] = {}
    elif mutation == "tie_embeddings":
        spec["tie_embeddings"] = False
    elif mutation == "norm_type":
        block["norm_type"] = "rmsnorm"
    elif mutation == "active_string_spelling":
        block["norm_type"] = "LayerNorm"
    elif mutation == "dropout":
        block["dropout_p"] = 0.1
    elif mutation == "dropout_spec_integer":
        block["dropout_p"] = 0
    elif mutation == "diff_lambda":
        block["diff_lambda_init"] = 0.7
    elif mutation == "inactive_jepa":
        spec["jepa_loss_coef"] = 1.0
    elif mutation == "inactive_ema":
        spec["ema_decay"] = 0.5
    elif mutation == "inactive_window":
        block["window_size"] = 7
    elif mutation == "inactive_route_evo":
        spec["route_evo_enabled"] = False
    elif mutation == "inactive_rope":
        block["rope_theta"] = 123.0
    elif mutation == "inactive_sparse":
        block["sparse_block_size"] = 32
    elif mutation == "inactive_adapter":
        block["adapter_dim"] = 4
    elif mutation == "unknown_spec_field":
        spec["unreviewed_semantic"] = True
    elif mutation == "unknown_block_field":
        block["unreviewed_semantic"] = True
    elif mutation == "unknown_template_field":
        template["unreviewed_semantic"] = True
    elif mutation == "root_training_method":
        payload["training_method"] = "evolutionary"
    elif mutation == "nested_training_method":
        model_graph["training_method"] = "evolutionary"
    elif mutation == "unreviewed_runtime":
        payload["runtime"] = "eager"
    elif mutation == "unknown_graph_field":
        payload["unreviewed_semantic"] = True
    elif mutation == "unknown_node_field":
        model_graph["nodes"]["token_embed"]["unreviewed_semantic"] = True
    elif mutation == "unknown_neuron_field":
        model_graph["nodes"]["token_embed"]["neuron_def"][
            "unreviewed_semantic"
        ] = True
    elif mutation == "unknown_edge_field":
        next(iter(model_graph["edges"].values()))["unreviewed_semantic"] = True
    elif mutation == "unknown_port_field":
        model_graph["nodes"]["token_embed"]["neuron_def"]["input_ports"][0][
            "unreviewed_semantic"
        ] = True
    elif mutation == "noncanonical_port_range":
        payload["nodes"]["tokens_in"]["neuron_def"]["input_ports"][0][
            "range"
        ] = [0, 65535]
    elif mutation == "noncanonical_source_code":
        payload["nodes"]["tokens_in"]["neuron_def"]["source_code"] = (
            "def input(x):\n    return x + 1\n"
        )
    elif mutation == "nested_subgraph_alias":
        model_graph["nodes"]["block_0"]["neuron_def"]["input_aliases"] = [
            "loss"
        ]
    elif mutation == "module_source_code":
        model_graph["nodes"]["token_embed"]["neuron_def"]["source_code"] = (
            "def input(x):\n    return x\n"
        )
    elif mutation == "subgraph_module_config":
        model_graph["nodes"]["block_0"]["neuron_def"]["module_config"] = {
            "future": True
        }
    elif mutation == "function_module_type":
        model_graph["nodes"]["tokens_in"]["neuron_def"]["module_type"] = (
            "future"
        )
    elif mutation == "function_module_config":
        model_graph["nodes"]["tokens_in"]["neuron_def"]["module_config"] = {
            "future": True
        }
    elif mutation == "subgraph_variant_ref":
        model_graph["nodes"]["block_0"]["neuron_def"]["variant_ref"] = {
            "family": "mamba_block",
            "version": "default",
        }
    elif mutation == "module_name":
        model_graph["nodes"]["token_embed"]["neuron_def"]["name"] = "future"
    elif mutation == "huge_ar_loss":
        spec["ar_loss_coef"] = 10**4000
    elif mutation == "huge_mlp_multiplier":
        block["mlp_multiplier"] = 10**4000
    elif mutation == "context_alias_conflict":
        spec["seq_len"] = 1024
        spec["max_seq_len"] = 2048
    elif mutation == "active_edge_transform":
        model_graph["edges"]["e_embed_add_block_0"]["weight"] = 0.5
    elif mutation == "is_causal_integer":
        attention_graph["nodes"]["sdpa"]["neuron_def"]["module_config"][
            "is_causal"
        ] = 1
    elif mutation == "dropout_boolean":
        attention_graph["nodes"]["sdpa"]["neuron_def"]["module_config"][
            "dropout_p"
        ] = False
    elif mutation == "dropout_integer":
        attention_graph["nodes"]["sdpa"]["neuron_def"]["module_config"][
            "dropout_p"
        ] = 0
    elif mutation == "reshape_heads_float":
        attention_graph["nodes"]["q_heads"]["neuron_def"]["module_config"][
            "num_heads"
        ] = 4.0
    elif mutation == "edge_integer_transforms":
        first_edge = next(iter(model_graph["edges"].values()))
        first_edge["weight"] = 1
        first_edge["bias"] = 0
    else:
        first_edge = next(iter(model_graph["edges"].values()))
        first_edge["weight"] = "1"
        first_edge["bias"] = "0"
    graph_path = tmp_path / f"{mutation}.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    artifact_dir = tmp_path / f"artifact-{mutation}"

    plan = plan_native_graph_training(
        graph_path,
        artifact_dir=artifact_dir,
        materialize=True,
    )

    assert plan.execution_ready is False
    assert plan.graph_preflight_proof is None
    assert plan.training_issues or plan.compatibility_report.issues
    assert not (artifact_dir / "native-training-proof.json").exists()


def test_only_exact_diff_materialization_issues_a_proof(tmp_path: Path) -> None:
    dense_graph = tmp_path / "dense.json"
    _write_graph(dense_graph, _preset_graph("gpt2", vocab_size=50257))
    dense_artifact = tmp_path / "dense-artifact"

    dense = plan_native_graph_training(
        dense_graph,
        artifact_dir=dense_artifact,
        materialize=True,
    )

    assert dense.execution_ready is True
    assert dense.graph_preflight_proof is None
    assert dense.artifact_metadata["graph_preflight_proof_path"] is None
    assert not (dense_artifact / "native-training-proof.json").exists()


@pytest.mark.parametrize("spelling", ("duplicate-key", "non-finite"))
def test_ambiguous_gpt2_diff_json_never_issues_a_proof(
    tmp_path: Path,
    spelling: str,
) -> None:
    canonical = json.dumps(
        _preset_graph("gpt2_diff", vocab_size=50257).to_dict(),
        separators=(",", ":"),
    ).encode("utf-8")
    if spelling == "duplicate-key":
        source = canonical.replace(b'"name":', b'"name":"decoy","name":', 1)
    else:
        source = canonical.replace(b'"ar_loss_coef":1.0', b'"ar_loss_coef":1e999', 1)
    graph_path = tmp_path / f"{spelling}.json"
    graph_path.write_bytes(source)
    artifact_dir = tmp_path / f"artifact-{spelling}"

    plan = plan_native_graph_training(
        graph_path,
        artifact_dir=artifact_dir,
        materialize=True,
    )

    assert plan.execution_ready is False
    assert plan.graph_preflight_proof is None
    assert any(
        issue.code == "ambiguous_training_graph_json"
        for issue in plan.training_issues
    )
    assert not (artifact_dir / "native-training-proof.json").exists()


@pytest.mark.parametrize("mutation", ("huge-position", "huge-edge"))
def test_huge_finite_json_integers_are_structured_incompatibilities_before_proof(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _preset_graph("gpt2_diff", vocab_size=50257).to_dict()
    if mutation == "huge-position":
        payload["nodes"]["tokens_in"]["position"][0] = 10**4000
        expected_code = "invalid_position"
    else:
        model_graph = payload["nodes"]["model"]["neuron_def"]["subgraph"]
        next(iter(model_graph["edges"].values()))["weight"] = 10**4000
        expected_code = "invalid_edge_parameter"
    graph_path = tmp_path / f"{mutation}.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = plan_native_graph_training(graph_path)

    assert plan.execution_ready is False
    assert plan.graph_preflight_proof is None
    assert any(
        issue.code == expected_code for issue in plan.compatibility_report.issues
    )


def test_gpt2_diff_proof_matches_native_source_and_geometry_bounds(tmp_path: Path) -> None:
    empty_graph = tmp_path / "empty.json"
    empty_graph.write_bytes(b"")
    with pytest.raises(ValueError, match="must be non-empty"):
        plan_native_graph_training(empty_graph)

    oversized_payload = _preset_graph("gpt2_diff", vocab_size=50257).to_dict()
    oversized_payload["proof_padding"] = "x" * (16 * 1024 * 1024)
    oversized_graph = tmp_path / "oversized.json"
    oversized_graph.write_text(json.dumps(oversized_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="verified-graph bound"):
        plan_native_graph_training(oversized_graph)

    short_payload = _preset_graph("gpt2_diff", vocab_size=50257).to_dict()
    short_spec = short_payload["torch_config"]["template_spec"]
    short_spec["max_seq_len"] = 8
    short_payload["nodes"]["model"]["neuron_def"]["subgraph"]["nodes"][
        "pos_embed"
    ]["neuron_def"]["module_config"]["max_seq_len"] = 8
    short_graph = tmp_path / "short.json"
    short_graph.write_text(json.dumps(short_payload), encoding="utf-8")
    short = plan_native_graph_training(short_graph)
    assert short.execution_ready is False
    assert any("max_seq_len" in issue.message for issue in short.training_issues)

    odd_head_graph = tmp_path / "odd-head.json"
    _write_graph(
        odd_head_graph,
        _preset_graph(
            "gpt2_diff",
            vocab_size=50257,
            model_dim=12,
            num_heads=4,
            num_kv_heads=4,
        ),
    )
    odd_head = plan_native_graph_training(odd_head_graph)
    assert odd_head.execution_ready is False
    assert any("even head_dim" in issue.message for issue in odd_head.training_issues)


def test_huge_declared_layer_count_is_rejected_before_expected_topology_expansion(
    tmp_path: Path,
) -> None:
    payload = _preset_graph("gpt2_diff", vocab_size=50257).to_dict()
    payload["torch_config"]["template_spec"]["num_layers"] = 2_147_483_647
    graph_path = tmp_path / "huge-layer-count.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = plan_native_graph_training(graph_path)

    assert plan.execution_ready is False
    assert plan.graph_preflight_proof is None
    assert any(
        issue.code == "unsupported_training_topology"
        and "contiguous active block topology" in issue.message
        for issue in plan.training_issues
    )


def test_source_growth_after_fstat_is_rejected_by_the_bounded_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_path = tmp_path / "sparse-growth-race.json"
    with graph_path.open("wb") as handle:
        handle.truncate(1 << 30)

    real_fstat = os.fstat

    def report_small_source(descriptor: int) -> SimpleNamespace:
        current = real_fstat(descriptor)
        return SimpleNamespace(st_mode=current.st_mode, st_size=2)

    monkeypatch.setattr("neuralfn.native_graph_train.os.fstat", report_small_source)

    with pytest.raises(ValueError, match="bound.*while it was read"):
        plan_native_graph_training(graph_path)


def test_canonical_llama_plan_is_execution_ready_with_authoritative_geometry(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "llama.json"
    _write_graph(graph_path, _preset_graph("llama", num_kv_heads=2))

    plan = preflight_native_graph_training(graph_path)

    assert plan.compatibility_report.compatible
    assert plan.trainer_family == "llama"
    assert plan.native_target == "nfn_llama_native_train"
    assert plan.trainer_registered is True
    assert plan.architecture_persistence_proven is True
    assert plan.execution_ready is True
    assert plan.trainer_consumes_native_ir is False
    assert plan.training_selector == "llama"
    assert plan.adapter_mode == "validated-canonical-llama-graph-file-v1"
    assert plan.training_issues == ()
    assert plan.blockers == ()
    fingerprint = plan.compatibility_report.graph_fingerprint
    assert fingerprint == hashlib.sha256(graph_path.read_bytes()).hexdigest()
    assert len(fingerprint) == 64
    assert set(fingerprint) <= set("0123456789abcdef")
    assert plan.trainer_arguments == (
        "--template-name",
        "llama",
        "--num-layers",
        "1",
        "--model-dim",
        "32",
        "--hidden-dim",
        "96",
        "--mlp-multiplier",
        "2.6666666666666665",
        "--multiple-of",
        "16",
        "--num-heads",
        "4",
        "--num-kv-heads",
        "2",
        "--vocab-size",
        "64",
        "--padded-vocab-size",
        "64",
        "--train-seq-len",
        "1024",
        "--rope-theta",
        "10000",
        "--rope-factor",
        "1",
        "--graph-fingerprint",
        fingerprint,
    )
    provenance = plan.artifact_metadata["architecture_provenance"]
    assert provenance["schema"] == "neuralfn.native_graph_training.llama_architecture"
    assert provenance["version"] == 1
    assert provenance["selector"] == "llama"
    assert provenance["graph_fingerprint"] == fingerprint
    assert provenance["geometry"] == {
        "max_seq_len": 1024,
        "vocab_size": 64,
        "padded_vocab_size": 64,
        "num_layers": 1,
        "model_dim": 32,
        "hidden_dim": 96,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 8,
        "rope_theta": 10_000.0,
        "rope_scaling_factor": 1.0,
        "rms_norm_eps": 1.0e-6,
        "mlp_multiplier": 8.0 / 3.0,
        "multiple_of": 16,
    }
    assert provenance["semantics"] == {
        "normalization": "rmsnorm",
        "position_encoding": "rope",
        "attention": "dense-gqa",
        "mlp": "gate-first-swiglu",
        "linear_bias": False,
        "dropout_p": 0.0,
        "tie_embeddings": False,
        "rope_scaling_type": "none",
    }
    assert provenance["sources"]["max_seq_len"] == "native-family-trainer-default"
    assert provenance["sources"]["padded_vocab_size"] == "template_spec.vocab_size"
    assert plan.artifact_metadata["materialized"] is False
    assert plan.artifact_metadata["artifact_dir"] is None


def test_llama_fast_plan_preserves_source_profile_and_normalizes_native_identity(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "llama-fast.json"
    _write_graph(graph_path, _preset_graph("llama_fast", num_kv_heads=2))

    plan = preflight_native_graph_training(graph_path)

    assert plan.compatibility_report.compatible
    assert plan.trainer_family == "llama"
    assert plan.native_target == "nfn_llama_native_train"
    assert plan.training_selector == "llama_fast"
    assert plan.adapter_mode == "validated-canonical-llama-graph-file-v1"
    assert plan.execution_ready is True
    assert plan.training_issues == ()
    assert plan.trainer_arguments[:2] == ("--template-name", "llama")
    fingerprint = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    assert plan.trainer_arguments[-2:] == ("--graph-fingerprint", fingerprint)
    provenance = plan.artifact_metadata["architecture_provenance"]
    assert provenance["selector"] == "llama_fast"
    assert provenance["source_selector"] == "llama_fast"
    assert provenance["source_preset"] == "llama_fast"
    assert provenance["source_runtime"] == "compile"
    assert provenance["native_template_name"] == "llama"
    assert provenance["checkpoint_identity"] == "llama"
    assert provenance["graph_fingerprint"] == fingerprint
    assert provenance["sources"]["source_preset"] == (
        "classified-exact-active-topology-and-runtime-profile"
    )
    assert provenance["sources"]["source_runtime"] == "template_spec.template.runtime"


@pytest.mark.parametrize(
    ("preset", "selector", "runtime", "native_template"),
    (
        ("moe", "moe", "eager", "mixllama"),
        ("mixllama", "mixllama", "eager", "mixllama"),
        ("mixllama_fast", "mixllama_fast", "compile", "mixllama-fast"),
    ),
)
def test_standard_moe_plan_preserves_alias_and_exact_graph_geometry(
    tmp_path: Path,
    preset: str,
    selector: str,
    runtime: str,
    native_template: str,
) -> None:
    graph_path = tmp_path / f"{preset}.json"
    _write_graph(
        graph_path,
        _preset_graph(
            preset,
            vocab_size=67,
            model_dim=48,
            num_heads=6,
            num_kv_heads=2,
            multiple_of=None,
            mlp_multiplier=2.5,
            experts=5,
            top_k=2,
            router_aux_loss_coef=0.0375,
        ),
    )

    plan = preflight_native_graph_training(graph_path)

    assert plan.compatibility_report.compatible
    assert plan.trainer_family == "mixllama"
    assert plan.native_target == "nfn_mixllama_native_train"
    assert plan.training_selector == selector
    assert plan.adapter_mode == "validated-standard-moe-graph-file-v1"
    assert plan.execution_ready is True
    assert plan.training_issues == ()
    arguments = dict(zip(plan.trainer_arguments[::2], plan.trainer_arguments[1::2], strict=True))
    assert arguments == {
        "--template-name": native_template,
        "--num-layers": "1",
        "--model-dim": "48",
        "--hidden-dim": "120",
        "--mlp-multiplier": "2.5",
        "--multiple-of": "0",
        "--num-heads": "6",
        "--num-kv-heads": "2",
        "--vocab-size": "67",
        "--padded-vocab-size": "67",
        "--train-seq-len": "1024",
        "--rope-theta": "10000",
        "--rope-factor": "1",
        "--experts": "5",
        "--top-k": "2",
        "--layers-per-expert": "1",
        "--router-aux-loss-coef": "0.037499999999999999",
        "--graph-fingerprint": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
    }
    provenance = plan.artifact_metadata["architecture_provenance"]
    assert provenance["schema"] == "neuralfn.native_graph_training.standard_moe_architecture"
    assert provenance["selector"] == selector
    assert provenance["source_runtime"] == runtime
    assert provenance["native_template_name"] == native_template
    assert provenance["checkpoint_identity"] == "mixllama"
    assert provenance["geometry"]["hidden_dim"] == 120
    assert provenance["geometry"]["multiple_of"] == 0
    assert provenance["geometry"]["experts"] == 5
    assert provenance["geometry"]["top_k"] == 2
    assert provenance["geometry"]["router_aux_loss_coef"] == pytest.approx(0.0375)
    assert provenance["sources"]["multiple_of"].startswith("native-zero-sentinel")


@pytest.mark.parametrize(
    "preset",
    ("mixllama_fast_megakernel", "moe_modern", "deepseek_v3", "moe_jepa_evo"),
)
def test_standard_moe_adapter_keeps_neighbor_profiles_closed(
    tmp_path: Path,
    preset: str,
) -> None:
    graph_path = tmp_path / f"{preset}.json"
    _write_graph(graph_path, _preset_graph(preset, num_kv_heads=2))

    plan = plan_native_graph_training(graph_path)

    assert plan.compatibility_report.compatible
    assert plan.execution_ready is False
    assert plan.adapter_mode is None
    assert plan.trainer_arguments == ()


def test_canonical_llama_plan_preserves_declared_context_padding_and_mlp_rounding(
    tmp_path: Path,
) -> None:
    payload = _preset_graph(
        "llama",
        vocab_size=67,
        model_dim=48,
        num_heads=6,
        num_kv_heads=2,
        multiple_of=32,
        mlp_multiplier=2.5,
    ).to_dict()
    template_spec = payload["torch_config"]["template_spec"]
    template_spec["max_seq_len"] = 384
    template_spec["padded_vocab_size"] = 80
    graph_path = tmp_path / "llama-explicit-geometry.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = plan_native_graph_training(graph_path)

    assert plan.execution_ready is True
    def value_after(flag: str) -> str:
        index = plan.trainer_arguments.index(flag)
        return plan.trainer_arguments[index + 1]

    assert value_after("--model-dim") == "48"
    assert value_after("--hidden-dim") == "128"
    assert value_after("--mlp-multiplier") == "2.5"
    assert value_after("--multiple-of") == "32"
    assert value_after("--num-heads") == "6"
    assert value_after("--num-kv-heads") == "2"
    assert value_after("--vocab-size") == "67"
    assert value_after("--padded-vocab-size") == "80"
    assert value_after("--train-seq-len") == "384"
    provenance = plan.artifact_metadata["architecture_provenance"]
    assert provenance["geometry"]["hidden_dim"] == 128
    assert provenance["geometry"]["head_dim"] == 8
    assert provenance["geometry"]["max_seq_len"] == 384
    assert provenance["geometry"]["padded_vocab_size"] == 80
    assert provenance["sources"]["max_seq_len"] == "template_spec.max_seq_len"
    assert provenance["sources"]["padded_vocab_size"] == "template_spec.padded_vocab_size"


@pytest.mark.parametrize("preset", ["llama", "llama_fast"])
def test_sdk_replans_reviewed_llama_graph_before_building_command(
    tmp_path: Path,
    preset: str,
) -> None:
    graph_path = tmp_path / f"{preset}.json"
    _write_graph(graph_path, _preset_graph(preset, num_kv_heads=2))
    plan = plan_native_graph_training(graph_path)
    config = build_native_train_run_config(
        "llama",
        [
            *plan.trainer_arguments,
            "--train-llama-dataset-loop",
            "--dataset-alias",
            "fixture",
        ],
        template_name=preset,
        graph_file=str(graph_path),
        native_train_cli="/tmp/nfn_native_train",
    )

    command = config.argv()

    assert command[:3] == ["/tmp/nfn_native_train", "--base-model", "llama"]
    assert command[command.index("--graph-file") + 1] == str(graph_path)
    assert command[command.index("--graph-fingerprint") + 1] == (
        plan.compatibility_report.graph_fingerprint
    )
    assert command[command.index("--template-name") + 1] == "llama"


def test_sdk_rejects_fabricated_or_conflicting_llama_graph_handoffs(
    tmp_path: Path,
) -> None:
    missing = build_native_train_run_config(
        "llama",
        ["--template-name", "llama", "--graph-fingerprint", "a" * 64],
        graph_file=str(tmp_path / "missing.json"),
        native_train_cli="/tmp/nfn_native_train",
    )
    with pytest.raises(ValueError, match="preflight failed"):
        missing.argv()

    graph_path = tmp_path / "llama.json"
    _write_graph(graph_path, _preset_graph("llama", num_kv_heads=2))
    plan = plan_native_graph_training(graph_path)
    conflicting_args = [*plan.trainer_arguments, "--model-dim", "999"]
    conflicting = build_native_train_run_config(
        "llama",
        conflicting_args,
        graph_file=str(graph_path),
        native_train_cli="/tmp/nfn_native_train",
    )
    with pytest.raises(ValueError, match="expected exactly --model-dim"):
        conflicting.argv()


@pytest.mark.parametrize("mutation", ["edge_transform", "embedded_state"])
def test_canonical_llama_training_adapter_rejects_unconsumed_payload_state(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _preset_graph("llama", num_kv_heads=2).to_dict()
    model_graph = payload["nodes"]["model"]["neuron_def"]["subgraph"]
    if mutation == "edge_transform":
        model_graph["edges"]["e_token_embed_block_0"]["weight"] = 0.5
        expected_code = "unsupported_training_topology"
    else:
        model_graph["nodes"]["token_embed"]["neuron_def"]["module_state"] = "{}"
        expected_code = "unsupported_training_state"
    graph_path = tmp_path / f"llama-{mutation}.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = plan_native_graph_training(graph_path)

    assert plan.compatibility_report.compatible
    assert plan.training_selector == "llama"
    assert plan.execution_ready is False
    assert len(plan.training_issues) == 1
    assert plan.training_issues[0].code == expected_code


def test_all_67_shipped_presets_have_canonical_registered_routing(tmp_path: Path) -> None:
    assert len(SHIPPED_GPT_TEMPLATE_PRESETS) == 67
    registry = {spec.family: spec for spec in native_trainer_specs()}
    execution_ready: list[str] = []
    execution_blocked: list[str] = []

    for preset in SHIPPED_GPT_TEMPLATE_PRESETS:
        graph_path = tmp_path / f"{preset}.json"
        _write_graph(graph_path, _preset_graph(preset, vocab_size=50257))

        plan = plan_native_graph_training(graph_path)

        assert plan.compatibility_report.compatible, (
            preset,
            plan.compatibility_report.to_dict(),
        )
        assert plan.manifest is not None
        assert plan.trainer_family in registry, preset
        expected = registry[plan.trainer_family]
        assert plan.native_target == expected.native_target
        assert plan.trainer_registered is True
        assert plan.trainer_consumes_native_ir is False
        assert plan.graph_preflight_enforced is True
        assert plan.manifest.capabilities["resident_inference"] is False
        (execution_ready if plan.execution_ready else execution_blocked).append(preset)
        assert plan.architecture_persistence_proven is plan.execution_ready
        if not plan.execution_ready:
            assert plan.training_issues, preset

    # Graph-training readiness is distinct from resident/migration readiness.
    # The exact planner can issue the gpt2_diff training handoff, while its
    # Native IR migration and resident differential consumer remain blocked.
    assert set(execution_ready) == {
        "gpt2",
        "gpt2_diff",
        "gpt2_megakernel",
        "gpt2_moa",
        "gpt2_qknorm",
        "gpt2_softcap",
        "gpt2_stable",
        "gpt2_zloss",
        "llama",
        "llama_fast",
        "moe",
        "mixllama",
        "mixllama_fast",
    }
    assert len(execution_blocked) == 54


@pytest.mark.parametrize(
    ("preset", "operation", "message_fragment"),
    (
        ("gpt2_modern", "rms_norm", "RMSNorm/RoPE/GeGLU"),
        ("nanogpt", "dropout", "bias-free linear/dropout"),
        ("nanogpt_megakernel", "dropout", "bias-free linear/dropout"),
    ),
)
def test_selector_labels_without_graph_faithful_adapters_fail_honestly(
    tmp_path: Path,
    preset: str,
    operation: str,
    message_fragment: str,
) -> None:
    graph_path = tmp_path / f"{preset}.json"
    _write_graph(graph_path, _preset_graph(preset, vocab_size=50257))

    plan = plan_native_graph_training(graph_path)

    assert plan.compatibility_report.compatible
    assert not plan.execution_ready
    assert plan.adapter_mode is None
    assert plan.trainer_arguments == ()
    assert len(plan.training_issues) == 1
    issue = plan.training_issues[0]
    assert issue.code == "unsupported_training_adapter"
    assert issue.operation == operation
    assert issue.path.startswith("root/nodes/model")
    assert message_fragment in issue.message


def test_dense_graph_adapter_rejects_ignored_edge_transforms_at_destination_node(
    tmp_path: Path,
) -> None:
    payload = _preset_graph("gpt2", vocab_size=50257).to_dict()
    model_graph = payload["nodes"]["model"]["neuron_def"]["subgraph"]
    model_graph["edges"]["e_embed_add_block_0"]["weight"] = 0.5
    graph_path = tmp_path / "weighted-edge.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = plan_native_graph_training(graph_path)

    assert plan.compatibility_report.compatible
    assert not plan.execution_ready
    assert len(plan.training_issues) == 1
    issue = plan.training_issues[0]
    assert issue.code == "unsupported_training_topology"
    assert issue.path.endswith("/nodes/block_0")
    assert "edge transforms" in issue.message


def test_dense_graph_adapter_rejects_geometry_the_compiled_loop_cannot_persist(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "small-vocab.json"
    _write_graph(graph_path, _preset_graph("gpt2", vocab_size=64))

    plan = plan_native_graph_training(graph_path)

    assert plan.compatibility_report.compatible
    assert not plan.execution_ready
    issues = [
        issue
        for issue in plan.training_issues
        if issue.code == "unsupported_training_geometry"
    ]
    assert len(issues) == 1
    assert issues[0].path == "root/nodes/model"
    assert "vocab_size=50257" in issues[0].message


def test_unsupported_nested_module_fails_closed_with_node_path(tmp_path: Path) -> None:
    graph = _preset_graph("gpt2")
    payload = graph.to_dict()
    payload["nodes"]["model"]["neuron_def"]["subgraph"]["nodes"]["token_embed"][
        "neuron_def"
    ]["module_type"] = "unregistered_future_op"
    graph_path = tmp_path / "unsupported.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    artifact_dir = tmp_path / "must-not-exist"

    plan = plan_native_graph_training(
        graph_path,
        artifact_dir=artifact_dir,
        materialize=True,
    )

    assert not plan.compatibility_report.compatible
    assert plan.execution_ready is False
    issues = [
        issue
        for issue in plan.compatibility_report.issues
        if issue.code == "unsupported_module"
    ]
    assert issues
    assert issues[0].operation == "unregistered_future_op"
    assert issues[0].path.endswith("/nodes/token_embed")
    assert plan.blockers == (f"unsupported_module:{issues[0].path}",)
    assert plan.artifact_metadata["materialized"] is False
    assert not artifact_dir.exists()


def test_custom_source_is_not_executed_during_training_preflight(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    payload = _custom_function_graph().to_dict()
    payload["nodes"]["custom"]["neuron_def"]["source_code"] = (
        f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')\n"
        "def custom_step(x):\n"
        "    return x\n"
    )
    graph_path = tmp_path / "custom.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = plan_native_graph_training(graph_path)

    assert not marker.exists()
    assert plan.execution_ready is False
    assert "root/nodes/custom" in plan.compatibility_report.unsupported_node_paths


def test_materialization_requires_new_explicit_directory(tmp_path: Path) -> None:
    graph_path = tmp_path / "gpt2.json"
    _write_graph(graph_path, _preset_graph("gpt2"))

    with pytest.raises(ValueError, match="explicit artifact_dir"):
        plan_native_graph_training(graph_path, materialize=True)

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        plan_native_graph_training(
            graph_path,
            artifact_dir=existing,
            materialize=True,
        )
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_graph_only_planner_import_does_not_load_torch() -> None:
    script = (
        "import sys; import neuralfn.native_graph_train; "
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
