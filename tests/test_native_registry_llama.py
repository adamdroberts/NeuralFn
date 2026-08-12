from __future__ import annotations

from copy import deepcopy

import pytest

from neuralfn.native_ir import compile_native_graph_payload
from neuralfn.native_registry import (
    capability_proof_for,
    classify_native_graph_training_selector,
    native_trainer_specs,
)
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


LLAMA_FAMILY_PRESETS = (
    "diff_transformer",
    "fp8_llama",
    "gemma3",
    "kv_pca_llama",
    "kv_pca_llama_modern",
    "llama",
    "llama_fast",
    "llama_fast_megakernel",
    "llama_megakernel",
    "llama_modern",
    "longctx_sparse_llama",
    "modern_norms_llama",
    "mxfp4_llama",
    "qwen3_longctx",
    "ternary_b158",
    "ternary_b158_modern",
)


def _manifest(
    preset: str = "llama",
    *,
    num_layers: int = 1,
    model_dim: int = 16,
    num_heads: int = 4,
    num_kv_heads: int = 2,
):
    spec = build_model_spec_from_config(
        {
            "preset": preset,
            "num_layers": num_layers,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "multiple_of": 8,
            "vocab_size": 32,
        },
        preview_defaults=True,
    )
    graph = build_gpt_root_graph(name=f"native_registry_{preset}", model_spec=spec)
    return compile_native_graph_payload(graph.to_dict())


def test_llama_family_trainer_registration_remains_diagnostic() -> None:
    trainer = next(spec for spec in native_trainer_specs() if spec.family == "llama")

    assert trainer.shipped_presets == LLAMA_FAMILY_PRESETS
    assert trainer.trainer_registered is True
    assert trainer.architecture_persistence_proven is False
    assert trainer.native_forward == "diagnostic-transition-only"
    assert trainer.resident_inference is False


@pytest.mark.parametrize("preset", LLAMA_FAMILY_PRESETS)
def test_only_reviewed_llama_runtime_profiles_receive_resident_capabilities(
    preset: str,
) -> None:
    manifest = _manifest(preset)
    proof = capability_proof_for(manifest.model, manifest.topology)
    expected_selector = {
        "llama": "llama",
        "llama_fast": "llama_fast",
    }.get(preset, "")
    expected = bool(expected_selector)

    assert classify_native_graph_training_selector(
        manifest.model,
        manifest.topology,
    ) == expected_selector
    assert proof.architecture_persistence_proven is expected
    assert proof.native_forward_proven is expected
    assert proof.resident_inference_proven is expected
    assert proof.lossless_cache_proven is expected
    assert proof.serving_proven is expected
    assert proof.turboquant_cache_proven is False

    if expected:
        assert "native-family-llama-float32-checkpoint-v1" in proof.evidence
        assert "native-family-llama-checkpoint-v2-inspector" in proof.evidence
        assert "resident-llama-cpu-reference-abi-v1" in proof.evidence
        assert "resident-llama-gqa-lossless-kv-cache-v1" in proof.evidence
        assert "resident-llama-rope-gqa-right-aligned-v1" in proof.evidence
        assert "resident-llama-lean-serving-abi-v1" in proof.evidence
        assert "resident-llama-chat-completions-serving-v1" not in proof.evidence
        assert "turboquant_native_cache" in proof.missing_gates


def test_llama_fast_requires_the_exact_compile_runtime_capability_profile() -> None:
    manifest = _manifest("llama_fast")
    model = deepcopy(manifest.model)
    template = model["template_spec"]["template"]
    template["backend_capabilities"]["compile"] = False

    proof = capability_proof_for(model, manifest.topology)

    assert classify_native_graph_training_selector(model, manifest.topology) == ""
    assert proof.architecture_persistence_proven is False
    assert proof.native_forward_proven is False
    assert proof.resident_inference_proven is False
    assert proof.lossless_cache_proven is False
    assert proof.serving_proven is False


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_canonical_llama_accepts_valid_mqa_gqa_and_mha_geometry(
    num_kv_heads: int,
) -> None:
    manifest = _manifest(num_layers=2, num_kv_heads=num_kv_heads)
    proof = capability_proof_for(manifest.model, manifest.topology)

    assert proof.resident_inference_proven is True
    assert proof.lossless_cache_proven is True
    assert proof.turboquant_cache_proven is False


@pytest.mark.parametrize(
    ("model_dim", "num_heads", "num_kv_heads"),
    [
        (12, 4, 2),  # odd head dimension
        (16, 4, 3),  # query heads are not divisible by KV heads
        (18, 6, 4),  # query heads are not divisible by KV heads
    ],
)
def test_canonical_llama_manifest_with_unsupported_geometry_stays_closed(
    model_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> None:
    manifest = _manifest(
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    proof = capability_proof_for(manifest.model, manifest.topology)

    assert proof.architecture_persistence_proven is False
    assert proof.native_forward_proven is False
    assert proof.resident_inference_proven is False
    assert proof.lossless_cache_proven is False
    assert proof.turboquant_cache_proven is False


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("template", "runtime"), "megakernel"),
        (("template", "sparsity"), "moe"),
        (("template", "compression"), "ternary_b158"),
        (("template", "adapter"), "lora"),
        (("block_spec", "norm_type"), "layernorm"),
        (("block_spec", "mlp_type"), "geglu"),
        (("block_spec", "pos_encoding"), "absolute"),
        (("block_spec", "attention_backend"), "math"),
        (("block_spec", "attention_variant"), "differential"),
        (("block_spec", "residual_type"), "mhc"),
        (("block_spec", "linear_bias"), True),
        (("block_spec", "dropout_p"), 0.1),
        (("block_spec", "use_qk_norm"), True),
        (("block_spec", "compression"), "fp8_e4m3"),
        (("block_spec", "adapter_type"), "lora"),
        (("block_spec", "adapter_dim"), 8),
        (("block_spec", "rope_scaling"), {"type": "yarn", "factor": 2.0}),
        (("block_spec", "rope_theta"), 20_000.0),
        (("tie_embeddings",), True),
        (("logit_softcap",), 30.0),
    ],
)
def test_canonical_llama_gate_rejects_neighboring_semantics(
    path: tuple[str, ...],
    bad_value: object,
) -> None:
    manifest = _manifest()
    model = deepcopy(manifest.model)
    target = model["template_spec"]
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = bad_value

    proof = capability_proof_for(model, manifest.topology)

    assert proof.architecture_persistence_proven is False
    assert proof.native_forward_proven is False
    assert proof.resident_inference_proven is False
    assert proof.lossless_cache_proven is False
    assert proof.serving_proven is False
    assert proof.turboquant_cache_proven is False


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model_dim", 15),
        ("num_heads", 3),
        ("num_kv_heads", 3),
        ("num_kv_heads", 0),
    ],
)
def test_canonical_llama_gate_rejects_invalid_geometry(
    field: str,
    bad_value: int,
) -> None:
    manifest = _manifest()
    model = deepcopy(manifest.model)
    template_spec = model["template_spec"]
    target = template_spec["block_spec"] if "heads" in field else template_spec
    target[field] = bad_value

    proof = capability_proof_for(model, manifest.topology)

    assert proof.architecture_persistence_proven is False
    assert proof.native_forward_proven is False
    assert proof.resident_inference_proven is False
    assert proof.lossless_cache_proven is False


@pytest.mark.parametrize("mutation", ["rope_bypass", "wrong_norm_epsilon"])
def test_canonical_llama_gate_requires_exact_active_topology(mutation: str) -> None:
    manifest = _manifest(num_layers=2)
    topology = deepcopy(manifest.topology)
    mutated = False

    for graph in topology["graphs"]:
        path = str(graph.get("path") or "")
        if mutation == "rope_bypass" and path.endswith("/attention/subgraph"):
            for edge in graph["edges"]:
                if edge["src_node"].endswith("/rope") and edge["dst_node"].endswith(
                    "/k_repeat"
                ):
                    edge["src_node"] = edge["src_node"].replace("/rope", "/k_heads")
                    edge["src_port"] = 0
                    mutated = True
                    break
        elif mutation == "wrong_norm_epsilon" and path.endswith("/block_0/subgraph"):
            for node in graph["nodes"]:
                if node.get("instance_id") == "attn_norm":
                    node["module_config"]["eps"] = 1.0e-5
                    mutated = True
                    break
        if mutated:
            break

    assert mutated
    proof = capability_proof_for(manifest.model, topology)

    assert proof.architecture_persistence_proven is False
    assert proof.native_forward_proven is False
    assert proof.resident_inference_proven is False
    assert proof.lossless_cache_proven is False
    assert proof.serving_proven is False
    assert proof.turboquant_cache_proven is False
