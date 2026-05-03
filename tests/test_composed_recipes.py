from __future__ import annotations

from neuralfn.config import build_composed_lm_spec
from neuralfn.semantic import NUM_VOCAB_DIMS
from neuralfn.torch_backend import CompiledTorchGraph
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


def test_build_composed_dense_gpt2_defaults_to_eager_runtime() -> None:
    spec = build_composed_lm_spec(
        base_model="gpt2",
        topology="dense",
        runtime="default",
        vocab_size=1024,
        num_layers=2,
        model_dim=128,
        num_heads=4,
    )
    assert spec.template.backbone == "gpt2"
    assert spec.template.runtime == "eager"
    assert spec.template.objective == "ar"
    graph = build_gpt_root_graph(name="dense_gpt2", model_spec=spec)
    CompiledTorchGraph(graph)
    assert graph.input_node_ids == ["tokens_in", "targets_in"]


def test_build_composed_standard_moe_jepa_uses_tokens_and_targets_contract() -> None:
    spec = build_composed_lm_spec(
        base_model="llama",
        topology="moe",
        router_mode="standard",
        use_jepa=True,
        runtime="default",
        vocab_size=1024,
        num_layers=2,
        model_dim=128,
        num_heads=4,
        num_kv_heads=4,
        experts=8,
        top_k=2,
    )
    assert spec.template.objective == "ar_jepa"
    graph = build_gpt_root_graph(name="llama_moe_jepa", model_spec=spec)
    CompiledTorchGraph(graph)
    assert graph.input_node_ids == ["tokens_in", "targets_in"]


def test_build_composed_semantic_router_jepa_graph_uses_semantic_inputs() -> None:
    spec = build_composed_lm_spec(
        base_model="nanogpt",
        topology="moe",
        router_mode="semantic",
        use_jepa=True,
        runtime="megakernel",
        vocab_size=1024,
        num_layers=2,
        model_dim=128,
        num_heads=4,
        experts=NUM_VOCAB_DIMS,
        top_k=2,
    )
    assert spec.template.objective == "semantic_router_jepa"
    graph = build_gpt_root_graph(name="nanogpt_semantic_router_jepa", model_spec=spec)
    CompiledTorchGraph(graph)
    assert graph.input_node_ids == ["dataset_source", "semantic_data_source"]


def test_build_model_spec_from_config_accepts_composed_fields() -> None:
    spec = build_model_spec_from_config(
        {
            "base_model": "gpt2",
            "topology": "moe",
            "router_mode": "semantic",
            "use_jepa": True,
            "runtime": "megakernel",
            "vocab_size": 1024,
            "num_layers": 2,
            "model_dim": 128,
            "num_heads": 4,
            "experts": NUM_VOCAB_DIMS,
            "top_k": 2,
        }
    )
    assert spec.template.backbone == "gpt2"
    assert spec.template.router_mode == "semantic"
    assert spec.template.objective == "semantic_router_jepa"
    assert spec.template.runtime == "megakernel"
