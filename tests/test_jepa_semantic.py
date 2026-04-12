"""Tests for the experimental JEPA Semantic Hybrid preset."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from unittest.mock import patch

from neuralfn.config import (
    build_jepa_semantic_hybrid_spec,
    build_semantic_router_moe_megakernel_spec,
    build_semantic_router_moe_spec,
)
from neuralfn.semantic import (
    DEFAULT_SEMANTIC_VOCAB_REF,
    DIMENSION_TO_EXPERT_ID,
    LEGACY_81D_SEMANTIC_VOCAB_REF,
    LEGACY_SEMANTIC_VOCAB_REF,
    NUM_SEMANTIC_DIMS,
    NUM_VOCAB_DIMS,
    SEMANTIC_IGNORE_INDEX,
    SEMANTIC_DIMS,
    ConversationalVocabulary,
    SemanticHasher,
    SemanticMatrix,
    build_semantic_targets_from_topics,
    generate_synthetic_semantic_data,
    load_training_data,
    load_training_targets,
    semantic_targets_to_router_vectors,
    semantic_vocab_ref_for_dim,
    semantic_vocab_ref_for_graph,
    signature_to_float,
)
from neuralfn.torch_backend import (
    AttentionlessDecoderStage,
    BroadcastExpertRoutesStage,
    CompiledTorchGraph,
    RoutedAttentionExpertsStage,
    SemanticAlignmentLossStage,
    SemanticHashRouterStage,
    SemanticHasherStage,
    SemanticMoERouterStage,
    SemanticProjectorStage,
    TorchTrainConfig,
    TorchTrainer,
)
from neuralfn.inference import export_to_pt
from neuralfn.serialization import save_graph
from neuralfn.torch_templates import (
    build_gpt_root_graph,
    build_gpt_template_payload,
    build_model_spec_from_config,
)
from server.models import ExecuteRequest, GPTTemplateRequest, LoadDatasetRequest
from server.services.graph_ops import apply_gpt_template, load_dataset_source_into_graph, trace_torch_graph


def _tiny_kwargs() -> dict:
    return {
        "num_layers": 1,
        "model_dim": 32,
        "num_heads": 4,
        "num_kv_heads": 4,
        "multiple_of": 16,
        "experts": NUM_VOCAB_DIMS,
        "top_k": 2,
    }


def _cpu_graph(graph):
    graph.torch_config = {**graph.torch_config, "device": "cpu", "amp_dtype": "bfloat16"}
    return graph


# -- Semantic data layer ---------------------------------------------------

def test_semantic_dims_match_default_vocab() -> None:
    assert len(SEMANTIC_DIMS) == NUM_SEMANTIC_DIMS
    assert all(d.index == i for i, d in enumerate(SEMANTIC_DIMS))
    assert SEMANTIC_DIMS[NUM_VOCAB_DIMS].name == "taxonomy_hash"


def test_semantic_vocab_ref_for_dim_handles_legacy_shapes() -> None:
    assert semantic_vocab_ref_for_dim(9) == LEGACY_SEMANTIC_VOCAB_REF
    assert semantic_vocab_ref_for_dim(82) == LEGACY_81D_SEMANTIC_VOCAB_REF
    assert semantic_vocab_ref_for_dim(NUM_SEMANTIC_DIMS) == DEFAULT_SEMANTIC_VOCAB_REF
    assert semantic_vocab_ref_for_dim(None) == DEFAULT_SEMANTIC_VOCAB_REF


def test_semantic_vocab_ref_for_graph_uses_81d_fallback_for_refless_legacy_graph() -> None:
    graph = SimpleNamespace(torch_config={"template_spec": {"semantic_dim": 82}}, nodes={})
    assert semantic_vocab_ref_for_graph(graph) == LEGACY_81D_SEMANTIC_VOCAB_REF

    sem_node = SimpleNamespace(neuron_def=SimpleNamespace(module_config={"seq_len": 82}))
    graph = SimpleNamespace(torch_config={}, nodes={"semantic_data_source": sem_node})
    assert semantic_vocab_ref_for_graph(graph) == LEGACY_81D_SEMANTIC_VOCAB_REF


def test_generate_synthetic_data_and_load_matrix(tmp_path) -> None:
    npz = generate_synthetic_semantic_data(n=500, out_dir=tmp_path)
    assert npz.exists()
    matrix = SemanticMatrix(npz)
    assert matrix.vectors.shape == (500, NUM_SEMANTIC_DIMS)
    norms = np.linalg.norm(matrix.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_semantic_matrix_neighbors(tmp_path) -> None:
    npz = generate_synthetic_semantic_data(n=1000, out_dir=tmp_path)
    matrix = SemanticMatrix(npz)
    query = matrix.vectors[0]
    ids, scores = matrix.neighbors(query, k=10)
    assert len(ids) == 10
    assert ids[0] == 0
    assert scores[0] >= scores[-1]


def test_semantic_hasher_roundtrip(tmp_path) -> None:
    npz = generate_synthetic_semantic_data(n=500, out_dir=tmp_path)
    matrix = SemanticMatrix(npz)
    hasher = matrix.build_hasher(tables=4, planes=8)
    query = matrix.vectors[42]
    ids, scores = hasher.query(query, k=20)
    assert 42 in ids


def test_semantic_hasher_projection_planes() -> None:
    hasher = SemanticHasher(dim=NUM_SEMANTIC_DIMS, tables=8, planes=12)
    planes = hasher.get_projection_planes()
    assert planes.shape == (8, 12, NUM_SEMANTIC_DIMS)


# -- Signature hash --------------------------------------------------------

def test_signature_to_float_deterministic() -> None:
    a = signature_to_float("per_dre_psy")
    b = signature_to_float("per_dre_psy")
    assert a == b
    c = signature_to_float("air_fly_avi")
    assert a != c
    assert 0.0 <= a <= 1.0
    assert 0.0 <= c <= 1.0


def test_signature_to_float_different_buckets() -> None:
    a = signature_to_float("per_dre_psy", n_buckets=256)
    b = signature_to_float("per_dre_psy", n_buckets=4096)
    assert 0.0 <= a <= 1.0
    assert 0.0 <= b <= 1.0


# -- Vocabulary and training data ------------------------------------------

def test_load_vocabulary_and_encode_row() -> None:
    vocab = ConversationalVocabulary()
    assert len(vocab.dim_names) == NUM_VOCAB_DIMS
    assert vocab.dim_names == [d.name for d in SEMANTIC_DIMS[:NUM_VOCAB_DIMS]]
    assert vocab.raw["term_counts"] == vocab.term_counts
    assert vocab.raw["total_terms"] == sum(vocab.term_counts.values())
    assert vocab.term_counts["entity_type"] > 40
    idx = vocab.term_to_index("entity_type", "person")
    assert idx >= 0
    assert vocab.term_to_index("entity_type", "NONEXISTENT_TERM") == -1

    row = {
        "entity_type": "person",
        "action": "fly",
        "property": "fast",
        "emotion_sentiment": "joy",
        "domain": "psychology",
        "temporal": "night",
        "causality": "trigger",
        "social_register": "casual",
        "semantic_signature": "per_fly_psy",
    }
    vec = vocab.encode_row(row)
    assert vec.shape == (NUM_SEMANTIC_DIMS,)
    assert vec.dtype == np.float32
    assert 0.0 <= vec[NUM_VOCAB_DIMS] <= 1.0

    decoded = vocab.decode_vector(vec)
    assert "entity_type" in decoded
    assert "taxonomy_hash" in decoded


def test_invalid_vocabulary_metadata_is_rejected(tmp_path) -> None:
    vocab_path = tmp_path / "bad_vocab.json"
    vocab = ConversationalVocabulary()
    bad = dict(vocab.raw)
    bad["term_counts"] = dict(vocab.term_counts)
    bad["term_counts"]["entity_type"] += 1
    vocab_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="term_counts mismatch"):
        ConversationalVocabulary(vocab_path)

    bad = dict(vocab.raw)
    bad["total_terms"] = int(vocab.raw["total_terms"]) + 1
    vocab_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="total_terms mismatch"):
        ConversationalVocabulary(vocab_path)


def test_load_training_data_vocab_wrapper() -> None:
    ids, vecs = load_training_data()
    assert ids.shape[0] == 100_000
    assert vecs.shape == (100_000, NUM_SEMANTIC_DIMS)
    assert vecs.dtype == np.float32
    assert np.all(vecs[:, NUM_VOCAB_DIMS] >= 0.0) and np.all(vecs[:, NUM_VOCAB_DIMS] <= 1.0)


def test_load_training_targets_vocab_only_and_deterministic() -> None:
    ids_a, targets_a = load_training_targets(n=32, seed=7, active_dims=2)
    ids_b, targets_b = load_training_targets(n=32, seed=7, active_dims=2)
    assert np.array_equal(ids_a, ids_b)
    assert np.array_equal(targets_a, targets_b)
    assert targets_a.shape == (32, NUM_SEMANTIC_DIMS)
    active_counts = (targets_a[:, :NUM_VOCAB_DIMS] != SEMANTIC_IGNORE_INDEX).sum(axis=1)
    assert np.all(active_counts == 2)
    assert np.all(targets_a[:, NUM_VOCAB_DIMS] >= 0)


def test_semantic_router_vecs_are_normalized_and_exclude_hash_slot() -> None:
    _ids, sem_targets = load_training_targets(n=8, seed=11, active_dims=3)
    router_vecs = semantic_targets_to_router_vectors(sem_targets)
    assert router_vecs.shape == (8, NUM_VOCAB_DIMS)
    assert router_vecs.dtype == np.float32
    assert np.all(router_vecs >= 0.0)
    assert np.all(router_vecs <= 1.0)
    inactive = sem_targets[:, :NUM_VOCAB_DIMS] == SEMANTIC_IGNORE_INDEX
    assert np.allclose(router_vecs[inactive], 0.0)


# -- Torch stages ----------------------------------------------------------

def test_semantic_projector_output_shapes() -> None:
    stage = SemanticProjectorStage(input_dim=32, semantic_dim=NUM_SEMANTIC_DIMS, residual_dim=64)
    x = torch.randn(2, 32)
    sem, res, topic_logits = stage(x)
    assert sem.shape == (2, NUM_SEMANTIC_DIMS)
    assert res.shape == (2, 1, 64)
    assert topic_logits.shape[0] == 2
    assert topic_logits.shape[1] == NUM_VOCAB_DIMS


def test_semantic_alignment_loss_refless_legacy_shape_uses_81d_vocab() -> None:
    legacy_vocab = ConversationalVocabulary(LEGACY_81D_SEMANTIC_VOCAB_REF)
    stage = SemanticAlignmentLossStage()
    pred = torch.randn(2, legacy_vocab.num_vocab_dims, legacy_vocab.max_terms)
    target = torch.full((2, legacy_vocab.vector_dim), SEMANTIC_IGNORE_INDEX, dtype=torch.long)
    target[:, 0] = 0
    loss = stage(pred, target)
    assert torch.isfinite(loss)
    assert len(stage.term_counts) == legacy_vocab.num_vocab_dims


def test_semantic_hasher_stage() -> None:
    stage = SemanticHasherStage(dim=NUM_SEMANTIC_DIMS, tables=4, planes=6)
    vec = torch.randn(3, NUM_SEMANTIC_DIMS)
    indices = stage(vec)
    assert indices.shape == (3, 4)
    assert indices.dtype == torch.long


def test_semantic_moe_router_entropy() -> None:
    stage = SemanticMoERouterStage(n_experts=NUM_VOCAB_DIMS, semantic_dim=NUM_SEMANTIC_DIMS, top_k=2)
    vec = torch.randn(16, NUM_SEMANTIC_DIMS)
    weights, indices = stage(vec)
    assert weights.shape == (16, 1, 2)
    assert indices.shape == (16, 1, 2)
    chosen = indices.squeeze(1).flatten()
    unique = len(set(chosen.tolist()))
    assert unique >= 2


def test_semantic_hash_router_shapes() -> None:
    vocab = ConversationalVocabulary()
    stage = SemanticHashRouterStage(
        n_experts=NUM_VOCAB_DIMS,
        semantic_dim=NUM_SEMANTIC_DIMS,
        top_k=2,
        tables=4,
        n_buckets=64,
    )
    vec = torch.randn(16, NUM_SEMANTIC_DIMS)
    bucket_indices = torch.randint(0, 64, (16, 4))
    topic_logits = torch.randn(16, NUM_VOCAB_DIMS, vocab.max_terms)
    sem_targets = torch.full((16, NUM_SEMANTIC_DIMS), SEMANTIC_IGNORE_INDEX, dtype=torch.long)
    weights, indices = stage(vec, bucket_indices, topic_logits, sem_targets)
    assert weights.shape == (16, 2)
    assert indices.shape == (16, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(16), atol=1e-5)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < NUM_VOCAB_DIMS


def test_semantic_hash_router_teacher_forces_dimension_map() -> None:
    vocab = ConversationalVocabulary()
    stage = SemanticHashRouterStage(
        n_experts=NUM_VOCAB_DIMS,
        semantic_dim=NUM_SEMANTIC_DIMS,
        top_k=2,
        tables=4,
        n_buckets=64,
    )
    vec = torch.randn(2, NUM_SEMANTIC_DIMS)
    bucket_indices = torch.randint(0, 64, (2, 4))
    topic_logits = torch.randn(2, NUM_VOCAB_DIMS, vocab.max_terms)
    sem_targets = torch.full((2, NUM_SEMANTIC_DIMS), SEMANTIC_IGNORE_INDEX, dtype=torch.long)
    sem_targets[0] = torch.from_numpy(
        build_semantic_targets_from_topics({"emotion_sentiment": "love", "domain": "psychology"})
    )
    sem_targets[1] = torch.from_numpy(
        build_semantic_targets_from_topics({"entity_type": "person", "action": "write"})
    )
    weights, indices = stage(vec, bucket_indices, topic_logits, sem_targets)
    assert set(indices[0].tolist()) == {
        DIMENSION_TO_EXPERT_ID["emotion_sentiment"],
        DIMENSION_TO_EXPERT_ID["domain"],
    }
    assert set(indices[1].tolist()) == {
        DIMENSION_TO_EXPERT_ID["entity_type"],
        DIMENSION_TO_EXPERT_ID["action"],
    }
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_broadcast_expert_routes_stage_expands_batch_routes_to_sequence() -> None:
    stage = BroadcastExpertRoutesStage()
    hidden = torch.randn(3, 7, 32)
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)
    indices = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long)
    routing_weights, routing_indices = stage(hidden, weights, indices)
    assert routing_weights.shape == (3, 7, 2)
    assert routing_indices.shape == (3, 7, 2)
    assert torch.allclose(routing_weights[:, 0], weights, atol=1e-6)
    assert torch.equal(routing_indices[:, 0], indices)


def test_build_semantic_targets_examples_survive_vocab_expansion() -> None:
    vocab = ConversationalVocabulary()
    assert vocab.term_to_index("emotion_sentiment", "love") >= 0
    assert vocab.term_to_index("domain", "psychology") >= 0
    assert vocab.term_to_index("entity_type", "person") >= 0
    assert vocab.term_to_index("action", "write") >= 0


def test_routed_attention_experts_output_and_gradients() -> None:
    stage = RoutedAttentionExpertsStage(
        model_dim=32,
        num_heads=4,
        num_kv_heads=4,
        rope_base=10_000.0,
        qk_gain_init=1.0,
        experts=4,
        top_k=2,
    )
    hidden = torch.randn(3, 8, 32, requires_grad=True)
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]], dtype=torch.float32)
    indices = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    out = stage(hidden, weights, indices)
    assert out.shape == hidden.shape
    out.sum().backward()
    assert hidden.grad is not None
    assert stage.q_proj.grad is not None


def test_attentionless_decoder_output_shape() -> None:
    stage = AttentionlessDecoderStage(
        semantic_dim=NUM_SEMANTIC_DIMS,
        residual_dim=64,
        vocab_size=128,
        n_buckets=64,
    )
    buckets = torch.randint(0, 64, (2, 4))
    expert_out = torch.randn(2, 64)
    logits = stage(buckets, expert_out)
    assert logits.shape == (2, 1, 128)


# -- Template preset integration -------------------------------------------

def test_jepa_semantic_hybrid_spec() -> None:
    spec = build_jepa_semantic_hybrid_spec(**_tiny_kwargs(), vocab_size=128)
    assert spec.template.objective == "jepa_semantic"
    assert spec.template.sparsity == "moe"
    assert spec.semantic_dim == NUM_SEMANTIC_DIMS
    assert spec.semantic_residual_dim == 64
    assert spec.semantic_vocab_ref == DEFAULT_SEMANTIC_VOCAB_REF
    assert spec.block_spec.experts == NUM_VOCAB_DIMS


def test_semantic_router_moe_spec() -> None:
    spec = build_semantic_router_moe_spec(**_tiny_kwargs(), vocab_size=128)
    assert spec.template.objective == "semantic_router"
    assert spec.template.backbone == "mixllama"
    assert spec.template.sparsity == "moe"
    assert spec.semantic_vocab_ref == DEFAULT_SEMANTIC_VOCAB_REF
    assert spec.block_spec.experts == NUM_VOCAB_DIMS
    assert spec.block_spec.router_aux_loss_coef == 0.0


def test_semantic_preset_preview_defaults_use_vocab_expert_count() -> None:
    jepa_spec = build_model_spec_from_config({"preset": "jepa_semantic_hybrid"}, preview_defaults=True)
    router_spec = build_model_spec_from_config({"preset": "semantic_router_moe"}, preview_defaults=True)
    assert jepa_spec.block_spec.experts == NUM_VOCAB_DIMS
    assert router_spec.block_spec.experts == NUM_VOCAB_DIMS


def test_jepa_semantic_hybrid_payload() -> None:
    payload = build_gpt_template_payload(
        name="jsh_payload", config={"preset": "jepa_semantic_hybrid"}
    )
    assert payload["node_def"]["kind"] == "subgraph"
    assert isinstance(payload["variant_library"], dict)


def test_semantic_router_moe_payload() -> None:
    payload = build_gpt_template_payload(name="srm_payload", config={"preset": "semantic_router_moe"})
    assert payload["node_def"]["kind"] == "subgraph"
    assert isinstance(payload["variant_library"], dict)


def test_jepa_semantic_hybrid_resolve_variants() -> None:
    spec = build_model_spec_from_config(
        {"preset": "jepa_semantic_hybrid", **_tiny_kwargs()}, preview_defaults=True
    )
    graph = build_gpt_root_graph(name="jsh_resolve", model_spec=spec)
    graph.resolve_variant_library()


def test_semantic_router_moe_resolve_variants() -> None:
    spec = build_model_spec_from_config(
        {"preset": "semantic_router_moe", **_tiny_kwargs()}, preview_defaults=True
    )
    graph = build_gpt_root_graph(name="srm_resolve", model_spec=spec)
    graph.resolve_variant_library()


def test_saved_semantic_graph_json_persists_vocab_ref_and_router_vec_contract(tmp_path) -> None:
    spec = build_model_spec_from_config(
        {
            "preset": "semantic_router_moe",
            "vocab_size": 256,
            "experimental_semantic_router_vecs": True,
            **_tiny_kwargs(),
        },
        preview_defaults=True,
    )
    graph = build_gpt_root_graph(name="srm_router_vecs_json", model_spec=spec)
    graph_path = tmp_path / "semantic_router_vecs.json"
    save_graph(graph, graph_path)

    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    template_spec = dict(payload.get("torch_config", {}).get("template_spec", {}) or {})
    sem_source = payload["nodes"]["semantic_data_source"]["neuron_def"]
    output_names = [port["name"] for port in sem_source["output_ports"]]

    assert template_spec["semantic_vocab_ref"] == DEFAULT_SEMANTIC_VOCAB_REF
    assert template_spec["experimental_semantic_router_vecs"] is True
    assert output_names == ["sem_targets", "semantic_router_vecs"]


def test_semantic_router_vec_graph_compile_and_forward() -> None:
    spec = build_model_spec_from_config(
        {
            "preset": "semantic_router_moe",
            "vocab_size": 256,
            "experimental_semantic_router_vecs": True,
            **_tiny_kwargs(),
        },
        preview_defaults=True,
    )
    graph = _cpu_graph(build_gpt_root_graph(name="srm_router_vecs_fwd", model_spec=spec))
    compiled = CompiledTorchGraph(graph)
    tokens = torch.randint(0, 256, (2, 9))
    targets = torch.randint(0, 256, (2, 9))
    _ids, sem_array = load_training_targets(n=2, active_dims=2)
    sem_targets = torch.from_numpy(sem_array)
    router_vecs = torch.from_numpy(semantic_targets_to_router_vectors(sem_array))
    outputs = compiled(tokens, targets, sem_targets, router_vecs)
    assert len(outputs) >= 1
    assert outputs[0].ndim == 0


def test_jepa_semantic_hybrid_compile_and_forward() -> None:
    spec = build_model_spec_from_config(
        {"preset": "jepa_semantic_hybrid", "vocab_size": 256, **_tiny_kwargs()},
        preview_defaults=True,
    )
    graph = _cpu_graph(build_gpt_root_graph(name="jsh_fwd", model_spec=spec))
    assert "semantic_data_source" in graph.nodes
    assert "dataset_source" in graph.nodes
    assert "tokens_in" not in graph.nodes
    compiled = CompiledTorchGraph(graph)
    tokens = torch.randint(0, 256, (2, 9))
    targets = torch.randint(0, 256, (2, 9))
    _ids, sem_array = load_training_targets(n=2, active_dims=2)
    sem_targets = torch.from_numpy(sem_array)
    outputs = compiled(tokens, targets, sem_targets)
    assert len(outputs) >= 1
    assert outputs[0].ndim == 0


def test_semantic_router_moe_compile_and_forward() -> None:
    spec = build_model_spec_from_config(
        {"preset": "semantic_router_moe", "vocab_size": 256, **_tiny_kwargs()},
        preview_defaults=True,
    )
    graph = _cpu_graph(build_gpt_root_graph(name="srm_fwd", model_spec=spec))
    assert "semantic_data_source" in graph.nodes
    assert "dataset_source" in graph.nodes
    compiled = CompiledTorchGraph(graph)
    tokens = torch.randint(0, 256, (2, 9))
    targets = torch.randint(0, 256, (2, 9))
    _ids, sem_array = load_training_targets(n=2, active_dims=2)
    sem_targets = torch.from_numpy(sem_array)
    outputs = compiled(tokens, targets, sem_targets)
    assert len(outputs) >= 1
    assert outputs[0].ndim == 0


def test_jepa_semantic_hybrid_apply_template() -> None:
    graph = apply_gpt_template(
        GPTTemplateRequest(name="jsh_apply", config={"preset": "jepa_semantic_hybrid"})
    )
    assert "model" in graph.nodes
    assert "semantic_data_source" in graph.nodes
    assert "dataset_source" in graph.nodes
    assert "tokens_in" not in graph.nodes
    assert graph.output_node_ids == ["loss_out"]
    assert "dataset_source" in graph.input_node_ids
    assert "semantic_data_source" in graph.input_node_ids
    ds_ports = [port.name for port in graph.nodes["dataset_source"].neuron_def.output_ports]
    assert ds_ports == ["tokens", "targets"]


def test_semantic_router_moe_apply_template() -> None:
    graph = apply_gpt_template(
        GPTTemplateRequest(name="srm_apply", config={"preset": "semantic_router_moe"})
    )
    assert "model" in graph.nodes
    assert "semantic_data_source" in graph.nodes
    assert "dataset_source" in graph.nodes
    assert "tokens_in" not in graph.nodes
    assert graph.output_node_ids == ["loss_out"]
    assert "dataset_source" in graph.input_node_ids
    assert "semantic_data_source" in graph.input_node_ids


def test_jepa_semantic_hybrid_template_payload_includes_extra_nodes() -> None:
    payload = build_gpt_template_payload(
        name="jsh_extra", config={"preset": "jepa_semantic_hybrid"}
    )
    assert "extra_nodes" in payload
    extra_ids = [n["instance_id"] for n in payload["extra_nodes"]]
    assert "semantic_data_source" in extra_ids


def test_jepa_semantic_hybrid_ema_training_with_shipped_data() -> None:
    spec = build_jepa_semantic_hybrid_spec(**_tiny_kwargs(), vocab_size=256, ema_decay=0.9)
    graph = _cpu_graph(build_gpt_root_graph(name="jsh_train", model_spec=spec))
    assert "semantic_data_source" in graph.nodes
    assert "dataset_source" in graph.nodes
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(epochs=1, batch_size=4, learning_rate=1e-3, max_steps=1, device="cpu"),
    )
    losses = trainer.train([], [])
    assert len(losses) == 1


def test_jepa_semantic_hybrid_real_5epoch_training_regression(tmp_path) -> None:
    ids, sem_targets = load_training_targets(n=16, active_dims=2)
    tiny_ids = ids[:16]
    tiny_targets = sem_targets[:16]

    spec = build_jepa_semantic_hybrid_spec(**_tiny_kwargs(), vocab_size=256, ema_decay=0.9)
    graph = _cpu_graph(build_gpt_root_graph(name="jsh_train_5ep", model_spec=spec))
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(epochs=5, batch_size=4, learning_rate=1e-3, device="cpu"),
    )

    with patch("neuralfn.semantic.load_training_targets", return_value=(tiny_ids, tiny_targets)):
        losses = trainer.train([], [])

    assert len(losses) == 5
    assert all(math.isfinite(float(loss)) for loss in losses)
    export_path = tmp_path / "jsh_train_5ep.pt"
    export_to_pt(graph, export_path)
    assert export_path.exists()


def test_jepa_semantic_hybrid_trace_uses_builtin_semantic_preview() -> None:
    spec = build_model_spec_from_config(
        {"preset": "jepa_semantic_hybrid", "vocab_size": 256, **_tiny_kwargs()},
        preview_defaults=True,
    )
    graph = _cpu_graph(build_gpt_root_graph(name="jsh_trace", model_spec=spec))
    result = trace_torch_graph(graph, ExecuteRequest())
    assert result["source"] == "dataset"
    assert "trace" in result
    assert "sample_inputs" in result


def test_jepa_semantic_hybrid_trace_uses_text_plus_semantic_preview() -> None:
    spec = build_model_spec_from_config(
        {"preset": "jepa_semantic_hybrid", "vocab_size": 256, **_tiny_kwargs()},
        preview_defaults=True,
    )
    graph = _cpu_graph(build_gpt_root_graph(name="jsh_trace_dual", model_spec=spec))
    load_dataset_source_into_graph(graph, LoadDatasetRequest(dataset_names=["dummy"], seq_len=8))

    tiny_text_inputs = [[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]]
    tiny_text_targets = [[2, 3, 4, 5, 6, 7, 8, 9], [3, 4, 5, 6, 7, 8, 9, 10]]
    ids, sem_targets = load_training_targets(n=4, active_dims=2)
    tiny_ids = ids[:4]
    tiny_targets = sem_targets[:4]

    with patch("server.services.graph_ops.load_dataset_tokens", return_value=(tiny_text_inputs, tiny_text_targets)):
        with patch("neuralfn.semantic.load_training_targets", return_value=(tiny_ids, tiny_targets)):
            result = trace_torch_graph(graph, ExecuteRequest())

    assert result["source"] == "dataset"
    assert "trace" in result
    assert "sample_inputs" in result


def test_jepa_semantic_hybrid_real_5epoch_hybrid_training_regression() -> None:
    spec = build_jepa_semantic_hybrid_spec(**_tiny_kwargs(), vocab_size=256, ema_decay=0.9)
    graph = _cpu_graph(build_gpt_root_graph(name="jsh_train_dual_5ep", model_spec=spec))
    load_dataset_source_into_graph(graph, LoadDatasetRequest(dataset_names=["dummy"], seq_len=8))

    text_x = torch.randint(0, 128, (16, 8), dtype=torch.long)
    text_y = torch.randint(0, 128, (16, 8), dtype=torch.long)
    _ids, sem_targets = load_training_targets(n=16, active_dims=2)
    sem_x = torch.from_numpy(sem_targets)

    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(epochs=5, batch_size=4, learning_rate=1e-3, device="cpu"),
    )

    with patch.object(TorchTrainer, "_load_dataset_for_graph", return_value=torch.utils.data.TensorDataset(text_x, text_y)) as text_mock:
        with patch.object(TorchTrainer, "_load_semantic_tensors", return_value={"sem_targets": sem_x}) as sem_mock:
            losses = trainer.train([], [])

    assert text_mock.called
    assert sem_mock.called
    assert len(losses) == 5
    assert all(math.isfinite(float(loss)) for loss in losses)


def test_jepa_semantic_hybrid_loss_connects_router_and_expert_branch() -> None:
    spec = build_model_spec_from_config(
        {"preset": "jepa_semantic_hybrid", "vocab_size": 128, **_tiny_kwargs()},
        preview_defaults=True,
    )
    graph = _cpu_graph(build_gpt_root_graph(name="jsh_grad_path", model_spec=spec))
    compiled = CompiledTorchGraph(graph)
    TorchTrainer._prepare_ema_targets(compiled)
    tokens = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    _ids, sem_array = load_training_targets(n=2, active_dims=2)
    sem_targets = torch.from_numpy(sem_array)
    loss = compiled(tokens, targets, sem_targets)[0]
    loss.backward()

    model = compiled.node_modules["model"]
    hash_router = model.node_modules["hash_router"]
    routed_experts = model.node_modules["routed_experts"]
    lm_head = model.node_modules["lm_head"]
    assert hash_router.dimension_bias.grad is not None
    assert routed_experts.q_proj.grad is not None
    assert next(lm_head.parameters()).grad is not None


def test_jepa_semantic_hybrid_early_stop_still_syncs_and_exports(tmp_path) -> None:
    spec = build_jepa_semantic_hybrid_spec(**_tiny_kwargs(), vocab_size=256, ema_decay=0.9)
    graph = _cpu_graph(build_gpt_root_graph(name="jsh_train_stop", model_spec=spec))
    load_dataset_source_into_graph(graph, LoadDatasetRequest(dataset_names=["dummy"], seq_len=8))

    text_x = torch.randint(0, 128, (16, 8), dtype=torch.long)
    text_y = torch.randint(0, 128, (16, 8), dtype=torch.long)
    _ids, sem_targets = load_training_targets(n=16, active_dims=2)
    sem_x = torch.from_numpy(sem_targets)

    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(epochs=5, batch_size=4, learning_rate=1e-3, device="cpu"),
    )

    def on_epoch(epoch: int, _loss: float) -> None:
        if epoch == 0:
            trainer.stop()

    with patch.object(TorchTrainer, "_load_dataset_for_graph", return_value=torch.utils.data.TensorDataset(text_x, text_y)):
        with patch.object(TorchTrainer, "_load_semantic_tensors", return_value={"sem_targets": sem_x}):
            losses = trainer.train([], [], on_epoch=on_epoch)

    assert len(losses) == 1
    model_graph = graph.nodes["model"].neuron_def.subgraph
    assert model_graph is not None
    predictor_state = model_graph.nodes["predictor"].neuron_def.module_state
    assert predictor_state not in ("", None)

    pt_path = tmp_path / "jsh_interrupted.pt"
    json_path = tmp_path / "jsh_interrupted.json"
    export_to_pt(graph, pt_path)
    save_graph(graph, json_path)
    assert pt_path.exists()
    assert json_path.exists()


def test_jepa_semantic_hybrid_parameter_golf_profile_smoke() -> None:
    spec = build_jepa_semantic_hybrid_spec(**_tiny_kwargs(), vocab_size=256, ema_decay=0.9)
    graph = _cpu_graph(build_gpt_root_graph(name="jsh_param_golf", model_spec=spec))
    load_dataset_source_into_graph(graph, LoadDatasetRequest(dataset_names=["dummy"], seq_len=8))

    text_x = torch.randint(0, 128, (8, 8), dtype=torch.long)
    text_y = torch.randint(0, 128, (8, 8), dtype=torch.long)
    _ids, sem_targets = load_training_targets(n=8, active_dims=2)
    sem_x = torch.from_numpy(sem_targets)
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=1,
            batch_size=2,
            learning_rate=3e-4,
            max_steps=1,
            device="cpu",
            optimizer_profile="parameter_golf",
            train_batch_tokens=32,
            warmup_steps=1,
            warmdown_iters=1,
            embed_lr=2e-4,
            head_lr=2e-4,
            matrix_lr=3e-4,
            scalar_lr=1e-4,
            grad_clip_norm=1.0,
        ),
    )

    with patch.object(TorchTrainer, "_load_dataset_for_graph", return_value=torch.utils.data.TensorDataset(text_x, text_y)):
        with patch.object(TorchTrainer, "_load_semantic_tensors", return_value={"sem_targets": sem_x}):
            losses = trainer.train([], [])

    assert len(losses) == 1
    assert math.isfinite(float(losses[0]))


def test_semantic_router_moe_megakernel_parameter_golf_smoke() -> None:
    spec = build_semantic_router_moe_megakernel_spec(**_tiny_kwargs(), vocab_size=256)
    graph = _cpu_graph(build_gpt_root_graph(name="srm_megakernel_param_golf", model_spec=spec))
    load_dataset_source_into_graph(graph, LoadDatasetRequest(dataset_names=["dummy"], seq_len=8))

    text_x = torch.randint(0, 128, (8, 8), dtype=torch.long)
    text_y = torch.randint(0, 128, (8, 8), dtype=torch.long)
    _ids, sem_targets = load_training_targets(n=8, active_dims=2)
    sem_x = torch.from_numpy(sem_targets)
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=1,
            batch_size=2,
            learning_rate=3e-4,
            max_steps=1,
            device="cpu",
            optimizer_profile="parameter_golf",
            train_batch_tokens=32,
            warmup_steps=1,
            warmdown_iters=1,
            embed_lr=2e-4,
            head_lr=2e-4,
            matrix_lr=3e-4,
            scalar_lr=1e-4,
            grad_clip_norm=1.0,
        ),
    )

    with patch.object(TorchTrainer, "_load_dataset_for_graph", return_value=torch.utils.data.TensorDataset(text_x, text_y)):
        with patch.object(TorchTrainer, "_load_semantic_tensors", return_value={"sem_targets": sem_x}):
            losses = trainer.train([], [])

    assert len(losses) == 1
    assert math.isfinite(float(losses[0]))
