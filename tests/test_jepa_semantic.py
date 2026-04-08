"""Tests for the experimental JEPA Semantic Hybrid preset."""

from __future__ import annotations

import numpy as np
import torch

from neuralfn.config import build_jepa_semantic_hybrid_spec
from neuralfn.semantic import (
    NUM_SEMANTIC_DIMS,
    SEMANTIC_DIMS,
    ConversationalVocabulary,
    SemanticHasher,
    SemanticMatrix,
    generate_synthetic_semantic_data,
    load_training_data,
    signature_to_float,
)
from neuralfn.torch_backend import (
    AttentionlessDecoderStage,
    CompiledTorchGraph,
    SemanticHasherStage,
    SemanticMoERouterStage,
    SemanticProjectorStage,
    TorchTrainConfig,
    TorchTrainer,
)
from neuralfn.torch_templates import (
    build_gpt_root_graph,
    build_gpt_template_payload,
    build_model_spec_from_config,
)
from server.models import GPTTemplateRequest
from server.services.graph_ops import apply_gpt_template


def _tiny_kwargs() -> dict:
    return {
        "num_layers": 1,
        "model_dim": 32,
        "num_heads": 4,
        "num_kv_heads": 4,
        "multiple_of": 16,
        "experts": 4,
        "top_k": 2,
    }


def _cpu_graph(graph):
    graph.torch_config = {**graph.torch_config, "device": "cpu", "amp_dtype": "bfloat16"}
    return graph


# -- Semantic data layer ---------------------------------------------------

def test_semantic_dims_has_9_entries() -> None:
    assert len(SEMANTIC_DIMS) == 9
    assert NUM_SEMANTIC_DIMS == 9
    assert all(d.index == i for i, d in enumerate(SEMANTIC_DIMS))
    assert SEMANTIC_DIMS[8].name == "taxonomy_hash"


def test_generate_synthetic_data_and_load_matrix(tmp_path) -> None:
    npz = generate_synthetic_semantic_data(n=500, out_dir=tmp_path)
    assert npz.exists()
    matrix = SemanticMatrix(npz)
    assert matrix.vectors.shape == (500, 9)
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
    hasher = SemanticHasher(dim=9, tables=8, planes=12)
    planes = hasher.get_projection_planes()
    assert planes.shape == (8, 12, 9)


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
    assert len(vocab.dim_names) == 8
    assert "entity_type" in vocab.dim_names
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
    assert vec.shape == (9,)
    assert vec.dtype == np.float32
    assert 0.0 <= vec[8] <= 1.0

    decoded = vocab.decode_vector(vec)
    assert "entity_type" in decoded
    assert "taxonomy_hash" in decoded


def test_load_training_csv() -> None:
    ids, vecs = load_training_data()
    assert ids.shape[0] == 100_000
    assert vecs.shape == (100_000, 9)
    assert vecs.dtype == np.float32
    assert np.all(vecs[:, 8] >= 0.0) and np.all(vecs[:, 8] <= 1.0)


# -- Torch stages ----------------------------------------------------------

def test_semantic_projector_output_shapes() -> None:
    stage = SemanticProjectorStage(input_dim=32, semantic_dim=9, residual_dim=64)
    x = torch.randn(2, 32)
    sem, res = stage(x)
    assert sem.shape == (2, 9)
    assert res.shape == (2, 1, 64)


def test_semantic_hasher_stage() -> None:
    stage = SemanticHasherStage(dim=9, tables=4, planes=6)
    vec = torch.randn(3, 9)
    indices = stage(vec)
    assert indices.shape == (3, 4)
    assert indices.dtype == torch.long


def test_semantic_moe_router_entropy() -> None:
    stage = SemanticMoERouterStage(n_experts=8, semantic_dim=9, top_k=2)
    vec = torch.randn(16, 9)
    weights, indices = stage(vec)
    assert weights.shape == (16, 1, 2)
    assert indices.shape == (16, 1, 2)
    chosen = indices.squeeze(1).flatten()
    unique = len(set(chosen.tolist()))
    assert unique >= 2


def test_attentionless_decoder_output_shape() -> None:
    stage = AttentionlessDecoderStage(semantic_dim=9, residual_dim=64, vocab_size=128, n_buckets=64)
    buckets = torch.randint(0, 64, (2, 4))
    expert_out = torch.randn(2, 64)
    logits = stage(buckets, expert_out)
    assert logits.shape == (2, 1, 128)


# -- Template preset integration -------------------------------------------

def test_jepa_semantic_hybrid_spec() -> None:
    spec = build_jepa_semantic_hybrid_spec(**_tiny_kwargs(), vocab_size=128)
    assert spec.template.objective == "jepa_semantic"
    assert spec.template.sparsity == "moe"
    assert spec.semantic_dim == 9
    assert spec.semantic_residual_dim == 64


def test_jepa_semantic_hybrid_payload() -> None:
    payload = build_gpt_template_payload(
        name="jsh_payload", config={"preset": "jepa_semantic_hybrid"}
    )
    assert payload["node_def"]["kind"] == "subgraph"
    assert isinstance(payload["variant_library"], dict)


def test_jepa_semantic_hybrid_resolve_variants() -> None:
    spec = build_model_spec_from_config(
        {"preset": "jepa_semantic_hybrid", **_tiny_kwargs()}, preview_defaults=True
    )
    graph = build_gpt_root_graph(name="jsh_resolve", model_spec=spec)
    graph.resolve_variant_library()


def test_jepa_semantic_hybrid_compile_and_forward() -> None:
    spec = build_model_spec_from_config(
        {"preset": "jepa_semantic_hybrid", "vocab_size": 128, **_tiny_kwargs()},
        preview_defaults=True,
    )
    graph = _cpu_graph(build_gpt_root_graph(name="jsh_fwd", model_spec=spec))
    compiled = CompiledTorchGraph(graph)
    tokens = torch.randint(0, 128, (2, 8))
    outputs = compiled(tokens)
    assert len(outputs) >= 1
    assert outputs[0].ndim == 0


def test_jepa_semantic_hybrid_apply_template() -> None:
    graph = apply_gpt_template(
        GPTTemplateRequest(name="jsh_apply", config={"preset": "jepa_semantic_hybrid"})
    )
    assert "model" in graph.nodes
    assert graph.output_node_ids == ["loss_out"]


def test_jepa_semantic_hybrid_ema_training() -> None:
    spec = build_jepa_semantic_hybrid_spec(**_tiny_kwargs(), vocab_size=128, ema_decay=0.9)
    graph = _cpu_graph(build_gpt_root_graph(name="jsh_train", model_spec=spec))
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(epochs=1, batch_size=2, learning_rate=1e-3, max_steps=1, device="cpu"),
    )
    tokens = torch.randint(0, 128, (4, 8))
    losses = trainer.train(tokens, tokens)
    assert len(losses) == 1
