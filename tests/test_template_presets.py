from __future__ import annotations

import uuid

import torch

from neuralfn.graph import Edge, NeuronGraph, NeuronInstance
from neuralfn.neuron import neuron_from_source, subgraph_neuron
from neuralfn.port import Port
from neuralfn.config import (
    build_hnet_lm_spec,
    build_llm_jepa_spec,
    build_ttt_llama_spec,
    build_universal_llama_spec,
)
from neuralfn.torch_backend import CompiledTorchGraph, TorchTrainConfig, TorchTrainer
from neuralfn.torch_templates import build_gpt_root_graph, build_gpt_template_payload, build_model_spec_from_config
from server.dataset_manager import DATASETS_DIR, load_dataset_bytes
from server.models import GPTTemplateRequest, LoadDatasetRequest
from server.services.graph_ops import apply_gpt_template, load_dataset_source_into_graph


PRESETS = [
    "nanogpt",
    "gpt2",
    "llama",
    "moe",
    "llama_fast",
    "mixllama_fast",
    "jamba",
    "ternary_b158",
    "seq2seq",
    "diffusion",
    "ttt_llama",
    "llm_jepa",
    "hnet_lm",
    "universal_llama",
]


def _cpu_graph(graph):
    graph.torch_config = {
        **graph.torch_config,
        "device": "cpu",
        "amp_dtype": "bfloat16",
    }
    return graph


def _tiny_kwargs() -> dict[str, int]:
    return {
        "num_layers": 1,
        "model_dim": 32,
        "num_heads": 4,
        "num_kv_heads": 4,
        "multiple_of": 16,
    }


def _make_terminal_def(role: str, port_name: str):
    source = f"def {role}(x):\n    return x\n"
    ports = [Port(port_name, range=(-1_000_000.0, 1_000_000.0), precision=0.001, dtype="float")]
    return neuron_from_source(source, role, ports, ports)


def _make_variant_graph(name: str) -> NeuronGraph:
    graph = NeuronGraph(name=name)
    graph.add_node(NeuronInstance(_make_terminal_def("input", "x"), instance_id="x_in", position=(0, 0)))
    graph.add_node(NeuronInstance(_make_terminal_def("output", "x"), instance_id="x_out", position=(200, 0)))
    graph.add_edge(Edge(src_node="x_in", src_port=0, dst_node="x_out", dst_port=0))
    graph.input_node_ids = ["x_in"]
    graph.output_node_ids = ["x_out"]
    return graph


def _make_alias_root(link_family: str, available_family: str) -> NeuronGraph:
    variant_graph = _make_variant_graph(f"{available_family}_default")
    root = NeuronGraph(name=f"{link_family}_root", variant_library={available_family: {"default": variant_graph}})
    root.add_node(NeuronInstance(_make_terminal_def("input", "x"), instance_id="x_in", position=(0, 0)))
    root.add_node(
        NeuronInstance(
            subgraph_neuron(
                variant_graph,
                name="block",
                input_aliases=["x"],
                output_aliases=["x"],
                variant_ref={"family": link_family, "version": "default"},
            ),
            instance_id="block",
            position=(200, 0),
        )
    )
    root.add_node(NeuronInstance(_make_terminal_def("output", "x"), instance_id="x_out", position=(400, 0)))
    root.add_edge(Edge(src_node="x_in", src_port=0, dst_node="block", dst_port=0))
    root.add_edge(Edge(src_node="block", src_port=0, dst_node="x_out", dst_port=0))
    root.input_node_ids = ["x_in"]
    root.output_node_ids = ["x_out"]
    return root


def test_build_gpt_template_payload_supports_all_presets() -> None:
    for preset in PRESETS:
        payload = build_gpt_template_payload(name=f"{preset}_payload", config={"preset": preset})
        assert payload["node_def"]["kind"] == "subgraph"
        assert isinstance(payload["variant_library"], dict)
        assert payload["graph_settings"]["torch_config"]["template_spec"]["template"]


def test_reported_presets_resolve_variant_libraries() -> None:
    for preset in ["moe", "mixllama_fast", "jamba", "ternary_b158", "seq2seq"]:
        spec = build_model_spec_from_config({"preset": preset, **_tiny_kwargs()}, preview_defaults=True)
        graph = build_gpt_root_graph(name=f"{preset}_resolve", model_spec=spec)
        graph.resolve_variant_library()


def test_seq2seq_blocks_reference_exported_variant_families() -> None:
    spec = build_model_spec_from_config({"preset": "seq2seq", **_tiny_kwargs()}, preview_defaults=True)
    graph = build_gpt_root_graph(name="seq2seq_refs", model_spec=spec)
    enc_block_graph = graph.variant_library["enc_block"]["default"]
    dec_block_graph = graph.variant_library["dec_block"]["default"]

    assert enc_block_graph.nodes["attention"].neuron_def.variant_ref == {"family": "enc_attention", "version": "default"}
    assert enc_block_graph.nodes["mlp"].neuron_def.variant_ref == {"family": "mlp_dense", "version": "default"}
    assert dec_block_graph.nodes["attention"].neuron_def.variant_ref == {"family": "dec_attention", "version": "default"}
    assert dec_block_graph.nodes["cross_attn"].neuron_def.variant_ref == {"family": "cross_attention", "version": "default"}
    assert dec_block_graph.nodes["mlp"].neuron_def.variant_ref == {"family": "mlp_moe", "version": "default"}


def test_legacy_variant_family_aliases_resolve_saved_graphs() -> None:
    for link_family, available_family in [
        ("attn_block", "transformer_block"),
        ("transformer_block", "attn_block"),
        ("mixllama", "attn_block"),
    ]:
        graph = _make_alias_root(link_family, available_family)
        graph.resolve_variant_library()
        assert graph.nodes["block"].neuron_def.subgraph is not None
        assert graph.nodes["block"].neuron_def.subgraph.name == f"{available_family}_default"


def test_apply_gpt_template_supports_all_presets() -> None:
    for preset in PRESETS:
        graph = apply_gpt_template(GPTTemplateRequest(name=f"{preset}_graph", config={"preset": preset}))
        assert "model" in graph.nodes
        assert graph.output_node_ids == ["loss_out"]


def test_ttt_llama_forward_smoke() -> None:
    spec = build_ttt_llama_spec(**_tiny_kwargs(), vocab_size=128, ttt_hidden_dim=24)
    graph = _cpu_graph(build_gpt_root_graph(name="ttt_smoke", model_spec=spec))
    attention_graph = graph.variant_library["attention"]["default"]
    assert any(node.neuron_def.module_type == "ttt_linear" for node in attention_graph.nodes.values())

    compiled = CompiledTorchGraph(graph)
    tokens = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    loss = compiled(tokens, targets)[0]
    assert loss.ndim == 0


def test_jepa_trainer_freezes_and_updates_ema_targets() -> None:
    spec = build_llm_jepa_spec(**_tiny_kwargs(), vocab_size=128, ema_decay=0.9)
    graph = _cpu_graph(build_gpt_root_graph(name="jepa_train", model_spec=spec))

    compiled = CompiledTorchGraph(graph)
    TorchTrainer._prepare_ema_targets(compiled)
    model = compiled.node_modules["model"]
    online = model.node_modules["online_encoder"]
    target = model.node_modules["target_encoder"]
    assert all(not param.requires_grad for param in target.parameters())
    initial_target_param = next(target.parameters()).detach().clone()
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        assert torch.equal(online_param, target_param)

    with torch.no_grad():
        next(online.parameters()).add_(0.5)
    TorchTrainer._ema_update_targets(compiled, 0.9)
    updated_target_param = next(target.parameters()).detach().clone()
    assert not torch.equal(updated_target_param, initial_target_param)
    assert not torch.equal(updated_target_param, next(online.parameters()).detach())

    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(epochs=1, batch_size=2, learning_rate=1e-3, max_steps=1, device="cpu"),
    )
    tokens = torch.randint(0, 128, (4, 8))
    losses = trainer.train(tokens, tokens)
    assert len(losses) == 1


def test_hnet_spec_enforces_byte_vocab_and_raw_byte_chunking() -> None:
    spec = build_hnet_lm_spec(**_tiny_kwargs(), vocab_size=1024, byte_patch_size=2, byte_patch_stride=2)
    assert spec.vocab_size == 256

    dataset_name = f"test_hnet_bytes_{uuid.uuid4().hex}"
    dataset_path = DATASETS_DIR / f"{dataset_name}.txt"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_bytes(b"abcdefghi")
    try:
        inputs, targets = load_dataset_bytes([dataset_name], seq_len=4)
    finally:
        dataset_path.unlink(missing_ok=True)

    assert inputs == [[97, 98, 99, 100], [101, 102, 103, 104]]
    assert targets == [[98, 99, 100, 101], [102, 103, 104, 105]]


def test_hnet_trainer_runs_one_step() -> None:
    spec = build_hnet_lm_spec(**_tiny_kwargs(), byte_patch_size=2, byte_patch_stride=2)
    graph = _cpu_graph(build_gpt_root_graph(name="hnet_train", model_spec=spec))

    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(epochs=1, batch_size=2, learning_rate=1e-3, max_steps=1, device="cpu"),
    )
    tokens = torch.randint(0, 256, (4, 8))
    targets = torch.randint(0, 256, (4, 8))
    losses = trainer.train(tokens, targets)
    assert len(losses) == 1


def test_universal_template_uses_single_shared_block_and_normalized_halting() -> None:
    spec = build_universal_llama_spec(**_tiny_kwargs(), vocab_size=128, max_recurrence_steps=3, halt_epsilon=0.01)
    graph = _cpu_graph(build_gpt_root_graph(name="universal_trace", model_spec=spec))
    model_subgraph = graph.nodes["model"].neuron_def.subgraph
    assert model_subgraph is not None
    assert sum(1 for node in model_subgraph.nodes.values() if node.neuron_def.module_type == "universal_transformer") == 1

    compiled = CompiledTorchGraph(graph)
    tokens = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    outputs, trace = compiled.trace(tokens, targets)
    halt_weights = trace["model/universal"][1]
    assert outputs[0].ndim == 0
    assert halt_weights.shape == (2, 3)
    assert torch.allclose(halt_weights.sum(dim=1), torch.ones(2), atol=1e-4)

    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(epochs=1, batch_size=2, learning_rate=1e-3, max_steps=1, device="cpu"),
    )
    losses = trainer.train(tokens, targets)
    assert len(losses) == 1


def test_dataset_source_role_wiring_covers_single_and_multi_input_templates() -> None:
    seq2seq_graph = apply_gpt_template(GPTTemplateRequest(name="seq2seq", config={"preset": "seq2seq", "num_layers": 1}))
    seq2seq_result = load_dataset_source_into_graph(seq2seq_graph, LoadDatasetRequest(dataset_names=["dummy"], seq_len=8))
    seq2seq_ports = [port.name for port in seq2seq_graph.nodes[seq2seq_result["dataset_source_node_id"]].neuron_def.output_ports]
    assert seq2seq_ports == ["enc_tokens", "dec_tokens", "targets"]

    jepa_graph = apply_gpt_template(GPTTemplateRequest(name="jepa", config={"preset": "llm_jepa", "num_layers": 1}))
    jepa_result = load_dataset_source_into_graph(jepa_graph, LoadDatasetRequest(dataset_names=["dummy"], seq_len=8))
    jepa_ports = [port.name for port in jepa_graph.nodes[jepa_result["dataset_source_node_id"]].neuron_def.output_ports]
    assert jepa_ports == ["tokens"]
