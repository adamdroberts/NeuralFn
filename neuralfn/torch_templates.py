from __future__ import annotations

from copy import deepcopy
from typing import Any

from .builtins import BuiltinNeurons
from .graph import Edge, NeuronGraph, NeuronInstance
from .neuron import NeuronDef, neuron_from_source, subgraph_neuron
from .port import Port
from .torch_backend import default_gpt_config


def make_terminal_def(
    *,
    role: str,
    port_name: str,
    dtype: str,
    neuron_id: str | None = None,
) -> NeuronDef:
    source = f"def {role}(x):\n    return x\n"
    ports = [Port(port_name, range=(-1_000_000.0, 1_000_000.0), precision=0.001, dtype=dtype)]
    return neuron_from_source(source, role, ports, ports, neuron_id=neuron_id)


def clone_neuron_def(ndef: NeuronDef, *, config: dict[str, Any] | None = None) -> NeuronDef:
    cloned = NeuronDef.from_dict(deepcopy(ndef.to_dict()))
    if config is not None:
        cloned.module_config.update(config)
    return cloned


def build_transformer_block_graph(
    *,
    name: str,
    model_dim: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_mult: int,
    rope_base: float,
    qk_gain_init: float,
) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 140)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x0", dtype="tensor"), instance_id="x0_in", position=(40, 260)))
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.residual_mix_module,
                config={"dim": model_dim, "primary_init": 1.0, "skip_init": 0.0},
            ),
            instance_id="resid_mix",
            position=(220, 200),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.rms_norm_module, config={"eps": 1e-6}),
            instance_id="attn_norm",
            position=(420, 140),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.causal_self_attention_module,
                config={
                    "model_dim": model_dim,
                    "num_heads": num_heads,
                    "num_kv_heads": num_kv_heads,
                    "rope_base": rope_base,
                    "qk_gain_init": qk_gain_init,
                },
            ),
            instance_id="attention",
            position=(620, 140),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.residual_add_module,
                config={"dim": model_dim, "init_scale": 1.0},
            ),
            instance_id="attn_residual",
            position=(820, 140),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.rms_norm_module, config={"eps": 1e-6}),
            instance_id="mlp_norm",
            position=(1020, 140),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.mlp_relu2_module,
                config={"model_dim": model_dim, "mlp_mult": mlp_mult},
            ),
            instance_id="mlp",
            position=(1220, 140),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.residual_add_module,
                config={"dim": model_dim, "init_scale": 1.0},
            ),
            instance_id="mlp_residual",
            position=(1420, 140),
        )
    )
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="x", dtype="tensor"), instance_id="x_out", position=(1620, 140)))

    edges = [
        ("e_x_mix", "x_in", 0, "resid_mix", 0),
        ("e_x0_mix", "x0_in", 0, "resid_mix", 1),
        ("e_mix_norm", "resid_mix", 0, "attn_norm", 0),
        ("e_norm_attn", "attn_norm", 0, "attention", 0),
        ("e_mix_attn_res", "resid_mix", 0, "attn_residual", 0),
        ("e_attn_attn_res", "attention", 0, "attn_residual", 1),
        ("e_attnres_mlpnorm", "attn_residual", 0, "mlp_norm", 0),
        ("e_mlpnorm_mlp", "mlp_norm", 0, "mlp", 0),
        ("e_attnres_mlpres", "attn_residual", 0, "mlp_residual", 0),
        ("e_mlp_mlpres", "mlp", 0, "mlp_residual", 1),
        ("e_mlpres_out", "mlp_residual", 0, "x_out", 0),
    ]
    for edge_id, src_node, src_port, dst_node, dst_port in edges:
        graph.add_edge(Edge(id=edge_id, src_node=src_node, src_port=src_port, dst_node=dst_node, dst_port=dst_port))

    graph.input_node_ids = ["x_in", "x0_in"]
    graph.output_node_ids = ["x_out"]
    return graph


def build_gpt_stage_graph(*, name: str = "gpt", config: dict[str, Any] | None = None) -> NeuronGraph:
    cfg = {**default_gpt_config(), **(config or {})}
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 140)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 360)))
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.token_embedding_module,
                config={"vocab_size": cfg["vocab_size"], "model_dim": cfg["model_dim"]},
            ),
            instance_id="token_embedding",
            position=(240, 140),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.rms_norm_module, config={"eps": 1e-6}),
            instance_id="embed_norm",
            position=(460, 140),
        )
    )

    graph.add_edge(Edge(id="e_tokens_embed", src_node="tokens_in", src_port=0, dst_node="token_embedding", dst_port=0))
    graph.add_edge(Edge(id="e_embed_hidden", src_node="token_embedding", src_port=0, dst_node="embed_norm", dst_port=0))

    encoder_count = int(cfg["num_layers"]) // 2
    decoder_count = int(cfg["num_layers"]) - encoder_count
    current_node = "embed_norm"
    skip_nodes: list[str] = []

    for idx in range(encoder_count):
        block_graph = build_transformer_block_graph(
            name=f"encoder_block_{idx}",
            model_dim=int(cfg["model_dim"]),
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=int(cfg["num_kv_heads"]),
            mlp_mult=int(cfg["mlp_mult"]),
            rope_base=float(cfg["rope_base"]),
            qk_gain_init=float(cfg["qk_gain_init"]),
        )
        block_node_id = f"encoder_block_{idx}"
        graph.add_node(
            NeuronInstance(
                subgraph_neuron(block_graph, name=block_node_id, input_aliases=["x", "x0"], output_aliases=["x"]),
                instance_id=block_node_id,
                position=(680 + idx * 220, 100),
            )
        )
        graph.add_edge(Edge(id=f"e_{block_node_id}_x", src_node=current_node, src_port=0, dst_node=block_node_id, dst_port=0))
        graph.add_edge(Edge(id=f"e_{block_node_id}_x0", src_node="embed_norm", src_port=0, dst_node=block_node_id, dst_port=1))
        current_node = block_node_id
        skip_nodes.append(block_node_id)

    for idx in range(decoder_count):
        if skip_nodes:
            skip_src = skip_nodes.pop()
            skip_add_id = f"skip_add_{idx}"
            graph.add_node(
                NeuronInstance(
                    clone_neuron_def(
                        BuiltinNeurons.residual_add_module,
                        config={"dim": cfg["model_dim"], "init_scale": 1.0},
                    ),
                    instance_id=skip_add_id,
                    position=(680 + (encoder_count + idx) * 220, 260),
                )
            )
            graph.add_edge(Edge(id=f"e_{skip_add_id}_current", src_node=current_node, src_port=0, dst_node=skip_add_id, dst_port=0))
            graph.add_edge(Edge(id=f"e_{skip_add_id}_skip", src_node=skip_src, src_port=0, dst_node=skip_add_id, dst_port=1))
            current_node = skip_add_id

        block_graph = build_transformer_block_graph(
            name=f"decoder_block_{idx}",
            model_dim=int(cfg["model_dim"]),
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=int(cfg["num_kv_heads"]),
            mlp_mult=int(cfg["mlp_mult"]),
            rope_base=float(cfg["rope_base"]),
            qk_gain_init=float(cfg["qk_gain_init"]),
        )
        block_node_id = f"decoder_block_{idx}"
        graph.add_node(
            NeuronInstance(
                subgraph_neuron(block_graph, name=block_node_id, input_aliases=["x", "x0"], output_aliases=["x"]),
                instance_id=block_node_id,
                position=(900 + (encoder_count + idx) * 220, 100),
            )
        )
        graph.add_edge(Edge(id=f"e_{block_node_id}_x", src_node=current_node, src_port=0, dst_node=block_node_id, dst_port=0))
        graph.add_edge(Edge(id=f"e_{block_node_id}_x0", src_node="embed_norm", src_port=0, dst_node=block_node_id, dst_port=1))
        current_node = block_node_id

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.rms_norm_module, config={"eps": 1e-6}),
            instance_id="final_norm",
            position=(1240 + int(cfg["num_layers"]) * 220, 140),
        )
    )
    graph.add_edge(Edge(id="e_final_norm", src_node=current_node, src_port=0, dst_node="final_norm", dst_port=0))

    head_node_id = "tied_lm_head" if bool(cfg["tie_embeddings"]) else "lm_head"
    head_def = (
        clone_neuron_def(BuiltinNeurons.tied_lm_head_module)
        if bool(cfg["tie_embeddings"])
        else clone_neuron_def(
            BuiltinNeurons.lm_head_module,
            config={"model_dim": cfg["model_dim"], "vocab_size": cfg["vocab_size"]},
        )
    )
    graph.add_node(NeuronInstance(head_def, instance_id=head_node_id, position=(1460 + int(cfg["num_layers"]) * 220, 140)))
    graph.add_edge(Edge(id="e_norm_head", src_node="final_norm", src_port=0, dst_node=head_node_id, dst_port=0))
    if bool(cfg["tie_embeddings"]):
        graph.add_edge(Edge(id="e_embed_weight_head", src_node="token_embedding", src_port=1, dst_node=head_node_id, dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.logit_softcap_module, config={"softcap": cfg["logit_softcap"]}),
            instance_id="logit_softcap",
            position=(1680 + int(cfg["num_layers"]) * 220, 140),
        )
    )
    graph.add_edge(Edge(id="e_head_softcap", src_node=head_node_id, src_port=0, dst_node="logit_softcap", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.token_cross_entropy_module),
            instance_id="token_cross_entropy",
            position=(1900 + int(cfg["num_layers"]) * 220, 220),
        )
    )
    graph.add_edge(Edge(id="e_softcap_ce", src_node="logit_softcap", src_port=0, dst_node="token_cross_entropy", dst_port=0))
    graph.add_edge(Edge(id="e_targets_ce", src_node="targets_in", src_port=0, dst_node="token_cross_entropy", dst_port=1))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(2120 + int(cfg["num_layers"]) * 220, 220)))
    graph.add_edge(Edge(id="e_ce_loss_out", src_node="token_cross_entropy", src_port=0, dst_node="loss_out", dst_port=0))

    graph.input_node_ids = ["tokens_in", "targets_in"]
    graph.output_node_ids = ["loss_out"]
    return graph


def build_gpt_subgraph(*, name: str = "gpt", config: dict[str, Any] | None = None) -> NeuronDef:
    graph = build_gpt_stage_graph(name=f"{name}_graph", config=config)
    return subgraph_neuron(graph, name=name, input_aliases=["tokens", "targets"], output_aliases=["loss"])


def build_gpt_root_graph(*, name: str = "gpt_root", config: dict[str, Any] | None = None) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch", torch_config={"device": "cuda", "amp_dtype": "bfloat16"})
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 120)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 300)))
    graph.add_node(NeuronInstance(build_gpt_subgraph(name="gpt", config=config), instance_id="gpt", position=(280, 180)))
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(560, 180)))
    graph.add_edge(Edge(id="e_tokens_gpt", src_node="tokens_in", src_port=0, dst_node="gpt", dst_port=0))
    graph.add_edge(Edge(id="e_targets_gpt", src_node="targets_in", src_port=0, dst_node="gpt", dst_port=1))
    graph.add_edge(Edge(id="e_gpt_out", src_node="gpt", src_port=0, dst_node="loss_out", dst_port=0))
    graph.input_node_ids = ["tokens_in", "targets_in"]
    graph.output_node_ids = ["loss_out"]
    return graph
