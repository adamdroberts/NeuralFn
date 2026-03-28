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


def link_variant_neuron(
    graph: NeuronGraph,
    *,
    family: str,
    version: str,
    name: str,
    input_aliases: list[str] | None = None,
    output_aliases: list[str] | None = None,
) -> NeuronDef:
    return subgraph_neuron(
        graph,
        name=name,
        input_aliases=input_aliases,
        output_aliases=output_aliases,
        variant_ref={"family": family, "version": version},
    )


def build_attention_variant_graph(
    *,
    name: str,
    model_dim: int,
    num_heads: int,
    num_kv_heads: int,
    rope_base: float,
    qk_gain_init: float,
) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 180)))
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.linear_module,
                config={"input_dim": model_dim, "output_dim": model_dim, "bias": False},
            ),
            instance_id="q_proj",
            position=(220, 40),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.linear_module,
                config={"input_dim": model_dim, "output_dim": kv_dim, "bias": False},
            ),
            instance_id="k_proj",
            position=(220, 160),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.linear_module,
                config={"input_dim": model_dim, "output_dim": kv_dim, "bias": False},
            ),
            instance_id="v_proj",
            position=(220, 280),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.reshape_heads_module, config={"num_heads": num_heads}),
            instance_id="q_heads",
            position=(420, 40),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.reshape_heads_module, config={"num_heads": num_kv_heads}),
            instance_id="k_heads",
            position=(420, 160),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.reshape_heads_module, config={"num_heads": num_kv_heads}),
            instance_id="v_heads",
            position=(420, 280),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.rms_norm_module, config={"eps": 1e-6}),
            instance_id="q_norm",
            position=(620, 40),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.rms_norm_module, config={"eps": 1e-6}),
            instance_id="k_norm",
            position=(620, 160),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.rotary_embedding_module,
                config={"head_dim": head_dim, "rope_base": rope_base},
            ),
            instance_id="rope",
            position=(820, 100),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.qk_gain_module,
                config={"num_heads": num_heads, "qk_gain_init": qk_gain_init},
            ),
            instance_id="q_gain",
            position=(1020, 40),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.repeat_kv_module,
                config={"num_heads": num_heads, "num_kv_heads": num_kv_heads},
            ),
            instance_id="k_repeat",
            position=(1020, 160),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.repeat_kv_module,
                config={"num_heads": num_heads, "num_kv_heads": num_kv_heads},
            ),
            instance_id="v_repeat",
            position=(1020, 280),
        )
    )
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.scaled_dot_product_attention_module),
            instance_id="attn",
            position=(1220, 160),
        )
    )
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.merge_heads_module), instance_id="merge", position=(1420, 160)))
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.linear_module,
                config={"input_dim": model_dim, "output_dim": model_dim, "bias": False},
            ),
            instance_id="out_proj",
            position=(1620, 160),
        )
    )
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="attn_out", dtype="tensor"), instance_id="x_out", position=(1820, 160)))

    edges = [
        ("e_x_q", "x_in", 0, "q_proj", 0),
        ("e_x_k", "x_in", 0, "k_proj", 0),
        ("e_x_v", "x_in", 0, "v_proj", 0),
        ("e_q_qheads", "q_proj", 0, "q_heads", 0),
        ("e_k_kheads", "k_proj", 0, "k_heads", 0),
        ("e_v_vheads", "v_proj", 0, "v_heads", 0),
        ("e_qheads_qnorm", "q_heads", 0, "q_norm", 0),
        ("e_kheads_knorm", "k_heads", 0, "k_norm", 0),
        ("e_qnorm_rope", "q_norm", 0, "rope", 0),
        ("e_knorm_rope", "k_norm", 0, "rope", 1),
        ("e_rope_qgain", "rope", 0, "q_gain", 0),
        ("e_rope_krepeat", "rope", 1, "k_repeat", 0),
        ("e_vheads_vrepeat", "v_heads", 0, "v_repeat", 0),
        ("e_qgain_attn", "q_gain", 0, "attn", 0),
        ("e_krepeat_attn", "k_repeat", 0, "attn", 1),
        ("e_vrepeat_attn", "v_repeat", 0, "attn", 2),
        ("e_attn_merge", "attn", 0, "merge", 0),
        ("e_merge_outproj", "merge", 0, "out_proj", 0),
        ("e_outproj_out", "out_proj", 0, "x_out", 0),
    ]
    for edge_id, src_node, src_port, dst_node, dst_port in edges:
        graph.add_edge(Edge(id=edge_id, src_node=src_node, src_port=src_port, dst_node=dst_node, dst_port=dst_port))

    graph.input_node_ids = ["x_in"]
    graph.output_node_ids = ["x_out"]
    return graph


def build_mlp_variant_graph(
    *,
    name: str,
    model_dim: int,
    mlp_mult: int,
) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    hidden_dim = model_dim * mlp_mult

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 160)))
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.linear_module,
                config={"input_dim": model_dim, "output_dim": hidden_dim, "bias": False},
            ),
            instance_id="fc",
            position=(220, 160),
        )
    )
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.relu), instance_id="relu", position=(420, 160)))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.multiply), instance_id="square", position=(620, 160)))
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.linear_module,
                config={"input_dim": hidden_dim, "output_dim": model_dim, "bias": False},
            ),
            instance_id="proj",
            position=(820, 160),
        )
    )
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="y", dtype="tensor"), instance_id="y_out", position=(1020, 160)))

    edges = [
        ("e_x_fc", "x_in", 0, "fc", 0),
        ("e_fc_relu", "fc", 0, "relu", 0),
        ("e_relu_square_a", "relu", 0, "square", 0),
        ("e_relu_square_b", "relu", 0, "square", 1),
        ("e_square_proj", "square", 0, "proj", 0),
        ("e_proj_out", "proj", 0, "y_out", 0),
    ]
    for edge_id, src_node, src_port, dst_node, dst_port in edges:
        graph.add_edge(Edge(id=edge_id, src_node=src_node, src_port=src_port, dst_node=dst_node, dst_port=dst_port))

    graph.input_node_ids = ["x_in"]
    graph.output_node_ids = ["y_out"]
    return graph


def build_transformer_block_graph(
    *,
    name: str,
    model_dim: int,
    attention_graph: NeuronGraph,
    mlp_graph: NeuronGraph,
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
            link_variant_neuron(
                attention_graph,
                family="attention",
                version="baseline",
                name="attention",
                input_aliases=["x"],
                output_aliases=["attn_out"],
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
            link_variant_neuron(
                mlp_graph,
                family="mlp",
                version="relu2",
                name="mlp",
                input_aliases=["x"],
                output_aliases=["y"],
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


def build_gpt_variant_library(config: dict[str, Any] | None = None) -> dict[str, dict[str, NeuronGraph]]:
    cfg = {**default_gpt_config(), **(config or {})}
    attention_graph = build_attention_variant_graph(
        name="attention_baseline",
        model_dim=int(cfg["model_dim"]),
        num_heads=int(cfg["num_heads"]),
        num_kv_heads=int(cfg["num_kv_heads"]),
        rope_base=float(cfg["rope_base"]),
        qk_gain_init=float(cfg["qk_gain_init"]),
    )
    mlp_graph = build_mlp_variant_graph(
        name="mlp_relu2",
        model_dim=int(cfg["model_dim"]),
        mlp_mult=int(cfg["mlp_mult"]),
    )
    block_graph = build_transformer_block_graph(
        name="transformer_block_baseline",
        model_dim=int(cfg["model_dim"]),
        attention_graph=attention_graph,
        mlp_graph=mlp_graph,
    )
    return {
        "attention": {"baseline": attention_graph},
        "mlp": {"relu2": mlp_graph},
        "transformer_block": {"baseline": block_graph},
    }


def build_gpt_stage_graph(
    *,
    name: str = "gpt",
    config: dict[str, Any] | None = None,
    attach_variant_library: bool = True,
) -> NeuronGraph:
    cfg = {**default_gpt_config(), **(config or {})}
    variant_library = build_gpt_variant_library(cfg)
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    if attach_variant_library:
        graph.variant_library = deepcopy(variant_library)

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
        block_node_id = f"encoder_block_{idx}"
        graph.add_node(
            NeuronInstance(
                link_variant_neuron(
                    deepcopy(variant_library["transformer_block"]["baseline"]),
                    family="transformer_block",
                    version="baseline",
                    name=block_node_id,
                    input_aliases=["x", "x0"],
                    output_aliases=["x"],
                ),
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

        block_node_id = f"decoder_block_{idx}"
        graph.add_node(
            NeuronInstance(
                link_variant_neuron(
                    deepcopy(variant_library["transformer_block"]["baseline"]),
                    family="transformer_block",
                    version="baseline",
                    name=block_node_id,
                    input_aliases=["x", "x0"],
                    output_aliases=["x"],
                ),
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


def build_gpt_template_payload(
    *,
    name: str = "gpt",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant_library = build_gpt_variant_library(config)
    node_def = subgraph_neuron(
        build_gpt_stage_graph(name=f"{name}_graph", config=config, attach_variant_library=False),
        name=name,
        input_aliases=["tokens", "targets"],
        output_aliases=["loss"],
    )
    return {
        "node_def": node_def.to_dict(),
        "variant_library": {
            family: {version: graph.to_dict() for version, graph in versions.items()}
            for family, versions in variant_library.items()
        },
        "graph_settings": {
            "training_method": "torch",
            "runtime": "torch",
            "torch_config": {"device": "cuda", "amp_dtype": "bfloat16"},
        },
    }


def build_gpt_root_graph(*, name: str = "gpt_root", config: dict[str, Any] | None = None) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch", torch_config={"device": "cuda", "amp_dtype": "bfloat16"})
    graph.variant_library = deepcopy(build_gpt_variant_library(config))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 120)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 300)))
    graph.add_node(
        NeuronInstance(
            subgraph_neuron(
                build_gpt_stage_graph(name="gpt_graph", config=config, attach_variant_library=False),
                name="gpt",
                input_aliases=["tokens", "targets"],
                output_aliases=["loss"],
            ),
            instance_id="gpt",
            position=(280, 180),
        )
    )
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(560, 180)))
    graph.add_edge(Edge(id="e_tokens_gpt", src_node="tokens_in", src_port=0, dst_node="gpt", dst_port=0))
    graph.add_edge(Edge(id="e_targets_gpt", src_node="targets_in", src_port=0, dst_node="gpt", dst_port=1))
    graph.add_edge(Edge(id="e_gpt_out", src_node="gpt", src_port=0, dst_node="loss_out", dst_port=0))
    graph.input_node_ids = ["tokens_in", "targets_in"]
    graph.output_node_ids = ["loss_out"]
    return graph
