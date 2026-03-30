import os

new_code = """from __future__ import annotations

from copy import deepcopy
from typing import Any

from .builtins import BuiltinNeurons
from .config import BlockSpec, ModelSpec
from .graph import Edge, NeuronGraph, NeuronInstance
from .neuron import NeuronDef, neuron_from_source, subgraph_neuron
from .port import Port


def make_terminal_def(
    *,
    role: str,
    port_name: str,
    dtype: str,
    neuron_id: str | None = None,
) -> NeuronDef:
    source = f"def {role}(x):\\n    return x\\n"
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


def build_dense_attention_graph(name: str, model_dim: int, spec: BlockSpec) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    
    num_heads = spec.num_heads
    num_kv_heads = spec.num_kv_heads if spec.num_kv_heads is not None else num_heads
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim

    # Input nodes
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 180)))
    if spec.attention_backend == "sdpa" or spec.attention_backend == "flex":
        # we can add cache nodes later, but leaving out for standard causal attention
        pass

    # Q,K,V Projections
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.linear_module, config={"input_dim": model_dim, "output_dim": model_dim, "bias": spec.linear_bias}), instance_id="q_proj", position=(220, 60)))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.linear_module, config={"input_dim": model_dim, "output_dim": kv_dim, "bias": spec.linear_bias}), instance_id="k_proj", position=(220, 180)))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.linear_module, config={"input_dim": model_dim, "output_dim": kv_dim, "bias": spec.linear_bias}), instance_id="v_proj", position=(220, 300)))
    
    graph.add_edge(Edge(id="e_x_q", src_node="x_in", src_port=0, dst_node="q_proj", dst_port=0))
    graph.add_edge(Edge(id="e_x_k", src_node="x_in", src_port=0, dst_node="k_proj", dst_port=0))
    graph.add_edge(Edge(id="e_x_v", src_node="x_in", src_port=0, dst_node="v_proj", dst_port=0))

    # Reshape
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.reshape_heads_module, config={"num_heads": num_heads}), instance_id="q_heads", position=(460, 60)))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.reshape_heads_module, config={"num_heads": num_kv_heads}), instance_id="k_heads", position=(460, 180)))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.reshape_heads_module, config={"num_heads": num_kv_heads}), instance_id="v_heads", position=(460, 300)))

    graph.add_edge(Edge(id="e_q_qheads", src_node="q_proj", src_port=0, dst_node="q_heads", dst_port=0))
    graph.add_edge(Edge(id="e_k_kheads", src_node="k_proj", src_port=0, dst_node="k_heads", dst_port=0))
    graph.add_edge(Edge(id="e_v_vheads", src_node="v_proj", src_port=0, dst_node="v_heads", dst_port=0))

    # Optional Pos Encoding (RoPE)
    curr_q = "q_heads"
    curr_k = "k_heads"
    if spec.pos_encoding == "rope":
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.rotary_embedding_module, config={"head_dim": head_dim, "rope_base": spec.rope_theta}), instance_id="rope", position=(700, 120)))
        graph.add_edge(Edge(id="e_qheads_rope", src_node="q_heads", src_port=0, dst_node="rope", dst_port=0))
        graph.add_edge(Edge(id="e_kheads_rope", src_node="k_heads", src_port=0, dst_node="rope", dst_port=1))
        curr_q = "rope"
        curr_k = "rope"
        k_port = 1
        q_port = 0
    else:
        k_port = 0
        q_port = 0

    # Repeat KV for GQA
    curr_k_attn = curr_k
    curr_v_attn = "v_heads"
    k_attn_port = k_port
    v_attn_port = 0
    if num_kv_heads < num_heads:
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.repeat_kv_module, config={"num_heads": num_heads, "num_kv_heads": num_kv_heads}), instance_id="k_repeat", position=(940, 180)))
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.repeat_kv_module, config={"num_heads": num_heads, "num_kv_heads": num_kv_heads}), instance_id="v_repeat", position=(940, 300)))
        graph.add_edge(Edge(id=f"e_{curr_k}_krepeat", src_node=curr_k, src_port=k_port, dst_node="k_repeat", dst_port=0))
        graph.add_edge(Edge(id="e_vheads_vrepeat", src_node="v_heads", src_port=0, dst_node="v_repeat", dst_port=0))
        curr_k_attn = "k_repeat"
        curr_v_attn = "v_repeat"
        k_attn_port = 0
        v_attn_port = 0

    # Attention Core
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.scaled_dot_product_attention_module, config={"is_causal": spec.is_causal}), instance_id="sdpa", position=(1180, 180)))
    graph.add_edge(Edge(id=f"e_{curr_q}_sdpa", src_node=curr_q, src_port=q_port, dst_node="sdpa", dst_port=0))
    graph.add_edge(Edge(id=f"e_{curr_k_attn}_sdpa", src_node=curr_k_attn, src_port=k_attn_port, dst_node="sdpa", dst_port=1))
    graph.add_edge(Edge(id=f"e_{curr_v_attn}_sdpa", src_node=curr_v_attn, src_port=v_attn_port, dst_node="sdpa", dst_port=2))
    curr_out = "sdpa"

    # Merge Heads
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.merge_heads_module), instance_id="merge", position=(1420, 180)))
    graph.add_edge(Edge(id="e_sdpa_merge", src_node=curr_out, src_port=0, dst_node="merge", dst_port=0))

    # Output Proj
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.linear_module, config={"input_dim": model_dim, "output_dim": model_dim, "bias": spec.linear_bias}), instance_id="out_proj", position=(1660, 180)))
    graph.add_edge(Edge(id="e_merge_outproj", src_node="merge", src_port=0, dst_node="out_proj", dst_port=0))
    curr_out = "out_proj"

    # Optional Dropout
    if spec.dropout_p > 0.0:
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.dropout_module, config={"p": spec.dropout_p}), instance_id="attn_dropout", position=(1900, 180)))
        graph.add_edge(Edge(id="e_outproj_dropout", src_node=curr_out, src_port=0, dst_node="attn_dropout", dst_port=0))
        curr_out = "attn_dropout"

    # Output Port
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="attn_out", dtype="tensor"), instance_id="attn_out", position=(2140, 180)))
    graph.add_edge(Edge(id="e_to_attn_out", src_node=curr_out, src_port=0, dst_node="attn_out", dst_port=0))

    graph.input_node_ids = ["x_in"]
    graph.output_node_ids = ["attn_out"]
    return graph


def build_dense_mlp_graph(name: str, model_dim: int, spec: BlockSpec) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 120)))
    
    curr_out = "x_in"
    if spec.mlp_type == "gelu":
        hidden = int(model_dim * spec.mlp_multiplier)
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.linear_module, config={"input_dim": model_dim, "output_dim": hidden, "bias": spec.linear_bias}), instance_id="fc1", position=(260, 120)))
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.gelu_module), instance_id="gelu", position=(480, 120)))
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.linear_module, config={"input_dim": hidden, "output_dim": model_dim, "bias": spec.linear_bias}), instance_id="fc2", position=(700, 120)))
        graph.add_edge(Edge(id="e_x_fc1", src_node=curr_out, src_port=0, dst_node="fc1", dst_port=0))
        graph.add_edge(Edge(id="e_fc1_gelu", src_node="fc1", src_port=0, dst_node="gelu", dst_port=0))
        graph.add_edge(Edge(id="e_gelu_fc2", src_node="gelu", src_port=0, dst_node="fc2", dst_port=0))
        curr_out = "fc2"
    elif spec.mlp_type == "swiglu":
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.swiglu_module, config={"model_dim": model_dim, "mlp_mult": spec.mlp_multiplier, "multiple_of": spec.multiple_of}), instance_id="swiglu", position=(480, 120)))
        graph.add_edge(Edge(id="e_x_swiglu", src_node=curr_out, src_port=0, dst_node="swiglu", dst_port=0))
        curr_out = "swiglu"
    else:
        raise ValueError(f"Unknown mlp_type: {spec.mlp_type}. Cannot build dense MLP.")

    # Optional Dropout
    if spec.dropout_p > 0.0:
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.dropout_module, config={"p": spec.dropout_p}), instance_id="mlp_dropout", position=(920, 120)))
        graph.add_edge(Edge(id="e_fc2_dropout", src_node=curr_out, src_port=0, dst_node="mlp_dropout", dst_port=0))
        curr_out = "mlp_dropout"

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="y", dtype="tensor"), instance_id="y_out", position=(1140, 120)))
    graph.add_edge(Edge(id="e_to_y_out", src_node=curr_out, src_port=0, dst_node="y_out", dst_port=0))

    graph.input_node_ids = ["x_in"]
    graph.output_node_ids = ["y_out"]
    return graph

def build_moe_mlp_graph(name: str, model_dim: int, spec: BlockSpec) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 160)))
    
    # Router
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.router_logits_module, config={"model_dim": model_dim, "experts": spec.experts}), instance_id="router", position=(260, 60)))
    graph.add_edge(Edge(id="e_x_router", src_node="x_in", src_port=0, dst_node="router", dst_port=0))

    # Top-K
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.topk_route_module, config={"top_k": spec.top_k}), instance_id="topk", position=(480, 60)))
    graph.add_edge(Edge(id="e_router_topk", src_node="router", src_port=0, dst_node="topk", dst_port=0))

    # Dispatch
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.expert_dispatch_module, config={"model_dim": model_dim, "experts": spec.experts, "mlp_mult": spec.mlp_multiplier}), instance_id="dispatch", position=(700, 160)))
    graph.add_edge(Edge(id="e_x_dispatch", src_node="x_in", src_port=0, dst_node="dispatch", dst_port=0))
    graph.add_edge(Edge(id="e_weights_dispatch", src_node="topk", src_port=0, dst_node="dispatch", dst_port=1))
    graph.add_edge(Edge(id="e_indices_dispatch", src_node="topk", src_port=1, dst_node="dispatch", dst_port=2))

    # Combine (Identity essentially)
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.expert_combine_module), instance_id="combine", position=(920, 160)))
    graph.add_edge(Edge(id="e_dispatch_combine", src_node="dispatch", src_port=0, dst_node="combine", dst_port=0))

    # Aux loss
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.load_balance_loss_module, config={"experts": spec.experts}), instance_id="lb_loss", position=(700, -40)))
    graph.add_edge(Edge(id="e_r_lb", src_node="router", src_port=0, dst_node="lb_loss", dst_port=0))
    graph.add_edge(Edge(id="e_w_lb", src_node="topk", src_port=0, dst_node="lb_loss", dst_port=1))
    graph.add_edge(Edge(id="e_i_lb", src_node="topk", src_port=1, dst_node="lb_loss", dst_port=2))

    # Outputs
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="y", dtype="tensor"), instance_id="y_out", position=(1140, 160)))
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="aux_loss", dtype="tensor"), instance_id="aux_loss_out", position=(1140, -40)))
    
    graph.add_edge(Edge(id="e_combine_yout", src_node="combine", src_port=0, dst_node="y_out", dst_port=0))
    graph.add_edge(Edge(id="e_lb_auxout", src_node="lb_loss", src_port=0, dst_node="aux_loss_out", dst_port=0))

    graph.input_node_ids = ["x_in"]
    graph.output_node_ids = ["y_out", "aux_loss_out"]
    return graph

def build_decoder_block_graph(name: str, model_dim: int, spec: BlockSpec, attention_graph: NeuronGraph, mlp_graph: NeuronGraph) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 140)))
    
    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module

    # Attn Path
    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_dim}), instance_id="attn_norm", position=(260, 80)))
    graph.add_edge(Edge(id="e_x_attn_norm", src_node="x_in", src_port=0, dst_node="attn_norm", dst_port=0))

    graph.add_node(NeuronInstance(link_variant_neuron(attention_graph, family="attention", version="default", name="attention", input_aliases=["x"], output_aliases=["attn_out"]), instance_id="attention", position=(480, 80)))
    graph.add_edge(Edge(id="e_norm_attn", src_node="attn_norm", src_port=0, dst_node="attention", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="attn_add", position=(700, 140)))
    graph.add_edge(Edge(id="e_x_attn_add", src_node="x_in", src_port=0, dst_node="attn_add", dst_port=0))
    graph.add_edge(Edge(id="e_attn_attn_add", src_node="attention", src_port=0, dst_node="attn_add", dst_port=1))

    # MLP Path
    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_dim}), instance_id="mlp_norm", position=(920, 80)))
    graph.add_edge(Edge(id="e_add_mlp_norm", src_node="attn_add", src_port=0, dst_node="mlp_norm", dst_port=0))

    out_aliases = ["y"]
    if spec.mlp_type == "moe":
        out_aliases.append("aux_loss")
    
    graph.add_node(NeuronInstance(link_variant_neuron(mlp_graph, family="mlp", version="default", name="mlp", input_aliases=["x"], output_aliases=out_aliases), instance_id="mlp", position=(1140, 80)))
    graph.add_edge(Edge(id="e_norm_mlp", src_node="mlp_norm", src_port=0, dst_node="mlp", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="mlp_add", position=(1360, 140)))
    graph.add_edge(Edge(id="e_attnadd_mlpadd", src_node="attn_add", src_port=0, dst_node="mlp_add", dst_port=0))
    graph.add_edge(Edge(id="e_mlp_mlpadd", src_node="mlp", src_port=0, dst_node="mlp_add", dst_port=1))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="x", dtype="tensor"), instance_id="x_out", position=(1580, 140)))
    graph.add_edge(Edge(id="e_mlpadd_xout", src_node="mlp_add", src_port=0, dst_node="x_out", dst_port=0))

    if spec.mlp_type == "moe":
        graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="aux_loss", dtype="tensor"), instance_id="aux_loss_out", position=(1580, 260)))
        graph.add_edge(Edge(id="e_mlp_auxout", src_node="mlp", src_port=1, dst_node="aux_loss_out", dst_port=0))
        graph.output_node_ids = ["x_out", "aux_loss_out"]
    else:
        graph.output_node_ids = ["x_out"]

    graph.input_node_ids = ["x_in"]
    return graph

def build_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    spec = model_spec.block_spec
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    
    # Pre-build variant libraries
    attn_graph = build_dense_attention_graph("attention_engine", model_spec.model_dim, spec)
    if spec.mlp_type == "moe":
        mlp_graph = build_moe_mlp_graph("mlp_moe", model_spec.model_dim, spec)
    else:
        mlp_graph = build_dense_mlp_graph("mlp_dense", model_spec.model_dim, spec)
    block_graph = build_decoder_block_graph("decoder_block", model_spec.model_dim, spec, attn_graph, mlp_graph)

    graph.variant_library = {
        "attention": {"default": attn_graph},
        "mlp": {"default": mlp_graph},
        "transformer_block": {"default": block_graph},
    }

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 140)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 460)))
    
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_embedding_module, config={"vocab_size": model_spec.vocab_size, "model_dim": model_spec.model_dim}), instance_id="token_embed", position=(260, 140)))
    graph.add_edge(Edge(id="e_tokens_embed", src_node="tokens_in", src_port=0, dst_node="token_embed", dst_port=0))

    curr_out = "token_embed"

    if spec.pos_encoding == "absolute":
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.absolute_position_embedding_module, config={"model_dim": model_spec.model_dim}), instance_id="pos_embed", position=(260, 260)))
        graph.add_edge(Edge(id="e_tokens_pos", src_node="tokens_in", src_port=0, dst_node="pos_embed", dst_port=0))
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="embed_add", position=(480, 140)))
        graph.add_edge(Edge(id="e_embed_add_a", src_node="token_embed", src_port=0, dst_node="embed_add", dst_port=0))
        graph.add_edge(Edge(id="e_pos_add_b", src_node="pos_embed", src_port=0, dst_node="embed_add", dst_port=1))
        curr_out = "embed_add"

    if spec.dropout_p > 0.0:
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.dropout_module, config={"p": spec.dropout_p}), instance_id="embed_dropout", position=(700, 140)))
        graph.add_edge(Edge(id="e_embed_dropout", src_node=curr_out, src_port=0, dst_node="embed_dropout", dst_port=0))
        curr_out = "embed_dropout"

    aux_losses = []
    
    for i in range(model_spec.num_layers):
        bname = f"block_{i}"
        
        out_aliases = ["x"]
        if spec.mlp_type == "moe":
            out_aliases.append("aux_loss")
            
        graph.add_node(NeuronInstance(link_variant_neuron(block_graph, family="transformer_block", version="default", name=bname, input_aliases=["x"], output_aliases=out_aliases), instance_id=bname, position=(920 + i*220, 140)))
        graph.add_edge(Edge(id=f"e_{curr_out}_{bname}", src_node=curr_out, src_port=0, dst_node=bname, dst_port=0))
        curr_out = bname
        if spec.mlp_type == "moe":
            aux_losses.append(bname)

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}), instance_id="final_norm", position=(1140 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(id="e_blocks_norm", src_node=curr_out, src_port=0, dst_node="final_norm", dst_port=0))

    head_id = "tied_lm_head" if model_spec.tie_embeddings else "lm_head"
    head_def = clone_neuron_def(BuiltinNeurons.tied_lm_head_module) if model_spec.tie_embeddings else clone_neuron_def(BuiltinNeurons.lm_head_module, config={"model_dim": model_spec.model_dim, "vocab_size": model_spec.vocab_size})
    
    graph.add_node(NeuronInstance(head_def, instance_id=head_id, position=(1360 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(id="e_norm_head", src_node="final_norm", src_port=0, dst_node=head_id, dst_port=0))
    if model_spec.tie_embeddings:
        graph.add_edge(Edge(id="e_tie", src_node="token_embed", src_port=1, dst_node=head_id, dst_port=1))
        
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.logit_softcap_module, config={"softcap": model_spec.logit_softcap}), instance_id="softcap", position=(1580 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(id="e_head_softcap", src_node=head_id, src_port=0, dst_node="softcap", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_cross_entropy_module), instance_id="ce", position=(1800 + model_spec.num_layers*220, 260)))
    graph.add_edge(Edge(id="e_softcap_ce", src_node="softcap", src_port=0, dst_node="ce", dst_port=0))
    graph.add_edge(Edge(id="e_targets_ce", src_node="targets_in", src_port=0, dst_node="ce", dst_port=1))

    main_loss = "ce"
    # Accumulate Aux Losses
    if spec.mlp_type == "moe" and aux_losses and spec.router_aux_loss_coef > 0.0:
        # Sum them up
        prev_loss = f"{aux_losses[0]}_loss"
        for i, b_id in enumerate(aux_losses):
            if i == 0:
                prev_loss = b_id
                continue
            add_id = f"add_aux_{i}"
            graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id=add_id, position=(1800 + model_spec.num_layers*220, 380 + i*80)))
            graph.add_edge(Edge(id=f"e_{b_id}_add", src_node=b_id, src_port=1, dst_node=add_id, dst_port=0))
            graph.add_edge(Edge(id=f"e_{prev_loss}_add", src_node=prev_loss, src_port=1 if i == 1 else 0, dst_node=add_id, dst_port=1))
            prev_loss = add_id
            
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.aux_loss_add_module, config={"coef": spec.router_aux_loss_coef}), instance_id="total_loss", position=(2020 + model_spec.num_layers*220, 260)))
        graph.add_edge(Edge(id="e_ce_total_loss", src_node="ce", src_port=0, dst_node="total_loss", dst_port=0))
        graph.add_edge(Edge(id="e_aux_total_loss", src_node=prev_loss, src_port=1 if len(aux_losses) == 1 else 0, dst_node="total_loss", dst_port=1))
        main_loss = "total_loss"

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(2240 + model_spec.num_layers*220, 260)))
    graph.add_edge(Edge(id="e_to_loss_out", src_node=main_loss, src_port=0, dst_node="loss_out", dst_port=0))

    graph.input_node_ids = ["tokens_in", "targets_in"]
    graph.output_node_ids = ["loss_out"]
    return graph

def build_gpt_root_graph(*, name: str = "model_root", model_spec: ModelSpec | None = None) -> NeuronGraph:
    if model_spec is None:
        model_spec = ModelSpec()
        
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch", torch_config={"device": "cuda", "amp_dtype": "bfloat16"})
    
    stage = build_model_stage_graph("model_stage", model_spec)
    graph.variant_library = deepcopy(stage.variant_library)
    stage.variant_library = {} # detach so it's not nested infinitely
    
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 120)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 300)))
    
    graph.add_node(NeuronInstance(subgraph_neuron(stage, name="model", input_aliases=["tokens", "targets"], output_aliases=["loss"]), instance_id="model", position=(280, 180)))
    
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(560, 180)))
    
    graph.add_edge(Edge(id="e_tokens_model", src_node="tokens_in", src_port=0, dst_node="model", dst_port=0))
    graph.add_edge(Edge(id="e_targets_model", src_node="targets_in", src_port=0, dst_node="model", dst_port=1))
    graph.add_edge(Edge(id="e_model_out", src_node="model", src_port=0, dst_node="loss_out", dst_port=0))
    
    graph.input_node_ids = ["tokens_in", "targets_in"]
    graph.output_node_ids = ["loss_out"]
    return graph
"""

with open("neuralfn/torch_templates.py", "w") as f:
    f.write(new_code)
