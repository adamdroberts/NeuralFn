from __future__ import annotations

from copy import deepcopy
from typing import Any

from .builtins import BuiltinNeurons
from .config import BlockSpec, ModelSpec, TemplateSpec, model_spec_to_dict
from .graph import Edge, NeuronGraph, NeuronInstance
from .neuron import NeuronDef, neuron_from_source, subgraph_neuron
from .port import Port
from .semantic import NUM_VOCAB_DIMS


def make_terminal_def(
    *,
    role: str,
    port_name: str,
    dtype: str,
    neuron_id: str | None = None,
) -> NeuronDef:
    source = f"def {role}(x):\n    return x\n"
    precision = 1.0 if dtype == "tokens" else None
    ports = [Port(port_name, precision=precision, dtype=dtype)]
    return neuron_from_source(source, role, ports, ports, neuron_id=neuron_id)


def clone_neuron_def(ndef: NeuronDef, *, config: dict[str, Any] | None = None) -> NeuronDef:
    cloned = NeuronDef.from_dict(deepcopy(ndef.to_dict()))
    if config is not None:
        cloned.module_config.update(config)
    return cloned


def get_linear_module_def(
    input_dim: int,
    output_dim: int,
    spec: BlockSpec,
    *,
    lora_role: str | None = None,
) -> NeuronDef:
    """Return a NeuronDef for a linear projection honouring ``BlockSpec``.

    ``lora_role`` identifies this projection's canonical name (e.g. ``"q_proj"``,
    ``"v_proj"``, ``"up_proj"``). When the role matches ``spec.lora_targets``
    and ``spec.adapter_type`` is ``"lora"`` or ``"qlora"``, the returned def is
    a LoRA (or nf4+LoRA) linear rather than a plain ``linear_module``.
    """
    if spec.compression == "ternary_b158":
        return clone_neuron_def(BuiltinNeurons.bitlinear_ternary_module, config={"input_dim": input_dim, "output_dim": output_dim})
    if spec.family == "ttt":
        return clone_neuron_def(
            BuiltinNeurons.ttt_linear_module,
            config={"input_dim": input_dim, "output_dim": output_dim, "hidden_dim": spec.ttt_hidden_dim},
        )
    adapter_type = getattr(spec, "adapter_type", "none")
    lora_targets = tuple(getattr(spec, "lora_targets", ()) or ())
    if lora_role is not None and lora_role in lora_targets and adapter_type in {"lora", "qlora"}:
        base_def = BuiltinNeurons.nf4_linear_module if adapter_type == "qlora" else BuiltinNeurons.lora_linear_module
        cfg: dict[str, Any] = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "rank": int(getattr(spec, "lora_rank", 8)),
            "alpha": float(getattr(spec, "lora_alpha", 16.0)),
            "dropout": float(getattr(spec, "lora_dropout", 0.0)),
            "bias": bool(getattr(spec, "lora_bias", False) or spec.linear_bias),
        }
        if adapter_type == "qlora":
            cfg["group_size"] = int(getattr(spec, "qlora_group_size", 64))
            cfg["compute_dtype"] = str(getattr(spec, "qlora_compute_dtype", "bf16"))
        return clone_neuron_def(base_def, config=cfg)
    return clone_neuron_def(BuiltinNeurons.linear_module, config={"input_dim": input_dim, "output_dim": output_dim, "bias": spec.linear_bias})


def maybe_wrap_with_adapter(graph: NeuronGraph, node_id: str, model_dim: int, spec: BlockSpec, *, position: tuple[float, float]) -> str:
    # The ``randmap`` adapter is a distinct, coarse-grained residual wrapper
    # that sits *after* a dense projection. ``lora``/``qlora`` adapters are
    # applied inside ``get_linear_module_def`` as a drop-in replacement for
    # the projection itself, so they do not need this wrapper.
    adapter_type = getattr(spec, "adapter_type", "none")
    if spec.adapter_dim <= 0 and adapter_type != "randmap":
        return node_id
    if adapter_type not in {"randmap", "none"}:
        return node_id

    adapter_id = f"{node_id}_adapter"
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.randmap_adapter_module, config={"model_dim": model_dim, "adapter_dim": spec.adapter_dim}), instance_id=adapter_id, position=position))
    graph.add_edge(Edge(src_node=node_id, src_port=0, dst_node=adapter_id, dst_port=0))

    to_remap = [eid for eid, e in graph.edges.items() if e.src_node == node_id and eid != f"e_{node_id}_to_{adapter_id}" and e.dst_node != adapter_id]
    for eid in to_remap:
        edge = graph.edges[eid]
        edge.src_node = adapter_id

    return adapter_id


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


def build_dense_attention_graph(
    name: str,
    model_dim: int,
    spec: BlockSpec,
    *,
    is_cross: bool = False,
    enable_cache: bool = False,
    enable_pca: bool = False,
    pca_compressed_dim: int | None = None,
    fused_megakernel: bool = False,
) -> NeuronGraph:
    # Megakernel fused path: single node replaces the whole Q/K/V -> SDPA -> out chain.
    # Only available for non-cross, RoPE-based causal attention (no cache/PCA in fused mode).
    if fused_megakernel and not is_cross and spec.pos_encoding == "rope":
        graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
        num_heads = spec.num_heads
        num_kv_heads = spec.num_kv_heads if spec.num_kv_heads is not None else num_heads
        fused_cfg = {
            "model_dim": model_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "rope_base": spec.rope_theta,
            "dropout_p": spec.dropout_p,
        }
        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 180)))
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.fused_causal_attention_module, config=fused_cfg), instance_id="fused_attn", position=(300, 180)))
        graph.add_edge(Edge(id="e_x_fused", src_node="x_in", src_port=0, dst_node="fused_attn", dst_port=0))
        graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="attn_out", dtype="tensor"), instance_id="attn_out", position=(560, 180)))
        graph.add_edge(Edge(id="e_fused_out", src_node="fused_attn", src_port=0, dst_node="attn_out", dst_port=0))
        graph.input_node_ids = ["x_in"]
        graph.output_node_ids = ["attn_out"]
        return graph

    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")

    num_heads = spec.num_heads
    num_kv_heads = spec.num_kv_heads if spec.num_kv_heads is not None else num_heads
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    compressed_dim = pca_compressed_dim or max(head_dim // 4, 1)

    # Input nodes
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 180)))
    if is_cross:
        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="context", dtype="tensor"), instance_id="context_in", position=(40, 300)))
    if enable_cache:
        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="cache_k", dtype="tensor"), instance_id="cache_k_in", position=(40, 420)))
        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="cache_v", dtype="tensor"), instance_id="cache_v_in", position=(40, 540)))

    # Q,K,V Projections
    graph.add_node(NeuronInstance(get_linear_module_def(model_dim, model_dim, spec), instance_id="q_proj", position=(220, 60)))
    graph.add_node(NeuronInstance(get_linear_module_def(model_dim, kv_dim, spec), instance_id="k_proj", position=(220, 180)))
    graph.add_node(NeuronInstance(get_linear_module_def(model_dim, kv_dim, spec), instance_id="v_proj", position=(220, 300)))

    curr_q = maybe_wrap_with_adapter(graph, "q_proj", model_dim, spec, position=(340, 60))
    curr_k = maybe_wrap_with_adapter(graph, "k_proj", kv_dim, spec, position=(340, 180))
    curr_v = maybe_wrap_with_adapter(graph, "v_proj", kv_dim, spec, position=(340, 300))

    graph.add_edge(Edge(id="e_x_q", src_node="x_in", src_port=0, dst_node="q_proj", dst_port=0))

    kv_src = "context_in" if is_cross else "x_in"
    graph.add_edge(Edge(id="e_kv_k", src_node=kv_src, src_port=0, dst_node="k_proj", dst_port=0))
    graph.add_edge(Edge(id="e_kv_v", src_node=kv_src, src_port=0, dst_node="v_proj", dst_port=0))

    # Reshape
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.reshape_heads_module, config={"num_heads": num_heads}), instance_id="q_heads", position=(460, 60)))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.reshape_heads_module, config={"num_heads": num_kv_heads}), instance_id="k_heads", position=(460, 180)))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.reshape_heads_module, config={"num_heads": num_kv_heads}), instance_id="v_heads", position=(460, 300)))

    graph.add_edge(Edge(id="e_q_qheads", src_node=curr_q, src_port=0, dst_node="q_heads", dst_port=0))
    graph.add_edge(Edge(id="e_k_kheads", src_node=curr_k, src_port=0, dst_node="k_heads", dst_port=0))
    graph.add_edge(Edge(id="e_v_vheads", src_node=curr_v, src_port=0, dst_node="v_heads", dst_port=0))

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

    # Track the K/V source nodes/ports feeding into attention (or into cache/PCA)
    curr_k_pre = curr_k
    k_pre_port = k_port
    curr_v_pre = "v_heads"
    v_pre_port = 0

    # Optional PCA encode (compress K/V before cache)
    if enable_pca:
        pca_cfg = {"head_dim": head_dim, "compressed_dim": compressed_dim}
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.kv_pca_encode_module, config=pca_cfg), instance_id="pca_encode", position=(820, 240)))
        graph.add_edge(Edge(id="e_k_pca_enc", src_node=curr_k_pre, src_port=k_pre_port, dst_node="pca_encode", dst_port=0))
        graph.add_edge(Edge(id="e_v_pca_enc", src_node=curr_v_pre, src_port=v_pre_port, dst_node="pca_encode", dst_port=1))
        curr_k_pre = "pca_encode"
        k_pre_port = 0
        curr_v_pre = "pca_encode"
        v_pre_port = 1

    # Optional KV cache: write current K/V then read with cached prefix
    if enable_cache:
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.kv_cache_write_module), instance_id="kv_cache_write", position=(940, 420)))
        graph.add_edge(Edge(id="e_k_cache_write", src_node=curr_k_pre, src_port=k_pre_port, dst_node="kv_cache_write", dst_port=0))
        graph.add_edge(Edge(id="e_v_cache_write", src_node=curr_v_pre, src_port=v_pre_port, dst_node="kv_cache_write", dst_port=1))

        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.kv_cache_read_module), instance_id="kv_cache_read", position=(940, 240)))
        graph.add_edge(Edge(id="e_k_cache_read", src_node=curr_k_pre, src_port=k_pre_port, dst_node="kv_cache_read", dst_port=0))
        graph.add_edge(Edge(id="e_v_cache_read", src_node=curr_v_pre, src_port=v_pre_port, dst_node="kv_cache_read", dst_port=1))
        graph.add_edge(Edge(id="e_ck_cache_read", src_node="cache_k_in", src_port=0, dst_node="kv_cache_read", dst_port=2))
        graph.add_edge(Edge(id="e_cv_cache_read", src_node="cache_v_in", src_port=0, dst_node="kv_cache_read", dst_port=3))

        curr_k_pre = "kv_cache_read"
        k_pre_port = 0
        curr_v_pre = "kv_cache_read"
        v_pre_port = 1

    # Optional PCA decode (decompress K/V after cache read, before attention)
    if enable_pca:
        pca_cfg = {"head_dim": head_dim, "compressed_dim": compressed_dim}
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.kv_pca_decode_module, config=pca_cfg), instance_id="pca_decode", position=(1060, 240)))
        graph.add_edge(Edge(id="e_k_pca_dec", src_node=curr_k_pre, src_port=k_pre_port, dst_node="pca_decode", dst_port=0))
        graph.add_edge(Edge(id="e_v_pca_dec", src_node=curr_v_pre, src_port=v_pre_port, dst_node="pca_decode", dst_port=1))
        curr_k_pre = "pca_decode"
        k_pre_port = 0
        curr_v_pre = "pca_decode"
        v_pre_port = 1

    # Repeat KV for GQA
    curr_k_attn = curr_k_pre
    curr_v_attn = curr_v_pre
    k_attn_port = k_pre_port
    v_attn_port = v_pre_port
    if num_kv_heads < num_heads:
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.repeat_kv_module, config={"num_heads": num_heads, "num_kv_heads": num_kv_heads}), instance_id="k_repeat", position=(1100, 180)))
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.repeat_kv_module, config={"num_heads": num_heads, "num_kv_heads": num_kv_heads}), instance_id="v_repeat", position=(1100, 300)))
        graph.add_edge(Edge(id=f"e_{curr_k_attn}_krepeat", src_node=curr_k_attn, src_port=k_attn_port, dst_node="k_repeat", dst_port=0))
        graph.add_edge(Edge(id=f"e_{curr_v_attn}_vrepeat", src_node=curr_v_attn, src_port=v_attn_port, dst_node="v_repeat", dst_port=0))
        curr_k_attn = "k_repeat"
        curr_v_attn = "v_repeat"
        k_attn_port = 0
        v_attn_port = 0

    # Attention Core
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.scaled_dot_product_attention_module, config={"is_causal": spec.is_causal and not is_cross, "backend": spec.attention_backend, "dropout_p": spec.dropout_p}), instance_id="sdpa", position=(1300, 180)))
    graph.add_edge(Edge(id=f"e_{curr_q}_sdpa_0", src_node=curr_q, src_port=q_port, dst_node="sdpa", dst_port=0))
    graph.add_edge(Edge(id=f"e_{curr_k_attn}_sdpa_1", src_node=curr_k_attn, src_port=k_attn_port, dst_node="sdpa", dst_port=1))
    graph.add_edge(Edge(id=f"e_{curr_v_attn}_sdpa_2", src_node=curr_v_attn, src_port=v_attn_port, dst_node="sdpa", dst_port=2))
    curr_out = "sdpa"

    # Merge Heads
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.merge_heads_module), instance_id="merge", position=(1540, 180)))
    graph.add_edge(Edge(id="e_sdpa_merge", src_node=curr_out, src_port=0, dst_node="merge", dst_port=0))

    # Output Proj
    graph.add_node(NeuronInstance(get_linear_module_def(model_dim, model_dim, spec), instance_id="out_proj", position=(1780, 180)))
    curr_out = maybe_wrap_with_adapter(graph, "out_proj", model_dim, spec, position=(1900, 180))
    graph.add_edge(Edge(id="e_merge_outproj", src_node="merge", src_port=0, dst_node="out_proj", dst_port=0))

    # Optional Dropout
    if spec.dropout_p > 0.0:
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.dropout_module, config={"p": spec.dropout_p}), instance_id="attn_dropout", position=(2020, 180)))
        graph.add_edge(Edge(id="e_outproj_dropout", src_node=curr_out, src_port=0, dst_node="attn_dropout", dst_port=0))
        curr_out = "attn_dropout"

    # Output Ports
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="attn_out", dtype="tensor"), instance_id="attn_out", position=(2260, 180)))
    graph.add_edge(Edge(id="e_to_attn_out", src_node=curr_out, src_port=0, dst_node="attn_out", dst_port=0))

    output_ids = ["attn_out"]
    if enable_cache:
        graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="new_cache_k", dtype="tensor"), instance_id="new_cache_k", position=(2260, 420)))
        graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="new_cache_v", dtype="tensor"), instance_id="new_cache_v", position=(2260, 540)))
        graph.add_edge(Edge(id="e_cache_write_k_out", src_node="kv_cache_write", src_port=0, dst_node="new_cache_k", dst_port=0))
        graph.add_edge(Edge(id="e_cache_write_v_out", src_node="kv_cache_write", src_port=1, dst_node="new_cache_v", dst_port=0))
        output_ids.extend(["new_cache_k", "new_cache_v"])

    input_ids = ["x_in"]
    if is_cross:
        input_ids.append("context_in")
    if enable_cache:
        input_ids.extend(["cache_k_in", "cache_v_in"])

    graph.input_node_ids = input_ids
    graph.output_node_ids = output_ids
    return graph


def build_dense_mlp_graph(name: str, model_dim: int, spec: BlockSpec) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 120)))

    curr_out = "x_in"
    if spec.mlp_type == "gelu":
        hidden = int(model_dim * spec.mlp_multiplier)
        graph.add_node(NeuronInstance(get_linear_module_def(model_dim, hidden, spec), instance_id="fc1", position=(260, 120)))
        curr_fc1 = maybe_wrap_with_adapter(graph, "fc1", hidden, spec, position=(370, 120))
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.gelu_module), instance_id="gelu", position=(480, 120)))
        graph.add_node(NeuronInstance(get_linear_module_def(hidden, model_dim, spec), instance_id="fc2", position=(700, 120)))
        curr_out = maybe_wrap_with_adapter(graph, "fc2", model_dim, spec, position=(810, 120))
        graph.add_edge(Edge(id="e_x_fc1", src_node="x_in", src_port=0, dst_node="fc1", dst_port=0))
        graph.add_edge(Edge(id="e_fc1_gelu", src_node=curr_fc1, src_port=0, dst_node="gelu", dst_port=0))
        graph.add_edge(Edge(id="e_gelu_fc2", src_node="gelu", src_port=0, dst_node="fc2", dst_port=0))
    elif spec.mlp_type == "swiglu":
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.swiglu_module, config={"model_dim": model_dim, "mlp_mult": spec.mlp_multiplier, "multiple_of": spec.multiple_of}), instance_id="swiglu", position=(480, 120)))
        curr_out = maybe_wrap_with_adapter(graph, "swiglu", model_dim, spec, position=(600, 120))
        graph.add_edge(Edge(id="e_x_swiglu", src_node="x_in", src_port=0, dst_node="swiglu", dst_port=0))
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

def build_mixllama_mlp_graph(name: str, model_dim: int, spec: BlockSpec) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 160)))

    # Router
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.router_logits_module, config={"model_dim": model_dim, "experts": spec.experts}), instance_id="router", position=(260, 60)))
    graph.add_edge(Edge(id="e_x_router", src_node="x_in", src_port=0, dst_node="router", dst_port=0))

    # Top-K
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.topk_route_module,
                config={"top_k": spec.top_k, "experts": spec.experts},
            ),
            instance_id="topk",
            position=(480, 60),
        )
    )
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


def build_semantic_router_mlp_graph(name: str, model_dim: int, spec: BlockSpec) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 180)))
    graph.add_node(
        NeuronInstance(make_terminal_def(role="input", port_name="routing_weights", dtype="tensor"), instance_id="routing_weights_in", position=(40, 60))
    )
    graph.add_node(
        NeuronInstance(make_terminal_def(role="input", port_name="routing_indices", dtype="tensor"), instance_id="routing_indices_in", position=(40, 300))
    )

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.expert_dispatch_module,
                config={"model_dim": model_dim, "experts": spec.experts, "mlp_mult": spec.mlp_multiplier},
            ),
            instance_id="dispatch",
            position=(320, 180),
        )
    )
    graph.add_edge(Edge(id="e_x_dispatch", src_node="x_in", src_port=0, dst_node="dispatch", dst_port=0))
    graph.add_edge(Edge(id="e_weights_dispatch", src_node="routing_weights_in", src_port=0, dst_node="dispatch", dst_port=1))
    graph.add_edge(Edge(id="e_indices_dispatch", src_node="routing_indices_in", src_port=0, dst_node="dispatch", dst_port=2))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.expert_combine_module), instance_id="combine", position=(560, 180)))
    graph.add_edge(Edge(id="e_dispatch_combine", src_node="dispatch", src_port=0, dst_node="combine", dst_port=0))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="y", dtype="tensor"), instance_id="y_out", position=(780, 180)))
    graph.add_edge(Edge(id="e_combine_y_out", src_node="combine", src_port=0, dst_node="y_out", dst_port=0))

    graph.input_node_ids = ["x_in", "routing_weights_in", "routing_indices_in"]
    graph.output_node_ids = ["y_out"]
    return graph

def build_mamba_graph(name: str, model_dim: int, spec: BlockSpec) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 120)))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.mamba_module, config={"model_dim": model_dim}), instance_id="mamba", position=(260, 120)))
    graph.add_edge(Edge(src_node="x_in", src_port=0, dst_node="mamba", dst_port=0))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="y", dtype="tensor"), instance_id="y_out", position=(480, 120)))
    graph.add_edge(Edge(src_node="mamba", src_port=0, dst_node="y_out", dst_port=0))

    graph.input_node_ids = ["x_in"]
    graph.output_node_ids = ["y_out"]
    return graph

def build_decoder_block_graph(
    name: str,
    model_dim: int,
    spec: BlockSpec,
    attention_graph: NeuronGraph,
    mlp_graph: NeuronGraph,
    *,
    cross_attention_graph: NeuronGraph | None = None,
    attention_family: str = "attention",
    cross_attention_family: str = "cross_attention",
    mlp_family: str = "mlp",
) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 140)))
    if cross_attention_graph:
        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="context", dtype="tensor"), instance_id="context_in", position=(40, 260)))

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module

    # Attn Path
    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_dim}), instance_id="attn_norm", position=(260, 80)))
    graph.add_edge(Edge(id="e_x_attn_norm", src_node="x_in", src_port=0, dst_node="attn_norm", dst_port=0))

    graph.add_node(NeuronInstance(link_variant_neuron(attention_graph, family=attention_family, version="default", name="attention", input_aliases=["x"], output_aliases=["attn_out"]), instance_id="attention", position=(480, 80)))
    graph.add_edge(Edge(id="e_norm_attn", src_node="attn_norm", src_port=0, dst_node="attention", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="attn_add", position=(700, 140)))
    graph.add_edge(Edge(id="e_x_attn_add", src_node="x_in", src_port=0, dst_node="attn_add", dst_port=0))
    graph.add_edge(Edge(id="e_attn_attn_add", src_node="attention", src_port=0, dst_node="attn_add", dst_port=1))

    curr_out = "attn_add"

    # Optional Cross Attention
    if cross_attention_graph:
        graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_dim}), instance_id="cross_norm", position=(700, 260)))
        graph.add_edge(Edge(id="e_add_cross_norm", src_node=curr_out, src_port=0, dst_node="cross_norm", dst_port=0))

        graph.add_node(NeuronInstance(link_variant_neuron(cross_attention_graph, family=cross_attention_family, version="default", name="cross_attention", input_aliases=["x", "context"], output_aliases=["attn_out"]), instance_id="cross_attn", position=(920, 260)))
        graph.add_edge(Edge(id="e_norm_cross", src_node="cross_norm", src_port=0, dst_node="cross_attn", dst_port=0))
        graph.add_edge(Edge(id="e_ctx_cross", src_node="context_in", src_port=0, dst_node="cross_attn", dst_port=1))

        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="cross_add", position=(1140, 140)))
        graph.add_edge(Edge(id="e_attnadd_crossadd", src_node=curr_out, src_port=0, dst_node="cross_add", dst_port=0))
        graph.add_edge(Edge(id="e_cross_crossadd", src_node="cross_attn", src_port=0, dst_node="cross_add", dst_port=1))
        curr_out = "cross_add"

    # MLP Path
    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_dim}), instance_id="mlp_norm", position=(920 if not cross_attention_graph else 1360, 80)))
    graph.add_edge(Edge(id="e_add_mlp_norm", src_node=curr_out, src_port=0, dst_node="mlp_norm", dst_port=0))

    out_aliases = ["y"]
    if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
        out_aliases.append("aux_loss")

    graph.add_node(NeuronInstance(link_variant_neuron(mlp_graph, family=mlp_family, version="default", name="mlp", input_aliases=["x"], output_aliases=out_aliases), instance_id="mlp", position=(1140 if not cross_attention_graph else 1580, 80)))
    graph.add_edge(Edge(id="e_norm_mlp", src_node="mlp_norm", src_port=0, dst_node="mlp", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="mlp_add", position=(1360 if not cross_attention_graph else 1800, 140)))
    graph.add_edge(Edge(id="e_attnadd_mlpadd", src_node=curr_out, src_port=0, dst_node="mlp_add", dst_port=0))
    graph.add_edge(Edge(id="e_mlp_mlpadd", src_node="mlp", src_port=0, dst_node="mlp_add", dst_port=1))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="x", dtype="tensor"), instance_id="x_out", position=(1580 if not cross_attention_graph else 2020, 140)))
    graph.add_edge(Edge(id="e_mlpadd_xout", src_node="mlp_add", src_port=0, dst_node="x_out", dst_port=0))

    if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
        graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="aux_loss", dtype="tensor"), instance_id="aux_loss_out", position=(1580 if not cross_attention_graph else 2020, 260)))
        graph.add_edge(Edge(id="e_mlp_auxout", src_node="mlp", src_port=1, dst_node="aux_loss_out", dst_port=0))
        graph.output_node_ids = ["x_out", "aux_loss_out"]
    else:
        graph.output_node_ids = ["x_out"]

    graph.input_node_ids = ["x_in", "context_in"] if cross_attention_graph else ["x_in"]
    return graph


def build_semantic_router_decoder_block_graph(
    name: str,
    model_dim: int,
    spec: BlockSpec,
    attention_graph: NeuronGraph,
    mlp_graph: NeuronGraph,
    *,
    attention_family: str = "attention",
    mlp_family: str = "semantic_router_mlp",
) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="x", dtype="tensor"), instance_id="x_in", position=(40, 160)))
    graph.add_node(
        NeuronInstance(make_terminal_def(role="input", port_name="routing_weights", dtype="tensor"), instance_id="routing_weights_in", position=(40, 40))
    )
    graph.add_node(
        NeuronInstance(make_terminal_def(role="input", port_name="routing_indices", dtype="tensor"), instance_id="routing_indices_in", position=(40, 280))
    )

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module

    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_dim}), instance_id="attn_norm", position=(260, 100)))
    graph.add_edge(Edge(id="e_x_attn_norm", src_node="x_in", src_port=0, dst_node="attn_norm", dst_port=0))

    graph.add_node(
        NeuronInstance(
            link_variant_neuron(
                attention_graph,
                family=attention_family,
                version="default",
                name="attention",
                input_aliases=["x"],
                output_aliases=["attn_out"],
            ),
            instance_id="attention",
            position=(480, 100),
        )
    )
    graph.add_edge(Edge(id="e_norm_attn", src_node="attn_norm", src_port=0, dst_node="attention", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="attn_add", position=(700, 160)))
    graph.add_edge(Edge(id="e_x_attn_add", src_node="x_in", src_port=0, dst_node="attn_add", dst_port=0))
    graph.add_edge(Edge(id="e_attn_attn_add", src_node="attention", src_port=0, dst_node="attn_add", dst_port=1))

    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_dim}), instance_id="mlp_norm", position=(920, 100)))
    graph.add_edge(Edge(id="e_attn_add_mlp_norm", src_node="attn_add", src_port=0, dst_node="mlp_norm", dst_port=0))

    graph.add_node(
        NeuronInstance(
            link_variant_neuron(
                mlp_graph,
                family=mlp_family,
                version="default",
                name="mlp",
                input_aliases=["x", "routing_weights", "routing_indices"],
                output_aliases=["y"],
            ),
            instance_id="mlp",
            position=(1140, 100),
        )
    )
    graph.add_edge(Edge(id="e_mlp_norm_mlp", src_node="mlp_norm", src_port=0, dst_node="mlp", dst_port=0))
    graph.add_edge(Edge(id="e_routing_weights_mlp", src_node="routing_weights_in", src_port=0, dst_node="mlp", dst_port=1))
    graph.add_edge(Edge(id="e_routing_indices_mlp", src_node="routing_indices_in", src_port=0, dst_node="mlp", dst_port=2))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="mlp_add", position=(1360, 160)))
    graph.add_edge(Edge(id="e_attn_add_mlp_add", src_node="attn_add", src_port=0, dst_node="mlp_add", dst_port=0))
    graph.add_edge(Edge(id="e_mlp_mlp_add", src_node="mlp", src_port=0, dst_node="mlp_add", dst_port=1))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="x", dtype="tensor"), instance_id="x_out", position=(1580, 160)))
    graph.add_edge(Edge(id="e_mlp_add_x_out", src_node="mlp_add", src_port=0, dst_node="x_out", dst_port=0))

    graph.input_node_ids = ["x_in", "routing_weights_in", "routing_indices_in"]
    graph.output_node_ids = ["x_out"]
    return graph


def _attn_flags(model_spec: ModelSpec) -> dict[str, Any]:
    """Derive enable_pca / fused_megakernel kwargs for build_dense_attention_graph.

    ``enable_cache`` is intentionally **not** set here.  KV cache nodes add
    extra input/output ports that break the single-alias subgraph wiring used
    by the training graph builders.  Cache is an inference-time concern --
    ``backend_capabilities["cache"]`` declares support, and the
    ``InferenceCache`` class (or a dedicated inference graph) handles actual
    cache management.
    """
    return {
        "enable_pca": model_spec.block_spec.compression == "kv_pca",
        "fused_megakernel": model_spec.template.runtime == "megakernel",
    }


def build_hidden_backbone_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    spec = model_spec.block_spec
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    block_family = "mixllama" if (spec.mlp_type == "mixllama" or spec.mlp_type == "moe") else "transformer_block"

    attn_graph = build_dense_attention_graph("attention_engine", model_spec.model_dim, spec, **_attn_flags(model_spec))
    mamba_graph = build_mamba_graph("mamba_engine", model_spec.model_dim, spec)
    if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
        mlp_graph = build_mixllama_mlp_graph("mlp_moe", model_spec.model_dim, spec)
    else:
        mlp_graph = build_dense_mlp_graph("mlp_dense", model_spec.model_dim, spec)

    attn_block_graph = build_decoder_block_graph("decoder_block_attn", model_spec.model_dim, spec, attn_graph, mlp_graph)
    mamba_block_graph = build_decoder_block_graph("decoder_block_mamba", model_spec.model_dim, spec, mamba_graph, mlp_graph)

    graph.variant_library = {
        "attention": {"default": attn_graph},
        "mamba": {"default": mamba_graph},
        "mlp": {"default": mlp_graph},
        "attn_block": {"default": attn_block_graph},
        "mamba_block": {"default": mamba_block_graph},
        block_family: {"default": attn_block_graph},
    }

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 140)))
    if model_spec.template.tokenization == "byte_hnet":
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(
                    BuiltinNeurons.byte_patch_embed_module,
                    config={
                        "model_dim": model_spec.model_dim,
                        "patch_size": spec.byte_patch_size,
                        "stride": spec.byte_patch_stride,
                        "vocab_size": model_spec.vocab_size,
                    },
                ),
                instance_id="token_embed",
                position=(260, 140),
            )
        )
        graph.add_edge(Edge(id="e_tokens_embed", src_node="tokens_in", src_port=0, dst_node="token_embed", dst_port=0))
        curr_out = "token_embed"
    else:
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(
                    BuiltinNeurons.token_embedding_module,
                    config={"vocab_size": model_spec.vocab_size, "model_dim": model_spec.model_dim},
                ),
                instance_id="token_embed",
                position=(260, 140),
            )
        )
        graph.add_edge(Edge(id="e_tokens_embed", src_node="tokens_in", src_port=0, dst_node="token_embed", dst_port=0))
        curr_out = "token_embed"

    if spec.pos_encoding == "absolute":
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.absolute_position_embedding_module, config={"model_dim": model_spec.model_dim}),
                instance_id="pos_embed",
                position=(260, 260),
            )
        )
        graph.add_edge(Edge(id="e_tokens_pos", src_node="tokens_in", src_port=0, dst_node="pos_embed", dst_port=0))
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="embed_add", position=(480, 140)))
        graph.add_edge(Edge(id="e_embed_add_a", src_node="token_embed", src_port=0, dst_node="embed_add", dst_port=0))
        graph.add_edge(Edge(id="e_pos_add_b", src_node="pos_embed", src_port=0, dst_node="embed_add", dst_port=1))
        curr_out = "embed_add"

    if spec.dropout_p > 0.0:
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.dropout_module, config={"p": spec.dropout_p}),
                instance_id="embed_dropout",
                position=(700, 140),
            )
        )
        graph.add_edge(Edge(id="e_embed_dropout", src_node=curr_out, src_port=0, dst_node="embed_dropout", dst_port=0))
        curr_out = "embed_dropout"

    for i in range(model_spec.num_layers):
        bname = f"block_{i}"
        is_attn = True
        if model_spec.template.backbone == "jamba":
            is_attn = (i % 4 == 0)
        if model_spec.template.backbone == "jamba":
            current_block_family = "attn_block" if is_attn else "mamba_block"
            current_block_graph = attn_block_graph if is_attn else mamba_block_graph
        else:
            current_block_family = block_family
            current_block_graph = attn_block_graph
        out_aliases = ["x"]
        if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
            out_aliases.append("aux_loss")
        graph.add_node(
            NeuronInstance(
                link_variant_neuron(current_block_graph, family=current_block_family, version="default", name=bname, input_aliases=["x"], output_aliases=out_aliases),
                instance_id=bname,
                position=(920 + i * 220, 140),
            )
        )
        graph.add_edge(Edge(id=f"e_{curr_out}_{bname}", src_node=curr_out, src_port=0, dst_node=bname, dst_port=0))
        curr_out = bname

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}),
            instance_id="final_norm",
            position=(1140 + model_spec.num_layers * 220, 140),
        )
    )
    graph.add_edge(Edge(id="e_blocks_norm", src_node=curr_out, src_port=0, dst_node="final_norm", dst_port=0))
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="hidden", dtype="tensor"), instance_id="hidden_out", position=(1360 + model_spec.num_layers * 220, 140)))
    graph.add_edge(Edge(id="e_norm_hidden", src_node="final_norm", src_port=0, dst_node="hidden_out", dst_port=0))
    graph.input_node_ids = ["tokens_in"]
    graph.output_node_ids = ["hidden_out"]
    return graph


def build_jepa_encoder_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    hidden_graph = build_hidden_backbone_graph("hidden_backbone", model_spec)
    graph.variant_library = deepcopy(hidden_graph.variant_library)
    hidden_graph.variant_library = {}

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 120)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="mask", dtype="tensor"), instance_id="mask_in", position=(40, 260)))
    graph.add_node(NeuronInstance(subgraph_neuron(hidden_graph, name="backbone", input_aliases=["tokens"], output_aliases=["hidden"]), instance_id="backbone", position=(260, 120)))
    graph.add_edge(Edge(id="e_tokens_backbone", src_node="tokens_in", src_port=0, dst_node="backbone", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.latent_pool_module), instance_id="pool", position=(520, 140)))
    graph.add_edge(Edge(id="e_backbone_pool", src_node="backbone", src_port=0, dst_node="pool", dst_port=0))
    graph.add_edge(Edge(id="e_mask_pool", src_node="mask_in", src_port=0, dst_node="pool", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.jepa_projector_module,
                config={"input_dim": model_spec.model_dim, "latent_dim": model_spec.jepa_latent_dim},
            ),
            instance_id="projector",
            position=(760, 140),
        )
    )
    graph.add_edge(Edge(id="e_pool_projector", src_node="pool", src_port=0, dst_node="projector", dst_port=0))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="latent", dtype="tensor"), instance_id="latent_out", position=(980, 140)))
    graph.add_edge(Edge(id="e_projector_out", src_node="projector", src_port=0, dst_node="latent_out", dst_port=0))
    graph.input_node_ids = ["tokens_in", "mask_in"]
    graph.output_node_ids = ["latent_out"]
    return graph

def build_seq2seq_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    spec = model_spec.block_spec
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")

    # Pre-build variant libraries
    encoder_spec = deepcopy(spec)
    encoder_spec.is_causal = False
    # Use SWIGLU for dense encoder MLP if the base is Llama-style (moe/swiglu)
    encoder_spec.mlp_type = "swiglu" if spec.mlp_type in ["moe", "mixllama", "swiglu"] else "gelu"

    af = _attn_flags(model_spec)
    enc_attn_graph = build_dense_attention_graph("enc_attention", model_spec.model_dim, encoder_spec, **af)
    dec_attn_graph = build_dense_attention_graph("dec_attention", model_spec.model_dim, spec, **af)
    cross_attn_graph = build_dense_attention_graph("cross_attention", model_spec.model_dim, spec, is_cross=True, **af)

    if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
        mlp_moe = build_mixllama_mlp_graph("mlp_moe", model_spec.model_dim, spec)
        mlp_dense = build_dense_mlp_graph("mlp_dense", model_spec.model_dim, encoder_spec)
    else:
        mlp_dense = build_dense_mlp_graph("mlp_dense", model_spec.model_dim, encoder_spec)
        mlp_moe = mlp_dense

    # Encoder block: Dense MLP always for encoder as per plan
    enc_block_graph = build_decoder_block_graph(
        "enc_block",
        model_spec.model_dim,
        encoder_spec,
        enc_attn_graph,
        mlp_dense,
        attention_family="enc_attention",
        mlp_family="mlp_dense",
    )
    # Decoder block: MoE MLP if requested
    dec_block_graph = build_decoder_block_graph(
        "dec_block",
        model_spec.model_dim,
        spec,
        dec_attn_graph,
        mlp_moe,
        cross_attention_graph=cross_attn_graph,
        attention_family="dec_attention",
        cross_attention_family="cross_attention",
        mlp_family="mlp_moe",
    )

    graph.variant_library = {
        "enc_attention": {"default": enc_attn_graph},
        "dec_attention": {"default": dec_attn_graph},
        "cross_attention": {"default": cross_attn_graph},
        "mlp_dense": {"default": mlp_dense},
        "mlp_moe": {"default": mlp_moe},
        "enc_block": {"default": enc_block_graph},
        "dec_block": {"default": dec_block_graph},
    }

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="enc_tokens", dtype="tokens"), instance_id="enc_tokens_in", position=(40, 140)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="dec_tokens", dtype="tokens"), instance_id="dec_tokens_in", position=(40, 300)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 460)))

    # Embeddings
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_embedding_module, config={"vocab_size": model_spec.vocab_size, "model_dim": model_spec.model_dim}), instance_id="enc_embed", position=(260, 140)))
    graph.add_edge(Edge(src_node="enc_tokens_in", src_port=0, dst_node="enc_embed", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_embedding_module, config={"vocab_size": model_spec.vocab_size, "model_dim": model_spec.model_dim}), instance_id="dec_embed", position=(260, 300)))
    graph.add_edge(Edge(src_node="dec_tokens_in", src_port=0, dst_node="dec_embed", dst_port=0))

    # Encoder stack
    curr_enc = "enc_embed"
    for i in range(model_spec.num_layers):
        bname = f"enc_block_{i}"
        graph.add_node(NeuronInstance(link_variant_neuron(enc_block_graph, family="enc_block", version="default", name=bname, input_aliases=["x"], output_aliases=["x"]), instance_id=bname, position=(480 + i*220, 140)))
        graph.add_edge(Edge(src_node=curr_enc, src_port=0, dst_node=bname, dst_port=0))
        curr_enc = bname

    # Decoder stack
    curr_dec = "dec_embed"
    aux_losses = []
    for i in range(model_spec.num_layers):
        bname = f"dec_block_{i}"
        out_aliases = ["x"]
        if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
            out_aliases.append("aux_loss")

        graph.add_node(NeuronInstance(link_variant_neuron(dec_block_graph, family="dec_block", version="default", name=bname, input_aliases=["x", "context"], output_aliases=out_aliases), instance_id=bname, position=(480 + i*220, 300)))
        graph.add_edge(Edge(src_node=curr_dec, src_port=0, dst_node=bname, dst_port=0))
        graph.add_edge(Edge(src_node=curr_enc, src_port=0, dst_node=bname, dst_port=1))
        curr_dec = bname
        if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
            aux_losses.append(bname)

    # Output head
    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}), instance_id="final_norm", position=(700 + model_spec.num_layers*220, 300)))
    graph.add_edge(Edge(src_node=curr_dec, src_port=0, dst_node="final_norm", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.lm_head_module, config={"model_dim": model_spec.model_dim, "vocab_size": model_spec.vocab_size}), instance_id="lm_head", position=(920 + model_spec.num_layers*220, 300)))
    graph.add_edge(Edge(src_node="final_norm", src_port=0, dst_node="lm_head", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_cross_entropy_module), instance_id="ce", position=(1140 + model_spec.num_layers*220, 300)))
    graph.add_edge(Edge(src_node="lm_head", src_port=0, dst_node="ce", dst_port=0))
    graph.add_edge(Edge(src_node="targets_in", src_port=0, dst_node="ce", dst_port=1))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(1360 + model_spec.num_layers*220, 300)))
    graph.add_edge(Edge(src_node="ce", src_port=0, dst_node="loss_out", dst_port=0))

    graph.input_node_ids = ["enc_tokens_in", "dec_tokens_in", "targets_in"]
    graph.output_node_ids = ["loss_out"]
    return graph

def build_diffusion_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    spec = model_spec.block_spec
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")

    # Pre-build variant libraries
    attn_graph = build_dense_attention_graph("attention_engine", model_spec.model_dim, spec, **_attn_flags(model_spec))
    mlp_graph = build_dense_mlp_graph("mlp_dense", model_spec.model_dim, spec)
    block_graph = build_decoder_block_graph("decoder_block", model_spec.model_dim, spec, attn_graph, mlp_graph)

    graph.variant_library = {
        "attention": {"default": attn_graph},
        "mlp": {"default": mlp_graph},
        "transformer_block": {"default": block_graph},
    }

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 140)))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.random_timesteps_module), instance_id="timesteps", position=(260, 300)))
    graph.add_edge(Edge(src_node="tokens_in", src_port=0, dst_node="timesteps", dst_port=0))

    # Mask Scheduler
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.mask_scheduler_module, config={"vocab_size": model_spec.vocab_size, "mask_token_id": 0}), instance_id="mask_sched", position=(480, 140)))
    graph.add_edge(Edge(src_node="tokens_in", src_port=0, dst_node="mask_sched", dst_port=0))
    graph.add_edge(Edge(src_node="timesteps", src_port=0, dst_node="mask_sched", dst_port=1))

    # Embeddings
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_embedding_module, config={"vocab_size": model_spec.vocab_size, "model_dim": model_spec.model_dim}), instance_id="embed", position=(700, 140)))
    graph.add_edge(Edge(src_node="mask_sched", src_port=0, dst_node="embed", dst_port=0))

    # Blocks
    curr_out = "embed"
    for i in range(model_spec.num_layers):
        bname = f"block_{i}"
        graph.add_node(NeuronInstance(link_variant_neuron(block_graph, family="transformer_block", version="default", name=bname, input_aliases=["x"], output_aliases=["x"]), instance_id=bname, position=(920 + i*220, 140)))
        graph.add_edge(Edge(src_node=curr_out, src_port=0, dst_node=bname, dst_port=0))
        curr_out = bname

    # Final Norm
    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}), instance_id="final_norm", position=(1140 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(src_node=curr_out, src_port=0, dst_node="final_norm", dst_port=0))

    # Denoise Head
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.denoise_head_module, config={"model_dim": model_spec.model_dim, "vocab_size": model_spec.vocab_size}), instance_id="denoise_head", position=(1360 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(src_node="final_norm", src_port=0, dst_node="denoise_head", dst_port=0))

    # Cross Entropy (Diffusion uses categorical XE over vocab)
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_cross_entropy_module), instance_id="ce", position=(1580 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(src_node="denoise_head", src_port=0, dst_node="ce", dst_port=0))
    graph.add_edge(Edge(src_node="tokens_in", src_port=0, dst_node="ce", dst_port=1))

    # Output Port
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(1800 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(src_node="ce", src_port=0, dst_node="loss_out", dst_port=0))

    graph.input_node_ids = ["tokens_in"]
    graph.output_node_ids = ["loss_out"]
    return graph


def build_jepa_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    encoder_graph = build_jepa_encoder_graph("jepa_encoder", model_spec)
    graph.variant_library = deepcopy(encoder_graph.variant_library)
    encoder_graph.variant_library = {}

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 180)))
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.jepa_mask_module,
                config={
                    "mask_ratio": model_spec.jepa_mask_ratio,
                    "mask_token_id": 0,
                    "mask_strategy": model_spec.jepa_mask_strategy,
                    "num_blocks": model_spec.jepa_num_blocks,
                    "min_block_ratio": model_spec.jepa_min_block_ratio,
                    "max_block_ratio": model_spec.jepa_max_block_ratio,
                },
            ),
            instance_id="mask",
            position=(260, 180),
        )
    )
    graph.add_edge(Edge(id="e_tokens_mask", src_node="tokens_in", src_port=0, dst_node="mask", dst_port=0))

    graph.add_node(
        NeuronInstance(
            subgraph_neuron(encoder_graph, name="online_encoder", input_aliases=["tokens", "mask"], output_aliases=["latent"]),
            instance_id="online_encoder",
            position=(520, 120),
        )
    )
    graph.add_edge(Edge(id="e_masked_online", src_node="mask", src_port=0, dst_node="online_encoder", dst_port=0))
    graph.add_edge(Edge(id="e_online_mask", src_node="mask", src_port=1, dst_node="online_encoder", dst_port=1))

    graph.add_node(
        NeuronInstance(
            subgraph_neuron(encoder_graph, name="target_encoder", input_aliases=["tokens", "mask"], output_aliases=["latent"]),
            instance_id="target_encoder",
            position=(520, 280),
        )
    )
    graph.add_edge(Edge(id="e_tokens_target", src_node="tokens_in", src_port=0, dst_node="target_encoder", dst_port=0))
    graph.add_edge(Edge(id="e_target_mask", src_node="mask", src_port=1, dst_node="target_encoder", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.jepa_predictor_module, config={"latent_dim": model_spec.jepa_latent_dim}),
            instance_id="predictor",
            position=(780, 120),
        )
    )
    graph.add_edge(Edge(id="e_online_predictor", src_node="online_encoder", src_port=0, dst_node="predictor", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.latent_mse_loss_module), instance_id="loss", position=(1020, 180)))
    graph.add_edge(Edge(id="e_pred_loss", src_node="predictor", src_port=0, dst_node="loss", dst_port=0))
    graph.add_edge(Edge(id="e_target_loss", src_node="target_encoder", src_port=0, dst_node="loss", dst_port=1))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(1240, 180)))
    graph.add_edge(Edge(id="e_loss_out", src_node="loss", src_port=0, dst_node="loss_out", dst_port=0))
    graph.input_node_ids = ["tokens_in"]
    graph.output_node_ids = ["loss_out"]
    return graph


def build_ar_jepa_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    spec = model_spec.block_spec
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")

    target_encoder_graph = build_jepa_encoder_graph("jepa_encoder", model_spec)
    graph.variant_library = deepcopy(target_encoder_graph.variant_library)
    target_encoder_graph.variant_library = {}

    block_family = "mixllama" if (spec.mlp_type == "mixllama" or spec.mlp_type == "moe") else "transformer_block"
    attn_graph = build_dense_attention_graph("attention_engine", model_spec.model_dim, spec, **_attn_flags(model_spec))
    mamba_graph = build_mamba_graph("mamba_engine", model_spec.model_dim, spec)
    if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
        mlp_graph = build_mixllama_mlp_graph("mlp_moe", model_spec.model_dim, spec)
    else:
        mlp_graph = build_dense_mlp_graph("mlp_dense", model_spec.model_dim, spec)
    attn_block_graph = build_decoder_block_graph("decoder_block_attn", model_spec.model_dim, spec, attn_graph, mlp_graph)
    mamba_block_graph = build_decoder_block_graph("decoder_block_mamba", model_spec.model_dim, spec, mamba_graph, mlp_graph)
    graph.variant_library.update(
        {
            "attention": {"default": attn_graph},
            "mamba": {"default": mamba_graph},
            "mlp": {"default": mlp_graph},
            "attn_block": {"default": attn_block_graph},
            "mamba_block": {"default": mamba_block_graph},
            block_family: {"default": attn_block_graph},
        }
    )

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 140)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 300)))
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.jepa_mask_module,
                config={
                    "mask_ratio": model_spec.jepa_mask_ratio,
                    "mask_token_id": 0,
                    "mask_strategy": model_spec.jepa_mask_strategy,
                    "num_blocks": model_spec.jepa_num_blocks,
                    "min_block_ratio": model_spec.jepa_min_block_ratio,
                    "max_block_ratio": model_spec.jepa_max_block_ratio,
                },
            ),
            instance_id="mask",
            position=(260, 180),
        )
    )
    graph.add_edge(Edge(id="e_tokens_mask", src_node="tokens_in", src_port=0, dst_node="mask", dst_port=0))

    graph.add_node(
        NeuronInstance(
            subgraph_neuron(target_encoder_graph, name="target_encoder", input_aliases=["tokens", "mask"], output_aliases=["latent"]),
            instance_id="target_encoder",
            position=(520, 440),
        )
    )
    graph.add_edge(Edge(id="e_tokens_target_encoder", src_node="tokens_in", src_port=0, dst_node="target_encoder", dst_port=0))
    graph.add_edge(Edge(id="e_mask_target_encoder", src_node="mask", src_port=1, dst_node="target_encoder", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.token_embedding_module,
                config={"vocab_size": model_spec.vocab_size, "model_dim": model_spec.model_dim},
            ),
            instance_id="token_embed",
            position=(520, 120),
        )
    )
    graph.add_edge(Edge(id="e_masked_tokens_embed", src_node="mask", src_port=0, dst_node="token_embed", dst_port=0))
    curr_out = "token_embed"

    if spec.pos_encoding == "absolute":
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.absolute_position_embedding_module, config={"model_dim": model_spec.model_dim}),
                instance_id="pos_embed",
                position=(520, 240),
            )
        )
        graph.add_edge(Edge(id="e_masked_tokens_pos", src_node="mask", src_port=0, dst_node="pos_embed", dst_port=0))
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="embed_add", position=(740, 120)))
        graph.add_edge(Edge(id="e_embed_add_a", src_node="token_embed", src_port=0, dst_node="embed_add", dst_port=0))
        graph.add_edge(Edge(id="e_pos_add_b", src_node="pos_embed", src_port=0, dst_node="embed_add", dst_port=1))
        curr_out = "embed_add"

    if spec.dropout_p > 0.0:
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.dropout_module, config={"p": spec.dropout_p}),
                instance_id="embed_dropout",
                position=(960, 120),
            )
        )
        graph.add_edge(Edge(id="e_embed_dropout", src_node=curr_out, src_port=0, dst_node="embed_dropout", dst_port=0))
        curr_out = "embed_dropout"

    aux_losses: list[str] = []
    for i in range(model_spec.num_layers):
        bname = f"block_{i}"
        is_attn = True
        if model_spec.template.backbone == "jamba":
            is_attn = (i % 4 == 0)
            current_block_family = "attn_block" if is_attn else "mamba_block"
            current_block_graph = attn_block_graph if is_attn else mamba_block_graph
        else:
            current_block_family = block_family
            current_block_graph = attn_block_graph
        out_aliases = ["x"]
        if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
            out_aliases.append("aux_loss")
        graph.add_node(
            NeuronInstance(
                link_variant_neuron(
                    current_block_graph,
                    family=current_block_family,
                    version="default",
                    name=bname,
                    input_aliases=["x"],
                    output_aliases=out_aliases,
                ),
                instance_id=bname,
                position=(1180 + i * 220, 120),
            )
        )
        graph.add_edge(Edge(id=f"e_{curr_out}_{bname}", src_node=curr_out, src_port=0, dst_node=bname, dst_port=0))
        curr_out = bname
        if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
            aux_losses.append(bname)

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}),
            instance_id="final_norm",
            position=(1400 + model_spec.num_layers * 220, 120),
        )
    )
    graph.add_edge(Edge(id="e_blocks_norm", src_node=curr_out, src_port=0, dst_node="final_norm", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.latent_pool_module), instance_id="online_pool", position=(1400 + model_spec.num_layers * 220, 320)))
    graph.add_edge(Edge(id="e_norm_online_pool", src_node="final_norm", src_port=0, dst_node="online_pool", dst_port=0))
    graph.add_edge(Edge(id="e_mask_online_pool", src_node="mask", src_port=1, dst_node="online_pool", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.jepa_projector_module,
                config={"input_dim": model_spec.model_dim, "latent_dim": model_spec.jepa_latent_dim},
            ),
            instance_id="online_projector",
            position=(1620 + model_spec.num_layers * 220, 320),
        )
    )
    graph.add_edge(Edge(id="e_pool_projector", src_node="online_pool", src_port=0, dst_node="online_projector", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.jepa_predictor_module, config={"latent_dim": model_spec.jepa_latent_dim}),
            instance_id="predictor",
            position=(1840 + model_spec.num_layers * 220, 320),
        )
    )
    graph.add_edge(Edge(id="e_online_projector_predictor", src_node="online_projector", src_port=0, dst_node="predictor", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.latent_mse_loss_module), instance_id="jepa_loss", position=(2060 + model_spec.num_layers * 220, 320)))
    graph.add_edge(Edge(id="e_predictor_jepa_loss", src_node="predictor", src_port=0, dst_node="jepa_loss", dst_port=0))
    graph.add_edge(Edge(id="e_target_jepa_loss", src_node="target_encoder", src_port=0, dst_node="jepa_loss", dst_port=1))

    head_id = "tied_lm_head" if model_spec.tie_embeddings else "lm_head"
    head_def = clone_neuron_def(BuiltinNeurons.tied_lm_head_module) if model_spec.tie_embeddings else clone_neuron_def(
        BuiltinNeurons.lm_head_module,
        config={"model_dim": model_spec.model_dim, "vocab_size": model_spec.vocab_size},
    )
    graph.add_node(NeuronInstance(head_def, instance_id=head_id, position=(1620 + model_spec.num_layers * 220, 120)))
    graph.add_edge(Edge(id="e_norm_head", src_node="final_norm", src_port=0, dst_node=head_id, dst_port=0))
    if model_spec.tie_embeddings:
        graph.add_edge(Edge(id="e_tie", src_node="token_embed", src_port=1, dst_node=head_id, dst_port=1))

    ce_input = head_id
    if model_spec.logit_softcap > 0.0:
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.logit_softcap_module, config={"softcap": model_spec.logit_softcap}),
                instance_id="softcap",
                position=(1840 + model_spec.num_layers * 220, 120),
            )
        )
        graph.add_edge(Edge(id="e_head_softcap", src_node=head_id, src_port=0, dst_node="softcap", dst_port=0))
        ce_input = "softcap"

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_cross_entropy_module), instance_id="ar_loss", position=(2060 + model_spec.num_layers * 220, 140)))
    graph.add_edge(Edge(id="e_logits_ar_loss", src_node=ce_input, src_port=0, dst_node="ar_loss", dst_port=0))
    graph.add_edge(Edge(id="e_targets_ar_loss", src_node="targets_in", src_port=0, dst_node="ar_loss", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.loss_scale_module, config={"coef": model_spec.ar_loss_coef}),
            instance_id="ar_scaled",
            position=(2280 + model_spec.num_layers * 220, 140),
        )
    )
    graph.add_edge(Edge(id="e_ar_loss_scaled", src_node="ar_loss", src_port=0, dst_node="ar_scaled", dst_port=0))

    ar_main_loss = "ar_scaled"
    if (spec.mlp_type == "mixllama" or spec.mlp_type == "moe") and aux_losses and spec.router_aux_loss_coef > 0.0:
        prev_loss = aux_losses[0]
        for i, b_id in enumerate(aux_losses):
            if i == 0:
                prev_loss = b_id
                continue
            add_id = f"add_aux_{i}"
            graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id=add_id, position=(2280 + model_spec.num_layers * 220, 480 + i * 80)))
            graph.add_edge(Edge(id=f"e_{b_id}_add_aux", src_node=b_id, src_port=1, dst_node=add_id, dst_port=0))
            graph.add_edge(Edge(id=f"e_{prev_loss}_add_aux", src_node=prev_loss, src_port=1 if i == 1 else 0, dst_node=add_id, dst_port=1))
            prev_loss = add_id
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.aux_loss_add_module, config={"coef": spec.router_aux_loss_coef}),
                instance_id="ar_plus_aux",
                position=(2500 + model_spec.num_layers * 220, 140),
            )
        )
        graph.add_edge(Edge(id="e_ar_plus_aux_main", src_node="ar_scaled", src_port=0, dst_node="ar_plus_aux", dst_port=0))
        graph.add_edge(Edge(id="e_ar_plus_aux_aux", src_node=prev_loss, src_port=1 if len(aux_losses) == 1 else 0, dst_node="ar_plus_aux", dst_port=1))
        ar_main_loss = "ar_plus_aux"

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.loss_scale_module, config={"coef": model_spec.jepa_loss_coef}),
            instance_id="jepa_scaled",
            position=(2280 + model_spec.num_layers * 220, 320),
        )
    )
    graph.add_edge(Edge(id="e_jepa_loss_scaled", src_node="jepa_loss", src_port=0, dst_node="jepa_scaled", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.aux_loss_add_module, config={"coef": 1.0}),
            instance_id="total_loss",
            position=(2720 + model_spec.num_layers * 220, 220),
        )
    )
    graph.add_edge(Edge(id="e_total_loss_main", src_node=ar_main_loss, src_port=0, dst_node="total_loss", dst_port=0))
    graph.add_edge(Edge(id="e_total_loss_aux", src_node="jepa_scaled", src_port=0, dst_node="total_loss", dst_port=1))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(2940 + model_spec.num_layers * 220, 220)))
    graph.add_edge(Edge(id="e_total_loss_out", src_node="total_loss", src_port=0, dst_node="loss_out", dst_port=0))

    graph.input_node_ids = ["tokens_in", "targets_in"]
    graph.output_node_ids = ["loss_out"]
    return graph


def build_jepa_semantic_encoder_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    """Experimental: JEPA encoder with pooled semantic vector + full hidden sequence."""
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    hidden_graph = build_hidden_backbone_graph("hidden_backbone", model_spec)
    graph.variant_library = deepcopy(hidden_graph.variant_library)
    hidden_graph.variant_library = {}

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 120)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="mask", dtype="tensor"), instance_id="mask_in", position=(40, 260)))
    graph.add_node(NeuronInstance(subgraph_neuron(hidden_graph, name="backbone", input_aliases=["tokens"], output_aliases=["hidden"]), instance_id="backbone", position=(260, 120)))
    graph.add_edge(Edge(id="e_tokens_backbone", src_node="tokens_in", src_port=0, dst_node="backbone", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.latent_pool_module), instance_id="pool", position=(520, 140)))
    graph.add_edge(Edge(id="e_backbone_pool", src_node="backbone", src_port=0, dst_node="pool", dst_port=0))
    graph.add_edge(Edge(id="e_mask_pool", src_node="mask_in", src_port=0, dst_node="pool", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_projector_module,
                config={
                    "input_dim": model_spec.model_dim,
                    "semantic_dim": model_spec.semantic_dim,
                    "residual_dim": model_spec.semantic_residual_dim,
                    "semantic_vocab_ref": model_spec.semantic_vocab_ref,
                },
            ),
            instance_id="sem_proj",
            position=(760, 140),
        )
    )
    graph.add_edge(Edge(id="e_pool_semproj", src_node="pool", src_port=0, dst_node="sem_proj", dst_port=0))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="semantic_vec", dtype="tensor"), instance_id="sem_out", position=(980, 80)))
    graph.add_edge(Edge(id="e_semproj_semout", src_node="sem_proj", src_port=0, dst_node="sem_out", dst_port=0))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="hidden", dtype="tensor"), instance_id="hidden_out", position=(980, 220)))
    graph.add_edge(Edge(id="e_backbone_hiddenout", src_node="backbone", src_port=0, dst_node="hidden_out", dst_port=0))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="topic_logits", dtype="tensor"), instance_id="topic_logits_out", position=(980, 340)))
    graph.add_edge(Edge(id="e_semproj_topic_logits", src_node="sem_proj", src_port=2, dst_node="topic_logits_out", dst_port=0))

    graph.input_node_ids = ["tokens_in", "mask_in"]
    graph.output_node_ids = ["sem_out", "hidden_out", "topic_logits_out"]
    return graph


def build_jepa_semantic_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    """Experimental: JEPA semantic hybrid with hash routing into attention experts."""
    spec = model_spec.block_spec
    use_router_vecs = bool(model_spec.experimental_semantic_router_vecs)
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    encoder_graph = build_jepa_semantic_encoder_graph("jepa_sem_encoder", model_spec)
    graph.variant_library = deepcopy(encoder_graph.variant_library)
    encoder_graph.variant_library = {}

    # -- Inputs: text tokens/targets plus semantic supervision --
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 140)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 260)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="sem_targets", dtype="tokens"), instance_id="sem_targets_in", position=(40, 380)))
    if use_router_vecs:
        graph.add_node(
            NeuronInstance(
                make_terminal_def(role="input", port_name="semantic_router_vecs", dtype="tensor"),
                instance_id="semantic_router_vecs_in",
                position=(40, 500),
            )
        )

    # -- JEPA mask --
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.jepa_mask_module,
                config={
                    "mask_ratio": model_spec.jepa_mask_ratio,
                    "mask_token_id": 0,
                    "mask_strategy": model_spec.jepa_mask_strategy,
                    "num_blocks": model_spec.jepa_num_blocks,
                    "min_block_ratio": model_spec.jepa_min_block_ratio,
                    "max_block_ratio": model_spec.jepa_max_block_ratio,
                },
            ),
            instance_id="mask",
            position=(260, 180),
        )
    )
    graph.add_edge(Edge(id="e_tokens_mask", src_node="tokens_in", src_port=0, dst_node="mask", dst_port=0))

    # -- Online encoder (masked tokens) --
    graph.add_node(
        NeuronInstance(
            subgraph_neuron(encoder_graph, name="online_encoder", input_aliases=["tokens", "mask"], output_aliases=["semantic_vec", "hidden", "topic_logits"]),
            instance_id="online_encoder",
            position=(520, 100),
        )
    )
    graph.add_edge(Edge(id="e_masked_online", src_node="mask", src_port=0, dst_node="online_encoder", dst_port=0))
    graph.add_edge(Edge(id="e_mask_online", src_node="mask", src_port=1, dst_node="online_encoder", dst_port=1))

    # -- Target encoder (full tokens, frozen via EMA) --
    graph.add_node(
        NeuronInstance(
            subgraph_neuron(encoder_graph, name="target_encoder", input_aliases=["tokens", "mask"], output_aliases=["semantic_vec", "hidden", "topic_logits"]),
            instance_id="target_encoder",
            position=(520, 320),
        )
    )
    graph.add_edge(Edge(id="e_tokens_target", src_node="tokens_in", src_port=0, dst_node="target_encoder", dst_port=0))
    graph.add_edge(Edge(id="e_mask_target", src_node="mask", src_port=1, dst_node="target_encoder", dst_port=1))

    # -- JEPA predictor on online semantic vec --
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.jepa_predictor_module, config={"latent_dim": model_spec.semantic_dim}),
            instance_id="predictor",
            position=(780, 60),
        )
    )
    graph.add_edge(Edge(id="e_online_sem_pred", src_node="online_encoder", src_port=0, dst_node="predictor", dst_port=0))

    # -- Latent MSE loss (JEPA self-supervised) --
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.latent_mse_loss_module), instance_id="jepa_loss", position=(1020, 100)))
    graph.add_edge(Edge(id="e_pred_jloss", src_node="predictor", src_port=0, dst_node="jepa_loss", dst_port=0))
    graph.add_edge(Edge(id="e_target_jloss", src_node="target_encoder", src_port=0, dst_node="jepa_loss", dst_port=1))

    # -- Hash-aware routing query --
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_hasher_module,
                config={"dim": model_spec.semantic_dim, "tables": model_spec.semantic_n_lsh_tables, "planes": model_spec.semantic_n_lsh_planes},
            ),
            instance_id="hasher",
            position=(780, 180),
        )
    )
    graph.add_edge(Edge(id="e_sem_hasher", src_node="online_encoder", src_port=0, dst_node="hasher", dst_port=0))

    n_experts = spec.experts or NUM_VOCAB_DIMS
    top_k = spec.top_k or 2
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_hash_router_module,
                config={
                    "n_experts": n_experts,
                    "semantic_dim": model_spec.semantic_dim,
                    "top_k": top_k,
                    "tables": model_spec.semantic_n_lsh_tables,
                    "n_buckets": 2 ** min(model_spec.semantic_n_lsh_planes, 12),
                    "semantic_vocab_ref": model_spec.semantic_vocab_ref,
                    "routing_source": "semantic_vec" if use_router_vecs else "topic_logits",
                },
            ),
            instance_id="hash_router",
            position=(1020, 220),
        )
    )
    if use_router_vecs:
        graph.add_edge(
            Edge(
                id="e_sem_router_vec_hash_router",
                src_node="semantic_router_vecs_in",
                src_port=0,
                dst_node="hash_router",
                dst_port=0,
            )
        )
    else:
        graph.add_edge(Edge(id="e_online_hash_router", src_node="online_encoder", src_port=0, dst_node="hash_router", dst_port=0))
    graph.add_edge(Edge(id="e_hasher_hash_router", src_node="hasher", src_port=0, dst_node="hash_router", dst_port=1))
    graph.add_edge(Edge(id="e_topic_logits_hash_router", src_node="online_encoder", src_port=2, dst_node="hash_router", dst_port=2))
    graph.add_edge(Edge(id="e_sem_targets_hash_router", src_node="sem_targets_in", src_port=0, dst_node="hash_router", dst_port=3))

    # -- Expert attention over the full masked hidden sequence --
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.routed_attention_experts_module,
                config={
                    "model_dim": model_spec.model_dim,
                    "num_heads": spec.num_heads,
                    "num_kv_heads": spec.num_kv_heads if spec.num_kv_heads is not None else spec.num_heads,
                    "rope_base": spec.rope_theta,
                    "qk_gain_init": spec.qk_gain_init,
                    "experts": n_experts,
                    "top_k": top_k,
                },
            ),
            instance_id="routed_experts",
            position=(1260, 220),
        )
    )
    graph.add_edge(Edge(id="e_hidden_routed_experts", src_node="online_encoder", src_port=1, dst_node="routed_experts", dst_port=0))
    graph.add_edge(Edge(id="e_weights_routed_experts", src_node="hash_router", src_port=0, dst_node="routed_experts", dst_port=1))
    graph.add_edge(Edge(id="e_indices_routed_experts", src_node="hash_router", src_port=1, dst_node="routed_experts", dst_port=2))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.residual_add_module, config={"dim": model_spec.model_dim}),
            instance_id="expert_residual",
            position=(1480, 220),
        )
    )
    graph.add_edge(Edge(id="e_hidden_expert_residual", src_node="online_encoder", src_port=1, dst_node="expert_residual", dst_port=0))
    graph.add_edge(Edge(id="e_routed_expert_residual", src_node="routed_experts", src_port=0, dst_node="expert_residual", dst_port=1))

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}),
            instance_id="final_norm",
            position=(1700, 220),
        )
    )
    graph.add_edge(Edge(id="e_expert_residual_norm", src_node="expert_residual", src_port=0, dst_node="final_norm", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.lm_head_module, config={"model_dim": model_spec.model_dim, "vocab_size": model_spec.vocab_size}),
            instance_id="lm_head",
            position=(1920, 220),
        )
    )
    graph.add_edge(Edge(id="e_norm_lm_head", src_node="final_norm", src_port=0, dst_node="lm_head", dst_port=0))

    ce_input = "lm_head"
    if model_spec.logit_softcap > 0.0:
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.logit_softcap_module, config={"softcap": model_spec.logit_softcap}),
                instance_id="softcap",
                position=(2140, 220),
            )
        )
        graph.add_edge(Edge(id="e_lm_head_softcap", src_node="lm_head", src_port=0, dst_node="softcap", dst_port=0))
        ce_input = "softcap"

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_cross_entropy_module), instance_id="ar_loss", position=(2360, 220)))
    graph.add_edge(Edge(id="e_logits_ar_loss", src_node=ce_input, src_port=0, dst_node="ar_loss", dst_port=0))
    graph.add_edge(Edge(id="e_targets_ar_loss", src_node="targets_in", src_port=0, dst_node="ar_loss", dst_port=1))

    # -- Semantic alignment loss (supervised: online vocab-topic logits vs sem_targets) --
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_alignment_loss_module,
                config={"semantic_vocab_ref": model_spec.semantic_vocab_ref},
            ),
            instance_id="sem_align_loss",
            position=(1260, 360),
        )
    )
    graph.add_edge(Edge(id="e_online_sem_align", src_node="online_encoder", src_port=2, dst_node="sem_align_loss", dst_port=0))
    graph.add_edge(Edge(id="e_targets_sem_align", src_node="sem_targets_in", src_port=0, dst_node="sem_align_loss", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.loss_scale_module, config={"coef": model_spec.ar_loss_coef}),
            instance_id="ar_scaled",
            position=(2580, 220),
        )
    )
    graph.add_edge(Edge(id="e_ar_loss_scaled", src_node="ar_loss", src_port=0, dst_node="ar_scaled", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.loss_scale_module, config={"coef": model_spec.jepa_loss_coef}),
            instance_id="jepa_scaled",
            position=(1260, 100),
        )
    )
    graph.add_edge(Edge(id="e_jepa_loss_scaled", src_node="jepa_loss", src_port=0, dst_node="jepa_scaled", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.loss_scale_module, config={"coef": model_spec.semantic_align_loss_coef}),
            instance_id="sem_align_scaled",
            position=(1480, 360),
        )
    )
    graph.add_edge(Edge(id="e_sem_align_scaled", src_node="sem_align_loss", src_port=0, dst_node="sem_align_scaled", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.aux_loss_add_module, config={"coef": 1.0}),
            instance_id="ar_plus_jepa",
            position=(2800, 180),
        )
    )
    graph.add_edge(Edge(id="e_ar_plus_jepa_main", src_node="ar_scaled", src_port=0, dst_node="ar_plus_jepa", dst_port=0))
    graph.add_edge(Edge(id="e_ar_plus_jepa_aux", src_node="jepa_scaled", src_port=0, dst_node="ar_plus_jepa", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.aux_loss_add_module, config={"coef": 1.0}),
            instance_id="total_loss",
            position=(3020, 180),
        )
    )
    graph.add_edge(Edge(id="e_total_loss_main", src_node="ar_plus_jepa", src_port=0, dst_node="total_loss", dst_port=0))
    graph.add_edge(Edge(id="e_total_loss_aux", src_node="sem_align_scaled", src_port=0, dst_node="total_loss", dst_port=1))

    # -- Output --
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(3240, 180)))
    graph.add_edge(Edge(id="e_total_out", src_node="total_loss", src_port=0, dst_node="loss_out", dst_port=0))

    graph.input_node_ids = ["tokens_in", "targets_in", "sem_targets_in"]
    if use_router_vecs:
        graph.input_node_ids.append("semantic_router_vecs_in")
    graph.output_node_ids = ["loss_out"]
    return graph


def build_semantic_router_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    """Experimental: AR MixLLaMA with one shared semantic router reused by every MoE block."""

    spec = model_spec.block_spec
    use_router_vecs = bool(model_spec.experimental_semantic_router_vecs)
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")

    attn_graph = build_dense_attention_graph("attention_engine", model_spec.model_dim, spec, **_attn_flags(model_spec))
    mlp_graph = build_semantic_router_mlp_graph("semantic_router_mlp", model_spec.model_dim, spec)
    block_graph = build_semantic_router_decoder_block_graph(
        "semantic_router_block",
        model_spec.model_dim,
        spec,
        attn_graph,
        mlp_graph,
    )
    graph.variant_library = {
        "attention": {"default": attn_graph},
        "semantic_router_mlp": {"default": mlp_graph},
        "semantic_router_block": {"default": block_graph},
    }

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 140)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 340)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="sem_targets", dtype="tokens"), instance_id="sem_targets_in", position=(40, 540)))
    if use_router_vecs:
        graph.add_node(
            NeuronInstance(
                make_terminal_def(role="input", port_name="semantic_router_vecs", dtype="tensor"),
                instance_id="semantic_router_vecs_in",
                position=(40, 740),
            )
        )

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.token_embedding_module,
                config={"vocab_size": model_spec.vocab_size, "model_dim": model_spec.model_dim},
            ),
            instance_id="token_embed",
            position=(260, 140),
        )
    )
    graph.add_edge(Edge(id="e_tokens_embed", src_node="tokens_in", src_port=0, dst_node="token_embed", dst_port=0))
    curr_out = "token_embed"

    if spec.pos_encoding == "absolute":
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.absolute_position_embedding_module, config={"model_dim": model_spec.model_dim}),
                instance_id="pos_embed",
                position=(260, 260),
            )
        )
        graph.add_edge(Edge(id="e_tokens_pos", src_node="tokens_in", src_port=0, dst_node="pos_embed", dst_port=0))
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="embed_add", position=(480, 140)))
        graph.add_edge(Edge(id="e_embed_add_a", src_node="token_embed", src_port=0, dst_node="embed_add", dst_port=0))
        graph.add_edge(Edge(id="e_pos_add_b", src_node="pos_embed", src_port=0, dst_node="embed_add", dst_port=1))
        curr_out = "embed_add"

    if spec.dropout_p > 0.0:
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.dropout_module, config={"p": spec.dropout_p}),
                instance_id="embed_dropout",
                position=(700, 140),
            )
        )
        graph.add_edge(Edge(id="e_embed_dropout", src_node=curr_out, src_port=0, dst_node="embed_dropout", dst_port=0))
        curr_out = "embed_dropout"

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_projector_module,
                config={
                    "input_dim": model_spec.model_dim,
                    "semantic_dim": model_spec.semantic_dim,
                    "residual_dim": model_spec.semantic_residual_dim,
                    "n_sig_buckets": 2 ** min(model_spec.semantic_n_lsh_planes, 12),
                    "semantic_vocab_ref": model_spec.semantic_vocab_ref,
                },
            ),
            instance_id="semantic_projector",
            position=(700, 380),
        )
    )
    graph.add_edge(Edge(id="e_hidden_semantic_projector", src_node=curr_out, src_port=0, dst_node="semantic_projector", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_hasher_module,
                config={
                    "dim": model_spec.semantic_dim,
                    "tables": model_spec.semantic_n_lsh_tables,
                    "planes": model_spec.semantic_n_lsh_planes,
                },
            ),
            instance_id="semantic_hasher",
            position=(940, 320),
        )
    )
    graph.add_edge(Edge(id="e_semantic_projector_hasher", src_node="semantic_projector", src_port=0, dst_node="semantic_hasher", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_hash_router_module,
                config={
                    "n_experts": spec.experts or NUM_VOCAB_DIMS,
                    "semantic_dim": model_spec.semantic_dim,
                    "top_k": spec.top_k or 2,
                    "tables": model_spec.semantic_n_lsh_tables,
                    "n_buckets": 2 ** min(model_spec.semantic_n_lsh_planes, 12),
                    "semantic_vocab_ref": model_spec.semantic_vocab_ref,
                    "routing_source": "semantic_vec" if use_router_vecs else "topic_logits",
                },
            ),
            instance_id="semantic_hash_router",
            position=(1160, 380),
        )
    )
    if use_router_vecs:
        graph.add_edge(
            Edge(
                id="e_semantic_router_vec_router",
                src_node="semantic_router_vecs_in",
                src_port=0,
                dst_node="semantic_hash_router",
                dst_port=0,
            )
        )
    else:
        graph.add_edge(Edge(id="e_semantic_vec_router", src_node="semantic_projector", src_port=0, dst_node="semantic_hash_router", dst_port=0))
    graph.add_edge(Edge(id="e_bucket_indices_router", src_node="semantic_hasher", src_port=0, dst_node="semantic_hash_router", dst_port=1))
    graph.add_edge(Edge(id="e_topic_logits_router", src_node="semantic_projector", src_port=2, dst_node="semantic_hash_router", dst_port=2))
    graph.add_edge(Edge(id="e_sem_targets_router", src_node="sem_targets_in", src_port=0, dst_node="semantic_hash_router", dst_port=3))

    graph.add_node(
        NeuronInstance(clone_neuron_def(BuiltinNeurons.broadcast_expert_routes_module), instance_id="broadcast_routes", position=(1380, 380))
    )
    graph.add_edge(Edge(id="e_hidden_broadcast_routes", src_node=curr_out, src_port=0, dst_node="broadcast_routes", dst_port=0))
    graph.add_edge(Edge(id="e_weights_broadcast_routes", src_node="semantic_hash_router", src_port=0, dst_node="broadcast_routes", dst_port=1))
    graph.add_edge(Edge(id="e_indices_broadcast_routes", src_node="semantic_hash_router", src_port=1, dst_node="broadcast_routes", dst_port=2))

    block_input = curr_out
    for i in range(model_spec.num_layers):
        bname = f"block_{i}"
        graph.add_node(
            NeuronInstance(
                link_variant_neuron(
                    block_graph,
                    family="semantic_router_block",
                    version="default",
                    name=bname,
                    input_aliases=["x", "routing_weights", "routing_indices"],
                    output_aliases=["x"],
                ),
                instance_id=bname,
                position=(940 + i * 220, 140),
            )
        )
        graph.add_edge(Edge(id=f"e_{block_input}_{bname}", src_node=block_input, src_port=0, dst_node=bname, dst_port=0))
        graph.add_edge(Edge(id=f"e_broadcast_weights_{bname}", src_node="broadcast_routes", src_port=0, dst_node=bname, dst_port=1))
        graph.add_edge(Edge(id=f"e_broadcast_indices_{bname}", src_node="broadcast_routes", src_port=1, dst_node=bname, dst_port=2))
        block_input = bname

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}),
            instance_id="final_norm",
            position=(1160 + model_spec.num_layers * 220, 140),
        )
    )
    graph.add_edge(Edge(id="e_blocks_norm", src_node=block_input, src_port=0, dst_node="final_norm", dst_port=0))

    head_id = "tied_lm_head" if model_spec.tie_embeddings else "lm_head"
    head_def = clone_neuron_def(BuiltinNeurons.tied_lm_head_module) if model_spec.tie_embeddings else clone_neuron_def(
        BuiltinNeurons.lm_head_module,
        config={"model_dim": model_spec.model_dim, "vocab_size": model_spec.vocab_size},
    )
    graph.add_node(NeuronInstance(head_def, instance_id=head_id, position=(1380 + model_spec.num_layers * 220, 140)))
    graph.add_edge(Edge(id="e_norm_head", src_node="final_norm", src_port=0, dst_node=head_id, dst_port=0))
    if model_spec.tie_embeddings:
        graph.add_edge(Edge(id="e_tie", src_node="token_embed", src_port=1, dst_node=head_id, dst_port=1))

    ce_input = head_id
    if model_spec.logit_softcap > 0.0:
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.logit_softcap_module, config={"softcap": model_spec.logit_softcap}),
                instance_id="softcap",
                position=(1600 + model_spec.num_layers * 220, 140),
            )
        )
        graph.add_edge(Edge(id="e_head_softcap", src_node=head_id, src_port=0, dst_node="softcap", dst_port=0))
        ce_input = "softcap"

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_cross_entropy_module), instance_id="ar_loss", position=(1820 + model_spec.num_layers * 220, 220)))
    graph.add_edge(Edge(id="e_logits_ar_loss", src_node=ce_input, src_port=0, dst_node="ar_loss", dst_port=0))
    graph.add_edge(Edge(id="e_targets_ar_loss", src_node="targets_in", src_port=0, dst_node="ar_loss", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_alignment_loss_module,
                config={"semantic_vocab_ref": model_spec.semantic_vocab_ref},
            ),
            instance_id="sem_align_loss",
            position=(1380, 520),
        )
    )
    graph.add_edge(Edge(id="e_topic_logits_sem_align", src_node="semantic_projector", src_port=2, dst_node="sem_align_loss", dst_port=0))
    graph.add_edge(Edge(id="e_sem_targets_sem_align", src_node="sem_targets_in", src_port=0, dst_node="sem_align_loss", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.loss_scale_module, config={"coef": model_spec.ar_loss_coef}),
            instance_id="ar_scaled",
            position=(2040 + model_spec.num_layers * 220, 220),
        )
    )
    graph.add_edge(Edge(id="e_ar_loss_scaled", src_node="ar_loss", src_port=0, dst_node="ar_scaled", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.loss_scale_module, config={"coef": model_spec.semantic_align_loss_coef}),
            instance_id="sem_align_scaled",
            position=(1600, 520),
        )
    )
    graph.add_edge(Edge(id="e_sem_align_scaled", src_node="sem_align_loss", src_port=0, dst_node="sem_align_scaled", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.aux_loss_add_module, config={"coef": 1.0}),
            instance_id="total_loss",
            position=(2260 + model_spec.num_layers * 220, 220),
        )
    )
    graph.add_edge(Edge(id="e_total_loss_main", src_node="ar_scaled", src_port=0, dst_node="total_loss", dst_port=0))
    graph.add_edge(Edge(id="e_total_loss_aux", src_node="sem_align_scaled", src_port=0, dst_node="total_loss", dst_port=1))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(2480 + model_spec.num_layers * 220, 220)))
    graph.add_edge(Edge(id="e_total_loss_out", src_node="total_loss", src_port=0, dst_node="loss_out", dst_port=0))

    graph.input_node_ids = ["tokens_in", "targets_in", "sem_targets_in"]
    if use_router_vecs:
        graph.input_node_ids.append("semantic_router_vecs_in")
    graph.output_node_ids = ["loss_out"]
    return graph


def build_semantic_router_jepa_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    spec = model_spec.block_spec
    use_router_vecs = bool(model_spec.experimental_semantic_router_vecs)
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")

    target_encoder_graph = build_jepa_semantic_encoder_graph("jepa_sem_encoder", model_spec)
    graph.variant_library = deepcopy(target_encoder_graph.variant_library)
    target_encoder_graph.variant_library = {}

    attn_graph = build_dense_attention_graph("attention_engine", model_spec.model_dim, spec, **_attn_flags(model_spec))
    mlp_graph = build_semantic_router_mlp_graph("semantic_router_mlp", model_spec.model_dim, spec)
    block_graph = build_semantic_router_decoder_block_graph(
        "semantic_router_block",
        model_spec.model_dim,
        spec,
        attn_graph,
        mlp_graph,
    )
    graph.variant_library.update(
        {
            "attention": {"default": attn_graph},
            "semantic_router_mlp": {"default": mlp_graph},
            "semantic_router_block": {"default": block_graph},
        }
    )

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 140)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 300)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="sem_targets", dtype="tokens"), instance_id="sem_targets_in", position=(40, 460)))
    if use_router_vecs:
        graph.add_node(
            NeuronInstance(
                make_terminal_def(role="input", port_name="semantic_router_vecs", dtype="tensor"),
                instance_id="semantic_router_vecs_in",
                position=(40, 620),
            )
        )

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.jepa_mask_module,
                config={
                    "mask_ratio": model_spec.jepa_mask_ratio,
                    "mask_token_id": 0,
                    "mask_strategy": model_spec.jepa_mask_strategy,
                    "num_blocks": model_spec.jepa_num_blocks,
                    "min_block_ratio": model_spec.jepa_min_block_ratio,
                    "max_block_ratio": model_spec.jepa_max_block_ratio,
                },
            ),
            instance_id="mask",
            position=(260, 220),
        )
    )
    graph.add_edge(Edge(id="e_tokens_mask", src_node="tokens_in", src_port=0, dst_node="mask", dst_port=0))

    graph.add_node(
        NeuronInstance(
            subgraph_neuron(
                target_encoder_graph,
                name="target_encoder",
                input_aliases=["tokens", "mask"],
                output_aliases=["semantic_vec", "hidden", "topic_logits"],
            ),
            instance_id="target_encoder",
            position=(520, 520),
        )
    )
    graph.add_edge(Edge(id="e_tokens_target_encoder", src_node="tokens_in", src_port=0, dst_node="target_encoder", dst_port=0))
    graph.add_edge(Edge(id="e_mask_target_encoder", src_node="mask", src_port=1, dst_node="target_encoder", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.token_embedding_module,
                config={"vocab_size": model_spec.vocab_size, "model_dim": model_spec.model_dim},
            ),
            instance_id="token_embed",
            position=(520, 120),
        )
    )
    graph.add_edge(Edge(id="e_masked_tokens_embed", src_node="mask", src_port=0, dst_node="token_embed", dst_port=0))
    curr_out = "token_embed"

    if spec.pos_encoding == "absolute":
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.absolute_position_embedding_module, config={"model_dim": model_spec.model_dim}),
                instance_id="pos_embed",
                position=(520, 240),
            )
        )
        graph.add_edge(Edge(id="e_masked_tokens_pos", src_node="mask", src_port=0, dst_node="pos_embed", dst_port=0))
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="embed_add", position=(740, 120)))
        graph.add_edge(Edge(id="e_embed_add_a", src_node="token_embed", src_port=0, dst_node="embed_add", dst_port=0))
        graph.add_edge(Edge(id="e_pos_add_b", src_node="pos_embed", src_port=0, dst_node="embed_add", dst_port=1))
        curr_out = "embed_add"

    if spec.dropout_p > 0.0:
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.dropout_module, config={"p": spec.dropout_p}),
                instance_id="embed_dropout",
                position=(960, 120),
            )
        )
        graph.add_edge(Edge(id="e_embed_dropout", src_node=curr_out, src_port=0, dst_node="embed_dropout", dst_port=0))
        curr_out = "embed_dropout"

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_projector_module,
                config={
                    "input_dim": model_spec.model_dim,
                    "semantic_dim": model_spec.semantic_dim,
                    "residual_dim": model_spec.semantic_residual_dim,
                    "n_sig_buckets": 2 ** min(model_spec.semantic_n_lsh_planes, 12),
                    "semantic_vocab_ref": model_spec.semantic_vocab_ref,
                },
            ),
            instance_id="semantic_projector",
            position=(960, 380),
        )
    )
    graph.add_edge(Edge(id="e_hidden_semantic_projector", src_node=curr_out, src_port=0, dst_node="semantic_projector", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_hasher_module,
                config={
                    "dim": model_spec.semantic_dim,
                    "tables": model_spec.semantic_n_lsh_tables,
                    "planes": model_spec.semantic_n_lsh_planes,
                },
            ),
            instance_id="semantic_hasher",
            position=(1180, 320),
        )
    )
    graph.add_edge(Edge(id="e_semantic_projector_hasher", src_node="semantic_projector", src_port=0, dst_node="semantic_hasher", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_hash_router_module,
                config={
                    "n_experts": spec.experts or NUM_VOCAB_DIMS,
                    "semantic_dim": model_spec.semantic_dim,
                    "top_k": spec.top_k or 2,
                    "tables": model_spec.semantic_n_lsh_tables,
                    "n_buckets": 2 ** min(model_spec.semantic_n_lsh_planes, 12),
                    "semantic_vocab_ref": model_spec.semantic_vocab_ref,
                    "routing_source": "semantic_vec" if use_router_vecs else "topic_logits",
                },
            ),
            instance_id="semantic_hash_router",
            position=(1400, 380),
        )
    )
    if use_router_vecs:
        graph.add_edge(Edge(id="e_semantic_router_vec_router", src_node="semantic_router_vecs_in", src_port=0, dst_node="semantic_hash_router", dst_port=0))
    else:
        graph.add_edge(Edge(id="e_semantic_vec_router", src_node="semantic_projector", src_port=0, dst_node="semantic_hash_router", dst_port=0))
    graph.add_edge(Edge(id="e_bucket_indices_router", src_node="semantic_hasher", src_port=0, dst_node="semantic_hash_router", dst_port=1))
    graph.add_edge(Edge(id="e_topic_logits_router", src_node="semantic_projector", src_port=2, dst_node="semantic_hash_router", dst_port=2))
    graph.add_edge(Edge(id="e_sem_targets_router", src_node="sem_targets_in", src_port=0, dst_node="semantic_hash_router", dst_port=3))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.broadcast_expert_routes_module), instance_id="broadcast_routes", position=(1620, 380)))
    graph.add_edge(Edge(id="e_hidden_broadcast_routes", src_node=curr_out, src_port=0, dst_node="broadcast_routes", dst_port=0))
    graph.add_edge(Edge(id="e_weights_broadcast_routes", src_node="semantic_hash_router", src_port=0, dst_node="broadcast_routes", dst_port=1))
    graph.add_edge(Edge(id="e_indices_broadcast_routes", src_node="semantic_hash_router", src_port=1, dst_node="broadcast_routes", dst_port=2))

    block_input = curr_out
    for i in range(model_spec.num_layers):
        bname = f"block_{i}"
        graph.add_node(
            NeuronInstance(
                link_variant_neuron(
                    block_graph,
                    family="semantic_router_block",
                    version="default",
                    name=bname,
                    input_aliases=["x", "routing_weights", "routing_indices"],
                    output_aliases=["x"],
                ),
                instance_id=bname,
                position=(1180 + i * 220, 120),
            )
        )
        graph.add_edge(Edge(id=f"e_{block_input}_{bname}", src_node=block_input, src_port=0, dst_node=bname, dst_port=0))
        graph.add_edge(Edge(id=f"e_broadcast_weights_{bname}", src_node="broadcast_routes", src_port=0, dst_node=bname, dst_port=1))
        graph.add_edge(Edge(id=f"e_broadcast_indices_{bname}", src_node="broadcast_routes", src_port=1, dst_node=bname, dst_port=2))
        block_input = bname

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(
        NeuronInstance(
            clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}),
            instance_id="final_norm",
            position=(1400 + model_spec.num_layers * 220, 120),
        )
    )
    graph.add_edge(Edge(id="e_blocks_norm", src_node=block_input, src_port=0, dst_node="final_norm", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.jepa_predictor_module, config={"latent_dim": model_spec.semantic_dim}),
            instance_id="predictor",
            position=(1620 + model_spec.num_layers * 220, 380),
        )
    )
    graph.add_edge(Edge(id="e_online_semantic_predictor", src_node="semantic_projector", src_port=0, dst_node="predictor", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.latent_mse_loss_module), instance_id="jepa_loss", position=(1840 + model_spec.num_layers * 220, 380)))
    graph.add_edge(Edge(id="e_predictor_jepa_loss", src_node="predictor", src_port=0, dst_node="jepa_loss", dst_port=0))
    graph.add_edge(Edge(id="e_target_jepa_loss", src_node="target_encoder", src_port=0, dst_node="jepa_loss", dst_port=1))

    head_id = "tied_lm_head" if model_spec.tie_embeddings else "lm_head"
    head_def = clone_neuron_def(BuiltinNeurons.tied_lm_head_module) if model_spec.tie_embeddings else clone_neuron_def(
        BuiltinNeurons.lm_head_module,
        config={"model_dim": model_spec.model_dim, "vocab_size": model_spec.vocab_size},
    )
    graph.add_node(NeuronInstance(head_def, instance_id=head_id, position=(1620 + model_spec.num_layers * 220, 120)))
    graph.add_edge(Edge(id="e_norm_head", src_node="final_norm", src_port=0, dst_node=head_id, dst_port=0))
    if model_spec.tie_embeddings:
        graph.add_edge(Edge(id="e_tie", src_node="token_embed", src_port=1, dst_node=head_id, dst_port=1))

    ce_input = head_id
    if model_spec.logit_softcap > 0.0:
        graph.add_node(
            NeuronInstance(
                clone_neuron_def(BuiltinNeurons.logit_softcap_module, config={"softcap": model_spec.logit_softcap}),
                instance_id="softcap",
                position=(1840 + model_spec.num_layers * 220, 120),
            )
        )
        graph.add_edge(Edge(id="e_head_softcap", src_node=head_id, src_port=0, dst_node="softcap", dst_port=0))
        ce_input = "softcap"

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_cross_entropy_module), instance_id="ar_loss", position=(2060 + model_spec.num_layers * 220, 160)))
    graph.add_edge(Edge(id="e_logits_ar_loss", src_node=ce_input, src_port=0, dst_node="ar_loss", dst_port=0))
    graph.add_edge(Edge(id="e_targets_ar_loss", src_node="targets_in", src_port=0, dst_node="ar_loss", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_alignment_loss_module,
                config={"semantic_vocab_ref": model_spec.semantic_vocab_ref},
            ),
            instance_id="sem_align_loss",
            position=(1840 + model_spec.num_layers * 220, 520),
        )
    )
    graph.add_edge(Edge(id="e_topic_logits_sem_align", src_node="semantic_projector", src_port=2, dst_node="sem_align_loss", dst_port=0))
    graph.add_edge(Edge(id="e_sem_targets_sem_align", src_node="sem_targets_in", src_port=0, dst_node="sem_align_loss", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.loss_scale_module, config={"coef": model_spec.ar_loss_coef}),
            instance_id="ar_scaled",
            position=(2280 + model_spec.num_layers * 220, 160),
        )
    )
    graph.add_edge(Edge(id="e_ar_loss_scaled", src_node="ar_loss", src_port=0, dst_node="ar_scaled", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.loss_scale_module, config={"coef": model_spec.jepa_loss_coef}),
            instance_id="jepa_scaled",
            position=(2060 + model_spec.num_layers * 220, 380),
        )
    )
    graph.add_edge(Edge(id="e_jepa_loss_scaled", src_node="jepa_loss", src_port=0, dst_node="jepa_scaled", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.loss_scale_module, config={"coef": model_spec.semantic_align_loss_coef}),
            instance_id="sem_align_scaled",
            position=(2060 + model_spec.num_layers * 220, 520),
        )
    )
    graph.add_edge(Edge(id="e_sem_align_scaled", src_node="sem_align_loss", src_port=0, dst_node="sem_align_scaled", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.aux_loss_add_module, config={"coef": 1.0}),
            instance_id="ar_plus_jepa",
            position=(2500 + model_spec.num_layers * 220, 220),
        )
    )
    graph.add_edge(Edge(id="e_ar_plus_jepa_main", src_node="ar_scaled", src_port=0, dst_node="ar_plus_jepa", dst_port=0))
    graph.add_edge(Edge(id="e_ar_plus_jepa_aux", src_node="jepa_scaled", src_port=0, dst_node="ar_plus_jepa", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.aux_loss_add_module, config={"coef": 1.0}),
            instance_id="total_loss",
            position=(2720 + model_spec.num_layers * 220, 220),
        )
    )
    graph.add_edge(Edge(id="e_total_loss_main", src_node="ar_plus_jepa", src_port=0, dst_node="total_loss", dst_port=0))
    graph.add_edge(Edge(id="e_total_loss_aux", src_node="sem_align_scaled", src_port=0, dst_node="total_loss", dst_port=1))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(2940 + model_spec.num_layers * 220, 220)))
    graph.add_edge(Edge(id="e_total_loss_out", src_node="total_loss", src_port=0, dst_node="loss_out", dst_port=0))

    graph.input_node_ids = ["tokens_in", "targets_in", "sem_targets_in"]
    if use_router_vecs:
        graph.input_node_ids.append("semantic_router_vecs_in")
    graph.output_node_ids = ["loss_out"]
    return graph


def build_semantic_dense_jepa_evo_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    """Experimental: dense decoder control for chunk semantic JEPA planning."""

    spec = model_spec.block_spec
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")

    encoder_graph = build_jepa_semantic_encoder_graph("semantic_dense_jepa_evo_encoder", model_spec)
    graph.variant_library = deepcopy(encoder_graph.variant_library)
    encoder_graph.variant_library = {}

    attn_graph = build_dense_attention_graph("semantic_dense_attention", model_spec.model_dim, spec, **_attn_flags(model_spec))
    mlp_graph = build_dense_mlp_graph("semantic_dense_mlp", model_spec.model_dim, spec)
    block_graph = build_decoder_block_graph(
        "semantic_dense_jepa_evo_block",
        model_spec.model_dim,
        spec,
        attn_graph,
        mlp_graph,
        attention_family="semantic_dense_attention",
        mlp_family="semantic_dense_mlp",
    )
    graph.variant_library.update(
        {
            "semantic_dense_attention": {"default": attn_graph},
            "semantic_dense_mlp": {"default": mlp_graph},
            "semantic_dense_jepa_evo_block": {"default": block_graph},
        }
    )

    chunk_size = int(model_spec.route_chunk_size)
    common_semantic_cfg = {
        "input_dim": model_spec.model_dim,
        "semantic_dim": model_spec.semantic_dim,
        "residual_dim": model_spec.semantic_residual_dim,
        "n_sig_buckets": 2 ** min(model_spec.semantic_n_lsh_planes, 12),
        "semantic_vocab_ref": model_spec.semantic_vocab_ref,
    }

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 120)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 300)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="sem_targets", dtype="tokens"), instance_id="sem_targets_in", position=(40, 480)))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.jepa_mask_module,
                config={
                    "mask_ratio": model_spec.jepa_mask_ratio,
                    "mask_token_id": 0,
                    "mask_strategy": model_spec.jepa_mask_strategy,
                    "num_blocks": model_spec.jepa_num_blocks,
                    "min_block_ratio": model_spec.jepa_min_block_ratio,
                    "max_block_ratio": model_spec.jepa_max_block_ratio,
                },
            ),
            instance_id="mask",
            position=(260, 300),
        )
    )
    graph.add_edge(Edge(id="e_tokens_mask", src_node="tokens_in", src_port=0, dst_node="mask", dst_port=0))

    graph.add_node(
        NeuronInstance(
            subgraph_neuron(encoder_graph, name="online_encoder", input_aliases=["tokens", "mask"], output_aliases=["semantic_vec", "hidden", "topic_logits"]),
            instance_id="online_encoder",
            position=(520, 360),
        )
    )
    graph.add_edge(Edge(id="e_masked_online_encoder", src_node="mask", src_port=0, dst_node="online_encoder", dst_port=0))
    graph.add_edge(Edge(id="e_mask_online_encoder", src_node="mask", src_port=1, dst_node="online_encoder", dst_port=1))

    graph.add_node(
        NeuronInstance(
            subgraph_neuron(encoder_graph, name="target_encoder", input_aliases=["tokens", "mask"], output_aliases=["semantic_vec", "hidden", "topic_logits"]),
            instance_id="target_encoder",
            position=(520, 620),
        )
    )
    graph.add_edge(Edge(id="e_tokens_target_encoder", src_node="tokens_in", src_port=0, dst_node="target_encoder", dst_port=0))
    graph.add_edge(Edge(id="e_mask_target_encoder", src_node="mask", src_port=1, dst_node="target_encoder", dst_port=1))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.causal_chunk_state_module, config={"chunk_size": chunk_size, "mode": "prefix"}), instance_id="online_chunk_state", position=(780, 360)))
    graph.add_edge(Edge(id="e_online_hidden_chunk_state", src_node="online_encoder", src_port=1, dst_node="online_chunk_state", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.semantic_chunk_projector_module, config=common_semantic_cfg), instance_id="online_chunk_projector", position=(1000, 360)))
    graph.add_edge(Edge(id="e_online_chunk_projector", src_node="online_chunk_state", src_port=0, dst_node="online_chunk_projector", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_embedding_module, config={"vocab_size": model_spec.vocab_size, "model_dim": model_spec.model_dim}), instance_id="token_embed", position=(520, 120)))
    graph.add_edge(Edge(id="e_tokens_embed", src_node="tokens_in", src_port=0, dst_node="token_embed", dst_port=0))
    curr_out = "token_embed"

    if spec.pos_encoding == "absolute":
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.absolute_position_embedding_module, config={"model_dim": model_spec.model_dim}), instance_id="pos_embed", position=(520, 240)))
        graph.add_edge(Edge(id="e_tokens_pos", src_node="tokens_in", src_port=0, dst_node="pos_embed", dst_port=0))
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="embed_add", position=(740, 120)))
        graph.add_edge(Edge(id="e_embed_add_a", src_node="token_embed", src_port=0, dst_node="embed_add", dst_port=0))
        graph.add_edge(Edge(id="e_pos_add_b", src_node="pos_embed", src_port=0, dst_node="embed_add", dst_port=1))
        curr_out = "embed_add"

    if spec.dropout_p > 0.0:
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.dropout_module, config={"p": spec.dropout_p}), instance_id="embed_dropout", position=(960, 120)))
        graph.add_edge(Edge(id="e_embed_dropout", src_node=curr_out, src_port=0, dst_node="embed_dropout", dst_port=0))
        curr_out = "embed_dropout"

    block_input = curr_out
    for i in range(model_spec.num_layers):
        bname = f"block_{i}"
        graph.add_node(
            NeuronInstance(
                link_variant_neuron(
                    block_graph,
                    family="semantic_dense_jepa_evo_block",
                    version="default",
                    name=bname,
                    input_aliases=["x"],
                    output_aliases=["x"],
                ),
                instance_id=bname,
                position=(1220 + i * 220, 120),
            )
        )
        graph.add_edge(Edge(id=f"e_{block_input}_{bname}", src_node=block_input, src_port=0, dst_node=bname, dst_port=0))
        block_input = bname

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}), instance_id="final_norm", position=(1440 + model_spec.num_layers * 220, 120)))
    graph.add_edge(Edge(id="e_blocks_norm", src_node=block_input, src_port=0, dst_node="final_norm", dst_port=0))

    head_id = "tied_lm_head" if model_spec.tie_embeddings else "lm_head"
    head_def = clone_neuron_def(BuiltinNeurons.tied_lm_head_module) if model_spec.tie_embeddings else clone_neuron_def(
        BuiltinNeurons.lm_head_module,
        config={"model_dim": model_spec.model_dim, "vocab_size": model_spec.vocab_size},
    )
    graph.add_node(NeuronInstance(head_def, instance_id=head_id, position=(1660 + model_spec.num_layers * 220, 120)))
    graph.add_edge(Edge(id="e_norm_head", src_node="final_norm", src_port=0, dst_node=head_id, dst_port=0))
    if model_spec.tie_embeddings:
        graph.add_edge(Edge(id="e_tie", src_node="token_embed", src_port=1, dst_node=head_id, dst_port=1))

    ce_input = head_id
    if model_spec.logit_softcap > 0.0:
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.logit_softcap_module, config={"softcap": model_spec.logit_softcap}), instance_id="softcap", position=(1880 + model_spec.num_layers * 220, 120)))
        graph.add_edge(Edge(id="e_head_softcap", src_node=head_id, src_port=0, dst_node="softcap", dst_port=0))
        ce_input = "softcap"

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_cross_entropy_module), instance_id="ar_loss", position=(2100 + model_spec.num_layers * 220, 160)))
    graph.add_edge(Edge(id="e_logits_ar_loss", src_node=ce_input, src_port=0, dst_node="ar_loss", dst_port=0))
    graph.add_edge(Edge(id="e_targets_ar_loss", src_node="targets_in", src_port=0, dst_node="ar_loss", dst_port=1))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.causal_chunk_state_module, config={"chunk_size": chunk_size, "mode": "mean"}), instance_id="target_chunk_state", position=(780, 620)))
    graph.add_edge(Edge(id="e_target_hidden_chunk_state", src_node="target_encoder", src_port=1, dst_node="target_chunk_state", dst_port=0))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.semantic_chunk_projector_module, config=common_semantic_cfg), instance_id="target_chunk_projector", position=(1000, 620)))
    graph.add_edge(Edge(id="e_target_chunk_projector", src_node="target_chunk_state", src_port=0, dst_node="target_chunk_projector", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.jepa_predictor_module, config={"latent_dim": model_spec.semantic_dim}), instance_id="semantic_predictor", position=(1220, 520)))
    graph.add_edge(Edge(id="e_online_semantic_predictor", src_node="online_chunk_projector", src_port=0, dst_node="semantic_predictor", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.latent_mse_loss_module), instance_id="jepa_loss", position=(1440, 560)))
    graph.add_edge(Edge(id="e_predictor_jepa_loss", src_node="semantic_predictor", src_port=0, dst_node="jepa_loss", dst_port=0))
    graph.add_edge(Edge(id="e_target_jepa_loss", src_node="target_chunk_projector", src_port=0, dst_node="jepa_loss", dst_port=1))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.semantic_alignment_loss_module, config={"semantic_vocab_ref": model_spec.semantic_vocab_ref}), instance_id="sem_align_loss", position=(1440, 700)))
    graph.add_edge(Edge(id="e_topic_logits_sem_align", src_node="online_chunk_projector", src_port=2, dst_node="sem_align_loss", dst_port=0))
    graph.add_edge(Edge(id="e_sem_targets_sem_align", src_node="sem_targets_in", src_port=0, dst_node="sem_align_loss", dst_port=1))

    scaled_terms: list[str] = []
    for loss_id, coef, y in [
        ("ar_loss", model_spec.ar_loss_coef, 180),
        ("jepa_loss", model_spec.jepa_loss_coef, 320),
        ("sem_align_loss", model_spec.semantic_align_loss_coef, 460),
    ]:
        scaled_id = f"{loss_id}_scaled"
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.loss_scale_module, config={"coef": coef}), instance_id=scaled_id, position=(2320 + model_spec.num_layers * 220, y)))
        graph.add_edge(Edge(id=f"e_{loss_id}_{scaled_id}", src_node=loss_id, src_port=0, dst_node=scaled_id, dst_port=0))
        scaled_terms.append(scaled_id)

    total = scaled_terms[0]
    for idx, loss_id in enumerate(scaled_terms[1:], start=1):
        add_id = f"total_loss_add_{idx}"
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.aux_loss_add_module, config={"coef": 1.0}), instance_id=add_id, position=(2540 + model_spec.num_layers * 220, 180 + idx * 80)))
        graph.add_edge(Edge(id=f"e_{total}_{add_id}_main", src_node=total, src_port=0, dst_node=add_id, dst_port=0))
        graph.add_edge(Edge(id=f"e_{loss_id}_{add_id}_aux", src_node=loss_id, src_port=0, dst_node=add_id, dst_port=1))
        total = add_id

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(2760 + model_spec.num_layers * 220, 340)))
    graph.add_edge(Edge(id="e_total_loss_out", src_node=total, src_port=0, dst_node="loss_out", dst_port=0))

    graph.input_node_ids = ["tokens_in", "targets_in", "sem_targets_in"]
    graph.output_node_ids = ["loss_out"]
    return graph


def build_semantic_moe_jepa_evo_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    """Experimental: chunk-routed semantic MoE with JEPA targets and route evolution."""

    spec = model_spec.block_spec
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")

    encoder_graph = build_jepa_semantic_encoder_graph("semantic_moe_jepa_evo_encoder", model_spec)
    graph.variant_library = deepcopy(encoder_graph.variant_library)
    encoder_graph.variant_library = {}

    attn_graph = build_dense_attention_graph("attention_engine", model_spec.model_dim, spec, **_attn_flags(model_spec))
    mlp_graph = build_semantic_router_mlp_graph("semantic_router_mlp", model_spec.model_dim, spec)
    block_graph = build_semantic_router_decoder_block_graph(
        "semantic_moe_jepa_evo_block",
        model_spec.model_dim,
        spec,
        attn_graph,
        mlp_graph,
    )
    graph.variant_library.update(
        {
            "attention": {"default": attn_graph},
            "semantic_router_mlp": {"default": mlp_graph},
            "semantic_moe_jepa_evo_block": {"default": block_graph},
        }
    )

    chunk_size = int(model_spec.route_chunk_size)
    shared_experts = int(model_spec.semantic_shared_experts)
    free_experts = int(model_spec.semantic_free_experts)
    route_aux_coef = float(spec.router_aux_loss_coef)
    common_semantic_cfg = {
        "input_dim": model_spec.model_dim,
        "semantic_dim": model_spec.semantic_dim,
        "residual_dim": model_spec.semantic_residual_dim,
        "n_sig_buckets": 2 ** min(model_spec.semantic_n_lsh_planes, 12),
        "semantic_vocab_ref": model_spec.semantic_vocab_ref,
    }

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 120)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 300)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="sem_targets", dtype="tokens"), instance_id="sem_targets_in", position=(40, 480)))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.jepa_mask_module,
                config={
                    "mask_ratio": model_spec.jepa_mask_ratio,
                    "mask_token_id": 0,
                    "mask_strategy": model_spec.jepa_mask_strategy,
                    "num_blocks": model_spec.jepa_num_blocks,
                    "min_block_ratio": model_spec.jepa_min_block_ratio,
                    "max_block_ratio": model_spec.jepa_max_block_ratio,
                },
            ),
            instance_id="mask",
            position=(260, 300),
        )
    )
    graph.add_edge(Edge(id="e_tokens_mask", src_node="tokens_in", src_port=0, dst_node="mask", dst_port=0))

    graph.add_node(
        NeuronInstance(
            subgraph_neuron(encoder_graph, name="online_encoder", input_aliases=["tokens", "mask"], output_aliases=["semantic_vec", "hidden", "topic_logits"]),
            instance_id="online_encoder",
            position=(520, 360),
        )
    )
    graph.add_edge(Edge(id="e_masked_online_encoder", src_node="mask", src_port=0, dst_node="online_encoder", dst_port=0))
    graph.add_edge(Edge(id="e_mask_online_encoder", src_node="mask", src_port=1, dst_node="online_encoder", dst_port=1))

    graph.add_node(
        NeuronInstance(
            subgraph_neuron(encoder_graph, name="target_encoder", input_aliases=["tokens", "mask"], output_aliases=["semantic_vec", "hidden", "topic_logits"]),
            instance_id="target_encoder",
            position=(520, 620),
        )
    )
    graph.add_edge(Edge(id="e_tokens_target_encoder", src_node="tokens_in", src_port=0, dst_node="target_encoder", dst_port=0))
    graph.add_edge(Edge(id="e_mask_target_encoder", src_node="mask", src_port=1, dst_node="target_encoder", dst_port=1))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.causal_chunk_state_module, config={"chunk_size": chunk_size, "mode": "prefix"}), instance_id="online_chunk_state", position=(780, 360)))
    graph.add_edge(Edge(id="e_online_hidden_chunk_state", src_node="online_encoder", src_port=1, dst_node="online_chunk_state", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.semantic_chunk_projector_module, config=common_semantic_cfg), instance_id="online_chunk_projector", position=(1000, 360)))
    graph.add_edge(Edge(id="e_online_chunk_projector", src_node="online_chunk_state", src_port=0, dst_node="online_chunk_projector", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_chunk_hasher_module,
                config={"dim": model_spec.semantic_dim, "tables": model_spec.semantic_n_lsh_tables, "planes": model_spec.semantic_n_lsh_planes},
            ),
            instance_id="online_chunk_hasher",
            position=(1220, 300),
        )
    )
    graph.add_edge(Edge(id="e_online_semantic_hash", src_node="online_chunk_projector", src_port=0, dst_node="online_chunk_hasher", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.semantic_moe_jepa_evo_router_module,
                config={
                    "semantic_dim": model_spec.semantic_dim,
                    "top_k": spec.top_k or 2,
                    "shared_experts": shared_experts,
                    "free_experts": free_experts,
                    "tables": model_spec.semantic_n_lsh_tables,
                    "n_buckets": 2 ** min(model_spec.semantic_n_lsh_planes, 12),
                    "semantic_vocab_ref": model_spec.semantic_vocab_ref,
                },
            ),
            instance_id="semantic_moe_router",
            position=(1440, 360),
        )
    )
    graph.add_edge(Edge(id="e_sem_vec_router", src_node="online_chunk_projector", src_port=0, dst_node="semantic_moe_router", dst_port=0))
    graph.add_edge(Edge(id="e_bucket_router", src_node="online_chunk_hasher", src_port=0, dst_node="semantic_moe_router", dst_port=1))
    graph.add_edge(Edge(id="e_topic_router", src_node="online_chunk_projector", src_port=2, dst_node="semantic_moe_router", dst_port=2))
    graph.add_edge(Edge(id="e_sem_targets_router", src_node="sem_targets_in", src_port=0, dst_node="semantic_moe_router", dst_port=3))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_embedding_module, config={"vocab_size": model_spec.vocab_size, "model_dim": model_spec.model_dim}), instance_id="token_embed", position=(520, 120)))
    graph.add_edge(Edge(id="e_tokens_embed", src_node="tokens_in", src_port=0, dst_node="token_embed", dst_port=0))
    curr_out = "token_embed"

    if spec.pos_encoding == "absolute":
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.absolute_position_embedding_module, config={"model_dim": model_spec.model_dim}), instance_id="pos_embed", position=(520, 240)))
        graph.add_edge(Edge(id="e_tokens_pos", src_node="tokens_in", src_port=0, dst_node="pos_embed", dst_port=0))
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.add), instance_id="embed_add", position=(740, 120)))
        graph.add_edge(Edge(id="e_embed_add_a", src_node="token_embed", src_port=0, dst_node="embed_add", dst_port=0))
        graph.add_edge(Edge(id="e_pos_add_b", src_node="pos_embed", src_port=0, dst_node="embed_add", dst_port=1))
        curr_out = "embed_add"

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.broadcast_chunk_routes_module, config={"chunk_size": chunk_size}), instance_id="broadcast_chunk_routes", position=(1660, 300)))
    graph.add_edge(Edge(id="e_hidden_broadcast_chunk_routes", src_node=curr_out, src_port=0, dst_node="broadcast_chunk_routes", dst_port=0))
    graph.add_edge(Edge(id="e_weights_broadcast_chunk_routes", src_node="semantic_moe_router", src_port=0, dst_node="broadcast_chunk_routes", dst_port=1))
    graph.add_edge(Edge(id="e_indices_broadcast_chunk_routes", src_node="semantic_moe_router", src_port=1, dst_node="broadcast_chunk_routes", dst_port=2))

    block_input = curr_out
    for i in range(model_spec.num_layers):
        bname = f"block_{i}"
        graph.add_node(
            NeuronInstance(
                link_variant_neuron(
                    block_graph,
                    family="semantic_moe_jepa_evo_block",
                    version="default",
                    name=bname,
                    input_aliases=["x", "routing_weights", "routing_indices"],
                    output_aliases=["x"],
                ),
                instance_id=bname,
                position=(1880 + i * 220, 120),
            )
        )
        graph.add_edge(Edge(id=f"e_{block_input}_{bname}", src_node=block_input, src_port=0, dst_node=bname, dst_port=0))
        graph.add_edge(Edge(id=f"e_broadcast_weights_{bname}", src_node="broadcast_chunk_routes", src_port=0, dst_node=bname, dst_port=1))
        graph.add_edge(Edge(id=f"e_broadcast_indices_{bname}", src_node="broadcast_chunk_routes", src_port=1, dst_node=bname, dst_port=2))
        block_input = bname

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}), instance_id="final_norm", position=(2100 + model_spec.num_layers * 220, 120)))
    graph.add_edge(Edge(id="e_blocks_norm", src_node=block_input, src_port=0, dst_node="final_norm", dst_port=0))

    head_id = "tied_lm_head" if model_spec.tie_embeddings else "lm_head"
    head_def = clone_neuron_def(BuiltinNeurons.tied_lm_head_module) if model_spec.tie_embeddings else clone_neuron_def(
        BuiltinNeurons.lm_head_module,
        config={"model_dim": model_spec.model_dim, "vocab_size": model_spec.vocab_size},
    )
    graph.add_node(NeuronInstance(head_def, instance_id=head_id, position=(2320 + model_spec.num_layers * 220, 120)))
    graph.add_edge(Edge(id="e_norm_head", src_node="final_norm", src_port=0, dst_node=head_id, dst_port=0))
    if model_spec.tie_embeddings:
        graph.add_edge(Edge(id="e_tie", src_node="token_embed", src_port=1, dst_node=head_id, dst_port=1))

    ce_input = head_id
    if model_spec.logit_softcap > 0.0:
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.logit_softcap_module, config={"softcap": model_spec.logit_softcap}), instance_id="softcap", position=(2540 + model_spec.num_layers * 220, 120)))
        graph.add_edge(Edge(id="e_head_softcap", src_node=head_id, src_port=0, dst_node="softcap", dst_port=0))
        ce_input = "softcap"

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_cross_entropy_module), instance_id="ar_loss", position=(2760 + model_spec.num_layers * 220, 160)))
    graph.add_edge(Edge(id="e_logits_ar_loss", src_node=ce_input, src_port=0, dst_node="ar_loss", dst_port=0))
    graph.add_edge(Edge(id="e_targets_ar_loss", src_node="targets_in", src_port=0, dst_node="ar_loss", dst_port=1))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.causal_chunk_state_module, config={"chunk_size": chunk_size, "mode": "mean"}), instance_id="target_chunk_state", position=(780, 620)))
    graph.add_edge(Edge(id="e_target_hidden_chunk_state", src_node="target_encoder", src_port=1, dst_node="target_chunk_state", dst_port=0))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.semantic_chunk_projector_module, config=common_semantic_cfg), instance_id="target_chunk_projector", position=(1000, 620)))
    graph.add_edge(Edge(id="e_target_chunk_projector", src_node="target_chunk_state", src_port=0, dst_node="target_chunk_projector", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.jepa_predictor_module, config={"latent_dim": model_spec.semantic_dim}), instance_id="route_predictor", position=(1220, 520)))
    graph.add_edge(Edge(id="e_online_semantic_route_predictor", src_node="online_chunk_projector", src_port=0, dst_node="route_predictor", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.latent_mse_loss_module), instance_id="jepa_loss", position=(1440, 560)))
    graph.add_edge(Edge(id="e_predictor_jepa_loss", src_node="route_predictor", src_port=0, dst_node="jepa_loss", dst_port=0))
    graph.add_edge(Edge(id="e_target_jepa_loss", src_node="target_chunk_projector", src_port=0, dst_node="jepa_loss", dst_port=1))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.semantic_alignment_loss_module, config={"semantic_vocab_ref": model_spec.semantic_vocab_ref}), instance_id="sem_align_loss", position=(1440, 700)))
    graph.add_edge(Edge(id="e_topic_logits_sem_align", src_node="online_chunk_projector", src_port=2, dst_node="sem_align_loss", dst_port=0))
    graph.add_edge(Edge(id="e_sem_targets_sem_align", src_node="sem_targets_in", src_port=0, dst_node="sem_align_loss", dst_port=1))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.route_balance_loss_module), instance_id="route_balance_loss", position=(1660, 500)))
    graph.add_edge(Edge(id="e_route_logits_balance", src_node="semantic_moe_router", src_port=2, dst_node="route_balance_loss", dst_port=0))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.route_selection_loss_module, config={"semantic_vocab_ref": model_spec.semantic_vocab_ref, "shared_experts": shared_experts, "free_experts": free_experts}), instance_id="route_selection_loss", position=(1660, 620)))
    graph.add_edge(Edge(id="e_route_logits_selection", src_node="semantic_moe_router", src_port=2, dst_node="route_selection_loss", dst_port=0))
    graph.add_edge(Edge(id="e_sem_targets_selection", src_node="sem_targets_in", src_port=0, dst_node="route_selection_loss", dst_port=1))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.route_distillation_loss_module, config={"semantic_vocab_ref": model_spec.semantic_vocab_ref, "shared_experts": shared_experts, "free_experts": free_experts}), instance_id="route_distill_loss", position=(1660, 740)))
    graph.add_edge(Edge(id="e_route_logits_distill", src_node="semantic_moe_router", src_port=2, dst_node="route_distill_loss", dst_port=0))
    graph.add_edge(Edge(id="e_target_topics_distill", src_node="target_chunk_projector", src_port=2, dst_node="route_distill_loss", dst_port=1))

    scaled_terms: list[str] = []
    for loss_id, coef, y in [
        ("ar_loss", model_spec.ar_loss_coef, 180),
        ("jepa_loss", model_spec.jepa_loss_coef, 320),
        ("sem_align_loss", model_spec.semantic_align_loss_coef, 460),
        ("route_balance_loss", route_aux_coef, 600),
        ("route_selection_loss", route_aux_coef, 740),
        ("route_distill_loss", route_aux_coef, 880),
    ]:
        scaled_id = f"{loss_id}_scaled"
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.loss_scale_module, config={"coef": coef}), instance_id=scaled_id, position=(2980 + model_spec.num_layers * 220, y)))
        graph.add_edge(Edge(id=f"e_{loss_id}_{scaled_id}", src_node=loss_id, src_port=0, dst_node=scaled_id, dst_port=0))
        scaled_terms.append(scaled_id)

    total = scaled_terms[0]
    for idx, loss_id in enumerate(scaled_terms[1:], start=1):
        add_id = f"total_loss_add_{idx}"
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.aux_loss_add_module, config={"coef": 1.0}), instance_id=add_id, position=(3200 + model_spec.num_layers * 220, 180 + idx * 80)))
        graph.add_edge(Edge(id=f"e_{total}_{add_id}_main", src_node=total, src_port=0, dst_node=add_id, dst_port=0))
        graph.add_edge(Edge(id=f"e_{loss_id}_{add_id}_aux", src_node=loss_id, src_port=0, dst_node=add_id, dst_port=1))
        total = add_id

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(3420 + model_spec.num_layers * 220, 420)))
    graph.add_edge(Edge(id="e_total_loss_out", src_node=total, src_port=0, dst_node="loss_out", dst_port=0))

    graph.input_node_ids = ["tokens_in", "targets_in", "sem_targets_in"]
    graph.output_node_ids = ["loss_out"]
    return graph


def build_hnet_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    hidden_graph = build_hidden_backbone_graph("hnet_backbone", model_spec)
    graph.variant_library = deepcopy(hidden_graph.variant_library)
    hidden_graph.variant_library = {}

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 140)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 300)))
    graph.add_node(NeuronInstance(subgraph_neuron(hidden_graph, name="backbone", input_aliases=["tokens"], output_aliases=["hidden"]), instance_id="backbone", position=(260, 140)))
    graph.add_edge(Edge(id="e_tokens_backbone", src_node="tokens_in", src_port=0, dst_node="backbone", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.byte_patch_merge_module), instance_id="patch_merge", position=(520, 140)))
    graph.add_edge(Edge(id="e_backbone_merge", src_node="backbone", src_port=0, dst_node="patch_merge", dst_port=0))
    graph.add_edge(Edge(id="e_targets_merge", src_node="targets_in", src_port=0, dst_node="patch_merge", dst_port=1))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.lm_head_module, config={"model_dim": model_spec.model_dim, "vocab_size": model_spec.vocab_size}),
            instance_id="lm_head",
            position=(760, 140),
        )
    )
    graph.add_edge(Edge(id="e_merge_head", src_node="patch_merge", src_port=0, dst_node="lm_head", dst_port=0))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_cross_entropy_module), instance_id="ce", position=(980, 140)))
    graph.add_edge(Edge(id="e_hnet_logits", src_node="lm_head", src_port=0, dst_node="ce", dst_port=0))
    graph.add_edge(Edge(id="e_hnet_targets", src_node="targets_in", src_port=0, dst_node="ce", dst_port=1))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(1200, 140)))
    graph.add_edge(Edge(id="e_hnet_loss_out", src_node="ce", src_port=0, dst_node="loss_out", dst_port=0))
    graph.input_node_ids = ["tokens_in", "targets_in"]
    graph.output_node_ids = ["loss_out"]
    return graph


def build_universal_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    spec = model_spec.block_spec
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 140)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 300)))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(BuiltinNeurons.token_embedding_module, config={"vocab_size": model_spec.vocab_size, "model_dim": model_spec.model_dim}),
            instance_id="token_embed",
            position=(260, 140),
        )
    )
    graph.add_edge(Edge(id="e_tokens_embed", src_node="tokens_in", src_port=0, dst_node="token_embed", dst_port=0))

    graph.add_node(
        NeuronInstance(
            clone_neuron_def(
                BuiltinNeurons.universal_transformer_module,
                config={
                    "model_dim": model_spec.model_dim,
                    "num_heads": spec.num_heads,
                    "mlp_mult": spec.mlp_multiplier,
                    "max_steps": model_spec.max_recurrence_steps,
                    "halt_epsilon": model_spec.halt_epsilon,
                },
            ),
            instance_id="universal",
            position=(520, 140),
        )
    )
    graph.add_edge(Edge(id="e_embed_universal", src_node="token_embed", src_port=0, dst_node="universal", dst_port=0))

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}), instance_id="final_norm", position=(760, 140)))
    graph.add_edge(Edge(id="e_universal_norm", src_node="universal", src_port=0, dst_node="final_norm", dst_port=0))

    head_id = "tied_lm_head" if model_spec.tie_embeddings else "lm_head"
    head_def = clone_neuron_def(BuiltinNeurons.tied_lm_head_module) if model_spec.tie_embeddings else clone_neuron_def(BuiltinNeurons.lm_head_module, config={"model_dim": model_spec.model_dim, "vocab_size": model_spec.vocab_size})
    graph.add_node(NeuronInstance(head_def, instance_id=head_id, position=(980, 140)))
    graph.add_edge(Edge(id="e_universal_head", src_node="final_norm", src_port=0, dst_node=head_id, dst_port=0))
    if model_spec.tie_embeddings:
        graph.add_edge(Edge(id="e_universal_tie", src_node="token_embed", src_port=1, dst_node=head_id, dst_port=1))

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_cross_entropy_module), instance_id="ce", position=(1200, 140)))
    graph.add_edge(Edge(id="e_universal_logits", src_node=head_id, src_port=0, dst_node="ce", dst_port=0))
    graph.add_edge(Edge(id="e_universal_targets", src_node="targets_in", src_port=0, dst_node="ce", dst_port=1))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(1420, 140)))
    graph.add_edge(Edge(id="e_universal_loss_out", src_node="ce", src_port=0, dst_node="loss_out", dst_port=0))
    graph.input_node_ids = ["tokens_in", "targets_in"]
    graph.output_node_ids = ["loss_out"]
    return graph


def _normalized_template_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config or {})
    if "n_layer" in normalized and "num_layers" not in normalized:
        normalized["num_layers"] = normalized["n_layer"]
    if "n_embd" in normalized and "model_dim" not in normalized:
        normalized["model_dim"] = normalized["n_embd"]
    if "n_head" in normalized and "num_heads" not in normalized:
        normalized["num_heads"] = normalized["n_head"]
    return normalized


def build_model_spec_from_config(config: dict[str, Any], *, preview_defaults: bool = False) -> ModelSpec:
    from neuralfn.config import (
        build_composed_lm_spec,
        build_decoder2encoder_moe_spec,
        build_diffllama_spec,
        build_gpt2_megakernel_spec,
        build_gpt2_spec,
        build_hnet_lm_spec,
        build_jamba_hybrid_spec,
        build_jepa_semantic_hybrid_megakernel_spec,
        build_jepa_semantic_hybrid_spec,
        build_kv_pca_llama_spec,
        build_dense_jepa_evo_spec,
        build_llama_fast_spec,
        build_llama_fast_megakernel_spec,
        build_llama_megakernel_spec,
        build_llama_spec,
        build_llm_jepa_spec,
        build_mixllama_fast_megakernel_spec,
        build_mixllama_fast_spec,
        build_mixllama_spec,
        build_moe_jepa_evo_spec,
        build_nanogpt_megakernel_spec,
        build_nanogpt_spec,
        build_semantic_dense_jepa_evo_spec,
        build_semantic_moe_jepa_evo_spec,
        build_semantic_router_moe_megakernel_spec,
        build_semantic_router_moe_spec,
        build_ternary_b158_spec,
        build_ttt_llama_spec,
        build_universal_llama_spec,
    )

    normalized = _normalized_template_config(config)
    if any(key in normalized for key in ("base_model", "topology", "router_mode", "use_jepa")):
        if preview_defaults and "num_layers" not in normalized:
            normalized["num_layers"] = 1
        topology = str(normalized.get("topology", "dense"))
        router_mode = str(normalized.get("router_mode", "none"))
        if preview_defaults and topology == "moe":
            normalized.setdefault("experts", NUM_VOCAB_DIMS if router_mode == "semantic" else 4)
            normalized.setdefault("top_k", 2)
        return build_composed_lm_spec(**normalized)

    preset = normalized.get("preset", "nanogpt")
    if preview_defaults and "num_layers" not in normalized:
        normalized["num_layers"] = 2 if preset == "jamba" else 1
    if preview_defaults and preset in {"mixllama", "moe", "mixllama_fast", "mixllama_fast_megakernel", "jamba", "seq2seq"}:
        normalized.setdefault("experts", 4)
        normalized.setdefault("top_k", 2)
    if preview_defaults and preset == "moe_jepa_evo":
        normalized.setdefault("experts", 4)
        normalized.setdefault("top_k", 2)
    if preview_defaults and preset in {"jepa_semantic_hybrid", "jepa_semantic_hybrid_megakernel"}:
        normalized.setdefault("experts", NUM_VOCAB_DIMS)
        normalized.setdefault("top_k", 2)
    if preview_defaults and preset in {"semantic_router_moe", "semantic_router_moe_megakernel"}:
        normalized.setdefault("experts", NUM_VOCAB_DIMS)
        normalized.setdefault("top_k", 2)
    if preview_defaults and preset == "semantic_moe_jepa_evo":
        normalized.setdefault("semantic_shared_experts", 2)
        normalized.setdefault("semantic_free_experts", 8)
        normalized.setdefault("experts", 2 + NUM_VOCAB_DIMS + 8)
        normalized.setdefault("top_k", 2)
        normalized.setdefault("route_chunk_size", 32)
    if preview_defaults and preset == "semantic_dense_jepa_evo":
        normalized.setdefault("route_chunk_size", 32)

    if preset == "gpt2_megakernel":
        return build_gpt2_megakernel_spec(**normalized)
    if preset == "gpt2":
        return build_gpt2_spec(**normalized)
    if preset == "nanogpt_megakernel":
        return build_nanogpt_megakernel_spec(**normalized)
    if preset == "nanogpt":
        return build_nanogpt_spec(**normalized)
    if preset == "llama":
        return build_llama_spec(**normalized)
    if preset in {"mixllama", "moe"}:
        return build_mixllama_spec(**normalized)
    if preset == "llama_fast":
        return build_llama_fast_spec(**normalized)
    if preset == "llama_fast_megakernel":
        return build_llama_fast_megakernel_spec(**normalized)
    if preset == "mixllama_fast":
        return build_mixllama_fast_spec(**normalized)
    if preset == "mixllama_fast_megakernel":
        return build_mixllama_fast_megakernel_spec(**normalized)
    if preset == "jamba":
        return build_jamba_hybrid_spec(**normalized)
    if preset == "ternary_b158":
        return build_ternary_b158_spec(**normalized)
    if preset == "llama_megakernel":
        return build_llama_megakernel_spec(**normalized)
    if preset == "kv_pca_llama":
        return build_kv_pca_llama_spec(**normalized)
    if preset == "seq2seq":
        return build_decoder2encoder_moe_spec(**normalized)
    if preset == "diffusion":
        return build_diffllama_spec(**normalized)
    if preset == "ttt_llama":
        return build_ttt_llama_spec(**normalized)
    if preset == "llm_jepa":
        return build_llm_jepa_spec(**normalized)
    if preset == "dense_jepa_evo":
        return build_dense_jepa_evo_spec(**normalized)
    if preset == "moe_jepa_evo":
        return build_moe_jepa_evo_spec(**normalized)
    if preset == "jepa_semantic_hybrid":
        return build_jepa_semantic_hybrid_spec(**normalized)
    if preset == "jepa_semantic_hybrid_megakernel":
        return build_jepa_semantic_hybrid_megakernel_spec(**normalized)
    if preset == "semantic_router_moe":
        return build_semantic_router_moe_spec(**normalized)
    if preset == "semantic_router_moe_megakernel":
        return build_semantic_router_moe_megakernel_spec(**normalized)
    if preset == "semantic_moe_jepa_evo":
        return build_semantic_moe_jepa_evo_spec(**normalized)
    if preset == "semantic_dense_jepa_evo":
        return build_semantic_dense_jepa_evo_spec(**normalized)
    if preset == "hnet_lm":
        return build_hnet_lm_spec(**normalized)
    if preset == "universal_llama":
        return build_universal_llama_spec(**normalized)
    return build_nanogpt_spec(**normalized)

def build_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    spec = model_spec.block_spec
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    block_family = "mixllama" if (spec.mlp_type == "mixllama" or spec.mlp_type == "moe") else "transformer_block"

    # Pre-build variant libraries
    attn_graph = build_dense_attention_graph("attention_engine", model_spec.model_dim, spec, **_attn_flags(model_spec))
    mamba_graph = build_mamba_graph("mamba_engine", model_spec.model_dim, spec)

    if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
        mlp_graph = build_mixllama_mlp_graph("mlp_moe", model_spec.model_dim, spec)
    else:
        mlp_graph = build_dense_mlp_graph("mlp_dense", model_spec.model_dim, spec)

    attn_block_graph = build_decoder_block_graph("decoder_block_attn", model_spec.model_dim, spec, attn_graph, mlp_graph)
    mamba_block_graph = build_decoder_block_graph("decoder_block_mamba", model_spec.model_dim, spec, mamba_graph, mlp_graph)

    graph.variant_library = {
        "attention": {"default": attn_graph},
        "mamba": {"default": mamba_graph},
        "mlp": {"default": mlp_graph},
        "attn_block": {"default": attn_block_graph},
        "mamba_block": {"default": mamba_block_graph},
        block_family: {"default": attn_block_graph}, # fallback
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

        is_attn = True
        if model_spec.template.backbone == "jamba":
            is_attn = (i % 4 == 0)
            current_block_family = "attn_block" if is_attn else "mamba_block"
            current_block_graph = attn_block_graph if is_attn else mamba_block_graph
        else:
            current_block_family = block_family
            current_block_graph = attn_block_graph

        out_aliases = ["x"]
        if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
            out_aliases.append("aux_loss")

        graph.add_node(NeuronInstance(link_variant_neuron(current_block_graph, family=current_block_family, version="default", name=bname, input_aliases=["x"], output_aliases=out_aliases), instance_id=bname, position=(920 + i*220, 140)))
        graph.add_edge(Edge(id=f"e_{curr_out}_{bname}", src_node=curr_out, src_port=0, dst_node=bname, dst_port=0))
        curr_out = bname
        if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
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
    ce_input = head_id
    if model_spec.logit_softcap > 0.0:
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.logit_softcap_module, config={"softcap": model_spec.logit_softcap}), instance_id="softcap", position=(1580 + model_spec.num_layers*220, 140)))
        graph.add_edge(Edge(id="e_head_softcap", src_node=head_id, src_port=0, dst_node="softcap", dst_port=0))
        ce_input = "softcap"

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.token_cross_entropy_module), instance_id="ce", position=(1800 + model_spec.num_layers*220, 260)))
    graph.add_edge(Edge(id="e_logits_ce", src_node=ce_input, src_port=0, dst_node="ce", dst_port=0))
    graph.add_edge(Edge(id="e_targets_ce", src_node="targets_in", src_port=0, dst_node="ce", dst_port=1))

    main_loss = "ce"
    # Accumulate Aux Losses
    if (spec.mlp_type == "mixllama" or spec.mlp_type == "moe") and aux_losses and spec.router_aux_loss_coef > 0.0:
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

def build_sft_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    """Mirror of ``build_model_stage_graph`` for supervised fine-tuning.

    Interface: ``(tokens, targets, loss_mask) -> loss``. Body is identical to
    the AR model stage; the only structural differences are:
      - Extra ``loss_mask_in`` input terminal.
      - ``masked_token_cross_entropy`` in place of ``token_cross_entropy``.
    """
    spec = model_spec.block_spec
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    block_family = "mixllama" if (spec.mlp_type == "mixllama" or spec.mlp_type == "moe") else "transformer_block"

    attn_graph = build_dense_attention_graph("attention_engine", model_spec.model_dim, spec, **_attn_flags(model_spec))
    mamba_graph = build_mamba_graph("mamba_engine", model_spec.model_dim, spec)
    if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
        mlp_graph = build_mixllama_mlp_graph("mlp_moe", model_spec.model_dim, spec)
    else:
        mlp_graph = build_dense_mlp_graph("mlp_dense", model_spec.model_dim, spec)
    attn_block_graph = build_decoder_block_graph("decoder_block_attn", model_spec.model_dim, spec, attn_graph, mlp_graph)
    mamba_block_graph = build_decoder_block_graph("decoder_block_mamba", model_spec.model_dim, spec, mamba_graph, mlp_graph)
    graph.variant_library = {
        "attention": {"default": attn_graph},
        "mamba": {"default": mamba_graph},
        "mlp": {"default": mlp_graph},
        "attn_block": {"default": attn_block_graph},
        "mamba_block": {"default": mamba_block_graph},
        block_family: {"default": attn_block_graph},
    }

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 140)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 360)))
    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="loss_mask", dtype="tensor"), instance_id="loss_mask_in", position=(40, 520)))

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

    for i in range(model_spec.num_layers):
        bname = f"block_{i}"
        out_aliases = ["x"]
        if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
            out_aliases.append("aux_loss")
        graph.add_node(NeuronInstance(link_variant_neuron(attn_block_graph, family=block_family, version="default", name=bname, input_aliases=["x"], output_aliases=out_aliases), instance_id=bname, position=(920 + i*220, 140)))
        graph.add_edge(Edge(id=f"e_{curr_out}_{bname}", src_node=curr_out, src_port=0, dst_node=bname, dst_port=0))
        curr_out = bname

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}), instance_id="final_norm", position=(1140 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(id="e_blocks_norm", src_node=curr_out, src_port=0, dst_node="final_norm", dst_port=0))

    head_id = "tied_lm_head" if model_spec.tie_embeddings else "lm_head"
    head_def = clone_neuron_def(BuiltinNeurons.tied_lm_head_module) if model_spec.tie_embeddings else clone_neuron_def(BuiltinNeurons.lm_head_module, config={"model_dim": model_spec.model_dim, "vocab_size": model_spec.vocab_size})
    graph.add_node(NeuronInstance(head_def, instance_id=head_id, position=(1360 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(id="e_norm_head", src_node="final_norm", src_port=0, dst_node=head_id, dst_port=0))
    if model_spec.tie_embeddings:
        graph.add_edge(Edge(id="e_tie", src_node="token_embed", src_port=1, dst_node=head_id, dst_port=1))
    ce_input = head_id
    if model_spec.logit_softcap > 0.0:
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.logit_softcap_module, config={"softcap": model_spec.logit_softcap}), instance_id="softcap", position=(1580 + model_spec.num_layers*220, 140)))
        graph.add_edge(Edge(id="e_head_softcap", src_node=head_id, src_port=0, dst_node="softcap", dst_port=0))
        ce_input = "softcap"

    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.masked_token_cross_entropy_module), instance_id="ce", position=(1800 + model_spec.num_layers*220, 260)))
    graph.add_edge(Edge(id="e_logits_ce", src_node=ce_input, src_port=0, dst_node="ce", dst_port=0))
    graph.add_edge(Edge(id="e_targets_ce", src_node="targets_in", src_port=0, dst_node="ce", dst_port=1))
    graph.add_edge(Edge(id="e_mask_ce", src_node="loss_mask_in", src_port=0, dst_node="ce", dst_port=2))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(2240 + model_spec.num_layers*220, 260)))
    graph.add_edge(Edge(id="e_to_loss_out", src_node="ce", src_port=0, dst_node="loss_out", dst_port=0))

    graph.input_node_ids = ["tokens_in", "targets_in", "loss_mask_in"]
    graph.output_node_ids = ["loss_out"]
    return graph


def build_sft_root_graph(*, name: str = "model_root", model_spec: ModelSpec | None = None) -> NeuronGraph:
    """Root graph for supervised fine-tuning: sft_dataset_source → model → loss."""
    if model_spec is None:
        model_spec = ModelSpec()
    if model_spec.template.objective != "sft":
        # Normalize so downstream consumers see a consistent template_spec.
        model_spec.template.objective = "sft"  # type: ignore[assignment]

    graph = NeuronGraph(
        name=name,
        training_method="torch",
        runtime="torch",
        torch_config={
            "device": "cuda",
            "amp_dtype": "float32",
            "template_spec": model_spec_to_dict(model_spec),
        },
    )

    stage = build_sft_model_stage_graph("model_stage", model_spec)
    graph.variant_library = deepcopy(stage.variant_library)
    stage.variant_library = {}

    dataset_def = clone_neuron_def(
        BuiltinNeurons.sft_dataset_source_module,
        config={"dataset_names": [], "seq_len": 64},
    )
    graph.add_node(NeuronInstance(dataset_def, instance_id="sft_dataset_source", position=(40, 180)))
    graph.add_node(NeuronInstance(
        subgraph_neuron(stage, name="model", input_aliases=["tokens", "targets", "loss_mask"], output_aliases=["loss"]),
        instance_id="model",
        position=(380, 180),
    ))
    graph.add_edge(Edge(id="e_tokens_model", src_node="sft_dataset_source", src_port=0, dst_node="model", dst_port=0))
    graph.add_edge(Edge(id="e_targets_model", src_node="sft_dataset_source", src_port=1, dst_node="model", dst_port=1))
    graph.add_edge(Edge(id="e_mask_model", src_node="sft_dataset_source", src_port=2, dst_node="model", dst_port=2))
    graph.input_node_ids = ["sft_dataset_source"]

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(720, 180)))
    graph.add_edge(Edge(id="e_model_out", src_node="model", src_port=0, dst_node="loss_out", dst_port=0))
    graph.output_node_ids = ["loss_out"]

    if model_spec.finetune is not None:
        graph.torch_config["finetune_spec"] = {
            "objective": model_spec.finetune.objective,
            "base_checkpoint": model_spec.finetune.base_checkpoint,
            "ref_checkpoint": model_spec.finetune.ref_checkpoint,
            "reward_checkpoint": model_spec.finetune.reward_checkpoint,
            "beta": model_spec.finetune.beta,
            "kl_coef": model_spec.finetune.kl_coef,
            "ppo_clip": model_spec.finetune.ppo_clip,
            "rollout_length": model_spec.finetune.rollout_length,
            "dpo_loss_type": model_spec.finetune.dpo_loss_type,
            "adapter_only_save": model_spec.finetune.adapter_only_save,
        }
    return graph


def _logits_model_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    """Variant of ``build_model_stage_graph`` whose terminal output is *logits* not loss.

    Used as a building block by the DPO and PPO root graphs — each of those
    composes multiple forward passes (policy + reference + reward) before
    combining them into a single loss.
    """
    spec = model_spec.block_spec
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    block_family = "mixllama" if (spec.mlp_type == "mixllama" or spec.mlp_type == "moe") else "transformer_block"

    attn_graph = build_dense_attention_graph("attention_engine", model_spec.model_dim, spec, **_attn_flags(model_spec))
    mamba_graph = build_mamba_graph("mamba_engine", model_spec.model_dim, spec)
    if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
        mlp_graph = build_mixllama_mlp_graph("mlp_moe", model_spec.model_dim, spec)
    else:
        mlp_graph = build_dense_mlp_graph("mlp_dense", model_spec.model_dim, spec)
    attn_block_graph = build_decoder_block_graph("decoder_block_attn", model_spec.model_dim, spec, attn_graph, mlp_graph)
    mamba_block_graph = build_decoder_block_graph("decoder_block_mamba", model_spec.model_dim, spec, mamba_graph, mlp_graph)
    graph.variant_library = {
        "attention": {"default": attn_graph},
        "mamba": {"default": mamba_graph},
        "mlp": {"default": mlp_graph},
        "attn_block": {"default": attn_block_graph},
        "mamba_block": {"default": mamba_block_graph},
        block_family: {"default": attn_block_graph},
    }

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 140)))
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

    for i in range(model_spec.num_layers):
        bname = f"block_{i}"
        out_aliases = ["x"]
        if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
            out_aliases.append("aux_loss")
        graph.add_node(NeuronInstance(link_variant_neuron(attn_block_graph, family=block_family, version="default", name=bname, input_aliases=["x"], output_aliases=out_aliases), instance_id=bname, position=(920 + i*220, 140)))
        graph.add_edge(Edge(id=f"e_{curr_out}_{bname}", src_node=curr_out, src_port=0, dst_node=bname, dst_port=0))
        curr_out = bname

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}), instance_id="final_norm", position=(1140 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(id="e_blocks_norm", src_node=curr_out, src_port=0, dst_node="final_norm", dst_port=0))

    head_id = "tied_lm_head" if model_spec.tie_embeddings else "lm_head"
    head_def = clone_neuron_def(BuiltinNeurons.tied_lm_head_module) if model_spec.tie_embeddings else clone_neuron_def(BuiltinNeurons.lm_head_module, config={"model_dim": model_spec.model_dim, "vocab_size": model_spec.vocab_size})
    graph.add_node(NeuronInstance(head_def, instance_id=head_id, position=(1360 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(id="e_norm_head", src_node="final_norm", src_port=0, dst_node=head_id, dst_port=0))
    if model_spec.tie_embeddings:
        graph.add_edge(Edge(id="e_tie", src_node="token_embed", src_port=1, dst_node=head_id, dst_port=1))
    logits_out = head_id
    if model_spec.logit_softcap > 0.0:
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.logit_softcap_module, config={"softcap": model_spec.logit_softcap}), instance_id="softcap", position=(1580 + model_spec.num_layers*220, 140)))
        graph.add_edge(Edge(id="e_head_softcap", src_node=head_id, src_port=0, dst_node="softcap", dst_port=0))
        logits_out = "softcap"

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="logits", dtype="tensor"), instance_id="logits_out", position=(1800 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(id="e_to_logits_out", src_node=logits_out, src_port=0, dst_node="logits_out", dst_port=0))
    graph.input_node_ids = ["tokens_in"]
    graph.output_node_ids = ["logits_out"]
    return graph


def build_dpo_root_graph(*, name: str = "model_root", model_spec: ModelSpec | None = None) -> NeuronGraph:
    """DPO root graph: paired policy + ref forwards with a pairwise preference loss.

    Interface (via ``dpo_dataset_source``): chosen + rejected token/target
    sequences with per-token loss masks. The policy and ``reference_forward``
    subgraphs share the same logits-emitting body; ``reference_forward`` loads
    its own checkpoint and is frozen.
    """
    if model_spec is None:
        model_spec = ModelSpec()
    model_spec.template.objective = "dpo"  # type: ignore[assignment]

    graph = NeuronGraph(
        name=name,
        training_method="torch",
        runtime="torch",
        torch_config={
            "device": "cuda",
            "amp_dtype": "float32",
            "template_spec": model_spec_to_dict(model_spec),
        },
    )

    stage = _logits_model_stage_graph("policy_stage", model_spec)
    graph.variant_library = deepcopy(stage.variant_library)
    stage.variant_library = {}

    dataset_def = clone_neuron_def(
        BuiltinNeurons.dpo_dataset_source_module,
        config={"dataset_names": [], "seq_len": 64},
    )
    graph.add_node(NeuronInstance(dataset_def, instance_id="dpo_dataset_source", position=(40, 200)))

    ref_cfg: dict[str, Any] = {
        "ref_graph_path": "",
        "ref_weights_path": "",
        "vocab_size": model_spec.vocab_size,
        "model_dim": model_spec.model_dim,
        "num_layers": model_spec.num_layers,
        "num_heads": model_spec.block_spec.num_heads,
    }
    if model_spec.finetune is not None and model_spec.finetune.ref_checkpoint:
        ref_cfg["ref_weights_path"] = model_spec.finetune.ref_checkpoint

    # Policy forwards
    graph.add_node(NeuronInstance(subgraph_neuron(stage, name="policy_chosen", input_aliases=["tokens"], output_aliases=["logits"]), instance_id="policy_chosen", position=(380, 80)))
    graph.add_node(NeuronInstance(subgraph_neuron(stage, name="policy_rejected", input_aliases=["tokens"], output_aliases=["logits"]), instance_id="policy_rejected", position=(380, 240)))
    # Reference forwards
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.reference_forward_module, config=dict(ref_cfg)), instance_id="ref_chosen", position=(380, 420)))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.reference_forward_module, config=dict(ref_cfg)), instance_id="ref_rejected", position=(380, 580)))

    graph.add_edge(Edge(id="e_chosen_tokens_policy", src_node="dpo_dataset_source", src_port=0, dst_node="policy_chosen", dst_port=0))
    graph.add_edge(Edge(id="e_rejected_tokens_policy", src_node="dpo_dataset_source", src_port=1, dst_node="policy_rejected", dst_port=0))
    graph.add_edge(Edge(id="e_chosen_tokens_ref", src_node="dpo_dataset_source", src_port=0, dst_node="ref_chosen", dst_port=0))
    graph.add_edge(Edge(id="e_rejected_tokens_ref", src_node="dpo_dataset_source", src_port=1, dst_node="ref_rejected", dst_port=0))

    # Sequence logp nodes.
    for role, logits_src, tgt_port, mask_port, yoff in (
        ("pc", "policy_chosen", 2, 4, 80),
        ("pr", "policy_rejected", 3, 5, 240),
        ("rc", "ref_chosen", 2, 4, 420),
        ("rr", "ref_rejected", 3, 5, 580),
    ):
        node_id = f"logp_{role}"
        graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.sequence_logp_module), instance_id=node_id, position=(700, yoff)))
        graph.add_edge(Edge(id=f"e_logits_{node_id}", src_node=logits_src, src_port=0, dst_node=node_id, dst_port=0))
        graph.add_edge(Edge(id=f"e_tgt_{node_id}", src_node="dpo_dataset_source", src_port=tgt_port, dst_node=node_id, dst_port=1))
        graph.add_edge(Edge(id=f"e_mask_{node_id}", src_node="dpo_dataset_source", src_port=mask_port, dst_node=node_id, dst_port=2))

    dpo_cfg: dict[str, Any] = {"beta": 0.1, "label_smoothing": 0.0, "loss_type": "sigmoid"}
    if model_spec.finetune is not None:
        dpo_cfg["beta"] = model_spec.finetune.beta
        dpo_cfg["loss_type"] = model_spec.finetune.dpo_loss_type
        dpo_cfg["label_smoothing"] = model_spec.finetune.dpo_label_smoothing
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.dpo_pairwise_loss_module, config=dpo_cfg), instance_id="dpo_loss", position=(1020, 260)))
    graph.add_edge(Edge(id="e_pc_loss", src_node="logp_pc", src_port=0, dst_node="dpo_loss", dst_port=0))
    graph.add_edge(Edge(id="e_pr_loss", src_node="logp_pr", src_port=0, dst_node="dpo_loss", dst_port=1))
    graph.add_edge(Edge(id="e_rc_loss", src_node="logp_rc", src_port=0, dst_node="dpo_loss", dst_port=2))
    graph.add_edge(Edge(id="e_rr_loss", src_node="logp_rr", src_port=0, dst_node="dpo_loss", dst_port=3))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(1240, 260)))
    graph.add_edge(Edge(id="e_dpo_loss_out", src_node="dpo_loss", src_port=0, dst_node="loss_out", dst_port=0))
    graph.input_node_ids = ["dpo_dataset_source"]
    graph.output_node_ids = ["loss_out"]

    if model_spec.finetune is not None:
        graph.torch_config["finetune_spec"] = {
            "objective": "dpo",
            "base_checkpoint": model_spec.finetune.base_checkpoint,
            "ref_checkpoint": model_spec.finetune.ref_checkpoint,
            "reward_checkpoint": model_spec.finetune.reward_checkpoint,
            "beta": model_spec.finetune.beta,
            "dpo_loss_type": model_spec.finetune.dpo_loss_type,
            "adapter_only_save": model_spec.finetune.adapter_only_save,
        }
    return graph


def build_reward_model_root_graph(*, name: str = "model_root", model_spec: ModelSpec | None = None) -> NeuronGraph:
    """Reward-model root graph: AR body + reward_head + preference_bce_loss.

    Consumes paired (chosen, rejected) token sequences from ``dpo_dataset_source``
    and outputs a Bradley-Terry preference loss.
    """
    if model_spec is None:
        model_spec = ModelSpec()
    model_spec.template.objective = "reward_model"  # type: ignore[assignment]

    graph = NeuronGraph(
        name=name,
        training_method="torch",
        runtime="torch",
        torch_config={
            "device": "cuda",
            "amp_dtype": "float32",
            "template_spec": model_spec_to_dict(model_spec),
        },
    )

    body_stage = _body_only_stage_graph("body_stage", model_spec)
    graph.variant_library = deepcopy(body_stage.variant_library)
    body_stage.variant_library = {}

    dataset_def = clone_neuron_def(
        BuiltinNeurons.dpo_dataset_source_module,
        config={"dataset_names": [], "seq_len": 64},
    )
    graph.add_node(NeuronInstance(dataset_def, instance_id="dpo_dataset_source", position=(40, 200)))
    graph.add_node(NeuronInstance(subgraph_neuron(body_stage, name="body_chosen", input_aliases=["tokens"], output_aliases=["hidden"]), instance_id="body_chosen", position=(380, 80)))
    graph.add_node(NeuronInstance(subgraph_neuron(body_stage, name="body_rejected", input_aliases=["tokens"], output_aliases=["hidden"]), instance_id="body_rejected", position=(380, 320)))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.reward_head_module, config={"model_dim": model_spec.model_dim, "pool": "last"}), instance_id="reward_chosen", position=(700, 80)))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.reward_head_module, config={"model_dim": model_spec.model_dim, "pool": "last"}), instance_id="reward_rejected", position=(700, 320)))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.preference_bce_loss_module), instance_id="pref_loss", position=(1020, 200)))
    graph.add_edge(Edge(id="e_chosen_body", src_node="dpo_dataset_source", src_port=0, dst_node="body_chosen", dst_port=0))
    graph.add_edge(Edge(id="e_rejected_body", src_node="dpo_dataset_source", src_port=1, dst_node="body_rejected", dst_port=0))
    graph.add_edge(Edge(id="e_chosen_head", src_node="body_chosen", src_port=0, dst_node="reward_chosen", dst_port=0))
    graph.add_edge(Edge(id="e_rejected_head", src_node="body_rejected", src_port=0, dst_node="reward_rejected", dst_port=0))
    graph.add_edge(Edge(id="e_chosen_loss", src_node="reward_chosen", src_port=0, dst_node="pref_loss", dst_port=0))
    graph.add_edge(Edge(id="e_rejected_loss", src_node="reward_rejected", src_port=0, dst_node="pref_loss", dst_port=1))
    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(1240, 200)))
    graph.add_edge(Edge(id="e_pref_loss_out", src_node="pref_loss", src_port=0, dst_node="loss_out", dst_port=0))
    graph.input_node_ids = ["dpo_dataset_source"]
    graph.output_node_ids = ["loss_out"]
    return graph


def _body_only_stage_graph(name: str, model_spec: ModelSpec) -> NeuronGraph:
    """Like ``_logits_model_stage_graph`` but stops at the post-norm hidden states.

    Used by reward-model and PPO value-head heads that read hidden states
    rather than logits.
    """
    spec = model_spec.block_spec
    graph = NeuronGraph(name=name, training_method="torch", runtime="torch")
    block_family = "mixllama" if (spec.mlp_type == "mixllama" or spec.mlp_type == "moe") else "transformer_block"

    attn_graph = build_dense_attention_graph("attention_engine", model_spec.model_dim, spec, **_attn_flags(model_spec))
    mamba_graph = build_mamba_graph("mamba_engine", model_spec.model_dim, spec)
    if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
        mlp_graph = build_mixllama_mlp_graph("mlp_moe", model_spec.model_dim, spec)
    else:
        mlp_graph = build_dense_mlp_graph("mlp_dense", model_spec.model_dim, spec)
    attn_block_graph = build_decoder_block_graph("decoder_block_attn", model_spec.model_dim, spec, attn_graph, mlp_graph)
    mamba_block_graph = build_decoder_block_graph("decoder_block_mamba", model_spec.model_dim, spec, mamba_graph, mlp_graph)
    graph.variant_library = {
        "attention": {"default": attn_graph},
        "mamba": {"default": mamba_graph},
        "mlp": {"default": mlp_graph},
        "attn_block": {"default": attn_block_graph},
        "mamba_block": {"default": mamba_block_graph},
        block_family: {"default": attn_block_graph},
    }

    graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 140)))
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

    for i in range(model_spec.num_layers):
        bname = f"block_{i}"
        out_aliases = ["x"]
        if spec.mlp_type == "mixllama" or spec.mlp_type == "moe":
            out_aliases.append("aux_loss")
        graph.add_node(NeuronInstance(link_variant_neuron(attn_block_graph, family=block_family, version="default", name=bname, input_aliases=["x"], output_aliases=out_aliases), instance_id=bname, position=(920 + i*220, 140)))
        graph.add_edge(Edge(id=f"e_{curr_out}_{bname}", src_node=curr_out, src_port=0, dst_node=bname, dst_port=0))
        curr_out = bname

    norm_module = BuiltinNeurons.rms_norm_module if spec.norm_type == "rmsnorm" else BuiltinNeurons.layer_norm_module
    graph.add_node(NeuronInstance(clone_neuron_def(norm_module, config={"model_dim": model_spec.model_dim}), instance_id="final_norm", position=(1140 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(id="e_blocks_norm", src_node=curr_out, src_port=0, dst_node="final_norm", dst_port=0))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="hidden", dtype="tensor"), instance_id="hidden_out", position=(1360 + model_spec.num_layers*220, 140)))
    graph.add_edge(Edge(id="e_to_hidden_out", src_node="final_norm", src_port=0, dst_node="hidden_out", dst_port=0))
    graph.input_node_ids = ["tokens_in"]
    graph.output_node_ids = ["hidden_out"]
    return graph


def build_ppo_root_graph(*, name: str = "model_root", model_spec: ModelSpec | None = None) -> NeuronGraph:
    """PPO root graph skeleton.

    A full PPO run is orchestrated by ``PPOTrainer`` (``neuralfn.torch_backend``);
    this graph represents one inner optimization pass over a rollout buffer
    populated by the trainer. The buffer source emits
    ``(tokens, targets, loss_mask, logp_old, value_old, advantages, returns)``
    and a frozen reference forward plus a frozen reward forward supply the
    KL-shaping terms.
    """
    if model_spec is None:
        model_spec = ModelSpec()
    model_spec.template.objective = "ppo"  # type: ignore[assignment]

    graph = NeuronGraph(
        name=name,
        training_method="torch",
        runtime="torch",
        torch_config={
            "device": "cuda",
            "amp_dtype": "float32",
            "template_spec": model_spec_to_dict(model_spec),
        },
    )

    logits_stage = _logits_model_stage_graph("policy_stage", model_spec)
    body_stage = _body_only_stage_graph("body_stage", model_spec)
    graph.variant_library = deepcopy(logits_stage.variant_library)
    logits_stage.variant_library = {}
    body_stage.variant_library = {}

    rollout_def = clone_neuron_def(
        BuiltinNeurons.ppo_rollout_source_module,
        config={"seq_len": 64, "rollout_length": model_spec.finetune.rollout_length if model_spec.finetune else 64},
    )
    graph.add_node(NeuronInstance(rollout_def, instance_id="ppo_rollout_source", position=(40, 200)))

    # Policy logits + sequence_logp_new.
    graph.add_node(NeuronInstance(subgraph_neuron(logits_stage, name="policy", input_aliases=["tokens"], output_aliases=["logits"]), instance_id="policy", position=(340, 80)))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.sequence_logp_module), instance_id="logp_new", position=(660, 80)))
    graph.add_edge(Edge(id="e_rollout_policy", src_node="ppo_rollout_source", src_port=0, dst_node="policy", dst_port=0))
    graph.add_edge(Edge(id="e_policy_logp_logits", src_node="policy", src_port=0, dst_node="logp_new", dst_port=0))
    graph.add_edge(Edge(id="e_policy_logp_tgt", src_node="ppo_rollout_source", src_port=1, dst_node="logp_new", dst_port=1))
    graph.add_edge(Edge(id="e_policy_logp_mask", src_node="ppo_rollout_source", src_port=2, dst_node="logp_new", dst_port=2))

    # Value head on policy body hidden states.
    graph.add_node(NeuronInstance(subgraph_neuron(body_stage, name="policy_body", input_aliases=["tokens"], output_aliases=["hidden"]), instance_id="policy_body", position=(340, 280)))
    graph.add_edge(Edge(id="e_rollout_body", src_node="ppo_rollout_source", src_port=0, dst_node="policy_body", dst_port=0))
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.value_head_module, config={"model_dim": model_spec.model_dim}), instance_id="value_new", position=(660, 280)))
    graph.add_edge(Edge(id="e_body_value", src_node="policy_body", src_port=0, dst_node="value_new", dst_port=0))

    # PPO clipped loss.
    ppo_cfg: dict[str, Any] = {"clip_range": 0.2, "vf_coef": 0.5, "ent_coef": 0.0}
    if model_spec.finetune is not None:
        ppo_cfg["clip_range"] = model_spec.finetune.ppo_clip
        ppo_cfg["vf_coef"] = model_spec.finetune.ppo_vf_coef
        ppo_cfg["ent_coef"] = model_spec.finetune.ppo_ent_coef
    graph.add_node(NeuronInstance(clone_neuron_def(BuiltinNeurons.ppo_clipped_loss_module, config=ppo_cfg), instance_id="ppo_loss", position=(980, 200)))
    graph.add_edge(Edge(id="e_logp_new_loss", src_node="logp_new", src_port=0, dst_node="ppo_loss", dst_port=0))
    graph.add_edge(Edge(id="e_logp_old_loss", src_node="ppo_rollout_source", src_port=3, dst_node="ppo_loss", dst_port=1))
    graph.add_edge(Edge(id="e_adv_loss", src_node="ppo_rollout_source", src_port=5, dst_node="ppo_loss", dst_port=2))
    graph.add_edge(Edge(id="e_value_new_loss", src_node="value_new", src_port=0, dst_node="ppo_loss", dst_port=3))
    graph.add_edge(Edge(id="e_value_old_loss", src_node="ppo_rollout_source", src_port=4, dst_node="ppo_loss", dst_port=4))
    graph.add_edge(Edge(id="e_returns_loss", src_node="ppo_rollout_source", src_port=6, dst_node="ppo_loss", dst_port=5))

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(1240, 200)))
    graph.add_edge(Edge(id="e_ppo_loss_out", src_node="ppo_loss", src_port=2, dst_node="loss_out", dst_port=0))
    graph.input_node_ids = ["ppo_rollout_source"]
    graph.output_node_ids = ["loss_out"]

    if model_spec.finetune is not None:
        graph.torch_config["finetune_spec"] = {
            "objective": "ppo",
            "base_checkpoint": model_spec.finetune.base_checkpoint,
            "ref_checkpoint": model_spec.finetune.ref_checkpoint,
            "reward_checkpoint": model_spec.finetune.reward_checkpoint,
            "kl_coef": model_spec.finetune.kl_coef,
            "ppo_clip": model_spec.finetune.ppo_clip,
            "rollout_length": model_spec.finetune.rollout_length,
            "ppo_epochs_per_rollout": model_spec.finetune.ppo_epochs_per_rollout,
            "gae_gamma": model_spec.finetune.gae_gamma,
            "gae_lambda": model_spec.finetune.gae_lambda,
            "adapter_only_save": model_spec.finetune.adapter_only_save,
        }
    return graph


def build_gpt_root_graph(*, name: str = "model_root", model_spec: ModelSpec | None = None) -> NeuronGraph:
    if model_spec is None:
        model_spec = ModelSpec()

    # Fine-tuning objectives dispatch into dedicated root-graph builders so
    # the CLI can call ``build_gpt_root_graph`` uniformly while the graph
    # topology (data source, loss composition, frozen sibling nets) follows
    # the objective.
    if model_spec.template.objective == "sft":
        return build_sft_root_graph(name=name, model_spec=model_spec)
    if model_spec.template.objective == "dpo":
        return build_dpo_root_graph(name=name, model_spec=model_spec)
    if model_spec.template.objective == "ppo":
        return build_ppo_root_graph(name=name, model_spec=model_spec)
    if model_spec.template.objective == "reward_model":
        return build_reward_model_root_graph(name=name, model_spec=model_spec)

    graph = NeuronGraph(
        name=name,
        training_method="torch",
        runtime="torch",
        torch_config={
            "device": "cuda",
            "amp_dtype": "float32",
            "template_spec": model_spec_to_dict(model_spec),
        },
    )

    if model_spec.template.objective == "seq2seq":
        stage = build_seq2seq_model_stage_graph("model_stage", model_spec)
        graph.variant_library = deepcopy(stage.variant_library)
        stage.variant_library = {}

        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="enc_tokens", dtype="tokens"), instance_id="enc_tokens_in", position=(40, 120)))
        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="dec_tokens", dtype="tokens"), instance_id="dec_tokens_in", position=(40, 240)))
        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 360)))

        graph.add_node(NeuronInstance(subgraph_neuron(stage, name="model", input_aliases=["enc_tokens", "dec_tokens", "targets"], output_aliases=["loss"]), instance_id="model", position=(280, 180)))

        graph.add_edge(Edge(src_node="enc_tokens_in", src_port=0, dst_node="model", dst_port=0))
        graph.add_edge(Edge(src_node="dec_tokens_in", src_port=0, dst_node="model", dst_port=1))
        graph.add_edge(Edge(src_node="targets_in", src_port=0, dst_node="model", dst_port=2))

        graph.input_node_ids = ["enc_tokens_in", "dec_tokens_in", "targets_in"]
    elif model_spec.template.objective == "diffusion":
        stage = build_diffusion_model_stage_graph("model_stage", model_spec)
        graph.variant_library = deepcopy(stage.variant_library)
        stage.variant_library = {}

        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 120)))

        graph.add_node(NeuronInstance(subgraph_neuron(stage, name="model", input_aliases=["tokens"], output_aliases=["loss"]), instance_id="model", position=(280, 180)))

        graph.add_edge(Edge(src_node="tokens_in", src_port=0, dst_node="model", dst_port=0))
        graph.input_node_ids = ["tokens_in"]
    elif model_spec.template.objective == "jepa":
        stage = build_jepa_model_stage_graph("model_stage", model_spec)
        graph.variant_library = deepcopy(stage.variant_library)
        stage.variant_library = {}

        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 180)))
        graph.add_node(NeuronInstance(subgraph_neuron(stage, name="model", input_aliases=["tokens"], output_aliases=["loss"]), instance_id="model", position=(280, 180)))
        graph.add_edge(Edge(id="e_tokens_model", src_node="tokens_in", src_port=0, dst_node="model", dst_port=0))
        graph.input_node_ids = ["tokens_in"]
    elif model_spec.template.objective == "ar_jepa":
        stage = build_ar_jepa_model_stage_graph("model_stage", model_spec)
        graph.variant_library = deepcopy(stage.variant_library)
        stage.variant_library = {}

        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 120)))
        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 300)))
        graph.add_node(NeuronInstance(subgraph_neuron(stage, name="model", input_aliases=["tokens", "targets"], output_aliases=["loss"]), instance_id="model", position=(280, 180)))
        graph.add_edge(Edge(id="e_tokens_model", src_node="tokens_in", src_port=0, dst_node="model", dst_port=0))
        graph.add_edge(Edge(id="e_targets_model", src_node="targets_in", src_port=0, dst_node="model", dst_port=1))
        graph.input_node_ids = ["tokens_in", "targets_in"]
    elif model_spec.template.objective == "jepa_semantic":
        stage = build_jepa_semantic_model_stage_graph("model_stage", model_spec)
        graph.variant_library = deepcopy(stage.variant_library)
        stage.variant_library = {}
        semantic_input_aliases = ["tokens", "targets", "sem_targets"]
        semantic_source_def = clone_neuron_def(
            BuiltinNeurons.semantic_data_source_module,
            config={
                "seq_len": model_spec.semantic_dim,
                "semantic_vocab_ref": model_spec.semantic_vocab_ref,
                "emit_router_vecs": model_spec.experimental_semantic_router_vecs
                and model_spec.template.objective != "semantic_moe_jepa_evo",
                "router_vec_dim": max(model_spec.semantic_dim - 1, 0),
            },
        )
        if model_spec.experimental_semantic_router_vecs and model_spec.template.objective != "semantic_moe_jepa_evo":
            semantic_source_def.output_ports = [
                Port("sem_targets", range=(-100, 65535), precision=1.0, dtype="tokens"),
                Port("semantic_router_vecs", range=(0, 1), precision=None, dtype="tensor"),
            ]
            semantic_input_aliases.append("semantic_router_vecs")

        dataset_def = clone_neuron_def(BuiltinNeurons.dataset_source_module, config={"dataset_names": [], "seq_len": 64})
        dataset_def.output_ports = [
            Port("tokens", range=(0, 65535), precision=1.0, dtype="tokens"),
            Port("targets", range=(0, 65535), precision=1.0, dtype="tokens"),
        ]
        graph.add_node(NeuronInstance(
            dataset_def,
            instance_id="dataset_source",
            position=(40, 120),
        ))
        graph.add_node(NeuronInstance(
            semantic_source_def,
            instance_id="semantic_data_source",
            position=(40, 300),
        ))
        graph.add_node(
            NeuronInstance(
                subgraph_neuron(stage, name="model", input_aliases=semantic_input_aliases, output_aliases=["loss"]),
                instance_id="model",
                position=(380, 180),
            )
        )
        graph.add_edge(Edge(id="e_tokens_model", src_node="dataset_source", src_port=0, dst_node="model", dst_port=0))
        graph.add_edge(Edge(id="e_targets_model", src_node="dataset_source", src_port=1, dst_node="model", dst_port=1))
        graph.add_edge(Edge(id="e_semds_model", src_node="semantic_data_source", src_port=0, dst_node="model", dst_port=2))
        if model_spec.experimental_semantic_router_vecs:
            graph.add_edge(Edge(id="e_semvec_model", src_node="semantic_data_source", src_port=1, dst_node="model", dst_port=3))
        graph.input_node_ids = ["dataset_source", "semantic_data_source"]
    elif model_spec.template.objective in {"semantic_router_jepa", "semantic_dense_jepa_evo", "semantic_moe_jepa_evo"}:
        if model_spec.template.objective == "semantic_moe_jepa_evo":
            stage = build_semantic_moe_jepa_evo_model_stage_graph("model_stage", model_spec)
        elif model_spec.template.objective == "semantic_dense_jepa_evo":
            stage = build_semantic_dense_jepa_evo_model_stage_graph("model_stage", model_spec)
        else:
            stage = build_semantic_router_jepa_model_stage_graph("model_stage", model_spec)
        graph.variant_library = deepcopy(stage.variant_library)
        stage.variant_library = {}
        semantic_input_aliases = ["tokens", "targets", "sem_targets"]
        emit_router_vecs = (
            model_spec.experimental_semantic_router_vecs
            and model_spec.template.objective == "semantic_router_jepa"
        )
        semantic_source_def = clone_neuron_def(
            BuiltinNeurons.semantic_data_source_module,
            config={
                "seq_len": model_spec.semantic_dim,
                "semantic_vocab_ref": model_spec.semantic_vocab_ref,
                "emit_router_vecs": emit_router_vecs,
                "router_vec_dim": max(model_spec.semantic_dim - 1, 0),
            },
        )
        if emit_router_vecs:
            semantic_source_def.output_ports = [
                Port("sem_targets", range=(-100, 65535), precision=1.0, dtype="tokens"),
                Port("semantic_router_vecs", range=(0, 1), precision=None, dtype="tensor"),
            ]
            semantic_input_aliases.append("semantic_router_vecs")

        dataset_def = clone_neuron_def(BuiltinNeurons.dataset_source_module, config={"dataset_names": [], "seq_len": 64})
        dataset_def.output_ports = [
            Port("tokens", range=(0, 65535), precision=1.0, dtype="tokens"),
            Port("targets", range=(0, 65535), precision=1.0, dtype="tokens"),
        ]
        graph.add_node(NeuronInstance(dataset_def, instance_id="dataset_source", position=(40, 120)))
        graph.add_node(NeuronInstance(semantic_source_def, instance_id="semantic_data_source", position=(40, 300)))
        graph.add_node(
            NeuronInstance(
                subgraph_neuron(stage, name="model", input_aliases=semantic_input_aliases, output_aliases=["loss"]),
                instance_id="model",
                position=(380, 180),
            )
        )
        graph.add_edge(Edge(id="e_tokens_model", src_node="dataset_source", src_port=0, dst_node="model", dst_port=0))
        graph.add_edge(Edge(id="e_targets_model", src_node="dataset_source", src_port=1, dst_node="model", dst_port=1))
        graph.add_edge(Edge(id="e_semds_model", src_node="semantic_data_source", src_port=0, dst_node="model", dst_port=2))
        if emit_router_vecs:
            graph.add_edge(Edge(id="e_semvec_model", src_node="semantic_data_source", src_port=1, dst_node="model", dst_port=3))
        graph.input_node_ids = ["dataset_source", "semantic_data_source"]
    elif model_spec.template.objective == "semantic_router":
        stage = build_semantic_router_model_stage_graph("model_stage", model_spec)
        graph.variant_library = deepcopy(stage.variant_library)
        stage.variant_library = {}
        semantic_input_aliases = ["tokens", "targets", "sem_targets"]
        semantic_source_def = clone_neuron_def(
            BuiltinNeurons.semantic_data_source_module,
            config={
                "seq_len": model_spec.semantic_dim,
                "semantic_vocab_ref": model_spec.semantic_vocab_ref,
                "emit_router_vecs": model_spec.experimental_semantic_router_vecs,
                "router_vec_dim": max(model_spec.semantic_dim - 1, 0),
            },
        )
        if model_spec.experimental_semantic_router_vecs:
            semantic_source_def.output_ports = [
                Port("sem_targets", range=(-100, 65535), precision=1.0, dtype="tokens"),
                Port("semantic_router_vecs", range=(0, 1), precision=None, dtype="tensor"),
            ]
            semantic_input_aliases.append("semantic_router_vecs")

        dataset_def = clone_neuron_def(BuiltinNeurons.dataset_source_module, config={"dataset_names": [], "seq_len": 64})
        dataset_def.output_ports = [
            Port("tokens", range=(0, 65535), precision=1.0, dtype="tokens"),
            Port("targets", range=(0, 65535), precision=1.0, dtype="tokens"),
        ]
        graph.add_node(NeuronInstance(dataset_def, instance_id="dataset_source", position=(40, 120)))
        graph.add_node(
            NeuronInstance(
                semantic_source_def,
                instance_id="semantic_data_source",
                position=(40, 300),
            )
        )
        graph.add_node(
            NeuronInstance(
                subgraph_neuron(stage, name="model", input_aliases=semantic_input_aliases, output_aliases=["loss"]),
                instance_id="model",
                position=(380, 180),
            )
        )
        graph.add_edge(Edge(id="e_tokens_model", src_node="dataset_source", src_port=0, dst_node="model", dst_port=0))
        graph.add_edge(Edge(id="e_targets_model", src_node="dataset_source", src_port=1, dst_node="model", dst_port=1))
        graph.add_edge(Edge(id="e_semds_model", src_node="semantic_data_source", src_port=0, dst_node="model", dst_port=2))
        if model_spec.experimental_semantic_router_vecs:
            graph.add_edge(Edge(id="e_semvec_model", src_node="semantic_data_source", src_port=1, dst_node="model", dst_port=3))
        graph.input_node_ids = ["dataset_source", "semantic_data_source"]
    elif model_spec.template.backbone == "hnet":
        stage = build_hnet_model_stage_graph("model_stage", model_spec)
        graph.variant_library = deepcopy(stage.variant_library)
        stage.variant_library = {}

        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 120)))
        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 300)))
        graph.add_node(NeuronInstance(subgraph_neuron(stage, name="model", input_aliases=["tokens", "targets"], output_aliases=["loss"]), instance_id="model", position=(280, 180)))
        graph.add_edge(Edge(id="e_tokens_model", src_node="tokens_in", src_port=0, dst_node="model", dst_port=0))
        graph.add_edge(Edge(id="e_targets_model", src_node="targets_in", src_port=0, dst_node="model", dst_port=1))
        graph.input_node_ids = ["tokens_in", "targets_in"]
    elif model_spec.template.backbone == "universal":
        stage = build_universal_model_stage_graph("model_stage", model_spec)
        graph.variant_library = deepcopy(stage.variant_library)
        stage.variant_library = {}

        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 120)))
        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 300)))
        graph.add_node(NeuronInstance(subgraph_neuron(stage, name="model", input_aliases=["tokens", "targets"], output_aliases=["loss"]), instance_id="model", position=(280, 180)))
        graph.add_edge(Edge(id="e_tokens_model", src_node="tokens_in", src_port=0, dst_node="model", dst_port=0))
        graph.add_edge(Edge(id="e_targets_model", src_node="targets_in", src_port=0, dst_node="model", dst_port=1))
        graph.input_node_ids = ["tokens_in", "targets_in"]
    else:
        stage = build_model_stage_graph("model_stage", model_spec)
        graph.variant_library = deepcopy(stage.variant_library)
        stage.variant_library = {} # detach so it's not nested infinitely

        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="tokens", dtype="tokens"), instance_id="tokens_in", position=(40, 120)))
        graph.add_node(NeuronInstance(make_terminal_def(role="input", port_name="targets", dtype="tokens"), instance_id="targets_in", position=(40, 300)))

        graph.add_node(NeuronInstance(subgraph_neuron(stage, name="model", input_aliases=["tokens", "targets"], output_aliases=["loss"]), instance_id="model", position=(280, 180)))

        graph.add_edge(Edge(id="e_tokens_model", src_node="tokens_in", src_port=0, dst_node="model", dst_port=0))
        graph.add_edge(Edge(id="e_targets_model", src_node="targets_in", src_port=0, dst_node="model", dst_port=1))

        graph.input_node_ids = ["tokens_in", "targets_in"]

    graph.add_node(NeuronInstance(make_terminal_def(role="output", port_name="loss", dtype="loss"), instance_id="loss_out", position=(560, 180)))
    graph.add_edge(Edge(id="e_model_out", src_node="model", src_port=0, dst_node="loss_out", dst_port=0))

    graph.output_node_ids = ["loss_out"]
    return graph

def build_gpt_template_payload(name: str, config: dict[str, Any]) -> dict[str, Any]:
    spec = build_model_spec_from_config(config, preview_defaults=True)
    graph = build_gpt_root_graph(name=name, model_spec=spec)

    node_def = graph.nodes["model"].neuron_def

    extra_nodes: list[dict[str, Any]] = []
    for nid, node in graph.nodes.items():
        if nid == "model":
            continue
        if node.neuron_def.kind == "module" or nid in graph.input_node_ids:
            extra_nodes.append(node.to_dict())

    extra_edges: list[dict[str, Any]] = []
    for eid, edge in graph.edges.items():
        extra_edges.append(edge.to_dict())

    return {
        "variant_library": {
            family: {version: vg.to_dict() for version, vg in versions.items()}
            for family, versions in graph.variant_library.items()
        },
        "graph_settings": {
            "training_method": graph.training_method,
            "runtime": graph.runtime,
            "torch_config": graph.torch_config,
        },
        "node_def": node_def.to_dict(),
        "extra_nodes": extra_nodes,
        "extra_edges": extra_edges,
    }
