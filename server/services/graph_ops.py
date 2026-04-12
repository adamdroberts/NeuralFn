from __future__ import annotations

from copy import deepcopy
from typing import Any
import uuid

import numpy as np
import torch

from neuralfn.builtins import BuiltinNeurons
from neuralfn.graph import Edge, NeuronGraph, NeuronInstance
from neuralfn.neuron import NeuronDef, module_neuron, neuron_from_source, subgraph_neuron
from neuralfn.port import Port
from neuralfn.surrogate import probe_neuron
from neuralfn.torch_backend import CompiledTorchGraph, TorchTrainer
from neuralfn.torch_templates import build_gpt_root_graph, build_gpt_template_payload, build_model_spec_from_config

from ..dataset_manager import (
    download_hf_dataset,
    load_dataset_bytes,
    load_dataset_tokens,
    raw_text_encoding_name_for_template_spec,
    validate_cached_tokenizer_contract,
)
from ..models import DownloadDatasetRequest, EdgeModel, EdgeUpdateModel, ExecuteRequest, GPTTemplateRequest, LoadDatasetRequest, NeuronDefModel, NodeModel


class GraphOperationError(ValueError):
    pass


def port_from_model(pm: Any) -> Port:
    return Port(name=pm.name, range=tuple(pm.range), precision=pm.precision, dtype=pm.dtype)


def neuron_def_from_model(nm: Any) -> NeuronDef:
    if getattr(nm, "kind", "function") == "subgraph":
        if nm.subgraph is None:
            raise GraphOperationError(f"Subgraph neuron '{nm.name}' is missing nested graph data")
        nested = NeuronGraph.from_dict(nm.subgraph.model_dump())
        return subgraph_neuron(
            nested,
            name=nm.name,
            input_aliases=list(getattr(nm, "input_aliases", [])) or None,
            output_aliases=list(getattr(nm, "output_aliases", [])) or None,
            variant_ref=(
                getattr(nm.variant_ref, "model_dump", lambda: None)()
                if getattr(nm, "variant_ref", None) is not None
                else None
            ),
            neuron_id=nm.id or None,
        )

    if getattr(nm, "kind", "function") == "module":
        return module_neuron(
            name=nm.name,
            module_type=getattr(nm, "module_type", "") or nm.name,
            input_ports=[port_from_model(p) for p in nm.input_ports],
            output_ports=[port_from_model(p) for p in nm.output_ports],
            module_config=dict(getattr(nm, "module_config", {}) or {}),
            module_state=getattr(nm, "module_state", ""),
            neuron_id=nm.id or None,
        )

    input_ports = [port_from_model(p) for p in nm.input_ports]
    output_ports = [port_from_model(p) for p in nm.output_ports]
    return neuron_from_source(
        nm.source_code,
        nm.name,
        input_ports,
        output_ports,
        neuron_id=nm.id or None,
    )


def clone_neuron_def(ndef: NeuronDef) -> NeuronDef:
    return NeuronDef.from_dict(deepcopy(ndef.to_dict()))


def summarize_graph_for_agent(graph: NeuronGraph) -> dict[str, Any]:
    gd = graph.to_dict()
    nodes_summary = {}
    for nid, node in gd.get("nodes", {}).items():
        ndef = node.get("neuron_def", {})
        nodes_summary[nid] = {"name": ndef.get("name", ""), "kind": ndef.get("kind", "function")}
    variant_families = {
        family: list(versions.keys())
        for family, versions in gd.get("variant_library", {}).items()
    }
    return {
        "name": gd.get("name", ""),
        "training_method": gd.get("training_method", ""),
        "runtime": gd.get("runtime", ""),
        "nodes": nodes_summary,
        "edge_count": len(gd.get("edges", {})),
        "input_node_ids": gd.get("input_node_ids", []),
        "output_node_ids": gd.get("output_node_ids", []),
        "variant_families": variant_families,
    }


def build_template_payload(body: GPTTemplateRequest) -> dict[str, Any]:
    return build_gpt_template_payload(name=body.name, config=dict(body.config or {}))


def apply_gpt_template(body: GPTTemplateRequest) -> NeuronGraph:
    cfg = dict(body.config or {})
    spec = build_model_spec_from_config(cfg)
    return build_gpt_root_graph(name=body.name, model_spec=spec)


def add_node_to_graph(graph: NeuronGraph, body: NodeModel) -> dict[str, Any]:
    ndef = neuron_def_from_model(body.neuron_def)
    inst = NeuronInstance(
        neuron_def=ndef,
        instance_id=body.instance_id or uuid.uuid4().hex[:12],
        position=tuple(body.position),
    )
    graph.add_node(inst)
    return inst.to_dict()


def update_node_in_graph(graph: NeuronGraph, node_id: str, body: NeuronDefModel) -> dict[str, Any]:
    if node_id not in graph.nodes:
        raise GraphOperationError("Node not found")
    graph.nodes[node_id].neuron_def = neuron_def_from_model(body)
    return graph.nodes[node_id].to_dict()


def delete_node_from_graph(graph: NeuronGraph, node_id: str) -> None:
    if node_id not in graph.nodes:
        raise GraphOperationError("Node not found")
    graph.remove_node(node_id)


def add_edge_to_graph(graph: NeuronGraph, body: EdgeModel) -> dict[str, Any]:
    edge = Edge(
        id=body.id or uuid.uuid4().hex[:12],
        src_node=body.src_node,
        src_port=body.src_port,
        dst_node=body.dst_node,
        dst_port=body.dst_port,
        weight=body.weight,
        bias=body.bias,
    )
    try:
        graph.add_edge(edge)
    except ValueError as exc:
        raise GraphOperationError(str(exc)) from exc
    return edge.to_dict()


def update_edge_in_graph(graph: NeuronGraph, edge_id: str, body: EdgeUpdateModel) -> dict[str, Any]:
    if edge_id not in graph.edges:
        raise GraphOperationError("Edge not found")
    edge = graph.edges[edge_id]
    if body.weight is not None:
        edge.weight = body.weight
    if body.bias is not None:
        edge.bias = body.bias
    return edge.to_dict()


def delete_edge_from_graph(graph: NeuronGraph, edge_id: str) -> None:
    if edge_id not in graph.edges:
        raise GraphOperationError("Edge not found")
    graph.remove_edge(edge_id)


def set_graph_io(graph: NeuronGraph, input_ids: list[str], output_ids: list[str]) -> dict[str, Any]:
    graph.input_node_ids = input_ids
    graph.output_node_ids = output_ids
    return {"input_node_ids": input_ids, "output_node_ids": output_ids}


def execute_graph(graph: NeuronGraph, body: ExecuteRequest) -> dict[str, Any]:
    inputs = {key: tuple(value) for key, value in body.inputs.items()}
    try:
        outputs = graph.execute(inputs)
    except Exception as exc:
        raise GraphOperationError(str(exc)) from exc
    return {key: list(value) for key, value in outputs.items()}


def execute_trace(graph: NeuronGraph, body: ExecuteRequest) -> dict[str, Any]:
    inputs = {key: tuple(value) for key, value in body.inputs.items()}
    try:
        outputs = graph.execute_trace(inputs)
    except Exception as exc:
        raise GraphOperationError(str(exc)) from exc
    return {key: list(value) for key, value in outputs.items()}


def _preview_tensor(value: Any, limit: int = 8) -> tuple[list[Any], list[int]]:
    if not hasattr(value, "shape"):
        return [], []
    tensor = value.detach().cpu()
    preview_tensor = tensor[0] if tensor.ndim > 1 else tensor
    preview_shape = list(preview_tensor.shape)
    flat = preview_tensor.reshape(-1)[:limit]
    if tensor.dtype.is_floating_point:
        preview = [round(float(item), 4) for item in flat.tolist()]
    else:
        preview = [int(item) for item in flat.tolist()]
    return preview, preview_shape


def _summarize_tensor_tuple(values: tuple[Any, ...]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for value in values:
        if not hasattr(value, "shape"):
            summary.append({"kind": type(value).__name__})
            continue
        tensor = value.detach().float()
        preview, preview_shape = _preview_tensor(value)
        summary.append(
            {
                "shape": list(tensor.shape),
                "dtype": str(value.dtype).replace("torch.", "") if hasattr(value, "dtype") else type(value).__name__,
                "mean": float(tensor.mean().item()) if tensor.numel() else 0.0,
                "std": float(tensor.std(unbiased=False).item()) if tensor.numel() else 0.0,
                "min": float(tensor.min().item()) if tensor.numel() else 0.0,
                "max": float(tensor.max().item()) if tensor.numel() else 0.0,
                "preview": preview,
                "preview_shape": preview_shape,
            }
        )
    return summary


def find_attached_text_dataset_config(graph: NeuronGraph) -> dict[str, Any] | None:
    for node in graph.nodes.values():
        mtype = getattr(node.neuron_def, "module_type", "")
        if mtype != "dataset_source":
            continue
        cfg = dict(node.neuron_def.module_config or {})
        if cfg.get("dataset_names"):
            return cfg
    return None


def find_attached_semantic_source_config(graph: NeuronGraph) -> dict[str, Any] | None:
    from neuralfn.semantic import NUM_SEMANTIC_DIMS

    for node in graph.nodes.values():
        if getattr(node.neuron_def, "module_type", "") != "semantic_data_source":
            continue
        cfg = dict(node.neuron_def.module_config or {})
        return {"dataset_names": ["__semantic_builtin__"], "seq_len": int(cfg.get("seq_len", NUM_SEMANTIC_DIMS))}
    return None


def find_attached_dataset_config(graph: NeuronGraph) -> dict[str, Any] | None:
    return find_attached_text_dataset_config(graph) or find_attached_semantic_source_config(graph)


def _coerce_trace_tensor(raw: Any, *, dtype: str) -> torch.Tensor:
    tensor_dtype = torch.long if dtype == "tokens" else torch.float32
    tensor = torch.as_tensor(raw, dtype=tensor_dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _coerce_trace_values_for_node(node: NeuronInstance, raw: Any) -> tuple[torch.Tensor, ...]:
    n_out = node.neuron_def.n_outputs
    if n_out == 1:
        raw_values = [raw]
    elif isinstance(raw, dict):
        raw_values = []
        for port in node.neuron_def.output_ports:
            if port.name not in raw:
                raise GraphOperationError(f"Missing trace values for '{node.instance_id}.{port.name}'")
            raw_values.append(raw[port.name])
    elif isinstance(raw, (list, tuple)) and len(raw) == n_out:
        raw_values = list(raw)
    else:
        raise GraphOperationError(
            f"Input node '{node.instance_id}' exposes {n_out} outputs and requires one payload per output"
        )

    tensors: list[torch.Tensor] = []
    for port, value in zip(node.neuron_def.output_ports, raw_values):
        tensors.append(_coerce_trace_tensor(value, dtype=port.dtype))
    return tuple(tensors)


def _build_dataset_trace_inputs(
    graph: NeuronGraph,
    dataset_names: list[str],
    *,
    seq_len: int,
    preview_batch_size: int,
) -> tuple[dict[str, tuple[torch.Tensor, ...]], dict[str, list[int]]]:
    template_spec = dict(graph.torch_config.get("template_spec", {}))
    tokenization = str(template_spec.get("template", {}).get("tokenization", "sp"))
    if tokenization == "byte_hnet":
        inputs_list, targets_list = load_dataset_bytes(dataset_names, seq_len=seq_len)
    else:
        inputs_list, targets_list = load_dataset_tokens(
            dataset_names,
            seq_len=seq_len,
            encoding_name=raw_text_encoding_name_for_template_spec(template_spec),
        )
    batch_size = max(1, min(int(preview_batch_size or 1), len(inputs_list)))
    x = torch.tensor(inputs_list[:batch_size], dtype=torch.long)
    y = torch.tensor(targets_list[:batch_size], dtype=torch.long)
    role_tensors = {
        "tokens": x,
        "enc_tokens": x,
        "dec_tokens": x,
        "targets": y,
    }

    provided: dict[str, tuple[torch.Tensor, ...]] = {}
    sample_inputs: dict[str, list[int]] = {}
    for nid in graph.input_node_ids:
        node = graph.nodes[nid]
        node_values: list[torch.Tensor] = []
        for port in node.neuron_def.output_ports:
            tensor = role_tensors.get(port.name)
            if tensor is None:
                raise GraphOperationError(
                    f"Dataset-backed tracing does not know how to populate graph input role '{port.name}'"
                )
            node_values.append(tensor if port.dtype == "tokens" else tensor.float())
            sample_inputs.setdefault(port.name, tensor[0].tolist())
        provided[nid] = tuple(node_values)

    return provided, sample_inputs


def _build_hybrid_trace_inputs(
    graph: NeuronGraph,
    text_dataset_names: list[str],
    *,
    seq_len: int,
    preview_batch_size: int,
) -> tuple[dict[str, tuple[torch.Tensor, ...]], dict[str, list[int]]]:
    template_spec = dict(graph.torch_config.get("template_spec", {}))
    tokenization = str(template_spec.get("template", {}).get("tokenization", "sp"))
    active_dims = max(1, min(int((template_spec.get("block_spec") or {}).get("top_k", 2) or 2), 8))
    if tokenization == "byte_hnet":
        inputs_list, targets_list = load_dataset_bytes(text_dataset_names, seq_len=seq_len)
    else:
        inputs_list, targets_list = load_dataset_tokens(
            text_dataset_names,
            seq_len=seq_len,
            encoding_name=raw_text_encoding_name_for_template_spec(template_spec),
        )

    from neuralfn.semantic import load_training_targets

    _ids, sem_targets = load_training_targets(active_dims=active_dims)
    sem_tokens = torch.from_numpy(sem_targets.astype("int64"))

    batch_size = max(1, min(int(preview_batch_size or 1), len(inputs_list), len(sem_tokens)))
    x = torch.tensor(inputs_list[:batch_size], dtype=torch.long)
    y = torch.tensor(targets_list[:batch_size], dtype=torch.long)
    sem = sem_tokens[:batch_size]
    role_tensors = {
        "tokens": x,
        "targets": y,
        "sem_targets": sem,
    }

    provided: dict[str, tuple[torch.Tensor, ...]] = {}
    sample_inputs: dict[str, list[int]] = {}
    for nid in graph.input_node_ids:
        node = graph.nodes[nid]
        node_values: list[torch.Tensor] = []
        for port in node.neuron_def.output_ports:
            tensor = role_tensors.get(port.name)
            if tensor is None:
                raise GraphOperationError(
                    f"Hybrid dataset-backed tracing does not know how to populate graph input role '{port.name}'"
                )
            node_values.append(tensor if port.dtype == "tokens" else tensor.float())
            sample_inputs.setdefault(port.name, tensor[0].tolist())
        provided[nid] = tuple(node_values)

    return provided, sample_inputs


def _build_semantic_trace_inputs(
    graph: NeuronGraph,
    *,
    preview_batch_size: int,
) -> tuple[dict[str, tuple[torch.Tensor, ...]], dict[str, list[int]]]:
    template_spec = dict(graph.torch_config.get("template_spec", {}))
    active_dims = max(1, min(int((template_spec.get("block_spec") or {}).get("top_k", 2) or 2), 8))
    from neuralfn.semantic import load_training_targets

    _ids, targets = load_training_targets(active_dims=active_dims)
    sem = torch.from_numpy(targets.astype("int64"))
    batch_size = max(1, min(int(preview_batch_size or 1), len(sem)))
    sem = sem[:batch_size]
    seq_len = sem.shape[1]
    x = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len).clone()
    y = torch.roll(x, shifts=-1, dims=1)
    role_tensors = {
        "tokens": x,
        "enc_tokens": x,
        "dec_tokens": x,
        "targets": y,
        "sem_targets": sem,
    }

    provided: dict[str, tuple[torch.Tensor, ...]] = {}
    sample_inputs: dict[str, list[int]] = {}
    for nid in graph.input_node_ids:
        node = graph.nodes[nid]
        node_values: list[torch.Tensor] = []
        for port in node.neuron_def.output_ports:
            tensor = role_tensors.get(port.name)
            if tensor is None:
                raise GraphOperationError(
                    f"Semantic preview does not know how to populate graph input role '{port.name}'"
                )
            node_values.append(tensor if port.dtype == "tokens" else tensor.float())
            sample_inputs.setdefault(port.name, tensor[0].tolist())
        provided[nid] = tuple(node_values)

    return provided, sample_inputs


def _resolve_torch_trace_inputs(
    graph: NeuronGraph,
    body: ExecuteRequest,
) -> tuple[dict[str, tuple[torch.Tensor, ...]], str, dict[str, list[int]]]:
    if body.inputs:
        provided: dict[str, tuple[torch.Tensor, ...]] = {}
        sample_inputs: dict[str, list[int]] = {}
        for nid in graph.input_node_ids:
            if nid not in body.inputs:
                raise GraphOperationError(f"Missing input values for '{nid}'")
            node = graph.nodes[nid]
            values = _coerce_trace_values_for_node(node, body.inputs[nid])
            provided[nid] = values
            sample_inputs[nid] = values[0][0].reshape(-1)[:8].detach().cpu().tolist()
        return provided, "manual", sample_inputs

    text_cfg = find_attached_text_dataset_config(graph)
    semantic_cfg = find_attached_semantic_source_config(graph)
    text_dataset_names = list(body.dataset_names or (text_cfg or {}).get("dataset_names") or [])
    if not text_dataset_names and semantic_cfg is None:
        raise GraphOperationError("Missing input values and no dataset-backed preview source is configured")

    seq_len = int(body.seq_len or (text_cfg or semantic_cfg or {}).get("seq_len", 64))
    if text_dataset_names and semantic_cfg is not None:
        provided, sample_inputs = _build_hybrid_trace_inputs(
            graph,
            text_dataset_names,
            seq_len=seq_len,
            preview_batch_size=body.preview_batch_size,
        )
        return provided, "dataset", sample_inputs
    if not text_dataset_names and semantic_cfg is not None:
        provided, sample_inputs = _build_semantic_trace_inputs(
            graph,
            preview_batch_size=body.preview_batch_size,
        )
        return provided, "dataset", sample_inputs
    provided, sample_inputs = _build_dataset_trace_inputs(
        graph,
        text_dataset_names,
        seq_len=seq_len,
        preview_batch_size=body.preview_batch_size,
    )
    return provided, "dataset", sample_inputs


def _max_text_token_id(
    graph: NeuronGraph,
    provided_inputs: dict[str, tuple[torch.Tensor, ...]],
) -> int:
    max_token = -1
    text_roles = {"tokens", "targets", "enc_tokens", "dec_tokens"}
    for node_id in graph.input_node_ids:
        node = graph.nodes[node_id]
        values = provided_inputs.get(node_id, ())
        for port, tensor in zip(node.neuron_def.output_ports, values):
            if port.name not in text_roles or tensor.dtype != torch.long or tensor.numel() == 0:
                continue
            max_token = max(max_token, int(tensor.max().item()))
    return max_token


def trace_torch_graph(graph: NeuronGraph, body: ExecuteRequest) -> dict[str, Any]:
    if graph.runtime != "torch" and not graph.has_module_nodes():
        raise GraphOperationError("Active graph is not a torch graph")
    try:
        text_cfg = find_attached_text_dataset_config(graph)
        text_dataset_names = list(body.dataset_names or (text_cfg or {}).get("dataset_names") or [])
        if text_dataset_names:
            model_vocab_size = TorchTrainer._configured_vocab_size(graph)
            for dataset_name in text_dataset_names:
                validate_cached_tokenizer_contract(
                    dataset_name,
                    model_vocab_size=model_vocab_size if model_vocab_size > 0 else None,
                )
        provided_inputs, trace_source, sample_inputs = _resolve_torch_trace_inputs(graph, body)
        max_token = _max_text_token_id(graph, provided_inputs)
        if max_token >= 0:
            TorchTrainer._adjust_vocab_size(graph, max_token + 1)
        compiled = CompiledTorchGraph(graph)
        flat_inputs: list[Any] = []
        for node_id in graph.input_node_ids:
            flat_inputs.extend(provided_inputs[node_id])
        _outputs, trace = compiled.trace(*flat_inputs)
    except GraphOperationError:
        raise
    except Exception as exc:
        raise GraphOperationError(str(exc)) from exc
    return {
        "source": trace_source,
        "sample_inputs": sample_inputs,
        "trace": {node_id: _summarize_tensor_tuple(values) for node_id, values in trace.items()},
    }


def probe_graph_node(graph: NeuronGraph, node_id: str, n_samples: int = 1000) -> dict[str, Any]:
    if node_id not in graph.nodes:
        raise GraphOperationError("Node not found")
    ndef = graph.nodes[node_id].neuron_def
    xs, ys = probe_neuron(ndef, n_samples)
    return {"inputs": xs.tolist(), "outputs": ys.tolist()}


def _find_dataset_source_node(graph: NeuronGraph) -> str | None:
    for nid, node in graph.nodes.items():
        if getattr(node.neuron_def, "module_type", "") == "dataset_source":
            return nid
    return None


def _find_semantic_source_node(graph: NeuronGraph) -> str | None:
    for nid, node in graph.nodes.items():
        if getattr(node.neuron_def, "module_type", "") == "semantic_data_source":
            return nid
    return None


def _find_input_ports(graph: NeuronGraph) -> list[tuple[str, Port]]:
    inputs: list[tuple[str, Port]] = []
    for nid in graph.input_node_ids:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if getattr(node.neuron_def, "module_type", "") == "semantic_data_source":
            continue
        for port in node.neuron_def.output_ports:
            if port.dtype == "tokens":
                inputs.append((nid, port))
    if not inputs:
        raise GraphOperationError(
            f"Active graph (inputs={graph.input_node_ids}) must expose token-typed inputs before a dataset can be loaded"
        )
    return inputs


def _dataset_source_position(graph: NeuronGraph, input_node_ids: list[str]) -> tuple[float, float]:
    positions = [tuple(graph.nodes[nid].position or (40.0, 120.0)) for nid in input_node_ids]
    x = min(float(pos[0]) for pos in positions) - 220.0
    y = sum(float(pos[1]) for pos in positions) / max(len(positions), 1)
    return (x, y)


def ensure_dataset_source_node(graph: NeuronGraph, *, node_id: str, seq_len: int) -> str:
    existing = _find_dataset_source_node(graph)
    if existing is not None:
        node = graph.nodes[existing]
        cfg = dict(node.neuron_def.module_config or {})
        node.neuron_def.module_config = {**cfg, "seq_len": seq_len}
        semantic_nid = _find_semantic_source_node(graph)
        graph.input_node_ids = [existing, semantic_nid] if semantic_nid is not None else [existing]
        return existing

    input_ports = _find_input_ports(graph)
    dataset_node_id = node_id or "dataset_source"
    if dataset_node_id in graph.nodes:
        dataset_node_id = f"{dataset_node_id}_{uuid.uuid4().hex[:6]}"

    dataset_def = clone_neuron_def(BuiltinNeurons.dataset_source_module)
    dataset_def.module_config = {"dataset_names": [], "seq_len": seq_len}
    dataset_def.output_ports = [
        Port(port.name, range=tuple(port.range), precision=port.precision, dtype=port.dtype)
        for _, port in input_ports
    ]
    graph.add_node(
        NeuronInstance(
            dataset_def,
            instance_id=dataset_node_id,
            position=_dataset_source_position(graph, [nid for nid, _ in input_ports]),
        )
    )

    input_port_map = {nid: idx for idx, (nid, _port) in enumerate(input_ports)}
    for edge in graph.edges.values():
        mapped_port = input_port_map.get(edge.src_node)
        if mapped_port is not None:
            edge.src_node = dataset_node_id
            edge.src_port = mapped_port

    for nid in dict.fromkeys(nid for nid, _ in input_ports):
        graph.remove_node(nid)
    semantic_nid = _find_semantic_source_node(graph)
    graph.input_node_ids = [dataset_node_id, semantic_nid] if semantic_nid is not None else [dataset_node_id]
    return dataset_node_id


def download_dataset(request: DownloadDatasetRequest) -> dict[str, Any]:
    return download_hf_dataset(
        request.hf_path,
        hf_split=request.hf_split,
        text_column=request.text_column,
        max_rows=request.max_rows,
        alias=request.alias,
        variant=request.variant,
        train_shards=request.train_shards,
        skip_manifest=request.skip_manifest,
        with_docs=request.with_docs,
        repo_id=request.repo_id,
        remote_root_prefix=request.remote_root_prefix,
    )


def load_dataset_source_into_graph(graph: NeuronGraph, body: LoadDatasetRequest) -> dict[str, Any]:
    dataset_names = list(body.dataset_names or [])
    downloaded: dict[str, Any] | None = None
    if body.hf_path:
        downloaded = download_hf_dataset(
            body.hf_path,
            hf_split=body.hf_split,
            text_column=body.text_column,
            max_rows=body.max_rows,
            alias=body.alias,
            variant=body.variant,
            train_shards=body.train_shards,
            skip_manifest=body.skip_manifest,
            with_docs=body.with_docs,
            repo_id=body.repo_id,
            remote_root_prefix=body.remote_root_prefix,
        )
        dataset_names.append(downloaded["name"])

    if not dataset_names:
        raise GraphOperationError("Provide either hf_path or dataset_names")

    dataset_node_id = ensure_dataset_source_node(graph, node_id=body.node_id, seq_len=body.seq_len)
    node = graph.nodes[dataset_node_id]
    cfg = dict(node.neuron_def.module_config or {})
    existing_names = list(cfg.get("dataset_names", [])) if body.append else []
    merged_names = list(dict.fromkeys(existing_names + dataset_names))
    node.neuron_def.module_config = {
        **cfg,
        "dataset_names": merged_names,
        "seq_len": body.seq_len,
    }
    semantic_nid = _find_semantic_source_node(graph)
    graph.input_node_ids = [dataset_node_id, semantic_nid] if semantic_nid is not None else [dataset_node_id]

    return {
        "dataset_source_node_id": dataset_node_id,
        "dataset_names": merged_names,
        "downloaded": downloaded,
        "graph": summarize_graph_for_agent(graph),
    }
