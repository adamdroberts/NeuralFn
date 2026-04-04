from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from starlette.responses import StreamingResponse

from neuralfn.builtins import BuiltinNeurons
from neuralfn.evolutionary import EvoConfig, EvolutionaryTrainer
from neuralfn.graph import Edge, NeuronGraph, NeuronInstance
from neuralfn.hybrid import HybridConfig, HybridTrainer
from neuralfn.neuron import NeuronDef, module_neuron, neuron_from_source, subgraph_neuron
from neuralfn.port import Port
from neuralfn.surrogate import probe_neuron
from neuralfn.trainer import SurrogateTrainer, TrainConfig
from neuralfn.torch_backend import CompiledTorchGraph, TorchTrainConfig, TorchTrainer
from neuralfn.torch_templates import build_gpt_root_graph, build_gpt_template_payload

from .dataset_manager import (
    list_local_datasets,
    download_hf_dataset,
    upload_local_file,
    load_dataset_tokens,
    delete_dataset,
)
from .models import (
    DownloadDatasetRequest,
    EdgeModel,
    EdgeUpdateModel,
    ExecuteRequest,
    GPTTemplateRequest,
    GraphModel,
    LoadDatasetRequest,
    NeuronDefModel,
    NodeModel,
    TrainRequest,
    AgentStatusModel,
)

router = APIRouter(prefix="/api")

_graph = NeuronGraph()
_training_thread: threading.Thread | None = None
_trainer_instance: SurrogateTrainer | EvolutionaryTrainer | HybridTrainer | TorchTrainer | None = None
_agent_last_active: float = 0.0
_training_status_lock = threading.Lock()
_training_events: list[dict[str, Any]] = []
_training_event_seq = 0
_training_status: dict[str, Any] = {
    "run_id": None,
    "status": "idle",
    "running": False,
    "done": False,
    "method": None,
    "requested_method": None,
    "graph_name": None,
    "graph_training_method": None,
    "runtime": None,
    "dataset_names": [],
    "seq_len": None,
    "event_id": 0,
    "history_length": 0,
    "last_event": None,
    "last_loss": None,
    "last_step": None,
    "started_at": None,
    "updated_at": None,
    "completed_at": None,
    "stop_requested": False,
    "error": None,
}


def _new_training_status() -> dict[str, Any]:
    return {
        "run_id": None,
        "status": "idle",
        "running": False,
        "done": False,
        "method": None,
        "requested_method": None,
        "graph_name": None,
        "graph_training_method": None,
        "runtime": None,
        "dataset_names": [],
        "seq_len": None,
        "event_id": 0,
        "history_length": 0,
        "last_event": None,
        "last_loss": None,
        "last_step": None,
        "started_at": None,
        "updated_at": None,
        "completed_at": None,
        "stop_requested": False,
        "error": None,
    }


def _begin_training_run(body: TrainRequest, *, resolved_method: str, graph: NeuronGraph) -> str:
    global _training_event_seq, _training_events, _training_status
    run_id = uuid.uuid4().hex[:12]
    now = time.time()
    with _training_status_lock:
        _training_event_seq = 0
        _training_events = []
        _training_status = {
            **_new_training_status(),
            "run_id": run_id,
            "status": "running",
            "running": True,
            "method": resolved_method,
            "requested_method": body.method,
            "graph_name": graph.name,
            "graph_training_method": graph.training_method,
            "runtime": graph.runtime,
            "dataset_names": list(body.dataset_names or []),
            "seq_len": body.seq_len,
            "started_at": now,
            "updated_at": now,
        }
    return run_id


def _record_training_event(event: dict[str, Any]) -> dict[str, Any]:
    global _training_event_seq
    now = time.time()
    with _training_status_lock:
        _training_event_seq += 1
        payload = {"event_id": _training_event_seq, "timestamp": now, **deepcopy(event)}
        _training_events.append(payload)
        _training_status["event_id"] = payload["event_id"]
        _training_status["history_length"] = len(_training_events)
        _training_status["last_event"] = deepcopy(payload)
        _training_status["updated_at"] = now
        loss = payload.get("loss")
        if loss is not None:
            _training_status["last_loss"] = float(loss)
        step = payload.get("local_step", payload.get("step"))
        if step is not None:
            _training_status["last_step"] = int(step)
        return deepcopy(payload)


def _mark_training_stop_requested() -> None:
    with _training_status_lock:
        if not _training_status.get("running"):
            return
        _training_status["stop_requested"] = True
        _training_status["updated_at"] = time.time()


def _finish_training_run(run_id: str, *, error: str | None = None) -> None:
    now = time.time()
    with _training_status_lock:
        if _training_status.get("run_id") != run_id:
            return
        status = "error" if error else ("stopped" if _training_status.get("stop_requested") else "completed")
        _training_status["status"] = status
        _training_status["running"] = False
        _training_status["done"] = True
        _training_status["updated_at"] = now
        _training_status["completed_at"] = now
        _training_status["error"] = error


def _training_status_snapshot(
    *,
    since_event_id: int | None = None,
    history_limit: int = 25,
) -> dict[str, Any]:
    with _training_status_lock:
        snapshot = deepcopy(_training_status)
        events = _training_events
        if since_event_id is not None:
            events = [event for event in events if int(event.get("event_id", 0)) > since_event_id]
        if history_limit >= 0:
            events = events[-history_limit:] if history_limit else []
        snapshot["events"] = deepcopy(events)
    snapshot["thread_alive"] = bool(_training_thread and _training_thread.is_alive())
    return snapshot


def _port_from_model(pm: Any) -> Port:
    return Port(name=pm.name, range=tuple(pm.range), precision=pm.precision, dtype=pm.dtype)


def _ndef_from_model(nm: Any) -> NeuronDef:
    if getattr(nm, "kind", "function") == "subgraph":
        if nm.subgraph is None:
            raise ValueError(f"Subgraph neuron '{nm.name}' is missing nested graph data")
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
            input_ports=[_port_from_model(p) for p in nm.input_ports],
            output_ports=[_port_from_model(p) for p in nm.output_ports],
            module_config=dict(getattr(nm, "module_config", {}) or {}),
            module_state=getattr(nm, "module_state", ""),
            neuron_id=nm.id or None,
        )

    input_ports = [_port_from_model(p) for p in nm.input_ports]
    output_ports = [_port_from_model(p) for p in nm.output_ports]
    return neuron_from_source(
        nm.source_code, nm.name, input_ports, output_ports, neuron_id=nm.id or None,
    )


def _clone_neuron_def(ndef: NeuronDef) -> NeuronDef:
    return NeuronDef.from_dict(deepcopy(ndef.to_dict()))


def _find_dataset_source_node(graph: NeuronGraph) -> str | None:
    for nid, node in graph.nodes.items():
        if getattr(node.neuron_def, "module_type", "") == "dataset_source":
            return nid
    return None


def _find_token_input_nodes(graph: NeuronGraph) -> tuple[str, str]:
    token_node_id: str | None = None
    target_node_id: str | None = None
    for nid in graph.input_node_ids:
        node = graph.nodes.get(nid)
        if node is None or not node.neuron_def.output_ports:
            continue
        port_name = node.neuron_def.output_ports[0].name.lower()
        if port_name == "tokens" or nid == "tokens_in":
            token_node_id = nid
        elif port_name == "targets" or nid == "targets_in":
            target_node_id = nid
    if token_node_id is None or target_node_id is None:
        raise ValueError(
            "Active graph must expose token inputs named 'tokens' and 'targets' before a dataset can be loaded"
        )
    return token_node_id, target_node_id


def _dataset_source_position(graph: NeuronGraph, token_node_id: str, target_node_id: str) -> tuple[float, float]:
    token_pos = tuple(graph.nodes[token_node_id].position or (40.0, 120.0))
    target_pos = tuple(graph.nodes[target_node_id].position or (40.0, 300.0))
    x = min(float(token_pos[0]), float(target_pos[0])) - 220.0
    y = (float(token_pos[1]) + float(target_pos[1])) / 2.0
    return (x, y)


def _ensure_dataset_source_node(graph: NeuronGraph, *, node_id: str, seq_len: int) -> str:
    existing = _find_dataset_source_node(graph)
    if existing is not None:
        node = graph.nodes[existing]
        cfg = dict(node.neuron_def.module_config or {})
        node.neuron_def.module_config = {**cfg, "seq_len": seq_len}
        graph.input_node_ids = [existing]
        return existing

    token_node_id, target_node_id = _find_token_input_nodes(graph)
    dataset_node_id = node_id or "dataset_source"
    if dataset_node_id in graph.nodes:
        dataset_node_id = f"{dataset_node_id}_{uuid.uuid4().hex[:6]}"

    dataset_def = _clone_neuron_def(BuiltinNeurons.dataset_source_module)
    dataset_def.module_config = {"dataset_names": [], "seq_len": seq_len}
    graph.add_node(
        NeuronInstance(
            dataset_def,
            instance_id=dataset_node_id,
            position=_dataset_source_position(graph, token_node_id, target_node_id),
        )
    )

    for edge in graph.edges.values():
        if edge.src_node == token_node_id:
            edge.src_node = dataset_node_id
            edge.src_port = 0
        elif edge.src_node == target_node_id:
            edge.src_node = dataset_node_id
            edge.src_port = 1

    graph.remove_node(token_node_id)
    if target_node_id != token_node_id:
        graph.remove_node(target_node_id)
    graph.input_node_ids = [dataset_node_id]
    return dataset_node_id


# ── Graph CRUD ────────────────────────────────────────────────────────

@router.get("/graph")
def get_graph() -> dict[str, Any]:
    return _graph.to_dict()


@router.get("/agent/status")
def get_agent_status() -> dict[str, bool]:
    return {"active": (time.time() - _agent_last_active) < 5.0}


@router.post("/agent/status")
def set_agent_status(body: AgentStatusModel) -> dict[str, bool]:
    global _agent_last_active
    if body.active:
        _agent_last_active = time.time()
    else:
        _agent_last_active = 0.0
    return {"active": body.active}


@router.put("/graph")
def put_graph(body: GraphModel) -> dict[str, Any]:
    global _graph
    _graph = NeuronGraph.from_dict(body.model_dump())
    return _graph.to_dict()


@router.post("/nodes")
def add_node(body: NodeModel) -> dict[str, Any]:
    ndef = _ndef_from_model(body.neuron_def)
    inst = NeuronInstance(
        neuron_def=ndef,
        instance_id=body.instance_id or uuid.uuid4().hex[:12],
        position=tuple(body.position),
    )
    _graph.add_node(inst)
    return inst.to_dict()


@router.delete("/nodes/{node_id}")
def delete_node(node_id: str) -> dict[str, str]:
    if node_id not in _graph.nodes:
        raise HTTPException(404, "Node not found")
    _graph.remove_node(node_id)
    return {"status": "deleted"}


@router.post("/edges")
def add_edge(body: EdgeModel) -> dict[str, Any]:
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
        _graph.add_edge(edge)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return edge.to_dict()


@router.delete("/edges/{edge_id}")
def delete_edge(edge_id: str) -> dict[str, str]:
    if edge_id not in _graph.edges:
        raise HTTPException(404, "Edge not found")
    _graph.remove_edge(edge_id)
    return {"status": "deleted"}


@router.put("/nodes/{node_id}")
def update_node(node_id: str, body: NeuronDefModel) -> dict[str, Any]:
    if node_id not in _graph.nodes:
        raise HTTPException(404, "Node not found")
    inst = _graph.nodes[node_id]
    inst.neuron_def = _ndef_from_model(body)
    return inst.to_dict()


@router.put("/edges/{edge_id}")
def update_edge(edge_id: str, body: EdgeUpdateModel) -> dict[str, Any]:
    if edge_id not in _graph.edges:
        raise HTTPException(404, "Edge not found")
    edge = _graph.edges[edge_id]
    if body.weight is not None:
        edge.weight = body.weight
    if body.bias is not None:
        edge.bias = body.bias
    return edge.to_dict()


# ── Execution ─────────────────────────────────────────────────────────

@router.post("/execute")
def execute(body: ExecuteRequest) -> dict[str, Any]:
    inputs = {k: tuple(v) for k, v in body.inputs.items()}
    try:
        outputs = _graph.execute(inputs)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {k: list(v) for k, v in outputs.items()}


@router.post("/execute-trace")
def execute_trace(body: ExecuteRequest) -> dict[str, Any]:
    inputs = {k: tuple(v) for k, v in body.inputs.items()}
    try:
        outputs = _graph.execute_trace(inputs)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {k: list(v) for k, v in outputs.items()}


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


def _find_attached_dataset_config(graph: NeuronGraph) -> dict[str, Any] | None:
    for node in graph.nodes.values():
        if getattr(node.neuron_def, "module_type", "") != "dataset_source":
            continue
        cfg = dict(node.neuron_def.module_config or {})
        if cfg.get("dataset_names"):
            return cfg
    return None


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
                raise ValueError(f"Missing trace values for '{node.instance_id}.{port.name}'")
            raw_values.append(raw[port.name])
    elif isinstance(raw, (list, tuple)) and len(raw) == n_out:
        raw_values = list(raw)
    else:
        raise ValueError(
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
    inputs_list, targets_list = load_dataset_tokens(dataset_names, seq_len=seq_len)
    batch_size = max(1, min(int(preview_batch_size or 1), len(inputs_list)))
    x = torch.tensor(inputs_list[:batch_size], dtype=torch.long)
    y = torch.tensor(targets_list[:batch_size], dtype=torch.long)

    flattened_expected = sum(graph.nodes[nid].neuron_def.n_outputs for nid in graph.input_node_ids)
    if flattened_expected != 2:
        raise ValueError(
            f"Dataset-backed tracing expects exactly 2 flattened graph inputs (tokens, targets), found {flattened_expected}"
        )

    supplied = [x, y]
    supplied_idx = 0
    provided: dict[str, tuple[torch.Tensor, ...]] = {}
    for nid in graph.input_node_ids:
        node = graph.nodes[nid]
        node_values: list[torch.Tensor] = []
        for port in node.neuron_def.output_ports:
            tensor = supplied[supplied_idx]
            node_values.append(tensor if port.dtype == "tokens" else tensor.float())
            supplied_idx += 1
        provided[nid] = tuple(node_values)

    return provided, {
        "tokens": x[0].tolist(),
        "targets": y[0].tolist(),
    }


def _resolve_torch_trace_inputs(
    graph: NeuronGraph,
    body: ExecuteRequest,
) -> tuple[dict[str, tuple[torch.Tensor, ...]], str, dict[str, list[int]]]:
    if body.inputs:
        provided: dict[str, tuple[torch.Tensor, ...]] = {}
        sample_inputs: dict[str, list[int]] = {}
        for nid in graph.input_node_ids:
            if nid not in body.inputs:
                raise ValueError(f"Missing input values for '{nid}'")
            node = graph.nodes[nid]
            values = _coerce_trace_values_for_node(node, body.inputs[nid])
            provided[nid] = values
            sample_inputs[nid] = values[0][0].reshape(-1)[:8].detach().cpu().tolist()
        return provided, "manual", sample_inputs

    dataset_cfg = _find_attached_dataset_config(graph)
    dataset_names_raw = body.dataset_names or (dataset_cfg or {}).get("dataset_names") or []
    dataset_names = list(dataset_names_raw)
    if not dataset_names:
        raise ValueError("Missing input values and no dataset-backed preview source is configured")

    seq_len = int(body.seq_len or (dataset_cfg or {}).get("seq_len", 64))
    provided, sample_inputs = _build_dataset_trace_inputs(
        graph,
        dataset_names,
        seq_len=seq_len,
        preview_batch_size=body.preview_batch_size,
    )
    return provided, "dataset", sample_inputs


@router.post("/trace/torch")
def torch_trace(body: ExecuteRequest) -> dict[str, Any]:
    if _graph.runtime != "torch" and not _graph.has_module_nodes():
        raise HTTPException(400, "Active graph is not a torch graph")
    try:
        provided_inputs, trace_source, sample_inputs = _resolve_torch_trace_inputs(_graph, body)
        max_token = max(
            (
                int(tensor.max().item())
                for values in provided_inputs.values()
                for tensor in values
                if tensor.dtype == torch.long and tensor.numel() > 0
            ),
            default=-1,
        )
        if max_token >= 0:
            TorchTrainer._adjust_vocab_size(_graph, max_token + 1)
        compiled = CompiledTorchGraph(_graph)
        flat_inputs: list[Any] = []
        for node_id in _graph.input_node_ids:
            flat_inputs.extend(provided_inputs[node_id])
        _outputs, trace = compiled.trace(*flat_inputs)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {
        "source": trace_source,
        "sample_inputs": sample_inputs,
        "trace": {node_id: _summarize_tensor_tuple(values) for node_id, values in trace.items()},
    }


# ── Builtins ──────────────────────────────────────────────────────────

@router.get("/builtins")
def list_builtins() -> list[dict[str, Any]]:
    return [n.to_dict() for n in BuiltinNeurons.all()]


@router.post("/templates/gpt")
def build_gpt_template(body: GPTTemplateRequest) -> dict[str, Any]:
    payload = build_gpt_template_payload(
        name=body.name,
        config=dict(body.config or {}),
    )
    return payload


def _summarize_graph_for_agent(g: NeuronGraph) -> dict[str, Any]:
    gd = g.to_dict()
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


@router.post("/templates/gpt/apply")
def apply_gpt_template(body: GPTTemplateRequest) -> dict[str, Any]:
    global _graph
    from neuralfn.config import (
        ModelSpec, build_gpt2_spec, build_llama_spec, build_moe_spec, build_nanogpt_spec,
    )

    cfg = dict(body.config or {})
    preset = cfg.get("preset", "nanogpt")
    kwargs = {
        k: cfg[k]
        for k in ("num_heads", "num_kv_heads", "tie_embeddings", "dropout_p",
                   "experts", "top_k", "router_aux_loss_coef", "mlp_multiplier", "multiple_of")
        if k in cfg
    }
    if preset == "gpt2":
        spec = build_gpt2_spec(**kwargs)
    elif preset == "llama":
        spec = build_llama_spec(**kwargs)
    elif preset == "moe":
        spec = build_moe_spec(**kwargs)
    else:
        spec = build_nanogpt_spec(**kwargs)

    if "n_layer" in cfg:
        spec.num_layers = cfg["n_layer"]
    if "n_embd" in cfg:
        spec.model_dim = cfg["n_embd"]
    if "n_head" in cfg:
        spec.block_spec.num_heads = cfg["n_head"]
    if "vocab_size" in cfg:
        spec.vocab_size = cfg["vocab_size"]

    _graph = build_gpt_root_graph(name=body.name, model_spec=spec)
    return _summarize_graph_for_agent(_graph)


# ── Probe ─────────────────────────────────────────────────────────────

@router.post("/probe/{node_id}")
def probe(node_id: str, n_samples: int = 1000) -> dict[str, Any]:
    if node_id not in _graph.nodes:
        raise HTTPException(404, "Node not found")
    ndef = _graph.nodes[node_id].neuron_def
    xs, ys = probe_neuron(ndef, n_samples)
    return {
        "inputs": xs.tolist(),
        "outputs": ys.tolist(),
    }


# ── Training (SSE) ───────────────────────────────────────────────────

@router.post("/train/start")
def train_start(body: TrainRequest) -> StreamingResponse:
    global _trainer_instance, _training_thread

    # ── Resolve datasets if specified ─────────────────────────────────
    if body.dataset_names and len(body.dataset_names) > 0:
        try:
            inputs, targets = load_dataset_tokens(
                body.dataset_names,
                seq_len=body.seq_len,
            )
        except Exception as e:
            raise HTTPException(400, f"Dataset loading failed: {e}")
        train_in = np.array(inputs, dtype=np.float32)
        train_tgt = np.array(targets, dtype=np.float32)
    else:
        train_in = np.array(body.train_inputs, dtype=np.float32)
        train_tgt = np.array(body.train_targets, dtype=np.float32)

    progress_queue: list[dict[str, Any]] = []
    done_event = threading.Event()

    def on_progress(step: int, loss: float) -> None:
        progress_queue.append(_record_training_event({"step": step, "loss": loss}))

    def on_hybrid_progress(info: dict[str, Any]) -> None:
        progress_queue.append(_record_training_event(info))

    def run_surrogate() -> None:
        cfg = TrainConfig(
            learning_rate=body.learning_rate,
            epochs=body.epochs,
            loss_fn=body.loss_fn,
        )
        trainer = SurrogateTrainer(_graph, cfg)
        global _trainer_instance
        _trainer_instance = trainer
        trainer.train(train_in, train_tgt, on_epoch=on_progress)

    def run_evolutionary() -> None:
        cfg = EvoConfig(
            population_size=body.population_size,
            generations=body.generations,
        )
        trainer = EvolutionaryTrainer(_graph, cfg)
        global _trainer_instance
        _trainer_instance = trainer
        trainer.train(train_in, train_tgt, on_generation=on_progress)

    def run_hybrid() -> None:
        cfg = HybridConfig(
            outer_rounds=body.outer_rounds,
            loss_fn=body.loss_fn,
            default_surrogate=TrainConfig(
                learning_rate=body.learning_rate,
                epochs=body.epochs,
                loss_fn=body.loss_fn,
            ),
            default_evolutionary=EvoConfig(
                population_size=body.population_size,
                generations=body.generations,
            ),
        )
        trainer = HybridTrainer(_graph, cfg)
        global _trainer_instance
        _trainer_instance = trainer
        trainer.train(train_in, train_tgt, on_step=on_hybrid_progress)

    def run_torch() -> None:
        cfg = TorchTrainConfig(
            learning_rate=body.learning_rate,
            epochs=body.epochs,
            batch_size=body.batch_size,
            weight_decay=body.weight_decay,
            device=str(_graph.torch_config.get("device", "cuda")),
            amp_dtype=str(_graph.torch_config.get("amp_dtype", "bfloat16")),
        )
        trainer = TorchTrainer(_graph, cfg)
        global _trainer_instance
        _trainer_instance = trainer
        # If datasets were specified, use the resolved arrays
        if body.dataset_names and len(body.dataset_names) > 0:
            trainer.train(
                train_in.astype(int).tolist(),
                train_tgt.astype(int).tolist(),
                on_epoch=on_progress,
            )
        else:
            trainer.train(body.train_inputs, body.train_targets, on_epoch=on_progress)

    use_torch = body.method == "torch" or _graph.training_method == "torch" or _graph.runtime == "torch" or _graph.has_module_nodes()
    use_legacy = body.method in {"surrogate", "evolutionary"} and not _graph.has_nested_subgraphs() and not use_torch
    resolved_method = "torch" if use_torch else (body.method if use_legacy else "hybrid")
    if use_torch:
        target = run_torch
    elif use_legacy:
        target = run_surrogate if body.method == "surrogate" else run_evolutionary
    else:
        target = run_hybrid
    run_id = _begin_training_run(body, resolved_method=resolved_method, graph=_graph)

    def run_target() -> None:
        global _trainer_instance
        try:
            target()
        except Exception as e:
            progress_queue.append(_record_training_event({"error": str(e)}))
            _finish_training_run(run_id, error=str(e))
        else:
            _finish_training_run(run_id)
        finally:
            _trainer_instance = None
            done_event.set()

    _training_thread = threading.Thread(target=run_target, daemon=True)
    _training_thread.start()

    async def event_stream():
        idx = 0
        while not done_event.is_set():
            await asyncio.sleep(0.1)
            while idx < len(progress_queue):
                msg = progress_queue[idx]
                yield f"data: {json.dumps(msg)}\n\n"
                idx += 1
        while idx < len(progress_queue):
            msg = progress_queue[idx]
            yield f"data: {json.dumps(msg)}\n\n"
            idx += 1
        yield "data: {\"done\": true}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/train/status")
def get_training_status(
    since_event_id: int | None = None,
    history_limit: int = 25,
) -> dict[str, Any]:
    capped_limit = max(0, min(history_limit, 500))
    return _training_status_snapshot(since_event_id=since_event_id, history_limit=capped_limit)


@router.post("/train/stop")
def train_stop() -> dict[str, str]:
    if _trainer_instance is not None:
        _mark_training_stop_requested()
        _trainer_instance.stop()
    return {"status": "stopped"}


# ── I/O designation ───────────────────────────────────────────────────

@router.put("/graph/io")
def set_io(input_ids: list[str], output_ids: list[str]) -> dict[str, Any]:
    _graph.input_node_ids = input_ids
    _graph.output_node_ids = output_ids
    return {"input_node_ids": input_ids, "output_node_ids": output_ids}


# ── Dataset Management ────────────────────────────────────────────────

@router.get("/datasets")
def get_datasets() -> list[dict[str, Any]]:
    """List all locally available datasets."""
    return list_local_datasets()


@router.post("/datasets/download")
def download_dataset(body: DownloadDatasetRequest) -> dict[str, Any]:
    """Download a HuggingFace dataset into server/datasets/."""
    try:
        result = download_hf_dataset(
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
        return result
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/datasets/load")
def load_dataset(body: LoadDatasetRequest) -> dict[str, Any]:
    """Download/load datasets and wire them into a dataset_source node on the active graph."""
    global _graph

    dataset_names = list(body.dataset_names or [])
    downloaded: dict[str, Any] | None = None
    if body.hf_path:
        try:
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
        except Exception as e:
            raise HTTPException(400, str(e))
        dataset_names.append(downloaded["name"])

    if not dataset_names:
        raise HTTPException(400, "Provide either hf_path or dataset_names")

    try:
        dataset_node_id = _ensure_dataset_source_node(
            _graph,
            node_id=body.node_id,
            seq_len=body.seq_len,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    node = _graph.nodes[dataset_node_id]
    cfg = dict(node.neuron_def.module_config or {})
    existing_names = list(cfg.get("dataset_names", [])) if body.append else []
    merged_names = list(dict.fromkeys(existing_names + dataset_names))
    node.neuron_def.module_config = {
        **cfg,
        "dataset_names": merged_names,
        "seq_len": body.seq_len,
    }
    _graph.input_node_ids = [dataset_node_id]

    return {
        "dataset_source_node_id": dataset_node_id,
        "dataset_names": merged_names,
        "downloaded": downloaded,
        "graph": _summarize_graph_for_agent(_graph),
    }


@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
) -> dict[str, Any]:
    """Upload a local file as a dataset."""
    try:
        content = await file.read()
        return upload_local_file(name, content, file.filename or "data.txt")
    except Exception as e:
        raise HTTPException(400, str(e))


@router.delete("/datasets/{ds_name}")
def remove_dataset(ds_name: str) -> dict[str, str]:
    """Delete a dataset from local storage."""
    if not delete_dataset(ds_name):
        raise HTTPException(404, f"Dataset '{ds_name}' not found")
    return {"status": "deleted"}
