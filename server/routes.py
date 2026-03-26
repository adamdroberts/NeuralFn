from __future__ import annotations

import asyncio
import json
import threading
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse

from neuralfn.builtins import BuiltinNeurons
from neuralfn.evolutionary import EvoConfig, EvolutionaryTrainer
from neuralfn.graph import Edge, NeuronGraph, NeuronInstance
from neuralfn.hybrid import HybridConfig, HybridTrainer
from neuralfn.neuron import NeuronDef, neuron_from_source, subgraph_neuron
from neuralfn.port import Port
from neuralfn.surrogate import probe_neuron
from neuralfn.trainer import SurrogateTrainer, TrainConfig

from .models import EdgeModel, ExecuteRequest, GraphModel, NodeModel, TrainRequest

router = APIRouter(prefix="/api")

_graph = NeuronGraph()
_training_thread: threading.Thread | None = None
_trainer_instance: SurrogateTrainer | EvolutionaryTrainer | HybridTrainer | None = None


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
            neuron_id=nm.id or None,
        )

    input_ports = [_port_from_model(p) for p in nm.input_ports]
    output_ports = [_port_from_model(p) for p in nm.output_ports]
    return neuron_from_source(
        nm.source_code, nm.name, input_ports, output_ports, neuron_id=nm.id or None,
    )


# ── Graph CRUD ────────────────────────────────────────────────────────

@router.get("/graph")
def get_graph() -> dict[str, Any]:
    return _graph.to_dict()


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


# ── Builtins ──────────────────────────────────────────────────────────

@router.get("/builtins")
def list_builtins() -> list[dict[str, Any]]:
    return [n.to_dict() for n in BuiltinNeurons.all()]


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

    train_in = np.array(body.train_inputs, dtype=np.float32)
    train_tgt = np.array(body.train_targets, dtype=np.float32)

    progress_queue: list[dict[str, Any]] = []
    done_event = threading.Event()

    def on_progress(step: int, loss: float) -> None:
        progress_queue.append({"step": step, "loss": loss})

    def on_hybrid_progress(info: dict[str, Any]) -> None:
        progress_queue.append(info)

    def run_surrogate() -> None:
        try:
            cfg = TrainConfig(
                learning_rate=body.learning_rate,
                epochs=body.epochs,
                loss_fn=body.loss_fn,
            )
            trainer = SurrogateTrainer(_graph, cfg)
            global _trainer_instance
            _trainer_instance = trainer
            trainer.train(train_in, train_tgt, on_epoch=on_progress)
        finally:
            done_event.set()

    def run_evolutionary() -> None:
        try:
            cfg = EvoConfig(
                population_size=body.population_size,
                generations=body.generations,
            )
            trainer = EvolutionaryTrainer(_graph, cfg)
            global _trainer_instance
            _trainer_instance = trainer
            trainer.train(train_in, train_tgt, on_generation=on_progress)
        finally:
            done_event.set()

    def run_hybrid() -> None:
        try:
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
        finally:
            done_event.set()

    use_legacy = body.method in {"surrogate", "evolutionary"} and not _graph.has_nested_subgraphs()
    if use_legacy:
        target = run_surrogate if body.method == "surrogate" else run_evolutionary
    else:
        target = run_hybrid
    _training_thread = threading.Thread(target=target, daemon=True)
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


@router.post("/train/stop")
def train_stop() -> dict[str, str]:
    if _trainer_instance is not None:
        _trainer_instance.stop()
    return {"status": "stopped"}


# ── I/O designation ───────────────────────────────────────────────────

@router.put("/graph/io")
def set_io(input_ids: list[str], output_ids: list[str]) -> dict[str, Any]:
    _graph.input_node_ids = input_ids
    _graph.output_node_ids = output_ids
    return {"input_node_ids": input_ids, "output_node_ids": output_ids}
