from __future__ import annotations

from pydantic import BaseModel, Field


class PortModel(BaseModel):
    name: str
    range: list[float] = [-1.0, 1.0]
    precision: float = 0.001
    dtype: str = "float"


class NeuronDefModel(BaseModel):
    id: str = ""
    name: str
    input_ports: list[PortModel]
    output_ports: list[PortModel]
    source_code: str = ""


class NodeModel(BaseModel):
    instance_id: str = ""
    neuron_def: NeuronDefModel
    position: list[float] = [0.0, 0.0]


class EdgeModel(BaseModel):
    id: str = ""
    src_node: str
    src_port: int = 0
    dst_node: str
    dst_port: int = 0
    weight: float = 1.0
    bias: float = 0.0


class GraphModel(BaseModel):
    nodes: dict[str, NodeModel] = {}
    edges: dict[str, EdgeModel] = {}
    input_node_ids: list[str] = []
    output_node_ids: list[str] = []


class ExecuteRequest(BaseModel):
    inputs: dict[str, list[float]]


class TrainRequest(BaseModel):
    method: str = "surrogate"  # "surrogate" | "evolutionary"
    train_inputs: list[list[float]]
    train_targets: list[list[float]]
    epochs: int = 200
    learning_rate: float = 0.001
    population_size: int = 50
    generations: int = 200
