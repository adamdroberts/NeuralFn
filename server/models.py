from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PortModel(BaseModel):
    name: str
    range: list[float] = Field(default_factory=lambda: [-1.0, 1.0])
    precision: float = 0.001
    dtype: str = "float"


class VariantRefModel(BaseModel):
    family: str
    version: str


class NeuronDefModel(BaseModel):
    id: str = ""
    name: str
    kind: str = "function"
    input_ports: list[PortModel] = Field(default_factory=list)
    output_ports: list[PortModel] = Field(default_factory=list)
    source_code: str = ""
    subgraph: GraphModel | None = None
    module_type: str = ""
    module_config: dict[str, Any] = Field(default_factory=dict)
    module_state: str = ""
    input_aliases: list[str] = Field(default_factory=list)
    output_aliases: list[str] = Field(default_factory=list)
    variant_ref: VariantRefModel | None = None


class NodeModel(BaseModel):
    instance_id: str = ""
    neuron_def: NeuronDefModel
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0])


class EdgeModel(BaseModel):
    id: str = ""
    src_node: str
    src_port: int = 0
    dst_node: str
    dst_port: int = 0
    weight: float = 1.0
    bias: float = 0.0


class GraphModel(BaseModel):
    name: str = "graph"
    training_method: str = "surrogate"
    runtime: str = "scalar"
    surrogate_config: dict[str, Any] = Field(default_factory=dict)
    evo_config: dict[str, Any] = Field(default_factory=dict)
    torch_config: dict[str, Any] = Field(default_factory=dict)
    variant_library: dict[str, dict[str, GraphModel]] = Field(default_factory=dict)
    nodes: dict[str, NodeModel] = Field(default_factory=dict)
    edges: dict[str, EdgeModel] = Field(default_factory=dict)
    input_node_ids: list[str] = Field(default_factory=list)
    output_node_ids: list[str] = Field(default_factory=list)


class ExecuteRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    dataset_names: list[str] | None = None
    seq_len: int | None = None
    preview_batch_size: int = 1


class TrainRequest(BaseModel):
    method: str | None = "surrogate"  # legacy single-graph training only
    train_inputs: list[list[float | int]] = Field(default_factory=list)
    train_targets: list[list[float | int]] = Field(default_factory=list)
    dataset_names: list[str] | None = None
    text_column: str = "text"
    seq_len: int = 64
    outer_rounds: int = 3
    loss_fn: str = "mse"
    epochs: int = 200
    learning_rate: float = 0.001
    population_size: int = 50
    generations: int = 200
    batch_size: int = 8
    weight_decay: float = 0.01


class DownloadDatasetRequest(BaseModel):
    hf_path: str
    hf_split: str = "train"
    text_column: str = "text"
    max_rows: int | None = None
    alias: str | None = None
    variant: str | None = None
    train_shards: int | None = None
    skip_manifest: bool = False
    with_docs: bool = False
    repo_id: str | None = None
    remote_root_prefix: str = "datasets"


class LoadDatasetRequest(BaseModel):
    dataset_names: list[str] | None = None
    hf_path: str | None = None
    hf_split: str = "train"
    text_column: str = "text"
    max_rows: int | None = None
    alias: str | None = None
    variant: str | None = None
    train_shards: int | None = None
    skip_manifest: bool = False
    with_docs: bool = False
    repo_id: str | None = None
    remote_root_prefix: str = "datasets"
    seq_len: int = 64
    node_id: str = "dataset_source"
    append: bool = False


class GPTTemplateRequest(BaseModel):
    name: str = "gpt"
    config: dict[str, Any] = Field(default_factory=dict)


class EdgeUpdateModel(BaseModel):
    weight: float | None = None
    bias: float | None = None


class AgentStatusModel(BaseModel):
    active: bool = False


NeuronDefModel.model_rebuild()
NodeModel.model_rebuild()
GraphModel.model_rebuild()
