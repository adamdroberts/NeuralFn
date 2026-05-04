from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PortModel(BaseModel):
    name: str
    range: list[float] | None = None
    precision: float | None = None
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


class FineTuneConfigModel(BaseModel):
    """Fine-tuning configuration attached to ``TrainRequest``.

    ``adapter_type`` selects between ``"none"`` (full fine-tune),
    ``"lora"``, ``"qlora"`` (int4 base + LoRA delta), or ``"randmap"``
    (Johnson–Lindenstrauss residual adapter).
    """
    adapter_type: str = "none"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_bias: bool = False
    qlora_group_size: int = 64
    qlora_compute_dtype: str = "bf16"
    # DPO
    beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    dpo_label_smoothing: float = 0.0
    # PPO
    kl_coef: float = 0.1
    ppo_clip: float = 0.2
    ppo_vf_coef: float = 0.5
    ppo_ent_coef: float = 0.0
    rollout_length: int = 64
    ppo_epochs_per_rollout: int = 4
    ppo_minibatch_size: int = 4
    gae_gamma: float = 1.0
    gae_lambda: float = 0.95


class TrainRequest(BaseModel):
    method: str | None = "surrogate"
    train_inputs: list[list[float | int]] = Field(default_factory=list)
    train_targets: list[list[float | int]] = Field(default_factory=list)
    dataset_names: list[str] | None = None
    text_column: str = "text"
    seq_len: int | None = None
    outer_rounds: int = 3
    loss_fn: str = "mse"
    epochs: int = 200
    learning_rate: float = 0.001
    population_size: int = 50
    generations: int = 200
    batch_size: int = 8
    weight_decay: float = 0.01
    # Fine-tuning fields (Phase 1-4). ``training_mode="pretrain"`` preserves
    # the classic behaviour; other modes attach a ``FineTuneSpec`` dict to the
    # graph's ``torch_config`` under ``finetune_spec`` so the trainer runs the
    # pre-train hook (base-checkpoint load + freeze-non-LoRA).
    training_mode: str = "pretrain"  # "pretrain" | "sft" | "dpo" | "ppo" | "reward_model"
    base_checkpoint_path: str | None = None
    ref_checkpoint_path: str | None = None
    reward_checkpoint_path: str | None = None
    adapter_only_save: bool = False
    finetune_config: FineTuneConfigModel | None = None


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
    project_ids: list[str] | None = None


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
    project_ids: list[str] | None = None


class GPTTemplateRequest(BaseModel):
    name: str = "gpt"
    config: dict[str, Any] = Field(default_factory=dict)


class EdgeUpdateModel(BaseModel):
    weight: float | None = None
    bias: float | None = None


class AgentStatusModel(BaseModel):
    active: bool = False


class BootstrapAdminRequest(BaseModel):
    email: str
    password: str
    display_name: str = "Admin"


class LoginRequest(BaseModel):
    email: str
    password: str


class CreateUserRequest(BaseModel):
    email: str
    password: str
    display_name: str
    is_admin: bool = False


class ProjectCreateRequest(BaseModel):
    name: str
    description: str | None = None


class SessionCreateRequest(BaseModel):
    name: str
    description: str | None = None


class ActiveSessionRequest(BaseModel):
    project_id: str | None = None
    session_id: str | None = None


class SessionGraphUpdateRequest(BaseModel):
    graph: GraphModel
    expected_revision: int | None = None
    persist_snapshot: bool = False
    snapshot_reason: str = "autosave"


class ProjectMembershipRequest(BaseModel):
    user_id: str | None = None
    email: str | None = None
    role: str = "data_scientist"


class DatasetAccessUpdateRequest(BaseModel):
    project_ids: list[str] = Field(default_factory=list)


NeuronDefModel.model_rebuild()
NodeModel.model_rebuild()
GraphModel.model_rebuild()
