# Pydantic Models

All request and response schemas are defined in `server/models.py`. These models handle JSON serialization, validation, and API documentation.

## Core Graph Models

### PortModel

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Port identifier |
| `range` | `list[float]` | Optional value range |
| `precision` | `str` | Optional precision hint |
| `dtype` | `str` | Optional data type |

### VariantRefModel

| Field | Type | Description |
|-------|------|-------------|
| `family` | `str` | Variant family name |
| `version` | `str` | Variant version within the family |

### NeuronDefModel

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique neuron definition ID |
| `name` | `str` | Display name |
| `kind` | `str` | Neuron kind (e.g. "custom", "builtin", "subgraph") |
| `input_ports` | `list[PortModel]` | Input port definitions |
| `output_ports` | `list[PortModel]` | Output port definitions |
| `source_code` | `str` | Optional, Python source for custom neurons |
| `subgraph` | `GraphModel` | Optional, nested graph for subgraph neurons |
| `module_type` | `str` | Optional, Torch module class name |
| `module_config` | `dict` | Optional, Torch module constructor args |
| `module_state` | `dict` | Optional, serialized module state |
| `input_aliases` | `list[str]` | Optional, maps parent ports to subgraph input nodes |
| `output_aliases` | `list[str]` | Optional, maps subgraph output nodes to parent ports |
| `variant_ref` | `VariantRefModel` | Optional, link to a variant library entry |

### NodeModel

| Field | Type | Description |
|-------|------|-------------|
| `instance_id` | `str` | Unique instance ID within the graph |
| `neuron_def` | `NeuronDefModel` | The neuron definition for this node |
| `position` | `dict` | `{x, y}` canvas coordinates |

### EdgeModel

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique edge ID |
| `src_node` | `str` | Source node instance ID |
| `src_port` | `str` | Source port name |
| `dst_node` | `str` | Destination node instance ID |
| `dst_port` | `str` | Destination port name |
| `weight` | `float` | Edge weight (default 1.0) |
| `bias` | `float` | Edge bias (default 0.0) |

### GraphModel

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Graph name |
| `training_method` | `str` | One of the supported training methods |
| `runtime` | `str` | Execution runtime identifier |
| `surrogate_config` | `dict` | Optional surrogate training configuration |
| `evo_config` | `dict` | Optional evolutionary training configuration |
| `torch_config` | `dict` | Optional Torch training configuration |
| `variant_library` | `dict` | Maps family names to version-keyed variant subgraphs |
| `nodes` | `list[NodeModel]` | Graph nodes |
| `edges` | `list[EdgeModel]` | Graph edges |
| `input_node_ids` | `list[str]` | Designated input node IDs |
| `output_node_ids` | `list[str]` | Designated output node IDs |

---

## Request Models

### ExecuteRequest

| Field | Type | Description |
|-------|------|-------------|
| `inputs` | `dict` | Input values keyed by port name |
| `dataset_names` | `list[str]` | Optional dataset names for data-driven execution |
| `seq_len` | `int` | Optional sequence length |
| `preview_batch_size` | `int` | Optional batch size for preview runs |

### TrainRequest

| Field | Type | Description |
|-------|------|-------------|
| `method` | `str` | Training method to use |
| `train_inputs` | `dict` | Optional explicit training inputs |
| `train_targets` | `dict` | Optional explicit training targets |
| `dataset_names` | `list[str]` | Dataset names for data-driven training |
| `text_column` | `str` | Column name for text data |
| `seq_len` | `int` | Sequence length for tokenization |
| `outer_rounds` | `int` | Outer optimization rounds (hybrid method) |
| `loss_fn` | `str` | Loss function identifier |
| `epochs` | `int` | Number of training epochs |
| `learning_rate` | `float` | Optimizer learning rate |
| `population_size` | `int` | Population size (evolutionary method) |
| `generations` | `int` | Number of generations (evolutionary method) |
| `batch_size` | `int` | Training batch size |
| `weight_decay` | `float` | Optimizer weight decay |

### DownloadDatasetRequest

| Field | Type | Description |
|-------|------|-------------|
| `hf_path` | `str` | HuggingFace dataset identifier |
| `hf_split` | `str` | Dataset split (e.g. "train") |
| `text_column` | `str` | Text column name |
| `max_rows` | `int` | Optional row limit |
| `alias` | `str` | Optional display name |
| `variant` | `str` | Optional variant identifier |
| `train_shards` | `int` | Optional shard count |
| `skip_manifest` | `bool` | Skip manifest generation |
| `with_docs` | `bool` | Include documentation |
| `repo_id` | `str` | Optional remote repository ID |
| `remote_root_prefix` | `str` | Optional remote path prefix |
| `project_ids` | `list[str]` | Projects to grant access to |

### LoadDatasetRequest

Extends `DownloadDatasetRequest` with additional fields:

| Field | Type | Description |
|-------|------|-------------|
| `dataset_names` | `list[str]` | Names of already-registered datasets to load |
| `seq_len` | `int` | Sequence length for tokenization |
| `node_id` | `str` | Optional target node ID in the graph |
| `append` | `bool` | Append to existing data vs. replace |

### GPTTemplateRequest

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Template preset name |
| `config` | `dict` | Template configuration overrides |

### EdgeUpdateModel

| Field | Type | Description |
|-------|------|-------------|
| `weight` | `float` | Optional new weight |
| `bias` | `float` | Optional new bias |

### AgentStatusModel

| Field | Type | Description |
|-------|------|-------------|
| `active` | `bool` | Whether the agent is currently active |

---

## Auth and Admin Request Models

### BootstrapAdminRequest

| Field | Type | Description |
|-------|------|-------------|
| `email` | `str` | Admin email |
| `password` | `str` | Admin password |
| `display_name` | `str` | Admin display name |

### LoginRequest

| Field | Type | Description |
|-------|------|-------------|
| `email` | `str` | User email |
| `password` | `str` | User password |

### CreateUserRequest

| Field | Type | Description |
|-------|------|-------------|
| `email` | `str` | New user email |
| `password` | `str` | New user password |
| `display_name` | `str` | New user display name |
| `is_admin` | `bool` | Whether the new user is an admin |

---

## Project and Session Request Models

### ProjectCreateRequest

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Project name |
| `description` | `str` | Optional project description |

### SessionCreateRequest

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Session name |
| `description` | `str` | Optional session description |

### ActiveSessionRequest

| Field | Type | Description |
|-------|------|-------------|
| `project_id` | `str` | Project to make active |
| `session_id` | `str` | Session to make active |

### SessionGraphUpdateRequest

| Field | Type | Description |
|-------|------|-------------|
| `graph` | `GraphModel` | Updated graph data |
| `expected_revision` | `int` | Optimistic concurrency revision |
| `persist_snapshot` | `bool` | Whether to create a snapshot |
| `snapshot_reason` | `str` | Optional reason for the snapshot |

### ProjectMembershipRequest

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | `str` | Optional, identify member by ID |
| `email` | `str` | Optional, identify member by email |
| `role` | `str` | Role to assign |

### DatasetAccessUpdateRequest

| Field | Type | Description |
|-------|------|-------------|
| `project_ids` | `list[str]` | Projects to grant dataset access to |
