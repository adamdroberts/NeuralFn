# Service Layer

## WorkspaceService

`server/services/session_service.py`

Manages projects, editor sessions, graphs, snapshots, and analytics. All methods accept a database session and (where needed) an `AuthContext` for permission checks.

### Helper Dataclasses

- **`ProjectAccess`** -- wraps a `Project` with the requesting user's role.
- **`SessionBundle`** -- groups an `EditorSession` with its parent `Project` and a list of snapshot summaries.

### Key Methods

| Method | Description |
|--------|-------------|
| `list_projects(user_id)` | Returns all projects the user is a member of, with roles. |
| `create_project(name, description, user_id)` | Creates a project and adds the creator as owner. |
| `get_session_bundle(project_id, session_id)` | Loads a session with its project context and snapshot history. |
| `get_session_detail(session_id)` | Returns session metadata, latest graph revision, and snapshot list. |
| `update_session_graph(session_id, graph, expected_revision, user_id)` | Writes a new graph revision with optimistic concurrency. Raises on revision conflict. |
| `mutate_session_graph(session_id, mutator_fn)` | Applies a mutation function to the current graph under a revision lock. |
| `create_snapshot(session_id, revision, reason, user_id)` | Persists a point-in-time snapshot of the session graph to disk. |
| `analytics_summary(project_id)` | Aggregates training run statistics for the project dashboard. |

---

## RunService

`server/services/run_service.py`

Manages training runs that execute in background threads.

### ActiveRunHandle

```python
@dataclass
class ActiveRunHandle:
    run_id: str
    session_id: str
    project_id: str
    progress_queue: Queue
    done_event: Event
    thread: Thread
    trainer: Any
```

Tracks an in-progress training run, providing a queue for progress events and a threading event for completion signaling.

### Key Methods

| Method | Description |
|--------|-------------|
| `list_runs(project_id, session_id)` | Returns run history filtered by project and/or session. |
| `get_latest_run(session_id)` | Returns the most recent run for a session. |
| `get_run_snapshot(run_id)` | Returns current status, loss, step count, and event log for a run. |
| `stop_run(run_id)` | Signals a running training thread to stop gracefully. |
| `start_run(session_id, project_id, user_id, request)` | Validates the request, spawns a background training thread, and returns a `RunSnapshot`. |

### Supported Training Methods

- **surrogate** -- lightweight surrogate-model-based optimization.
- **evolutionary** -- evolutionary strategy over network weights.
- **hybrid** -- combines surrogate and evolutionary approaches.
- **torch** -- standard PyTorch gradient-based training.

---

## DatasetService

`server/services/dataset_service.py`

Manages the dataset catalog, project-level access grants, and data loading.

### Key Methods

| Method | Description |
|--------|-------------|
| `list_datasets(project_id)` | Lists datasets, optionally filtered to those granted to a project. |
| `download_dataset(request)` | Downloads a dataset from HuggingFace and registers it in the catalog. |
| `upload_dataset(file, metadata)` | Accepts a file upload and registers it as a dataset. |
| `update_dataset_access(dataset_id, project_ids)` | Sets which projects can use a dataset. |
| `delete_dataset(dataset_id)` | Removes a dataset and its access grants. |
| `load_dataset_tokens_for_project(project_id, dataset_names, seq_len)` | Loads tokenized training data for the specified datasets. |
| `load_dataset_bytes_for_project(project_id, dataset_names)` | Loads raw byte data for datasets. |

---

## LiveStateStore

`server/services/live_state.py`

Abstract interface for ephemeral, high-frequency state. Two implementations:

- **`MemoryLiveStateStore`** -- in-process dictionaries; suitable for single-instance deployments.
- **`RedisLiveStateStore`** -- Redis-backed; required for multi-process or horizontally scaled deployments.

### SessionGraphState

```python
@dataclass
class SessionGraphState:
    graph: dict
    revision: int
    updated_at: datetime
```

### Exceptions

- **`RevisionConflict`** -- raised when a graph update supplies a stale `expected_revision`.

### Methods

| Method | Description |
|--------|-------------|
| `ensure_session_graph(session_id, loader_fn)` | Loads the graph into the store if not already present. |
| `get_session_graph(session_id)` | Returns the current `SessionGraphState` or `None`. |
| `put_session_graph(session_id, graph, expected_revision)` | Updates the graph with optimistic locking. |
| `overwrite_session_graph(session_id, graph, revision)` | Unconditionally replaces the graph (used during hydration). |
| `touch_agent(session_id)` | Records agent activity for the session (heartbeat). |
| `is_agent_active(session_id)` | Returns whether an agent has been active recently. |
| `set_active_run(session_id, run_id)` | Associates a session with an active training run. |
| `get_active_run(session_id)` | Returns the active run ID for a session, if any. |
| `initialize_run(run_id)` | Sets up initial run state in the store. |
| `patch_run_status(run_id, updates)` | Merges partial updates into a run's ephemeral state. |
| `append_run_event(run_id, event)` | Appends a progress event to the run's event log. |
| `get_run_snapshot(run_id)` | Returns the full ephemeral state of a run. |

---

## PersistenceWorker

`server/services/persistence_worker.py`

Decouples hot-path operations from database writes using an async queue.

### Methods

| Method | Description |
|--------|-------------|
| `enqueue_update(session_id, graph, revision)` | Queues a session graph update for DB persistence. |
| `enqueue_run_update(run_id, updates)` | Queues a training run status update for DB persistence. |
| `start()` | Starts the background consumer thread. |
| `stop()` | Signals the worker to drain the queue and shut down. |

---

## graph_ops

`server/services/graph_ops.py`

Stateless graph-manipulation functions used by route handlers.

### Key Functions

| Function | Description |
|----------|-------------|
| `summarize_graph_for_agent(graph)` | Returns a compact text summary of a graph suitable for LLM context. |
| `build_template_payload(name, config)` | Constructs a GPT-template JSON payload with graph, variant library, and template spec. |
| `apply_gpt_template(session_id, name, config, store)` | Builds and applies a GPT template to a session's graph in the live store. |
| `add_node_to_graph(graph, neuron_def, position)` | Inserts a new node into a graph and returns the updated graph with the new node ID. |
| `update_node_in_graph(graph, node_id, updates)` | Applies partial updates to an existing node. |
| `delete_node_from_graph(graph, node_id)` | Removes a node and all connected edges. |
| `execute_graph(graph, inputs, builtins)` | Runs a forward pass through the graph and returns outputs. |
| `execute_trace(graph, inputs, builtins)` | Like `execute_graph`, but also records per-edge activation values. |
| `trace_torch_graph(graph, dataset_names, seq_len, preview_batch_size)` | Compiles the graph to a Torch model and runs a traced forward pass. |
| `probe_graph_node(graph, node_id, inputs)` | Executes a single node in isolation and returns its outputs. |
| `ensure_dataset_source_node(graph, dataset_name)` | Adds a dataset-source node to the graph if one does not already exist. |
| `load_dataset_source_into_graph(graph, dataset_names, seq_len)` | Loads dataset tokens and wires them into the graph's input nodes. |
