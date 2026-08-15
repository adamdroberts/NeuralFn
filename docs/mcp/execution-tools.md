# Execution Tools

Tools for running forward passes, tracing, probing, loading templates, and managing training.

---

## execute_graph

Runs a single forward pass through the graph with the provided inputs.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `inputs` | dict[string, list[float]] | yes | Map of input node ID to input values. |

**Returns:** execution result with output values.

---

## execute_trace

Runs a forward pass and returns a full trace showing every node's input and output values.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `inputs` | dict[string, list[float]] | yes | Map of input node ID to input values. |

**Returns:** trace dictionary keyed by node ID, with per-node input/output snapshots.

---

## trace_torch

Runs a forward pass through the compiled Torch graph and returns a Torch-specific trace.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |
| `inputs` | dict[string, list[float]] | yes | Map of input node ID to input values. |

**Returns:** Torch-specific trace dictionary.

---

## probe_node

Probes a single node by running multiple random forward passes and collecting activation statistics.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `project_id` | string | yes | | Project ID. |
| `session_id` | string | yes | | Session ID. |
| `node_id` | string | yes | | The node to probe. |
| `n_samples` | int | no | `1000` | Number of forward-pass samples to collect. |

**Returns:** probe statistics including mean, standard deviation, min, max, and histogram data.

---

## load_gpt_template

Loads a GPT template preset into the session, replacing the current graph.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `project_id` | string | yes | | Project ID. |
| `session_id` | string | yes | | Session ID. |
| `name` | string | no | `"gpt"` | Template name. |
| `preset` | string | no | `"nanogpt"` | Preset configuration (e.g. `"nanogpt"`, `"gpt2"`, `"llama"`, `"moe"`, `"dense_jepa_evo"`, `"moe_jepa_evo"`, `"semantic_dense_jepa_evo"`, `"semantic_moe_jepa_evo"`). |
| `config` | dict | no | | Additional configuration overrides (e.g. `n_layer`, `n_head`, `n_embd`). |

**Returns:** a summary of the generated graph.

---

## train_start

Starts a training run on the current graph.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `project_id` | string | yes | | Project ID. |
| `session_id` | string | yes | | Session ID. |
| `method` | string | no | `"surrogate"` | Training method. |
| `epochs` | int | no | `10` | Number of training epochs. |
| `learning_rate` | float | no | `0.001` | Learning rate. |
| `train_inputs` | list[list[number]] | no | | Explicit training input data. |
| `train_targets` | list[list[number]] | no | | Explicit training target data. |
| `dataset_names` | list[string] | no | | Names of datasets to train on (alternative to explicit inputs/targets). |
| `runtime` | string | no | graph runtime | `scalar`, `torch`, or `native-cuda`; appended to preserve existing positional callers. |

**Returns:** a synchronous server acknowledgement. A successful launch keeps
the established `status="started"` value, exposes the initial REST state as
`run_status` (normally `running`), and includes `run_id`, `runtime`,
`compatibility_report`, `artifact_metadata`, and `checkpoint_path` when
available. Use `poll_training_status` to monitor progress.

For `runtime="native-cuda"`, `train_start` first calls the same Native IR
preflight used by the editor. An incompatible graph returns
`status="incompatible"`, `issues[]` with exact node paths, and no run ID; no
trainer is launched. A compatible run materializes Native IR under the server
artifact root and dispatches through the canonical native trainer registry.
There is no Torch fallback.

The current execution-ready graph-file adapters are the exact reviewed GPT-2
profiles `gpt2`, `gpt2_megakernel`, `gpt2_moa`, `gpt2_qknorm`,
`gpt2_softcap`, `gpt2_stable`, and `gpt2_zloss`, plus canonical `llama`,
`llama_fast`, exact standard-MoE `moe`, `mixllama`, and `mixllama_fast`, and
trusted-planner proof-bound `gpt2_diff` training. `gpt2_diff` migration and
resident inference remain unavailable because they do not consume its
graph-bound learned-lambda bundle and its exact low-level native path is
packed-QKV-only. MCP materialization requires the exact
graph/fingerprint/proof triplet; its unkeyed proof digest is local-handoff
integrity, not caller authenticity. The low-level
bundle's version-2 metadata is continuation-only and binds all five binaries,
the source graph, training-only shard identity, counters/sampler, seed,
accumulation, optimizer/LR horizon, BF16 routes, and a canonical profile of
supported effective numerics before Tile/CUDA/H2D. Completion returns the
strict `.diff.json` only after full bundle validation. The graph-training total
is 13 ready and 54 blocked; migration/resident stay 12 ready and 55 blocked.
LLaMA preflight derives and binds its complete trainer geometry and source SHA;
the server accepts its final checkpoint only when validated v2 metadata carries
the same source digest. A completed graph-bound `gpt2_moa` run similarly returns the
validated `model_XXXXXXXX.moa.json` path only after its named dense-v5 model,
empty DONE marker, source hash, canonical candidates, selected activation, and
positive interval pass inspection. Resuming a MoA run requires that same
sidecar, restores its activation without a fresh probe, and rejects missing or
changed metadata. This is graph-bound behavior; direct selector-only first-leg
training may still write ordinary dense-v5, but its unbound output cannot
resume exactly. Native runs require exactly
one cached project dataset alias passed in `dataset_names` and support
pretraining only. The compiled training ABI currently has no cooperative
cancellation, so `train_stop` returns `status="unsupported"` for an active
native run.

---

## get_training_status

Returns the latest snapshot of the current or most recent training run.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |

**Returns:** run status snapshot including current epoch, loss, and run state.

---

## poll_training_status

Blocks until a training status update is available or the timeout expires. Useful for agents that want to wait for progress without busy-polling.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `project_id` | string | yes | | Project ID. |
| `session_id` | string | yes | | Session ID. |
| `since_event_id` | string | no | | Only return events after this event ID. |
| `timeout_seconds` | int | no | `30` | Maximum time to wait before returning. |
| `interval_seconds` | int | no | `1` | Polling interval within the server. |

**Returns:** the next training status update, or the current status if the timeout expires.

---

## train_stop

Stops the currently active training run.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `session_id` | string | yes | Session ID. |

**Returns:** confirmation that the run was stopped, or `status="unsupported"`
for a native run whose compiled trainer ABI cannot be cancelled cooperatively.

---

## Experimental semantic tools

These research tools target the `jepa_semantic_hybrid` stack.

| Tool | Description |
|------|-------------|
| `reverse_engineer_to_semantic` | Encodes text to the vocab-grounded 9-D semantic space exposed by the hybrid preset. |
| `semantic_search` | Queries the experimental semantic search endpoint with a 9-D vector. |
| `train_jepa_semantic` | Starts torch training for a `jepa_semantic_hybrid` graph. |
| `generate_with_semantics` | Reserved for future semantic-conditioned generation workflows. |
