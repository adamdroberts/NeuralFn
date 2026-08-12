# Training Runs

Training runs are scoped to one authenticated project and editor session. Every
endpoint below uses this prefix:

```text
/api/projects/{project_id}/sessions/{session_id}/runs
```

The caller must have access to the project/session. A hidden or inaccessible
scope is reported as `404`.

## Runtime and request contract

`TrainRequest.runtime` accepts `scalar`, `torch`, or `native-cuda`. If omitted,
the saved graph runtime is used. The common fields are:

| Field | Type | Default | Notes |
|---|---|---|---|
| `method` | string or null | `surrogate` | Requested training method. Native runs resolve to `native-cuda`. |
| `runtime` | string or null | graph runtime | `scalar`, `torch`, or `native-cuda`. |
| `dataset_names` | list[string] or null | null | Project-accessible cached dataset aliases. |
| `train_inputs` | list[list[number]] | `[]` | Inline inputs for non-native runs. |
| `train_targets` | list[list[number]] | `[]` | Inline targets for non-native runs. |
| `seq_len` | integer or null | null | Sequence length override. |
| `epochs` | integer | `200` | Native training interprets this as maximum optimizer steps. |
| `learning_rate` | number | `0.001` | Optimizer learning rate. |
| `batch_size` | integer | `8` | Native trainer microbatch size. |
| `weight_decay` | number | `0.01` | Optimizer weight decay. |
| `training_mode` | string | `pretrain` | Native training currently accepts only `pretrain`. |

Native runs require exactly one project-accessible persisted dataset alias.
They do not accept inline tensors or the in-memory `__semantic_builtin__`
dataset. The alias is passed out of band so adding a `dataset_source` node does
not change an exact reviewed graph topology.

## POST /preflight

Checks the saved graph with the same Native IR compiler and trainer-adapter
registry used by the CLI. It does not create a run or persistent artifacts.

```json
{
  "runtime": "native-cuda",
  "dataset_names": ["tiny_tokens"],
  "epochs": 10,
  "batch_size": 8,
  "training_mode": "pretrain"
}
```

A compatible response is `200`:

```json
{
  "runtime": "native-cuda",
  "compatible": true,
  "execution_ready": true,
  "trainer_family": "gpt2",
  "training_selector": "gpt2",
  "native_target": "nfn_gpt_native_train",
  "issues": [],
  "compatibility_report": {
    "compatible": true,
    "graph_fingerprint": "..."
  },
  "training_compatibility": {
    "compatible": true,
    "issues": []
  },
  "artifact_metadata": {
    "source_graph": "editor-session",
    "materialized": false
  }
}
```

Graph incompatibility is also a `200` preflight result with
`execution_ready: false`. Each error retains its stable graph location:

```json
{
  "runtime": "native-cuda",
  "compatible": false,
  "execution_ready": false,
  "issues": [
    {
      "path": "root/nodes/model/subgraph/nodes/token_embed",
      "code": "unsupported_module",
      "operation": "future_op",
      "message": "No reviewed Native IR lowerer is registered.",
      "severity": "error"
    }
  ]
}
```

Invalid request options, such as a non-pretrain native mode, return `409` with
a string `detail`. Preflight never falls back to Torch.

The currently execution-ready graph-file adapters are `gpt2`,
`gpt2_megakernel`, `gpt2_moa`, `gpt2_qknorm`, `gpt2_softcap`, `gpt2_stable`,
`gpt2_zloss`, canonical `llama`, and its exact compile-runtime alias
`llama_fast`, exact standard-MoE `moe`, `mixllama`, and `mixllama_fast`, plus
trusted-planner proof-bound `gpt2_diff` training. That is 13 graph-training
ready and 53 blocked shipped presets. `gpt2_diff` migration and resident
inference remain blocked because its
graph-bound learned-lambda bundle is not consumed by migration or resident
inference and exact low-level differential execution is packed-QKV-only. Its
low-level `neuralfn.native_gpt2_diff.training_checkpoint` version-2 metadata is
a strict continuation contract over the source, five binaries, training shards,
counters/sampler, seed, accumulation, optimizer/LR horizon, BF16 routes, and
canonical numerics profile of supported effective routes before Tile/CUDA/H2D. Validation
shards are excluded. REST preparation materializes the exact
graph/fingerprint/proof triplet and completion accepts only the fully validated
`.diff.json`; the proof digest is local-handoff integrity, not caller
authenticity. Migration/resident readiness remains 12/54.
LLaMA preparation derives all trainer
geometry plus the source SHA-256 from the graph, and the public SDK re-runs the
same planner while constructing the command. The compile alias uses the
canonical `llama` native ABI while retaining source-profile provenance. A graph being structurally lowerable does not make another
adapter executable.

## POST /

Starts a run and returns a Server-Sent Events stream (`text/event-stream`). A
native request should use the same body first sent to `/preflight`:

```json
{
  "method": "torch",
  "runtime": "native-cuda",
  "dataset_names": ["tiny_tokens"],
  "epochs": 10,
  "learning_rate": 0.001,
  "batch_size": 8,
  "weight_decay": 0.01,
  "training_mode": "pretrain"
}
```

Each SSE frame is a JSON object:

```text
data: {"event_id":1,"status":"starting","message":"Training session started using native-cuda method"}

data: {"event_id":2,"status":"checkpoint_persisted","checkpoint_path":".../model_00000010.bin"}

data: {"done":true}
```

The final frame closes the stream. Poll `GET /active` for the authoritative
terminal status and persisted metadata.

Ordinary dense runs persist a recognized non-empty `.bin`. Graph-bound
`gpt2_moa` runs persist validated `model_XXXXXXXX.moa.json`, not the sibling `.bin`, after the
named model, empty DONE marker, source hash, candidates, selected activation,
and positive interval validate. MoA resume requires the same sidecar and
restores the selected activation without a fresh probe; missing or changed
metadata fails closed. This is the graph-bound REST workflow; direct
selector-only first-leg output remains ordinary dense-v5 and cannot resume
exactly. Canonical LLaMA runs persist
the validated `*_native_family_model_00000000.json` metadata path, not its raw
`.f32` sidecar. Discovery validates the v2 tensor/sidecar/DONE contract and
requires `training.source_graph.sha256` to match the prepared plan before the
run can complete. The corresponding source record uses
`byte_identity_verified: true`; topology preflight remains the Python planner's
separate `graph_preflight_enforced` assertion.

If native compatibility changes between preflight and launch, or the saved
graph has no reviewed adapter, start returns `409` before a trainer is
launched. Its response is `{"detail": <preflight metadata>}` and includes the
same node-specific `issues`. Other request conflicts use a string `detail`.

## POST /start

Starts the same background run as `POST /`, but returns the initial JSON run
snapshot instead of holding an SSE connection. This is the acknowledgement
endpoint used by MCP automation.

```json
{
  "run_id": "run-uuid",
  "status": "running",
  "running": true,
  "runtime": "native-cuda",
  "compatibility_report": {"compatible": true, "graph_fingerprint": "..."},
  "artifact_metadata": {
    "materialized": true,
    "manifest_path": ".../native-ir/native-execution-manifest.json",
    "training_plan_path": ".../native-ir/native-training-plan.json",
    "checkpoint_dir": ".../checkpoints",
    "checkpoint_path": null
  },
  "checkpoint_path": null
}
```

The compatibility and conflict behavior is identical to `POST /`.

## GET /

Lists up to 100 persisted runs, newest first. Each row includes `id`, `status`,
`requested_method`, `resolved_method`, `runtime`, dataset/step/loss fields,
`compatibility_report`, `artifact_metadata`, `checkpoint_path`, and timestamps.

## GET /active

Returns the live run snapshot when a run is active, otherwise the most recent
persisted run. If the session has never run, the response is:

```json
{
  "status": "idle",
  "running": false,
  "done": false,
  "events": []
}
```

Native snapshots keep the Native IR compatibility report and artifact paths
from run creation. On successful completion, both top-level `checkpoint_path`
and `artifact_metadata.checkpoint_path` identify the verified, non-empty,
contained checkpoint: a recognized `.bin` for ordinary dense runs, validated
`.moa.json` for MoA, or validated inference-checkpoint metadata `.json` for
canonical LLaMA and standard MoE.

## POST /{run_id}/stop

Requests cooperative cancellation for a running non-native trainer:

```json
{"status": "stopping"}
```

The current compiled native trainer ABI does not expose cooperative
cancellation. An active native run therefore returns:

```json
{
  "status": "unsupported",
  "message": "The current compiled native trainer ABI does not expose cooperative cancellation."
}
```

This response does not claim the native process stopped. A run not active in
the server process returns `{"status": "not_running"}`.

## Native artifact persistence

Successful launch materializes immutable per-run data below
`NEURALFN_ARTIFACTS_DIR/runs/{run_id}/`:

```text
editor-graph.json
native-ir/
  source-graph.json
  native-execution-manifest.json
  compatibility-report.json
  native-training-plan.json
checkpoints/
  model_*.bin
  llama_native_family_model_00000000.json
  llama_native_family_parameters_00000000.f32
  llama_native_family_optimizer_00000000.bin
  llama_native_family_model_DONE
```

The server stores the runtime, compatibility report, artifact metadata, and
checkpoint path on `training_runs`. Migration `20260804_0003` adds those
columns for existing databases.
