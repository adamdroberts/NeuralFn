# Training Runs

Endpoints for starting, monitoring, and stopping training runs.

All endpoints are prefixed with:

```
/api/projects/{project_id}/sessions/{session_id}/runs
```

**Authentication:** required for all endpoints.

---

## GET /

Lists all training runs for the session.

**Response:**

```json
[
  {
    "id": "r_001",
    "status": "completed",
    "method": "surrogate",
    "epochs": 10,
    "final_loss": 0.042,
    "created_at": "2025-03-15T10:30:00Z"
  },
  {
    "id": "r_002",
    "status": "stopped",
    "method": "torch",
    "epochs": 50,
    "final_loss": 0.15,
    "created_at": "2025-03-16T14:00:00Z"
  }
]
```

---

## GET /active

Returns a snapshot of the currently running training run. If no run is active, returns an idle placeholder.

**Response (active run):**

```json
{
  "id": "r_003",
  "status": "running",
  "method": "torch",
  "current_epoch": 7,
  "total_epochs": 50,
  "current_loss": 0.089,
  "events": []
}
```

**Response (no active run):**

```json
{
  "id": null,
  "status": "idle"
}
```

---

## POST /

Starts a new training run. The response is a **Server-Sent Event (SSE)** stream (`text/event-stream`) that delivers progress updates in real time.

**Request Body:** `TrainRequest`

```json
{
  "method": "surrogate",
  "epochs": 10,
  "learning_rate": 0.001,
  "dataset_names": ["tiny_shakespeare"],
  "train_inputs": null,
  "train_targets": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `method` | string | no | Training method (`"surrogate"`, `"torch"`, etc.). |
| `epochs` | int | no | Number of training epochs. |
| `learning_rate` | float | no | Learning rate. |
| `dataset_names` | list[string] | no | Datasets to train on. |
| `train_inputs` | dict | no | Explicit training inputs (alternative to datasets). |
| `train_targets` | dict | no | Explicit training targets (alternative to datasets). |

### SSE Stream Format

Each event is a JSON object. Events include epoch completions, loss updates, and a final summary.

```
data: {"event": "epoch", "epoch": 1, "loss": 0.95, "lr": 0.001}

data: {"event": "epoch", "epoch": 2, "loss": 0.42, "lr": 0.001}

data: {"event": "done", "done": true, "final_loss": 0.042, "run_id": "r_003"}
```

The stream closes after the `done` event.

---

## POST /{run_id}/stop

Stops a running training run. The run transitions to `"stopped"` status and the SSE stream closes.

### Path Parameters

| Parameter | Description |
|-----------|-------------|
| `run_id` | The run to stop. |

**Response:** stop confirmation with final run state.

Returns `404` if the run does not exist. Returns `400` if the run is not currently active.
