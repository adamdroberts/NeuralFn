# API Client

`editor/src/api/client.ts` provides the typed HTTP client used by the editor to communicate with the NeuralFn server. All requests include `credentials: "include"` so the session cookie is sent automatically.

## ApiError

```typescript
class ApiError extends Error {
  status: number;
  detail: string;
}
```

Thrown by API methods when the server returns a non-2xx response. Route-level error handling can inspect `status` for 401/403/404 distinctions.

## DTO Interfaces

These interfaces mirror the server-side Pydantic models and are used for request/response typing throughout the editor.

| Interface | Key Fields |
|-----------|------------|
| `PortData` | `name`, `range`, `precision`, `dtype` |
| `NeuronDefData` | `id`, `name`, `kind`, `input_ports`, `output_ports`, `source_code`, `subgraph`, `module_type`, `module_config`, `module_state`, `input_aliases`, `output_aliases`, `variant_ref` |
| `NodeData` | `instance_id`, `neuron_def`, `position` |
| `EdgeData` | `id`, `src_node`, `src_port`, `dst_node`, `dst_port`, `weight`, `bias` |
| `GraphData` | `name`, `training_method`, `runtime`, `surrogate_config`, `evo_config`, `torch_config`, `variant_library`, `nodes`, `edges`, `input_node_ids`, `output_node_ids` |
| `VariantRefData` | `family`, `version` |
| `TrainingMessage` | Training progress event from SSE stream |
| `TorchTraceStat` | Per-layer timing/shape statistics from a traced forward pass |
| `TorchTraceResponse` | Wraps trace stats with metadata |
| `GPTTemplateResponse` | Template graph, variant library, and template spec |
| `DatasetInfo` | Dataset catalog entry with metadata |
| `UserData` | User profile (id, email, display_name, is_admin) |
| `ProjectSummary` | Project listing entry |
| `SessionSummary` | Session listing entry |
| `SessionDetail` | Full session with graph and snapshot list |
| `ProjectCreateResponse` | Newly created project with first session |
| `MembershipInfo` | Project membership record |
| `ProjectAnalytics` | Aggregated training statistics |
| `RunSnapshot` | Training run status, loss, step, events |
| `BootstrapResponse` | First-run bootstrap result |

## Types

- **`TrainingMethod`** -- string union of supported training methods.
- **`VariantLibraryData`** -- `Record<string, Record<string, GraphData>>`, mapping family names to version-keyed subgraphs.

## API Methods

The exported `api` object provides the following methods:

### Bootstrap and Auth

| Method | HTTP | Description |
|--------|------|-------------|
| `getBootstrap()` | `GET /api/bootstrap` | Returns app status (setup required, current user). |
| `bootstrapAdmin(req)` | `POST /api/bootstrap` | Creates the first admin user. |
| `login(req)` | `POST /api/auth/login` | Authenticates and sets the session cookie. |
| `logout()` | `POST /api/auth/logout` | Clears the session. |
| `me()` | `GET /api/auth/me` | Returns the current authenticated user. |

### Projects and Sessions

| Method | HTTP | Description |
|--------|------|-------------|
| `setActiveSession(req)` | `POST /api/auth/active-session` | Sets the user's active project and session. |
| `listProjects()` | `GET /api/projects` | Lists projects the user can access. |
| `createProject(req)` | `POST /api/projects` | Creates a new project. |
| `getProjectAnalytics(id)` | `GET /api/projects/:id/analytics` | Returns project-level training analytics. |
| `listSessions(projectId)` | `GET /api/projects/:id/sessions` | Lists sessions in a project. |
| `createSession(projectId, req)` | `POST /api/projects/:id/sessions` | Creates a new editor session. |
| `getSession(sessionId)` | `GET /api/sessions/:id` | Returns session detail with graph. |
| `getSessionGraph(sessionId)` | `GET /api/sessions/:id/graph` | Returns just the session graph. |
| `putSessionGraph(sessionId, req)` | `PUT /api/sessions/:id/graph` | Updates the session graph with revision. |

### Execution and Training

| Method | HTTP | Description |
|--------|------|-------------|
| `execute(sessionId, req)` | `POST /api/sessions/:id/execute` | Runs a forward pass and returns outputs. |
| `executeTrace(sessionId, req)` | `POST /api/sessions/:id/execute-trace` | Runs a traced forward pass with edge telemetry. |
| `traceTorchPreview(sessionId, req)` | `POST /api/sessions/:id/trace-torch` | Compiles to Torch and runs a traced preview. |
| `startTraining(sessionId, req)` | `POST /api/sessions/:id/train` | Starts training via SSE. Returns an `AbortController` for cancellation. |
| `getActiveRun(sessionId)` | `GET /api/sessions/:id/active-run` | Returns the active run for a session. |
| `listRuns(projectId)` | `GET /api/runs?project_id=...` | Lists training runs. |
| `stopTraining(runId)` | `POST /api/runs/:id/stop` | Stops a running training job. |

### Catalog and Datasets

| Method | HTTP | Description |
|--------|------|-------------|
| `getBuiltins()` | `GET /api/catalog/builtins` | Returns all builtin neuron definitions. |
| `buildGPTTemplate(req)` | `POST /api/catalog/gpt-template` | Builds a GPT template graph from a preset. |
| `getAgentStatus(sessionId)` | `GET /api/sessions/:id/agent-status` | Checks if an MCP agent is active. |
| `getDatasets(projectId)` | `GET /api/datasets` | Lists datasets, optionally filtered by project. |
| `downloadDataset(req)` | `POST /api/datasets/download` | Downloads a HuggingFace dataset. |
| `uploadDataset(formData)` | `POST /api/datasets/upload` | Uploads a dataset file. |
| `setDatasetAccess(id, req)` | `PUT /api/datasets/:id/access` | Updates project access for a dataset. |
| `deleteDataset(id)` | `DELETE /api/datasets/:id` | Deletes a dataset. |

### Users and Admin

| Method | HTTP | Description |
|--------|------|-------------|
| `listUsers()` | `GET /api/admin/users` | Lists all registered users. |
| `createUser(req)` | `POST /api/admin/users` | Creates a new user account. |
| `listMemberships(projectId)` | `GET /api/projects/:id/members` | Lists project memberships. |
| `addMembership(projectId, req)` | `POST /api/projects/:id/members` | Adds a member to a project. |
