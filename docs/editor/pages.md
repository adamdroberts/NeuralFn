# Pages and Routing

## Pages

### EditorPage

**Default export.**

The main graph editor view. Composes `GraphCanvas`, `LibraryPanel`, `CodePanel`, `TrainingPanel`, `PortConfig`, `DatasetSourcePanel`, and `Toolbar` into a full editing workspace.

Includes an `AgentBanner` that displays a notification when an MCP agent is actively modifying the session graph (detected via the `getAgentStatus` API call).

### LoginPage

**Default export.**

Dual-mode login screen:

- **Normal mode**: email and password form that calls `api.login()`.
- **Bootstrap mode**: shown on first run when no admin user exists. Presents a create-admin form that calls `api.bootstrapAdmin()`.

Mode is determined by the `setup_required` flag from `getBootstrap()`.

### AdminPage

**Default export.**

Administration dashboard for managing users and project memberships:

- User list with role indicators.
- Create-user form (email, password, display name, admin flag).
- Per-project membership table with add/remove controls.

### DatasetsPage

**Default export.**

Dataset catalog and management:

- Table of registered datasets with metadata (source, tokens, rows, format).
- Download form for pulling datasets from HuggingFace.
- Upload form for local dataset files.
- Access management: grant or revoke dataset access per project.
- Delete action with confirmation.

### RunsPage

**Default export.**

Training run history and monitoring:

- Sortable table of past and current training runs with status, method, loss, and timestamps.
- Active run detail panel with live progress updates.
- Stop button for running jobs.

### AnalyticsPage

**Default export.**

Project-level analytics dashboard:

- Summary statistics (total runs, total steps, best loss).
- Charts for loss trends across runs.
- Per-session breakdown of training activity.

---

## App State

`routes/AppState.tsx`

### AppStateProvider

React context provider that wraps the application and supplies authentication and navigation state. Fetches bootstrap status and user data on mount, manages the active project/session selection, and provides the project list.

### useAppState()

Hook returning an `AppStateValue` object:

| Field | Type | Description |
|-------|------|-------------|
| `user` | `UserData \| null` | Current authenticated user |
| `projects` | `ProjectSummary[]` | Projects the user can access |
| `activeProjectId` | `string \| null` | Currently selected project |
| `activeSessionId` | `string \| null` | Currently selected session |
| `setupRequired` | `boolean` | Whether first-run bootstrap is needed |
| `loading` | `boolean` | Whether initial data is still loading |
| `setActiveScope(projectId, sessionId)` | `function` | Updates the active project and session |
| `refreshProjects()` | `function` | Reloads the project list |

---

## Session Sync

`routes/sessionSync.ts`

Utility functions for synchronizing the client-side graph store with the server.

### reloadActiveSessionGraph()

Loads the current session's graph from the server and replaces the store's `rootGraph`. Used on initial hydration and after external changes (e.g. an MCP agent updating the graph).

### syncActiveSessionGraph(options?)

Saves the store's current graph to the server via `putSessionGraph`. Accepts optional parameters:

- `persistSnapshot` -- whether to create a server-side snapshot.
- `snapshotReason` -- human-readable reason for the snapshot.

Handles revision conflicts by re-fetching the server graph and retrying.

---

## Routing

Defined in `App.tsx` using React Router.

| Path | Component | Description |
|------|-----------|-------------|
| `/` | Redirect | Redirects to the user's active workspace. |
| `/login` | `LoginPage` | Authentication (or bootstrap on first run). |
| `/app/projects/:projectId/sessions/:sessionId/editor` | `EditorPage` | Graph editor. |
| `/app/projects/:projectId/sessions/:sessionId/datasets` | `DatasetsPage` | Dataset management. |
| `/app/projects/:projectId/sessions/:sessionId/runs` | `RunsPage` | Training run history. |
| `/app/projects/:projectId/sessions/:sessionId/analytics` | `AnalyticsPage` | Project analytics. |
| `/app/admin` | `AdminPage` | User and membership administration. |

All `/app/*` routes are wrapped in `AppShell` which provides the layout, navigation, and auth guard. Unauthenticated users are redirected to `/login`.
