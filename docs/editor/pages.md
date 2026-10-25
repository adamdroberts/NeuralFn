# Pages and Routing

## Pages

### EditorPage

**Default export.**

The main graph editor view. The stage below `Toolbar` is given entirely to `GraphCanvas`; the side panels are no longer docked but open as modal dialogs driven by `activePanelDialog` / `rightPanelTab` in [`useSettingsStore`](store.md#settings-store-usesettingsstore). See [Cyber Mode, Full Graph View, and Panel Dialogs](cyber-mode.md#full-screen-canvas-and-panel-dialogs) for the user-facing walkthrough.

Layout, top to bottom:

1. `AgentBanner` -- polls `api.getAgentStatus(projectId, sessionId)` every 2 s and shows a purple banner while an MCP agent controls the session; when the agent releases the session it calls `reloadActiveSessionGraph()` so the canvas picks up the agent's edits.
2. A one-line status strip (`Loading session...`, `Saving...`, the last save error, `Unsaved changes`, or `Revision <n>`). In cyber mode it also shows `SYS.MODE // FULL CYBER 3D COCKPIT`.
3. `Toolbar`.
4. The canvas stage: `GraphCanvas` plus three floating launcher groups positioned over the canvas edges:
   - left edge `◈ Variants` with the variant-family count, opens the **left** dialog;
   - right edge `◧ Inspector` (shows the selected node's label and a `Selected` badge when a node is selected) and `💬 Model Chat`, open the **right** dialog on the `inspector` or `chat` tab;
   - bottom edge `⚡ Training & Trace` (or `⚡ Training Active` while a run is live) with the active graph's `training_method · runtime` and the latest loss, opens the **bottom** dialog.

Dialogs:

| Dialog | Contents | Mounting |
|--------|----------|----------|
| left | `LibraryPanel` (`onDone` closes the dialog after a variant is opened or inserted) | Rendered only while open. |
| bottom | `TrainingPanel` | Always mounted, hidden with `display: none` when closed, so active training runs, SSE loss streaming, and torch-trace polling keep running. |
| right | `CodePanel` (`inspector` tab) and `ChatPanel` (`chat` tab) | Always mounted; the inactive tab is hidden so Monaco editor state and the chat transcript persist. |

Every dialog closes on `Escape`, on its `✕ Close (ESC)` button, or on a backdrop click; `EditorPage` registers the `keydown` listener only while a dialog is open.

Session lifecycle is unchanged: on mount it calls `api.getSession` and `hydrateSession`, and while `hydrationState === "ready"` and the store is dirty it debounces `syncActiveSessionGraph({ skipIfClean: true })` by 600 ms. `PortConfig` and `DatasetSourcePanel` are reached through `CodePanel` inside the right dialog.

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
