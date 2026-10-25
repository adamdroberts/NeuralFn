# Graph Store

`editor/src/store/graphStore.ts` exports `useGraphStore`, a **Zustand** store hook that holds all client-side state for the graph editor.

## State Fields

### Session State

| Field | Type | Description |
|-------|------|-------------|
| `projectId` | `string \| null` | Active project ID |
| `sessionId` | `string \| null` | Active editor session ID |
| `revision` | `number` | Current graph revision for optimistic concurrency |
| `hydrationState` | `"idle" \| "loading" \| "ready" \| "error"` | Session loading lifecycle |
| `isDirty` | `boolean` | Whether the graph has unsaved changes |
| `isSaving` | `boolean` | Whether a save is in progress |
| `saveError` | `string \| null` | Last save error message |

### Graph State

| Field | Type | Description |
|-------|------|-------------|
| `rootGraph` | `GraphData` | The top-level graph for the session |
| `currentPath` | `GraphPathSegment[]` | Navigation path into nested subgraphs (empty = root). Each segment is `{kind: "node", nodeId}` or `{kind: "variant", family, version}`; see [Graph Utilities](graph-utils.md#graphpathsegment). |
| `selectedNodeId` | `string \| null` | Currently selected node. In View Full Graph mode this may be a scoped full-graph id (`fullg::<path>::<nodeId>`) pointing at a nested leaf; `normalizeState` keeps such ids only while the target node still exists. |
| `preferredInsertPosition` | `{x, y} \| null` | Hint for where to place new nodes |
| `insertSequence` | `number` | Auto-incrementing counter for insert positioning |

### Catalog State

| Field | Type | Description |
|-------|------|-------------|
| `builtins` | `NeuronDefData[]` | Loaded builtin neuron definitions |

### Training State

| Field | Type | Description |
|-------|------|-------------|
| `lossHistory` | `LossPoint[]` | Accumulated loss values for charting |
| `isTraining` | `boolean` | Whether a training run is active |
| `edgeTelemetry` | `Record<string, number>` | Per-edge activation magnitudes |
| `torchTrace` | `TorchTraceResponse \| null` | Latest Torch trace result |
| `torchTraceSource` | `string \| null` | Which action produced the trace |
| `lastError` | `string \| null` | Last execution or training error |

## Actions

### Session Actions

| Action | Description |
|--------|-------------|
| `hydrateSession(projectId, sessionId)` | Loads the session graph from the server and transitions to "ready". |
| `setHydrationState(state)` | Manually sets the hydration lifecycle state. |
| `markSessionSaved(revision)` | Clears `isDirty` and updates the revision after a successful save. |
| `setSaving(flag)` | Sets the `isSaving` flag. |
| `setSaveError(error)` | Records a save error message. |
| `setRootGraph(graph)` | Replaces the root graph (used during hydration). |
| `setPreferredInsertPosition(pos)` | Sets the hint position for the next node insertion. |

### Canvas Actions

| Action | Description |
|--------|-------------|
| `applyActiveNodeChanges(changes)` | Applies React Flow node change events (position, selection, removal) to the active subgraph. |
| `applyActiveEdgeChanges(changes)` | Applies React Flow edge change events to the active subgraph. |
| `connectActiveGraph(connection)` | Creates a new edge from a React Flow connection event. |

### Node Actions

| Action | Description |
|--------|-------------|
| `addBuiltinNode(defId)` | Adds an instance of a builtin neuron to the active graph. |
| `addCustomNode(name, source_code, inputs, outputs)` | Creates and adds a custom neuron with user-provided source code. |
| `addSubgraphNode(name?)` | Adds an empty subgraph neuron. |
| `addVariantNode(family, version)` | Adds a node linked to a variant library entry. |
| `mergeVariantLibrary(library)` | Merges external variant definitions into the root graph's variant library. |
| `saveNodeAsVariant(nodeId, family, version)` | Saves a node's subgraph into the variant library. |
| `swapNodeVariant(nodeId, version)` | Swaps a variant node to a different version within its family. |
| `removeNode(nodeId)` | Removes a node and all connected edges. Accepts a plain id (active graph) or a scoped full-graph id, which is resolved with `parseFullGraphNodeId` + `resolveEditableGraphPath` so a nested leaf selected in View Full Graph mode is deleted inside its own subgraph or variant-library entry. |
| `updateNodeData(nodeId, updates)` | Applies partial updates (`label` and/or `neuronDef`) to a node's neuron definition. Accepts plain or scoped full-graph ids, resolved the same way as `removeNode`. |
| `selectNode(nodeId)` | Sets the selected node. |
| `setBuiltins(defs)` | Stores the loaded builtin neuron definitions. |

### Training Actions

| Action | Description |
|--------|-------------|
| `addLossPoint(point)` | Appends a data point to the loss history. |
| `clearLoss()` | Resets the loss history. |
| `setTraining(flag)` | Sets the training-active flag. |
| `updateEdgeTelemetry(data)` | Updates per-edge activation values for visualization. |
| `updateTorchTrace(trace, source)` | Stores a Torch trace result. |
| `clearError()` | Clears the last error message. |

### Navigation Actions

| Action | Description |
|--------|-------------|
| `toggleInput(nodeId)` | Toggles a node as a graph input. |
| `toggleOutput(nodeId)` | Toggles a node as a graph output. |
| `openSubgraph(nodeId)` | Navigates into a node's subgraph by extending `currentPath`. |
| `openVariant(nodeId)` | Navigates into a variant node's resolved subgraph. |
| `setPath(path)` | Sets the subgraph navigation path directly. |
| `updateActiveGraphSettings(updates)` | Updates settings (name, training method, `scalar | torch | native-cuda` runtime, configs) on the active subgraph. |

## Selectors

Exported selector functions for use with `useGraphStore(selector)`:

| Selector | Returns |
|----------|---------|
| `selectCurrentPath(state)` | The current subgraph path. |
| `selectActiveGraph(state)` | The `GraphData` at the current path. |
| `selectBreadcrumbs(state)` | `Breadcrumb[]` for the subgraph navigation trail. |
| `selectFlowNodes(state)` | React Flow `Node[]` for the active graph. |
| `selectFlowEdges(state)` | React Flow `Edge[]` for the active graph. |
| `selectSelectedNode(state)` | The React Flow `Node<NeuronNodeData>` for the selected node, or `null`. If `selectedNodeId` is a scoped full-graph id, the node is synthesized from the nested graph with `subgraphPath`, `instancePath`, `localNodeId`, and `breadcrumbLabel` filled in so `CodePanel` can edit it without navigating. |
| `torchTracePathPrefix(state)` | The trace path prefix matching the current subgraph depth. |
| `resolveTorchTraceStats(state)` | Filtered `TorchTraceStat[]` for the active subgraph level. |

## Supporting Types

### LossPoint

```typescript
interface LossPoint {
  step: number;
  loss: number;
}
```

### NeuronNodeData

Type alias for `FlowNodeData`, used as the data payload on React Flow nodes.

## Settings Store (`useSettingsStore`)

`editor/src/store/settingsStore.ts` exports `useSettingsStore`, a second Zustand store that holds client-side feature flags, the Volumetric 3D Cyber mode visual settings, the **View Full Graph** toggle, and the editor's panel-dialog UI state. It is independent of `useGraphStore`, is never sent to the server, and is read by `GraphCanvas`, `ParticleGraphCanvas`, `Toolbar`, `EditorPage`, `AppShell`, `CodePanel`, `LibraryPanel`, and `SettingsModal`. The feature walkthrough lives in [Cyber Mode, Full Graph View, and Panel Dialogs](cyber-mode.md).

### Persistence

The `PersistedEditorSettings` subset is written to `window.localStorage` under the key `neuralfn:editor_settings:v1` after every change and read back once when the store module is first evaluated. Loading is defensive: each field is validated against its allowed values and falls back to its default when the stored value is missing or invalid, and a corrupt payload or a storage error (private mode, quota) yields the defaults. Transient fields are never persisted and reset on every page load.

### Persisted fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `particleGraphEditorEnabled` | `boolean` | `false` | Feature flag for the Volumetric 3D Cyber graph editor. When `true`, `GraphCanvas` renders `ParticleGraphCanvas` instead of the standard React Flow canvas and the shell, toolbar, and panels switch to the cyber cockpit styling. |
| `particleVisualMode` | `"hybrid_hud" \| "particle_nebula"` | `"hybrid_hud"` | `hybrid_hud` renders multi-plane CSS 3D node chassis (`ParticleNeuronNode`) with orbital particle halos; `particle_nebula` renders compact floating 3D sculptures with laser callout banners. |
| `particleDensity` | `"balanced" \| "high" \| "ultra"` | `"high"` | Particle count budget across the ambient field, node-orbital swarms, and edge photon streams. |
| `colorPalette` | `"cyber_plasma" \| "solaris_gold" \| "quantum_violet" \| "crimson_protocol"` | `"cyber_plasma"` | Active holographic palette; resolved through `PARTICLE_PALETTES`. |
| `holoPerspective3D` | `boolean` | `false` | Tilts the whole canvas into a 3D holographic-table projection. |
| `motionIntensity` | `"smooth" \| "cinematic"` | `"cinematic"` | Spring damping and vortex-transition impulse strength for particle retargeting. |
| `hudTelemetryOverlay` | `boolean` | `true` | Shows the background telemetry schematic and the bottom-right tactical radar in cyber mode. |
| `viewFullGraph` | `boolean` | `false` | View Full Graph mode: both canvases render `buildFullExpandedGraph(rootGraph)` instead of the active subgraph. Applies to the standard and cyber canvases alike. |

### Transient fields

| Field | Type | Initial | Description |
|-------|------|---------|-------------|
| `isSettingsOpen` | `boolean` | `false` | Whether `SettingsModal` is shown. |
| `activePanelDialog` | `"left" \| "bottom" \| "right" \| null` | `null` | Which `EditorPage` panel dialog is open: left = Variant Library, bottom = Training & Trace, right = Inspector / Chat. `null` closes all. |
| `rightPanelTab` | `"inspector" \| "chat"` | `"inspector"` | Active tab inside the right dialog (`CodePanel` or `ChatPanel`). |
| `pulseSequence` | `number` | `0` | Monotonic counter; each increment makes `ParticleGraphCanvas` fire a shockwave and particle surge across the graph. |

### Actions

| Action | Effect |
|--------|--------|
| `setParticleGraphEditorEnabled(enabled)` | Sets the cyber-mode flag and persists. |
| `toggleParticleGraphEditor()` | Flips the cyber-mode flag and persists. Bound to the toolbar `✦ Cyber 3D` button. |
| `setParticleVisualMode(mode)` | Sets `particleVisualMode` and persists. |
| `setParticleDensity(density)` | Sets `particleDensity` and persists. |
| `setColorPalette(palette)` | Sets `colorPalette` and persists. The cyber HUD color-matrix button cycles through the four palettes with this action. |
| `setHoloPerspective3D(enabled)` | Sets `holoPerspective3D` and persists. |
| `setMotionIntensity(intensity)` | Sets `motionIntensity` and persists. |
| `setHudTelemetryOverlay(enabled)` | Sets `hudTelemetryOverlay` and persists. |
| `setViewFullGraph(enabled)` / `toggleViewFullGraph()` | Set or flip View Full Graph mode and persist. Bound to the toolbar `⊞ View Full Graph` button, the cyber HUD `FULL GRAPH` button, the `SettingsModal` checkbox, and the `⤢ Isolate` zone action (which turns it off before navigating). |
| `setSettingsOpen(open)` | Shows or hides `SettingsModal`. |
| `setActivePanelDialog(dialog)` | Opens one panel dialog (`"left"`, `"bottom"`, `"right"`) or closes all (`null`). `EditorPage` also calls this with `null` on `Escape` and on backdrop click. |
| `openRightPanelDialog(tab = "inspector")` | Opens the right dialog and selects the `inspector` or `chat` tab in one update. |
| `triggerPulse()` | Increments `pulseSequence`. Bound to the cyber HUD `⚡ PULSE` button. |
| `resetSettings()` | Restores every persisted field to its default and persists; transient fields are untouched. Bound to the reset button in `SettingsModal`. |

### Palettes

`PARTICLE_PALETTES` is a `Record<ParticleColorPalette, PaletteColors>` exported from the same module. Each `PaletteColors` entry carries `id`, `name`, `subtitle`, hex and `[r, g, b]` pairs for `primary`, `secondary`, `accent`, `subgraph`, and `module`, plus `terminalIn`, `terminalOut`, `bgCore`, and `gridColor` strings. The canvases, `WebGLCyberScene`, and `SettingsModal` all read colors from this table rather than hardcoding them.

| Palette id | Display name | Primary | Secondary | Accent |
|------------|--------------|---------|-----------|--------|
| `cyber_plasma` | Cyber Plasma Core | `#00f0ff` | `#38bdf8` | `#f59e0b` |
| `solaris_gold` | Solaris Nanotech | `#fbbf24` | `#f97316` | `#00f0ff` |
| `quantum_violet` | Quantum Nebula | `#a855f7` | `#06b6d4` | `#f472b6` |
| `crimson_protocol` | Crimson Tactical HUD | `#fb7185` | `#f97316` | `#facc15` |

### Exported types

`ParticleVisualMode`, `ParticleDensity`, `ParticleColorPalette`, `MotionIntensity`, `PaletteColors`, `PersistedEditorSettings`, `EditorPanelDialog`, `RightPanelTab`, and `SettingsState` (the full store shape including actions).
