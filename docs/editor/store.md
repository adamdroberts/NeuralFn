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
| `currentPath` | `number[]` | Path into nested subgraphs (empty = root) |
| `selectedNodeId` | `string \| null` | Currently selected node |
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
| `removeNode(nodeId)` | Removes a node and all connected edges from the active graph. |
| `updateNodeData(nodeId, updates)` | Applies partial updates to a node's neuron definition. |
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
| `updateActiveGraphSettings(updates)` | Updates settings (name, training method, configs) on the active subgraph. |

## Selectors

Exported selector functions for use with `useGraphStore(selector)`:

| Selector | Returns |
|----------|---------|
| `selectCurrentPath(state)` | The current subgraph path. |
| `selectActiveGraph(state)` | The `GraphData` at the current path. |
| `selectBreadcrumbs(state)` | `Breadcrumb[]` for the subgraph navigation trail. |
| `selectFlowNodes(state)` | React Flow `Node[]` for the active graph. |
| `selectFlowEdges(state)` | React Flow `Edge[]` for the active graph. |
| `selectSelectedNode(state)` | The full `NodeData` of the selected node, or `null`. |
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
