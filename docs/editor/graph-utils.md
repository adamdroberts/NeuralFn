# Graph Utilities

`editor/src/store/graphUtils.ts` provides pure functions for creating, normalizing, converting, and navigating neural-network graphs. These utilities are used by the Zustand store and by components that need to inspect or transform graph data.

## Interfaces and Types

### FlowNodeData

The `data` payload carried by every React Flow node the editor renders (`NeuronNodeData` in the store is an alias for it). It extends `Record<string, unknown>` with:

| Field | Type | Set by | Purpose |
|-------|------|--------|---------|
| `label` | `string` | `graphToFlowNodes` | Display name (`neuron_def.name`). |
| `neuronDef` | `NeuronDefData` | `graphToFlowNodes` | Normalized neuron definition for the node. |
| `isInput` / `isOutput` | `boolean` | `graphToFlowNodes` | Whether the node is listed in the graph's `input_node_ids` / `output_node_ids`. |
| `subgraphPath?` | `GraphPathSegment[]` | `buildFullExpandedGraph`, `selectSelectedNode` | Navigation path of the graph that owns the node (variant-linked ancestors become `variant` segments). |
| `instancePath?` | `GraphPathSegment[]` | same | Instance path of the owning graph (always `node` segments, as the graph is nested inline). |
| `localNodeId?` | `string` | same | The node's id inside its owning graph, as opposed to the scoped full-graph id used as the React Flow `id`. |
| `breadcrumbLabel?` | `string` | same | `"Parent ▸ Child"` trail shown on nested leaves in View Full Graph mode. |
| `spawnScale?` | `number` | `useSubgraphExpansion` | 0..1 scale applied while a node is being spawned by the expansion animation. |
| `titleDetached?` | `number` | `useSubgraphExpansion` | 0..1 progress of the parent node's title plate lifting away during expansion/collapse. |
| `coreOnlyMorph?` | `number` | `useSubgraphExpansion` | 0..1; while non-zero the node renders only its 3D core (no shell, halo, or ports). |
| `coreMorphProgress?` | `number` | `useSubgraphExpansion` | 0..1 interpolation of the node's 3D wireframe from `coreMorphFrom` to its own archetype. |
| `coreMorphFrom?` | `CoreMorphSource` | `useSubgraphExpansion` | The parent's `{archetypeId, neuronName, primary, secondary}` the wireframe morphs from. |
| `shellExpandProgress?` | `number` | `useSubgraphExpansion` | 0..1 growth of the outer box, border, ports, and labels outward from the core. |

The animation-only fields are consumed by `ParticleNeuronNode` / `VolumetricNodeCore` (cyber mode) and `NeuronNode` (standard mode) and are never written back to `GraphData`.

### CoreMorphSource

```typescript
interface CoreMorphSource {
  archetypeId: string;   // NodeArchetypeId of the parent node
  neuronName: string;
  primary: string;       // hex colors used while morphing
  secondary: string;
}
```

### GraphPathSegment

```typescript
type GraphPathSegment =
  | { kind: "node"; nodeId: string }
  | { kind: "variant"; family: string; version: string };
```

A `node` segment descends into a node's inline `neuron_def.subgraph`; a `variant` segment descends into `rootGraph.variant_library[family][version]`. `useGraphStore.currentPath` and every path helper below use this type. Two flavours of path appear in the code:

- **Navigation path** (`subgraphPath`, `currentPath`, `Breadcrumb.path`): what the breadcrumb bar and `getGraphAtPath` follow. Variant-linked subgraph nodes appear as `variant` segments so edits land in the variant library entry.
- **Instance path** (`instancePath`): the same descent expressed purely as `node` segments through the inline graph snapshots. `resolveEditableGraphPath` converts an instance path into a navigation path.

### Breadcrumb

```typescript
interface Breadcrumb {
  id: string;
  label: string;
  path: GraphPathSegment[];
}
```

Used by the breadcrumb bar to show the current position within nested subgraphs. Clicking a parent crumb calls `setPath(crumb.path)` (and turns View Full Graph off first); the canvases detect the shorter path and play the collapse animation.

### SubgraphZone

Describes one expanded subgraph container in View Full Graph mode (see [Full Graph Expansion](#full-graph-expansion)).

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | `makeFullGraphNodeId(parentInstancePath, "zone:<nodeId>")`. |
| `label` | `string` | Subgraph node name (falls back to the node id). |
| `breadcrumb` | `string` | `"Outer ▸ Inner"` trail of subgraph names down to this zone. |
| `path` | `GraphPathSegment[]` | Navigation path to open the zone's graph (used by `⤢ Isolate`). |
| `instancePath` | `GraphPathSegment[]` | Instance path of the zone's graph. |
| `x`, `y`, `width`, `height` | `number` | Zone rectangle in flow coordinates, including padding. |
| `depth` | `number` | Nesting depth (0 = direct child of the start graph). Zones are sorted ascending so outer zones render behind inner ones. |
| `variantLabel` | `string \| null` | `"family@version"` when the subgraph node carries a `variant_ref`. |
| `nodeCount` | `number` | Number of leaf nodes rendered inside the zone (recursively). |

### FullExpandedGraphResult

```typescript
interface FullExpandedGraphResult {
  nodes: Node<FlowNodeData>[];
  edges: RFEdge[];
  zones: SubgraphZone[];
}
```

## Factory Functions

| Function | Description |
|----------|-------------|
| `createEmptyGraph()` | Returns a new `GraphData` with empty nodes, edges, and default settings. |
| `createCustomNeuronDef(name, source_code, input_ports, output_ports)` | Builds a `NeuronDefData` with `kind: "custom"` and the given source code and ports. |
| `createSubgraphNeuronDef(name?)` | Builds a `NeuronDefData` with `kind: "subgraph"` and an empty nested graph. |
| `createLinkedVariantNeuronDef(family, version, graph)` | Builds a `NeuronDefData` linked to a variant library entry, with a snapshot of the variant's graph as inline subgraph. |

## Normalization

| Function | Description |
|----------|-------------|
| `normalizeGraph(graph)` | Fills in missing fields with defaults, ensuring a complete `GraphData` shape. Recursive -- also normalizes nested subgraphs. |
| `normalizeNeuronDef(def)` | Fills in missing fields on a `NeuronDefData` with defaults. |

## Port Derivation and Compatibility

| Function | Description |
|----------|-------------|
| `deriveExternalPorts(graph)` | Inspects a subgraph's input/output node IDs and returns `{inputs, outputs}` port lists representing the subgraph's external interface. |
| `areGraphInterfacesCompatible(a, b)` | Returns `true` if two graphs have the same number and names of input and output ports. Used to check whether a variant swap is safe. |

## Variant Library

| Function | Description |
|----------|-------------|
| `mergeVariantLibraries(target, source)` | Merges entries from `source` into `target`, overwriting families and versions that exist in both. Returns a new library object. |
| `listCompatibleVariantVersions(library, family)` | Returns all version names within a family whose graph interface is compatible with the family's default version. |

## Flow Conversion

These functions bridge between the NeuralFn `GraphData` format and React Flow's node/edge format.

| Function | Description |
|----------|-------------|
| `graphToFlowNodes(graph, builtins)` | Converts `NodeData[]` to React Flow `Node[]` with `FlowNodeData` payloads. Resolves builtin neuron definitions from the builtins catalog. |
| `graphToFlowEdges(graph)` | Converts `EdgeData[]` to React Flow `Edge[]`. |
| `flowNodesToGraphNodes(nodes)` | Extracts `NodeData[]` from React Flow nodes (inverse of `graphToFlowNodes`). |
| `flowEdgesToGraphEdges(edges)` | Extracts `EdgeData[]` from React Flow edges (inverse of `graphToFlowEdges`). |

## Path Navigation

Functions for navigating into and manipulating nested subgraph hierarchies.

| Function | Description |
|----------|-------------|
| `getGraphAtPath(root, path)` | Traverses the root graph along the given path and returns the subgraph at that depth. Returns the root graph if the path is empty. |
| `clampGraphPath(root, path)` | Validates a path against the current graph structure, truncating it if any segment points to a node that no longer has a subgraph. |
| `updateGraphAtPath(root, path, updater)` | Immutably updates the subgraph at the given path by applying an updater function, then reconstructs the full root graph. |
| `breadcrumbsForPath(root, path)` | Generates a `Breadcrumb[]` array for the given path, with labels derived from node names at each level. |

## Subgraph Detection

| Function | Description |
|----------|-------------|
| `graphContainsSubgraphs(graph)` | Returns `true` if any node in the graph has a non-empty subgraph. Used to determine whether subgraph navigation UI should be shown. |

## Full Graph Expansion

These helpers power **View Full Graph** mode (`viewFullGraph` in [`useSettingsStore`](store.md#settings-store-usesettingsstore)): both canvases replace the active subgraph with a single flattened graph in which every nested subgraph is drawn inline inside a labelled zone. The user-facing walkthrough is in [Cyber Mode, Full Graph View, and Panel Dialogs](cyber-mode.md#view-full-graph-mode).

### Scoped node ids

| Function | Description |
|----------|-------------|
| `makeFullGraphNodeId(instancePath, nodeId)` | Returns `nodeId` unchanged when `instancePath` is empty (root-level nodes keep their plain ids). Otherwise returns `fullg::<path>::<nodeId>` where `<path>` joins the instance path with `/`, encoding node segments as `encodeURIComponent(nodeId)` and variant segments as `v@<family>:<version>`. |
| `parseFullGraphNodeId(fullId)` | Inverse of the above. Returns `{instancePath, nodeId}` for ids that start with `fullg::`, or `null` for plain ids. `useGraphStore` uses it in `normalizeState`, `selectSelectedNode`, `removeNode`, and `updateNodeData` so a nested leaf can be selected and edited straight from the full-graph view. |
| `resolveEditableGraphPath(rootGraph, instancePath)` | Walks the instance path from the root and rewrites each `node` segment whose node has a `variant_ref` into the matching `variant` segment. The result is the navigation path to pass to `updateGraphAtPath` so edits reach the variant-library entry rather than the inline snapshot. Stops early (returns the prefix) if a segment no longer resolves. |

### buildFullExpandedGraph(rootGraph, startPath = [])

Returns a `FullExpandedGraphResult` for the graph at `startPath` with every nested subgraph expanded recursively. The algorithm, per graph level:

1. **Pass-through terminals.** In non-root graphs, `input` function nodes in `input_node_ids` that have at least one outgoing edge, and `output` function nodes in `output_node_ids` that have at least one incoming edge, are treated as pass-through terminals and are not rendered; their edges are bridged to the parent graph's wiring instead. If an edge connects a pass-through input directly to a pass-through output, both are kept visible so the wire is not lost.
2. **Leaves vs. zones.** Every other node becomes either a leaf (a React Flow node with id `makeFullGraphNodeId(instancePath, nodeId)`) or, when it is a `subgraph` neuron with a non-empty `subgraph` and the instance path is shorter than 6 segments, a zone that contains the recursive expansion of its subgraph. Deeper subgraphs are rendered as ordinary leaves.
3. **Layout.** Items are sorted by original X then Y and grouped into columns whose original X values lie within 75 px of the column anchor. Leaves are laid out at 226 x 118 px with 76 px column gaps and 38 px row gaps; each column is vertically centred on the tallest column. Zones are sized to their content plus 32 px horizontal, 48 px top, and 28 px bottom padding, and child nodes/zones are offset into the zone.
4. **Port bridging.** External input/output port indices are computed the same way as `deriveExternalPorts` (one external port per output port of each input node, and per output port of each output node). For every edge, the source's resolved output endpoints and the target's resolved input endpoints are cross-joined into edges with id `fe::<src>:<port>-><dst>:<port>`, `sourceHandle: out-<n>`, `targetHandle: in-<n>`, and the original `weight` / `bias` in `data`. A zone's endpoints are the endpoints of its internal pass-through terminals, so residual streams connect end-to-end across nested blocks.
5. **Zone ordering.** Collected zones are sorted by `depth` ascending so outer containers paint behind inner ones in `FullGraphZonesLayer`.

Leaf nodes carry `subgraphPath`, `instancePath`, `localNodeId`, and `breadcrumbLabel` in their `FlowNodeData`; `isInput` / `isOutput` are only set for root-level nodes. The result is memoized by the canvases on `[viewFullGraph, rootGraph]` and is purely visual: node drags and edge edits are ignored while View Full Graph is on, and only selection is forwarded to the store.
