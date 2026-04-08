# Graph Utilities

`editor/src/store/graphUtils.ts` provides pure functions for creating, normalizing, converting, and navigating neural-network graphs. These utilities are used by the Zustand store and by components that need to inspect or transform graph data.

## Interfaces and Types

### FlowNodeData

Extends React Flow's node data with NeuralFn-specific fields:

- `neuron_def` -- the full `NeuronDefData` for the node.
- Graph path information for breadcrumb navigation and subgraph resolution.

### GraphPathSegment

Type alias for a single entry in a subgraph navigation path (numeric index into a parent node's subgraph children).

### Breadcrumb

```typescript
interface Breadcrumb {
  label: string;
  path: number[];
}
```

Used by the breadcrumb bar to show the current position within nested subgraphs.

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
