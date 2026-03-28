import { create } from "zustand";
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge as RFEdge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from "@xyflow/react";
import type { GraphData, NeuronDefData, TrainingMethod, VariantLibraryData } from "../api/client";
import {
  type Breadcrumb,
  type FlowNodeData,
  type GraphPathSegment,
  breadcrumbsForPath,
  clampGraphPath,
  createCustomNeuronDef,
  createEmptyGraph,
  createLinkedVariantNeuronDef,
  createSubgraphNeuronDef,
  flowEdgesToGraphEdges,
  flowNodesToGraphNodes,
  getGraphAtPath,
  graphToFlowEdges,
  graphToFlowNodes,
  listCompatibleVariantVersions,
  mergeVariantLibraries,
  normalizeGraph,
  normalizeNeuronDef,
  updateGraphAtPath,
} from "./graphUtils";

export type NeuronNodeData = FlowNodeData;

export interface LossPoint {
  step: number;
  loss: number;
  graphName?: string;
  method?: string;
}

interface GraphState {
  rootGraph: GraphData;
  currentPath: GraphPathSegment[];
  selectedNodeId: string | null;
  builtins: NeuronDefData[];
  lossHistory: LossPoint[];
  isTraining: boolean;
  edgeTelemetry: Record<string, number[]>;
  lastError: string | null;

  setRootGraph: (graph: GraphData) => void;
  applyActiveNodeChanges: (changes: NodeChange[]) => void;
  applyActiveEdgeChanges: (changes: EdgeChange[]) => void;
  connectActiveGraph: (connection: Connection) => void;
  addBuiltinNode: (ndef: NeuronDefData, pos?: { x: number; y: number }) => void;
  addCustomNode: (pos?: { x: number; y: number }) => void;
  addSubgraphNode: (pos?: { x: number; y: number }) => void;
  addVariantNode: (family: string, version: string, pos?: { x: number; y: number }) => void;
  mergeVariantLibrary: (library: VariantLibraryData) => void;
  saveNodeAsVariant: (nodeId: string, family: string, version: string, linkNode?: boolean) => void;
  swapNodeVariant: (nodeId: string, family: string, version: string) => void;
  removeNode: (id: string) => void;
  updateNodeData: (id: string, data: Partial<NeuronNodeData>) => void;
  selectNode: (id: string | null) => void;
  setBuiltins: (b: NeuronDefData[]) => void;
  addLossPoint: (p: LossPoint) => void;
  clearLoss: () => void;
  setTraining: (v: boolean) => void;
  updateEdgeTelemetry: (t: Record<string, number[]>) => void;
  clearError: () => void;

  toggleInput: (id: string) => void;
  toggleOutput: (id: string) => void;
  openSubgraph: (id: string) => void;
  openVariant: (family: string, version: string) => void;
  setPath: (path: GraphPathSegment[]) => void;
  updateActiveGraphSettings: (patch: {
    name?: string;
    training_method?: TrainingMethod;
    runtime?: "scalar" | "torch";
    surrogate_config?: Record<string, unknown>;
    evo_config?: Record<string, unknown>;
    torch_config?: Record<string, unknown>;
  }) => void;
}

const initialGraph = createEmptyGraph("Root graph");

function createNodeId(prefix = "n"): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
}

function createNodePosition(baseX = 100, baseY = 100) {
  return [baseX + Math.random() * 280, baseY + Math.random() * 220] as [number, number];
}

function normalizeState(
  state: GraphState,
  rootGraph: GraphData,
  currentPath: GraphPathSegment[],
  selectedNodeId: string | null,
) {
  const normalizedRoot = normalizeGraph(rootGraph, rootGraph.name || "Root graph");
  const nextPath = clampGraphPath(normalizedRoot, currentPath);
  const activeGraph = getGraphAtPath(normalizedRoot, nextPath);
  return {
    ...state,
    rootGraph: normalizedRoot,
    currentPath: nextPath,
    selectedNodeId: selectedNodeId && activeGraph.nodes[selectedNodeId] ? selectedNodeId : null,
    lastError: null,
  };
}

function withSafety(
  state: GraphState,
  build: () => {
    rootGraph: GraphData;
    currentPath?: GraphPathSegment[];
    selectedNodeId?: string | null;
  },
): GraphState {
  try {
    const next = build();
    return normalizeState(
      state,
      next.rootGraph,
      next.currentPath ?? state.currentPath,
      Object.prototype.hasOwnProperty.call(next, "selectedNodeId")
        ? (next.selectedNodeId ?? null)
        : state.selectedNodeId,
    );
  } catch (error) {
    return {
      ...state,
      lastError: error instanceof Error ? error.message : "Unknown graph error",
    };
  }
}

function mutateActiveGraph(
  state: GraphState,
  updater: (graph: GraphData) => GraphData,
  selectedNodeId = state.selectedNodeId,
): GraphState {
  return withSafety(state, () => {
    const currentPath = clampGraphPath(state.rootGraph, state.currentPath);
    return {
      rootGraph: updateGraphAtPath(state.rootGraph, currentPath, updater),
      currentPath,
      selectedNodeId,
    };
  });
}

function addNodeToGraph(
  state: GraphState,
  ndef: NeuronDefData,
  options?: {
    idPrefix?: string;
    position?: [number, number];
  },
) {
  const instanceId = createNodeId(options?.idPrefix);
  const nodePosition = options?.position ?? createNodePosition();
  const neuronDef = normalizeNeuronDef({
    ...ndef,
    id: instanceId,
  });

  return {
    instanceId,
    nextState: mutateActiveGraph(
      state,
      (graph) => ({
        ...graph,
        nodes: {
          ...graph.nodes,
          [instanceId]: {
            instance_id: instanceId,
            position: nodePosition,
            neuron_def: neuronDef,
          },
        },
      }),
      instanceId,
    ),
  };
}

function toActiveGraph(state: GraphState): GraphData {
  return getGraphAtPath(state.rootGraph, clampGraphPath(state.rootGraph, state.currentPath));
}

export function selectCurrentPath(state: GraphState): GraphPathSegment[] {
  return clampGraphPath(state.rootGraph, state.currentPath);
}

export function selectActiveGraph(state: GraphState): GraphData {
  return toActiveGraph(state);
}

export function selectBreadcrumbs(state: GraphState): Breadcrumb[] {
  return breadcrumbsForPath(state.rootGraph, state.currentPath);
}

export function selectFlowNodes(state: GraphState): Node<NeuronNodeData>[] {
  const nodes = graphToFlowNodes(selectActiveGraph(state));
  for (const node of nodes) {
    if (node.id === state.selectedNodeId) {
      node.selected = true;
    }
  }
  return nodes;
}

export function selectFlowEdges(state: GraphState): RFEdge[] {
  return graphToFlowEdges(selectActiveGraph(state));
}

export function selectSelectedNode(state: GraphState): Node<NeuronNodeData> | null {
  const selectedNodeId = state.selectedNodeId;
  if (!selectedNodeId) {
    return null;
  }
  return selectFlowNodes(state).find((node) => node.id === selectedNodeId) ?? null;
}

export const useGraphStore = create<GraphState>((set) => ({
  ...normalizeState(
    {
      rootGraph: initialGraph,
      currentPath: [],
      selectedNodeId: null,
      builtins: [],
      lossHistory: [],
      isTraining: false,
      edgeTelemetry: {},
      lastError: null,
      setRootGraph: () => undefined,
      applyActiveNodeChanges: () => undefined,
      applyActiveEdgeChanges: () => undefined,
      connectActiveGraph: () => undefined,
      addBuiltinNode: () => undefined,
      addCustomNode: () => undefined,
      addSubgraphNode: () => undefined,
      addVariantNode: () => undefined,
      mergeVariantLibrary: () => undefined,
      saveNodeAsVariant: () => undefined,
      swapNodeVariant: () => undefined,
      removeNode: () => undefined,
      updateNodeData: () => undefined,
      selectNode: () => undefined,
      setBuiltins: () => undefined,
      addLossPoint: () => undefined,
      clearLoss: () => undefined,
      setTraining: () => undefined,
      updateEdgeTelemetry: () => undefined,
      clearError: () => undefined,
      toggleInput: () => undefined,
      toggleOutput: () => undefined,
      openSubgraph: () => undefined,
      openVariant: () => undefined,
      setPath: () => undefined,
      updateActiveGraphSettings: () => undefined,
    },
    initialGraph,
    [],
    null,
  ),
  setRootGraph: (graph) => set((state) => withSafety(state, () => ({ rootGraph: graph }))),

  applyActiveNodeChanges: (changes) =>
    set((state) => {
      const activeGraph = toActiveGraph(state);
      const nextNodes = applyNodeChanges(
        changes,
        graphToFlowNodes(activeGraph),
      ) as Node<NeuronNodeData>[];

      return mutateActiveGraph(state, (graph) => {
        const graphNodes = flowNodesToGraphNodes(graph, nextNodes);
        const validIds = new Set(Object.keys(graphNodes));
        return {
          ...graph,
          nodes: graphNodes,
          edges: Object.fromEntries(
            Object.entries(graph.edges).filter(
              ([, edge]) => validIds.has(edge.src_node) && validIds.has(edge.dst_node),
            ),
          ),
          input_node_ids: graph.input_node_ids.filter((id) => validIds.has(id)),
          output_node_ids: graph.output_node_ids.filter((id) => validIds.has(id)),
        };
      });
    }),

  applyActiveEdgeChanges: (changes) =>
    set((state) => {
      const activeGraph = toActiveGraph(state);
      const nextEdges = applyEdgeChanges(changes, graphToFlowEdges(activeGraph));
      return mutateActiveGraph(state, (graph) => ({
        ...graph,
        edges: flowEdgesToGraphEdges(nextEdges),
      }));
    }),

  connectActiveGraph: (connection) =>
    set((state) => {
      const activeGraph = toActiveGraph(state);
      const nextEdges = addEdge(
        {
          ...connection,
          id: createNodeId("e"),
          type: "default",
          data: { weight: 1.0, bias: 0.0 },
        } as RFEdge,
        graphToFlowEdges(activeGraph),
      );
      return mutateActiveGraph(state, (graph) => ({
        ...graph,
        edges: flowEdgesToGraphEdges(nextEdges),
      }));
    }),

  addBuiltinNode: (ndef, pos) =>
    set((state) => {
      const { nextState } = addNodeToGraph(state, ndef, {
        idPrefix: "n",
        position: pos ? [pos.x, pos.y] : createNodePosition(100, 100),
      });
      return nextState;
    }),

  addCustomNode: (pos) =>
    set((state) => {
      const { nextState } = addNodeToGraph(state, createCustomNeuronDef("custom"), {
        idPrefix: "n",
        position: pos ? [pos.x, pos.y] : createNodePosition(200, 160),
      });
      return nextState;
    }),

  addSubgraphNode: (pos) =>
    set((state) => {
      const subgraphIndex =
        Object.values(toActiveGraph(state).nodes).filter(
          (node) => node.neuron_def.kind === "subgraph",
        ).length + 1;
      const { nextState } = addNodeToGraph(state, createSubgraphNeuronDef(`subgraph_${subgraphIndex}`), {
        idPrefix: "g",
        position: pos ? [pos.x, pos.y] : createNodePosition(180, 120),
      });
      return nextState;
    }),

  addVariantNode: (family, version, pos) =>
    set((state) => {
      const variantGraph = state.rootGraph.variant_library[family]?.[version];
      if (!variantGraph) {
        return {
          ...state,
          lastError: `Variant '${family}@${version}' not found`,
        };
      }
      const { nextState } = addNodeToGraph(
        state,
        createLinkedVariantNeuronDef(`${family}_${version}`, { family, version }, variantGraph),
        {
          idPrefix: "g",
          position: pos ? [pos.x, pos.y] : createNodePosition(220, 180),
        },
      );
      return nextState;
    }),

  mergeVariantLibrary: (library) =>
    set((state) =>
      withSafety(state, () => ({
        rootGraph: {
          ...state.rootGraph,
          variant_library: mergeVariantLibraries(state.rootGraph.variant_library, library),
        },
      })),
    ),

  saveNodeAsVariant: (nodeId, family, version, linkNode = true) =>
    set((state) =>
      withSafety(state, () => {
        const activeGraph = toActiveGraph(state);
        const node = activeGraph.nodes[nodeId];
        const subgraph = node?.neuron_def.subgraph;
        if (!node || node.neuron_def.kind !== "subgraph" || !subgraph) {
          throw new Error("Selected node is not a subgraph");
        }
        const rootWithVariant = {
          ...state.rootGraph,
          variant_library: {
            ...state.rootGraph.variant_library,
            [family]: {
              ...(state.rootGraph.variant_library[family] ?? {}),
              [version]: subgraph,
            },
          },
        };
        if (!linkNode) {
          return { rootGraph: rootWithVariant };
        }
        return {
          rootGraph: updateGraphAtPath(rootWithVariant, clampGraphPath(rootWithVariant, state.currentPath), (graph) => ({
            ...graph,
            nodes: {
              ...graph.nodes,
              [nodeId]: {
                ...graph.nodes[nodeId],
                neuron_def: createLinkedVariantNeuronDef(
                  graph.nodes[nodeId].neuron_def.name,
                  { family, version },
                  subgraph,
                  graph.nodes[nodeId].neuron_def.input_aliases,
                  graph.nodes[nodeId].neuron_def.output_aliases,
                ),
              },
            },
          })),
          selectedNodeId: nodeId,
        };
      }),
    ),

  swapNodeVariant: (nodeId, family, version) =>
    set((state) =>
      withSafety(state, () => {
        const variantGraph = state.rootGraph.variant_library[family]?.[version];
        const activeGraph = toActiveGraph(state);
        const node = activeGraph.nodes[nodeId];
        const currentGraph = node?.neuron_def.subgraph;
        if (!variantGraph || !node || node.neuron_def.kind !== "subgraph" || !currentGraph) {
          throw new Error("Cannot swap variant for the selected node");
        }
        const compatibleVersions = listCompatibleVariantVersions(state.rootGraph, family, currentGraph);
        if (!compatibleVersions.includes(version)) {
          throw new Error(`Variant '${family}@${version}' is not interface-compatible with '${node.neuron_def.name}'`);
        }
        return {
          rootGraph: updateGraphAtPath(state.rootGraph, clampGraphPath(state.rootGraph, state.currentPath), (graph) => ({
            ...graph,
            nodes: {
              ...graph.nodes,
              [nodeId]: {
                ...graph.nodes[nodeId],
                neuron_def: createLinkedVariantNeuronDef(
                  graph.nodes[nodeId].neuron_def.name,
                  { family, version },
                  variantGraph,
                  graph.nodes[nodeId].neuron_def.input_aliases,
                  graph.nodes[nodeId].neuron_def.output_aliases,
                ),
              },
            },
          })),
          selectedNodeId: nodeId,
        };
      }),
    ),

  removeNode: (id) =>
    set((state) =>
      mutateActiveGraph(
        state,
        (graph) => {
          const nextNodes = { ...graph.nodes };
          delete nextNodes[id];
          const validIds = new Set(Object.keys(nextNodes));
          return {
            ...graph,
            nodes: nextNodes,
            edges: Object.fromEntries(
              Object.entries(graph.edges).filter(
                ([, edge]) => validIds.has(edge.src_node) && validIds.has(edge.dst_node),
              ),
            ),
            input_node_ids: graph.input_node_ids.filter((nodeId) => nodeId !== id),
            output_node_ids: graph.output_node_ids.filter((nodeId) => nodeId !== id),
          };
        },
        state.selectedNodeId === id ? null : state.selectedNodeId,
      ),
    ),

  updateNodeData: (id, data) =>
    set((state) =>
      mutateActiveGraph(state, (graph) => {
        const node = graph.nodes[id];
        if (!node) {
          return graph;
        }
        const nextNeuronDef = data.neuronDef
          ? normalizeNeuronDef(data.neuronDef)
          : normalizeNeuronDef({
              ...node.neuron_def,
              name: data.label ?? node.neuron_def.name,
            });
        return {
          ...graph,
          nodes: {
            ...graph.nodes,
            [id]: {
              ...node,
              neuron_def: nextNeuronDef,
            },
          },
        };
      }),
    ),

  selectNode: (selectedNodeId) => set((state) => ({ ...state, selectedNodeId })),
  setBuiltins: (builtins) => set((state) => ({ ...state, builtins })),
  addLossPoint: (lossPoint) =>
    set((state) => ({ ...state, lossHistory: [...state.lossHistory, lossPoint] })),
  clearLoss: () => set((state) => ({ ...state, lossHistory: [] })),
  setTraining: (isTraining) => set((state) => ({ ...state, isTraining })),
  updateEdgeTelemetry: (edgeTelemetry) => set((state) => ({ ...state, edgeTelemetry })),
  clearError: () => set((state) => ({ ...state, lastError: null })),

  toggleInput: (id) =>
    set((state) =>
      mutateActiveGraph(state, (graph) => ({
        ...graph,
        input_node_ids: graph.input_node_ids.includes(id)
          ? graph.input_node_ids.filter((nodeId) => nodeId !== id)
          : [...graph.input_node_ids, id],
      })),
    ),

  toggleOutput: (id) =>
    set((state) =>
      mutateActiveGraph(state, (graph) => ({
        ...graph,
        output_node_ids: graph.output_node_ids.includes(id)
          ? graph.output_node_ids.filter((nodeId) => nodeId !== id)
          : [...graph.output_node_ids, id],
      })),
    ),

  openSubgraph: (id) =>
    set((state) => {
      const activeGraph = toActiveGraph(state);
      const node = activeGraph.nodes[id];
      if (!node || node.neuron_def.kind !== "subgraph" || !node.neuron_def.subgraph) {
        return state;
      }
      if (node.neuron_def.variant_ref) {
        const variantRef = node.neuron_def.variant_ref;
        return withSafety(state, () => ({
          rootGraph: state.rootGraph,
          currentPath: [
            ...selectCurrentPath(state),
            { kind: "variant", family: variantRef.family, version: variantRef.version },
          ],
          selectedNodeId: null,
        }));
      }
      return withSafety(state, () => ({
        rootGraph: state.rootGraph,
        currentPath: [...selectCurrentPath(state), { kind: "node", nodeId: id }],
        selectedNodeId: null,
      }));
    }),

  openVariant: (family, version) =>
    set((state) =>
      withSafety(state, () => ({
        rootGraph: state.rootGraph,
        currentPath: [{ kind: "variant", family, version }],
        selectedNodeId: null,
      })),
    ),

  setPath: (path) =>
    set((state) =>
      withSafety(state, () => ({
        rootGraph: state.rootGraph,
        currentPath: path,
        selectedNodeId: null,
      })),
    ),

  updateActiveGraphSettings: (patch) =>
    set((state) =>
      mutateActiveGraph(state, (graph) => ({
        ...graph,
        ...patch,
        runtime:
          patch.runtime ??
          (patch.training_method === "torch" ? "torch" : graph.runtime),
        surrogate_config: patch.surrogate_config ?? graph.surrogate_config,
        evo_config: patch.evo_config ?? graph.evo_config,
        torch_config: patch.torch_config ?? graph.torch_config,
      })),
    ),
}));
