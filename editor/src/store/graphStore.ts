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
import type { GraphData, NeuronDefData, TrainingMethod } from "../api/client";
import {
  type Breadcrumb,
  type FlowNodeData,
  breadcrumbsForPath,
  clampGraphPath,
  createCustomNeuronDef,
  createEmptyGraph,
  createSubgraphNeuronDef,
  flowEdgesToGraphEdges,
  flowNodesToGraphNodes,
  getGraphAtPath,
  graphToFlowEdges,
  graphToFlowNodes,
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
  currentPath: string[];
  selectedNodeId: string | null;
  builtins: NeuronDefData[];
  lossHistory: LossPoint[];
  isTraining: boolean;

  setRootGraph: (graph: GraphData) => void;
  applyActiveNodeChanges: (changes: NodeChange[]) => void;
  applyActiveEdgeChanges: (changes: EdgeChange[]) => void;
  connectActiveGraph: (connection: Connection) => void;
  addBuiltinNode: (ndef: NeuronDefData) => void;
  addCustomNode: () => void;
  addSubgraphNode: () => void;
  removeNode: (id: string) => void;
  updateNodeData: (id: string, data: Partial<NeuronNodeData>) => void;
  selectNode: (id: string | null) => void;
  setBuiltins: (b: NeuronDefData[]) => void;
  addLossPoint: (p: LossPoint) => void;
  clearLoss: () => void;
  setTraining: (v: boolean) => void;

  toggleInput: (id: string) => void;
  toggleOutput: (id: string) => void;
  openSubgraph: (id: string) => void;
  setPath: (path: string[]) => void;
  updateActiveGraphSettings: (patch: {
    name?: string;
    training_method?: TrainingMethod;
    surrogate_config?: Record<string, unknown>;
    evo_config?: Record<string, unknown>;
  }) => void;
}

const initialGraph = createEmptyGraph("Root graph");

function createNodeId(prefix = "n"): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
}

function createNodePosition(baseX = 100, baseY = 100) {
  return [
    baseX + Math.random() * 280,
    baseY + Math.random() * 220,
  ] as [number, number];
}

function finalizeState(
  rootGraph: GraphData,
  currentPath: string[],
  selectedNodeId: string | null,
) {
  const normalizedRoot = normalizeGraph(rootGraph, rootGraph.name || "Root graph");
  const nextPath = clampGraphPath(normalizedRoot, currentPath);
  const activeGraph = getGraphAtPath(normalizedRoot, nextPath);
  return {
    rootGraph: normalizedRoot,
    currentPath: nextPath,
    selectedNodeId:
      selectedNodeId && activeGraph.nodes[selectedNodeId] ? selectedNodeId : null,
  };
}

function mutateActiveGraph(
  state: GraphState,
  updater: (graph: GraphData) => GraphData,
  selectedNodeId = state.selectedNodeId,
) {
  const currentPath = clampGraphPath(state.rootGraph, state.currentPath);
  const rootGraph = updateGraphAtPath(state.rootGraph, currentPath, updater);
  return finalizeState(rootGraph, currentPath, selectedNodeId);
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

export function selectCurrentPath(state: GraphState): string[] {
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
  ...finalizeState(initialGraph, [], null),
  builtins: [],
  lossHistory: [],
  isTraining: false,

  setRootGraph: (graph) =>
    set((state) => ({
      ...state,
      ...finalizeState(graph, state.currentPath, state.selectedNodeId),
    })),

  applyActiveNodeChanges: (changes) =>
    set((state) => {
      const activeGraph = toActiveGraph(state);
      const nextNodes = applyNodeChanges(
        changes,
        graphToFlowNodes(activeGraph),
      ) as Node<NeuronNodeData>[];

      return {
        ...state,
        ...mutateActiveGraph(state, (graph) => {
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
        }),
      };
    }),

  applyActiveEdgeChanges: (changes) =>
    set((state) => {
      const activeGraph = toActiveGraph(state);
      const nextEdges = applyEdgeChanges(changes, graphToFlowEdges(activeGraph));
      return {
        ...state,
        ...mutateActiveGraph(state, (graph) => ({
          ...graph,
          edges: flowEdgesToGraphEdges(nextEdges),
        })),
      };
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

      return {
        ...state,
        ...mutateActiveGraph(state, (graph) => ({
          ...graph,
          edges: flowEdgesToGraphEdges(nextEdges),
        })),
      };
    }),

  addBuiltinNode: (ndef) =>
    set((state) => {
      const { nextState } = addNodeToGraph(state, ndef, {
        idPrefix: "n",
        position: createNodePosition(100, 100),
      });
      return { ...state, ...nextState };
    }),

  addCustomNode: () =>
    set((state) => {
      const { nextState } = addNodeToGraph(state, createCustomNeuronDef("custom"), {
        idPrefix: "n",
        position: createNodePosition(200, 160),
      });
      return { ...state, ...nextState };
    }),

  addSubgraphNode: () =>
    set((state) => {
      const subgraphIndex =
        Object.values(toActiveGraph(state).nodes).filter(
          (node) => node.neuron_def.kind === "subgraph",
        ).length + 1;
      const { nextState } = addNodeToGraph(
        state,
        createSubgraphNeuronDef(`subgraph_${subgraphIndex}`),
        {
          idPrefix: "g",
          position: createNodePosition(180, 120),
        },
      );
      return { ...state, ...nextState };
    }),

  removeNode: (id) =>
    set((state) => ({
      ...state,
      ...mutateActiveGraph(
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
    })),

  updateNodeData: (id, data) =>
    set((state) => ({
      ...state,
      ...mutateActiveGraph(state, (graph) => {
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
    })),

  selectNode: (selectedNodeId) => set({ selectedNodeId }),
  setBuiltins: (builtins) => set({ builtins }),
  addLossPoint: (lossPoint) =>
    set((state) => ({ lossHistory: [...state.lossHistory, lossPoint] })),
  clearLoss: () => set({ lossHistory: [] }),
  setTraining: (isTraining) => set({ isTraining }),

  toggleInput: (id) =>
    set((state) => ({
      ...state,
      ...mutateActiveGraph(state, (graph) => ({
        ...graph,
        input_node_ids: graph.input_node_ids.includes(id)
          ? graph.input_node_ids.filter((nodeId) => nodeId !== id)
          : [...graph.input_node_ids, id],
      })),
    })),

  toggleOutput: (id) =>
    set((state) => ({
      ...state,
      ...mutateActiveGraph(state, (graph) => ({
        ...graph,
        output_node_ids: graph.output_node_ids.includes(id)
          ? graph.output_node_ids.filter((nodeId) => nodeId !== id)
          : [...graph.output_node_ids, id],
      })),
    })),

  openSubgraph: (id) =>
    set((state) => {
      const activeGraph = toActiveGraph(state);
      const node = activeGraph.nodes[id];
      if (!node || node.neuron_def.kind !== "subgraph" || !node.neuron_def.subgraph) {
        return state;
      }

      return {
        ...state,
        ...finalizeState(state.rootGraph, [...selectCurrentPath(state), id], null),
      };
    }),

  setPath: (path) =>
    set((state) => ({
      ...state,
      ...finalizeState(state.rootGraph, path, null),
    })),

  updateActiveGraphSettings: (patch) =>
    set((state) => ({
      ...state,
      ...mutateActiveGraph(state, (graph) => ({
        ...graph,
        ...patch,
        surrogate_config: patch.surrogate_config ?? graph.surrogate_config,
        evo_config: patch.evo_config ?? graph.evo_config,
      })),
    })),
}));
