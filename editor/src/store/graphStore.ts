import { create } from "zustand";
import type { Node, Edge as RFEdge } from "@xyflow/react";
import type { NeuronDefData, PortData } from "../api/client";

export interface NeuronNodeData extends Record<string, unknown> {
  label: string;
  neuronDef: NeuronDefData;
  isInput: boolean;
  isOutput: boolean;
}

export interface LossPoint {
  step: number;
  loss: number;
}

interface GraphState {
  nodes: Node<NeuronNodeData>[];
  edges: RFEdge[];
  selectedNodeId: string | null;
  builtins: NeuronDefData[];
  lossHistory: LossPoint[];
  isTraining: boolean;

  setNodes: (nodes: Node<NeuronNodeData>[]) => void;
  setEdges: (edges: RFEdge[]) => void;
  addNode: (node: Node<NeuronNodeData>) => void;
  removeNode: (id: string) => void;
  updateNodeData: (id: string, data: Partial<NeuronNodeData>) => void;
  selectNode: (id: string | null) => void;
  setBuiltins: (b: NeuronDefData[]) => void;
  addLossPoint: (p: LossPoint) => void;
  clearLoss: () => void;
  setTraining: (v: boolean) => void;

  toggleInput: (id: string) => void;
  toggleOutput: (id: string) => void;
}

export const useGraphStore = create<GraphState>((set) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,
  builtins: [],
  lossHistory: [],
  isTraining: false,

  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),

  addNode: (node) =>
    set((s) => ({ nodes: [...s.nodes, node] })),

  removeNode: (id) =>
    set((s) => ({
      nodes: s.nodes.filter((n) => n.id !== id),
      edges: s.edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: s.selectedNodeId === id ? null : s.selectedNodeId,
    })),

  updateNodeData: (id, data) =>
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, ...data } } : n
      ),
    })),

  selectNode: (id) => set({ selectedNodeId: id }),
  setBuiltins: (builtins) => set({ builtins }),
  addLossPoint: (p) => set((s) => ({ lossHistory: [...s.lossHistory, p] })),
  clearLoss: () => set({ lossHistory: [] }),
  setTraining: (isTraining) => set({ isTraining }),

  toggleInput: (id) =>
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, isInput: !n.data.isInput } } : n
      ),
    })),

  toggleOutput: (id) =>
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, isOutput: !n.data.isOutput } } : n
      ),
    })),
}));
