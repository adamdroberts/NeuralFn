import React, { useCallback } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  type Connection,
  type Edge as RFEdge,
  type Node,
  type NodeTypes,
  type OnConnect,
  type NodeChange,
  type EdgeChange,
} from "@xyflow/react";
import { NeuronNode } from "./NeuronNode";
import { useGraphStore, type NeuronNodeData } from "../store/graphStore";

const nodeTypes: NodeTypes = { neuron: NeuronNode as any };

export default function GraphCanvas() {
  const {
    nodes,
    edges,
    setNodes,
    setEdges,
    selectNode,
  } = useGraphStore();

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      setNodes(applyNodeChanges(changes, nodes) as Node<NeuronNodeData>[]);
    },
    [nodes, setNodes]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      setEdges(applyEdgeChanges(changes, edges));
    },
    [edges, setEdges]
  );

  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      const newEdge: RFEdge = {
        ...connection,
        id: `e-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        type: "default",
        data: { weight: 1.0, bias: 0.0 },
      } as unknown as RFEdge;
      setEdges(addEdge(newEdge, edges));
    },
    [edges, setEdges]
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      selectNode(node.id);
    },
    [selectNode]
  );

  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  return (
    <div className="flex-1 h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        className="bg-gray-950"
        defaultEdgeOptions={{
          style: { stroke: "#6b7280", strokeWidth: 2 },
          animated: true,
        }}
      >
        <Background color="#374151" gap={20} />
        <Controls className="!bg-gray-800 !border-gray-700 !text-gray-200 [&>button]:!bg-gray-800 [&>button]:!border-gray-700 [&>button]:!text-gray-200" />
        <MiniMap
          nodeColor={() => "#3b82f6"}
          maskColor="rgba(0,0,0,0.7)"
          className="!bg-gray-900 !border-gray-700"
        />
      </ReactFlow>
    </div>
  );
}
