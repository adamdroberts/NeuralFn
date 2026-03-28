import React, { useCallback } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Connection,
  type EdgeChange,
  type Node,
  type NodeChange,
  type NodeTypes,
  type OnConnect,
} from "@xyflow/react";
import { NeuronNode } from "./NeuronNode";
import InteractiveEdge from "./InteractiveEdge";
import {
  selectFlowEdges,
  selectFlowNodes,
  useGraphStore,
  type NeuronNodeData,
} from "../store/graphStore";

const nodeTypes: NodeTypes = { neuron: NeuronNode as any };
const edgeTypes = { default: InteractiveEdge };

export default function GraphCanvas() {
  const nodes = useGraphStore(selectFlowNodes);
  const edges = useGraphStore(selectFlowEdges);
  const applyActiveNodeChanges = useGraphStore((state) => state.applyActiveNodeChanges);
  const applyActiveEdgeChanges = useGraphStore((state) => state.applyActiveEdgeChanges);
  const connectActiveGraph = useGraphStore((state) => state.connectActiveGraph);
  const selectNode = useGraphStore((state) => state.selectNode);
  const openSubgraph = useGraphStore((state) => state.openSubgraph);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      applyActiveNodeChanges(changes);
    },
    [applyActiveNodeChanges]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      applyActiveEdgeChanges(changes);
    },
    [applyActiveEdgeChanges]
  );

  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      connectActiveGraph(connection);
    },
    [connectActiveGraph]
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      selectNode(node.id);
    },
    [selectNode]
  );

  const onNodeDoubleClick = useCallback(
    (_: React.MouseEvent, node: Node<NeuronNodeData>) => {
      if (node.data.neuronDef.kind === "subgraph") {
        openSubgraph(node.id);
      }
    },
    [openSubgraph]
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
        onNodeDoubleClick={onNodeDoubleClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        className="bg-gray-950"
        defaultEdgeOptions={{
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
