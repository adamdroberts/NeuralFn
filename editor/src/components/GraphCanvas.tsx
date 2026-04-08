import React, { useCallback, useEffect, useRef } from "react";
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
  useReactFlow,
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
  const containerRef = useRef<HTMLDivElement | null>(null);
  const { screenToFlowPosition } = useReactFlow();
  const nodes = useGraphStore(selectFlowNodes);
  const edges = useGraphStore(selectFlowEdges);
  const applyActiveNodeChanges = useGraphStore((state) => state.applyActiveNodeChanges);
  const applyActiveEdgeChanges = useGraphStore((state) => state.applyActiveEdgeChanges);
  const connectActiveGraph = useGraphStore((state) => state.connectActiveGraph);
  const selectNode = useGraphStore((state) => state.selectNode);
  const openSubgraph = useGraphStore((state) => state.openSubgraph);
  const currentPath = useGraphStore((state) => state.currentPath);
  const setPreferredInsertPosition = useGraphStore((state) => state.setPreferredInsertPosition);

  const updatePreferredInsertPosition = useCallback(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    const rect = container.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return;
    }
    setPreferredInsertPosition(
      screenToFlowPosition({
        x: rect.left + rect.width / 2,
        y: rect.top + rect.height / 2,
      }),
    );
  }, [screenToFlowPosition, setPreferredInsertPosition]);

  useEffect(() => {
    const frameId = window.requestAnimationFrame(updatePreferredInsertPosition);
    const container = containerRef.current;
    let resizeObserver: ResizeObserver | null = null;
    if (container && typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => {
        updatePreferredInsertPosition();
      });
      resizeObserver.observe(container);
    }
    window.addEventListener("resize", updatePreferredInsertPosition);
    return () => {
      window.cancelAnimationFrame(frameId);
      resizeObserver?.disconnect();
      window.removeEventListener("resize", updatePreferredInsertPosition);
    };
  }, [currentPath, updatePreferredInsertPosition]);

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

  const onMoveEnd = useCallback(() => {
    updatePreferredInsertPosition();
  }, [updatePreferredInsertPosition]);

  return (
    <div ref={containerRef} className="flex-1 h-full min-h-0">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onNodeDoubleClick={onNodeDoubleClick}
        onPaneClick={onPaneClick}
        onMoveEnd={onMoveEnd}
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
