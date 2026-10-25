import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  type Viewport,
  useReactFlow,
} from "@xyflow/react";
import { NeuronNode } from "./NeuronNode";
import InteractiveEdge from "./InteractiveEdge";
import ParticleGraphCanvas from "./ParticleGraphCanvas";
import {
  FullGraphZonesLayer,
  SubgraphExpansionOverlay,
  useSubgraphExpansion,
} from "./SubgraphExpansionOverlay";
import {
  selectFlowEdges,
  selectFlowNodes,
  useGraphStore,
  type NeuronNodeData,
} from "../store/graphStore";
import {
  buildFullExpandedGraph,
  type GraphPathSegment,
  type SubgraphZone,
} from "../store/graphUtils";
import { useSettingsStore } from "../store/settingsStore";

const nodeTypes: NodeTypes = { neuron: NeuronNode as any };
const edgeTypes = { default: InteractiveEdge };

export default function GraphCanvas() {
  const particleGraphEditorEnabled = useSettingsStore(
    (state) => state.particleGraphEditorEnabled,
  );
  if (particleGraphEditorEnabled) {
    return <ParticleGraphCanvas />;
  }
  return <StandardGraphCanvas />;
}

function StandardGraphCanvas() {
  const viewFullGraph = useSettingsStore((state) => state.viewFullGraph);
  const setViewFullGraph = useSettingsStore((state) => state.setViewFullGraph);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const { screenToFlowPosition, fitView, setViewport, getViewport } = useReactFlow();
  const baseNodes = useGraphStore(selectFlowNodes);
  const baseEdges = useGraphStore(selectFlowEdges);
  const rootGraph = useGraphStore((state) => state.rootGraph);
  const selectedNodeId = useGraphStore((state) => state.selectedNodeId);
  const applyActiveNodeChanges = useGraphStore((state) => state.applyActiveNodeChanges);
  const applyActiveEdgeChanges = useGraphStore((state) => state.applyActiveEdgeChanges);
  const connectActiveGraph = useGraphStore((state) => state.connectActiveGraph);
  const selectNode = useGraphStore((state) => state.selectNode);
  const openSubgraph = useGraphStore((state) => state.openSubgraph);
  const setPath = useGraphStore((state) => state.setPath);
  const currentPath = useGraphStore((state) => state.currentPath);
  const setPreferredInsertPosition = useGraphStore((state) => state.setPreferredInsertPosition);

  const [viewportState, setViewportState] = useState<Viewport>({
    x: 0,
    y: 0,
    zoom: 1,
  });

  // Full Graph Mode expansion
  const fullExpanded = useMemo(() => {
    if (!viewFullGraph) return null;
    return buildFullExpandedGraph(rootGraph, []);
  }, [viewFullGraph, rootGraph]);

  const fullGraphNodes = useMemo(() => {
    if (!fullExpanded) return baseNodes;
    return fullExpanded.nodes.map((node) => ({
      ...node,
      selected: node.id === selectedNodeId,
    }));
  }, [fullExpanded, baseNodes, selectedNodeId]);

  const syncViewportFromAnim = useCallback((vp: Viewport) => {
    setViewportState(vp);
  }, []);

  // Animated Subgraph Click Expansion & Breadcrumb Collapse
  const { expansionState, triggerSubgraphExpansion } = useSubgraphExpansion({
    containerRef,
    rootGraph,
    currentPath,
    activeNodes: baseNodes,
    activeEdges: baseEdges,
    viewFullGraph,
    openSubgraph,
    setViewport,
    getViewport,
    onViewportSync: syncViewportFromAnim,
  });

  const nodes = useMemo(() => {
    if (viewFullGraph && fullExpanded) {
      return fullGraphNodes;
    }
    if (expansionState) {
      const measuredMap = new Map(baseNodes.map((n) => [n.id, n.measured]));
      return expansionState.animatedNodes.map((n) => ({
        ...n,
        measured: measuredMap.get(n.id) ?? n.measured,
      }));
    }
    return baseNodes;
  }, [viewFullGraph, fullExpanded, fullGraphNodes, expansionState, baseNodes]);

  const edges = useMemo(() => {
    if (viewFullGraph && fullExpanded) {
      return fullExpanded.edges;
    }
    if (expansionState) {
      return expansionState.animatedEdges;
    }
    return baseEdges;
  }, [viewFullGraph, fullExpanded, expansionState, baseEdges]);

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

  // Fit view when toggling Full Graph mode
  const prevFullGraphRef = useRef(viewFullGraph);
  useEffect(() => {
    if (prevFullGraphRef.current !== viewFullGraph) {
      prevFullGraphRef.current = viewFullGraph;
      const timer = window.setTimeout(() => {
        fitView({ duration: 480, padding: 0.18 });
      }, 60);
      return () => window.clearTimeout(timer);
    }
  }, [viewFullGraph, fitView]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      if (viewFullGraph) {
        const selectionChanges = changes.filter((c) => c.type === "select");
        if (selectionChanges.length > 0) {
          const selectedChange = selectionChanges.find(
            (c) => c.type === "select" && c.selected,
          );
          if (selectedChange && "id" in selectedChange) {
            selectNode(selectedChange.id);
          }
        }
        return;
      }
      applyActiveNodeChanges(changes);
    },
    [applyActiveNodeChanges, viewFullGraph, selectNode],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      if (viewFullGraph) return;
      applyActiveEdgeChanges(changes);
    },
    [applyActiveEdgeChanges, viewFullGraph],
  );

  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      if (viewFullGraph) return;
      connectActiveGraph(connection);
    },
    [connectActiveGraph, viewFullGraph],
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node<NeuronNodeData>) => {
      selectNode(node.id);
      if (!viewFullGraph && node.data?.neuronDef?.kind === "subgraph") {
        triggerSubgraphExpansion(node);
      }
    },
    [selectNode, viewFullGraph, triggerSubgraphExpansion],
  );

  const onNodeDoubleClick = useCallback(
    (_: React.MouseEvent, node: Node<NeuronNodeData>) => {
      if (node.data.neuronDef.kind === "subgraph") {
        if (!viewFullGraph) {
          triggerSubgraphExpansion(node);
        } else {
          openSubgraph(node.id);
        }
      }
    },
    [openSubgraph, viewFullGraph, triggerSubgraphExpansion],
  );

  const handleIsolateZone = useCallback(
    (path: GraphPathSegment[]) => {
      setViewFullGraph(false);
      setPath(path);
    },
    [setViewFullGraph, setPath],
  );

  const handleFitZone = useCallback(
    (zone: SubgraphZone) => {
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      const pad = 70;
      const targetZoom = Math.max(
        0.25,
        Math.min(
          1.4,
          Math.min(
            (rect.width - pad * 2) / Math.max(240, zone.width),
            (rect.height - pad * 2) / Math.max(180, zone.height),
          ),
        ),
      );
      const centerX = zone.x + zone.width / 2;
      const centerY = zone.y + zone.height / 2;
      setViewport(
        {
          x: rect.width / 2 - centerX * targetZoom,
          y: rect.height / 2 - centerY * targetZoom,
          zoom: targetZoom,
        },
        { duration: 480 },
      );
    },
    [setViewport],
  );

  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  const onMove = useCallback((_: any, vp: Viewport) => {
    setViewportState(vp);
  }, []);

  const onMoveEnd = useCallback(
    (_: any, vp: Viewport) => {
      setViewportState(vp);
      updatePreferredInsertPosition();
    },
    [updatePreferredInsertPosition],
  );

  return (
    <div ref={containerRef} className="relative flex-1 h-full min-h-0 overflow-hidden">
      <FullGraphZonesLayer
        zones={viewFullGraph && fullExpanded ? fullExpanded.zones : []}
        viewport={viewportState}
        cyberMode={false}
        onIsolateZone={handleIsolateZone}
        onFitZone={handleFitZone}
      />
      <SubgraphExpansionOverlay
        expansionState={expansionState}
        cyberMode={false}
      />
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onNodeDoubleClick={onNodeDoubleClick}
        onPaneClick={onPaneClick}
        onMove={onMove}
        onMoveEnd={onMoveEnd}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        nodeDragThreshold={5}
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
