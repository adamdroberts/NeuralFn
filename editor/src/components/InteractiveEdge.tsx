import React, { useState } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from '@xyflow/react';
import {
  resolveTorchTraceStats,
  selectCurrentPath,
  useGraphStore,
} from '../store/graphStore';
import NeuronIcon from './NeuronIcon';

export default function InteractiveEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
  source,
  target
}: EdgeProps) {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const [isHovered, setIsHovered] = useState(false);
  const activeGraph = useGraphStore((state) => state.rootGraph);
  const currentPath = useGraphStore(selectCurrentPath);
  const edgeTelemetry = useGraphStore(state => state.edgeTelemetry);
  const torchTrace = useGraphStore((state) => state.torchTrace);
  const torchTraceSource = useGraphStore((state) => state.torchTraceSource);
  
  // Find the exact node object in the graph
  const searchNode = () => {
     // A proper recursive search or relying on flat namespace
     // For typical operation, we can just look up the target node 
     // since hover shows the transform of the target node.
     let targetNode = null;
     const findIn = (g: any) => {
       if (g.nodes[target]) targetNode = g.nodes[target];
       Object.values(g.nodes).forEach((n: any) => {
         if (n.neuron_def.kind === "subgraph" && n.neuron_def.subgraph) {
           findIn(n.neuron_def.subgraph);
         }
       });
     };
     findIn(activeGraph);
     // If the user wants the transform of the target node:
     return targetNode;
  };

  const targetNode: any = searchNode();
  const functionName = targetNode?.neuron_def?.name || 'identity';
  // Get telemetry array for the source node, since the edge carries the source node's output.
  const telemetryValues = edgeTelemetry[source] || [];
  const sourceTrace = resolveTorchTraceStats(torchTrace, currentPath, source);
  const targetTrace = resolveTorchTraceStats(torchTrace, currentPath, target);
  const scalarPreview = telemetryValues.length > 0;
  const torchPreview = sourceTrace.length > 0 || targetTrace.length > 0;

  const renderTraceCard = (label: string, stats: any[]) => {
    if (stats.length === 0) return null;
    return (
      <div className="w-full rounded border border-gray-800 bg-gray-900/80 px-2 py-1">
        <div className="mb-1 text-[9px] font-semibold uppercase tracking-wider text-gray-500">{label}</div>
        {stats.slice(0, 2).map((stat, idx) => (
          <div key={`${label}-${idx}`} className="mb-1 last:mb-0 text-[9px] text-gray-400 font-mono">
            {stat.kind ? (
              <div>{stat.kind}</div>
            ) : (
              <>
                <div>{JSON.stringify(stat.shape)} {stat.dtype ? `• ${stat.dtype}` : ""}</div>
                <div>
                  mean={stat.mean?.toFixed(4)} std={stat.std?.toFixed(4)} min={stat.min?.toFixed(4)} max={stat.max?.toFixed(4)}
                </div>
                {Array.isArray(stat.preview) && stat.preview.length > 0 && (
                  <div className="truncate text-cyan-300">
                    preview {JSON.stringify(stat.preview)}
                  </div>
                )}
              </>
            )}
          </div>
        ))}
      </div>
    );
  };

  return (
    <>
      <BaseEdge path={edgePath} markerEnd={markerEnd} style={{
        ...style,
        strokeWidth: isHovered ? 3 : 2,
        stroke: isHovered ? '#60a5fa' : '#6b7280',
        transition: 'stroke 0.2s, stroke-width 0.2s',
      }} />

      {/* Invisible bounding path for easy hovering */}
      <path
        d={edgePath}
        fill="none"
        strokeOpacity={0}
        strokeWidth={20}
        className="react-flow__edge-interaction cursor-crosshair"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      />

      {isHovered && targetNode && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -100%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: 'none',
            }}
            className="nodrag nopan flex flex-col items-center bg-gray-950 border border-gray-700 px-3 py-2 rounded shadow-2xl z-[9999]"
          >
            <div className="text-[10px] font-bold text-blue-300 mb-1">
              Target Transform: {functionName}
            </div>
            {targetNode?.neuron_def?.kind === "function" && (
              <div className="bg-gray-900 rounded p-1 mb-1">
                <NeuronIcon
                  name={functionName}
                  expanded={true}
                  animated={scalarPreview}
                  telemetry={telemetryValues}
                />
              </div>
            )}
            {scalarPreview && (
              <div className="text-[9px] text-gray-500 font-mono">
                {telemetryValues.length} samples streaming
              </div>
            )}
            {torchPreview && (
              <div className="mt-1 flex w-full flex-col gap-1">
                {renderTraceCard("Input Sample", sourceTrace)}
                {renderTraceCard("Output Sample", targetTrace)}
                <div className="text-[8px] text-emerald-300">
                  {torchTraceSource === "dataset" ? "Previewing dataset sample" : "Previewing traced tensor flow"}
                </div>
              </div>
            )}
            {!scalarPreview && !torchPreview && (
              <div className="text-[8px] text-amber-500/80">
                Preview unavailable yet
              </div>
            )}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
