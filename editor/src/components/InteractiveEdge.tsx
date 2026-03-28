import React, { useState } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from '@xyflow/react';
import { useGraphStore } from '../store/graphStore';
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
  const graph = useGraphStore((state) => state.rootGraph); // Assuming flat scope for now, we could use toActiveGraph
  const activeGraph = useGraphStore((state) => {
    // Basic lookup for the node
    return state.rootGraph; 
  }); 
  const edgeTelemetry = useGraphStore(state => state.edgeTelemetry);
  
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

  // Get telemetry array for the source node, since the edge carries the source node's output!
  // Note: the backend trace returns the values for the instance_id.
  const telemetryValues = edgeTelemetry[source] || [];

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
            <div className="bg-gray-900 rounded p-1 mb-1">
              {/* Force expanded state to show the high res curve + animation */}
              <NeuronIcon 
                name={functionName} 
                expanded={true} 
                animated={true} 
                telemetry={telemetryValues} 
              />
            </div>
            {telemetryValues.length > 0 && (
              <div className="text-[9px] text-gray-500 font-mono">
                {telemetryValues.length} samples streaming
              </div>
            )}
            {telemetryValues.length === 0 && (
              <div className="text-[8px] text-amber-500/80">
                Awaiting Inputs (JSON)...
              </div>
            )}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
