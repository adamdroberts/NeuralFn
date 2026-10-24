import React, { useState } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from "@xyflow/react";
import {
  resolveTorchTraceStats,
  selectCurrentPath,
  useGraphStore,
} from "../store/graphStore";
import { PARTICLE_PALETTES, useSettingsStore } from "../store/settingsStore";
import NeuronIcon from "./NeuronIcon";

export default function ParticleEdge({
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
  target,
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
  const edgeTelemetry = useGraphStore((state) => state.edgeTelemetry);
  const torchTrace = useGraphStore((state) => state.torchTrace);
  const torchTraceSource = useGraphStore((state) => state.torchTraceSource);
  const isTraining = useGraphStore((state) => state.isTraining);

  const colorPalette = useSettingsStore((state) => state.colorPalette);
  const palette = PARTICLE_PALETTES[colorPalette];

  const searchNode = () => {
    let targetNode = null;
    const findIn = (g: any) => {
      if (g?.nodes?.[target]) targetNode = g.nodes[target];
      Object.values(g?.nodes ?? {}).forEach((n: any) => {
        if (n.neuron_def?.kind === "subgraph" && n.neuron_def.subgraph) {
          findIn(n.neuron_def.subgraph);
        }
      });
    };
    findIn(activeGraph);
    return targetNode;
  };

  const targetNode: any = searchNode();
  const functionName = targetNode?.neuron_def?.name || "identity";
  const telemetryValues = edgeTelemetry[source] || [];
  const sourceTrace = resolveTorchTraceStats(torchTrace, currentPath, source);
  const targetTrace = resolveTorchTraceStats(torchTrace, currentPath, target);
  const scalarPreview = telemetryValues.length > 0;
  const torchPreview = sourceTrace.length > 0 || targetTrace.length > 0;

  const renderTraceCard = (label: string, stats: any[]) => {
    if (stats.length === 0) return null;
    return (
      <div className="w-full rounded border border-cyan-500/30 bg-slate-900/90 px-2 py-1">
        <div className="mb-1 font-mono text-[9px] font-semibold uppercase tracking-wider text-cyan-400">
          {label}
        </div>
        {stats.slice(0, 2).map((stat, idx) => (
          <div
            key={`${label}-${idx}`}
            className="mb-1 last:mb-0 font-mono text-[9px] text-gray-300"
          >
            {stat.kind ? (
              <div>{stat.kind}</div>
            ) : (
              <>
                <div className="text-cyan-200">
                  {JSON.stringify(stat.shape)} {stat.dtype ? `• ${stat.dtype}` : ""}
                </div>
                <div className="text-gray-400">
                  μ={stat.mean?.toFixed(4)} σ={stat.std?.toFixed(4)} [
                  {stat.min?.toFixed(3)}, {stat.max?.toFixed(3)}]
                </div>
                {Array.isArray(stat.preview) && stat.preview.length > 0 && (
                  <div className="truncate text-amber-300">
                    vec {JSON.stringify(stat.preview)}
                  </div>
                )}
              </>
            )}
          </div>
        ))}
      </div>
    );
  };

  const activeStroke = isHovered ? palette.accent : palette.primary;

  return (
    <>
      {/* Outer Holographic Plasma Glow Path */}
      <path
        d={edgePath}
        fill="none"
        stroke={activeStroke}
        strokeWidth={isHovered ? 8 : isTraining ? 5 : 3.5}
        strokeOpacity={isHovered ? 0.38 : 0.16}
        className="pointer-events-none transition-all duration-300"
      />

      {/* Core Energy Conduit */}
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          ...style,
          strokeWidth: isHovered ? 2.6 : 1.5,
          stroke: activeStroke,
          strokeOpacity: isHovered ? 0.95 : 0.55,
          strokeDasharray: isHovered ? "none" : "6 6",
          transition: "stroke 0.25s, stroke-width 0.25s, stroke-opacity 0.25s",
        }}
      />

      {/* Midpoint HUD Micro-Reticle */}
      <g
        transform={`translate(${labelX}, ${labelY})`}
        className="pointer-events-none"
      >
        <circle
          r={isHovered ? 7 : 3}
          fill="none"
          stroke={activeStroke}
          strokeWidth={isHovered ? 1.5 : 1}
          strokeOpacity={isHovered ? 0.95 : 0.45}
        />
        {isHovered && (
          <circle
            r={12}
            fill="none"
            stroke={palette.primary}
            strokeWidth={1}
            strokeDasharray="4 4"
          />
        )}
      </g>

      {/* Invisible Wide Interaction Path */}
      <path
        d={edgePath}
        fill="none"
        strokeOpacity={0}
        strokeWidth={24}
        className="react-flow__edge-interaction cursor-crosshair"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      />

      {isHovered && targetNode && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: "absolute",
              transform: `translate(-50%, -105%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: "none",
              borderColor: palette.primary,
              boxShadow: `0 0 28px ${palette.primary}55`,
            }}
            className="nodrag nopan z-[9999] flex flex-col items-center rounded-lg border bg-slate-950/95 px-3 py-2.5 backdrop-blur-md"
          >
            <div className="mb-1 flex items-center gap-1.5 font-mono text-[10px] font-bold uppercase tracking-wider text-cyan-300">
              <span
                className="h-1.5 w-1.5 rounded-full"
                style={{ backgroundColor: palette.accent }}
              />
              <span>SYNAPSE // {functionName}</span>
            </div>
            {targetNode?.neuron_def?.kind === "function" && (
              <div className="mb-1 rounded border border-cyan-500/20 bg-slate-900/90 p-1">
                <NeuronIcon
                  name={functionName}
                  expanded={true}
                  animated={scalarPreview}
                  telemetry={telemetryValues}
                />
              </div>
            )}
            {scalarPreview && (
              <div className="font-mono text-[9px] text-cyan-400">
                {telemetryValues.length} telemetry packets active
              </div>
            )}
            {torchPreview && (
              <div className="mt-1 flex w-full flex-col gap-1">
                {renderTraceCard("Source Tensor", sourceTrace)}
                {renderTraceCard("Target Tensor", targetTrace)}
                <div className="font-mono text-[8px] uppercase text-emerald-300">
                  {torchTraceSource === "dataset"
                    ? "Dataset Stream Active"
                    : "Traced Tensor Conduit"}
                </div>
              </div>
            )}
            {!scalarPreview && !torchPreview && (
              <div className="font-mono text-[8px] uppercase tracking-wider text-amber-300/85">
                Conduit Ready • No Live Trace
              </div>
            )}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
