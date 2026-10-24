import React, { memo, useCallback, useMemo, useState } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import {
  resolveTorchTraceStats,
  selectCurrentPath,
  type NeuronNodeData,
  useGraphStore,
} from "../store/graphStore";
import { useSettingsStore } from "../store/settingsStore";
import {
  type NodeArchetypeId,
  resolveNodeArchetype,
} from "./nodeArchetypes";
import VolumetricNodeCore from "./VolumetricNodeCore";

/**
 * Live 60 FPS 3D Volumetric Hologram + Counter-Rotating Reticle Assembly
 * used both on graph nodes and inside the Node Picker Modal.
 */
export function SemanticNodeGlyph({
  archetypeId,
  neuronName,
  primary,
  secondary,
  size = 56,
  selected = false,
}: {
  archetypeId: NodeArchetypeId;
  neuronName: string;
  primary: string;
  secondary: string;
  size?: number;
  selected?: boolean;
}) {
  const outerSize = size + 14;
  return (
    <div
      className="relative flex items-center justify-center select-none"
      style={{
        width: outerSize,
        height: outerSize,
        transformStyle: "preserve-3d",
      }}
    >
      {/* Rear Counter-Rotating Mechanical Turbine Ring (Z = -8px) */}
      <svg
        width={outerSize}
        height={outerSize}
        viewBox="0 0 100 100"
        className="pointer-events-none absolute inset-0 animate-[spin_16s_linear_infinite_reverse] opacity-65"
        style={{ transform: "translateZ(-8px)" }}
      >
        <circle
          cx="50"
          cy="50"
          r="46"
          fill="none"
          stroke={secondary}
          strokeWidth="1.2"
          strokeDasharray="6 8 2 6"
        />
        <circle
          cx="50"
          cy="50"
          r="39"
          fill="none"
          stroke={primary}
          strokeWidth="0.9"
          strokeDasharray="16 10"
        />
      </svg>

      {/* Mid Clockwise-Rotating Precision Azimuth Ring (Z = +4px) */}
      <svg
        width={outerSize}
        height={outerSize}
        viewBox="0 0 100 100"
        className="pointer-events-none absolute inset-0 animate-[spin_11s_linear_infinite] opacity-80"
        style={{ transform: "translateZ(4px)" }}
      >
        <circle
          cx="50"
          cy="50"
          r="43"
          fill="none"
          stroke={primary}
          strokeWidth="1.4"
          strokeDasharray="22 18 4 14"
        />
        <polygon points="50,2 47,8 53,8" fill={primary} />
        <polygon points="50,98 47,92 53,92" fill={secondary} />
      </svg>

      {/* Center 60 FPS 3D Rotating & Twisting Volumetric Mesh Core (Z = +10px) */}
      <div className="relative z-10" style={{ transform: "translateZ(10px)" }}>
        <VolumetricNodeCore
          archetypeId={archetypeId}
          neuronName={neuronName}
          primaryColor={primary}
          secondaryColor={secondary}
          size={size}
          selected={selected}
        />
      </div>
    </div>
  );
}

function ParticleNeuronNodeInner({ id, data, selected }: NodeProps) {
  const nodeData = data as unknown as NeuronNodeData;
  const { neuronDef, label, isInput, isOutput } = nodeData;
  const inputs = neuronDef.input_ports;
  const outputs = neuronDef.output_ports;
  const isSubgraph = neuronDef.kind === "subgraph";
  const isModule = neuronDef.kind === "module";
  const trainingMethod = neuronDef.subgraph?.training_method;
  const variantRef = neuronDef.variant_ref;

  const updateNodeData = useGraphStore((state) => state.updateNodeData);
  const torchTrace = useGraphStore((state) => state.torchTrace);
  const torchTraceSource = useGraphStore((state) => state.torchTraceSource);
  const isTraining = useGraphStore((state) => state.isTraining);
  const currentPath = useGraphStore(selectCurrentPath);

  const visualMode = useSettingsStore((state) => state.particleVisualMode);

  const [isHovered, setIsHovered] = useState(false);
  const [tilt, setTilt] = useState<{ rx: number; ry: number }>({ rx: 6, ry: -6 });
  const traceStats = resolveTorchTraceStats(torchTrace, currentPath, id);

  // Resolve distinct semantic archetype for this node
  const archetype = useMemo(
    () => resolveNodeArchetype(label, neuronDef, isInput, isOutput),
    [label, neuronDef, isInput, isOutput],
  );

  const accentColor = archetype.primary;
  const secondaryColor = archetype.secondary;

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const nx = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
    const ny = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
    setTilt({
      rx: Math.max(-16, Math.min(16, -ny * 13)),
      ry: Math.max(-16, Math.min(16, nx * 13)),
    });
  }, []);

  const onMouseLeave = useCallback(() => {
    setIsHovered(false);
    setTilt({ rx: 6, ry: -6 });
  }, []);

  const roleTag = isInput ? "IN" : isOutput ? "OUT" : null;
  const isOverride = neuronDef.source_code?.includes("def override(x):") ?? false;
  const overrideMatch = neuronDef.source_code?.match(/return\s+([-\d.]+)/);
  const overrideValue = overrideMatch ? parseFloat(overrideMatch[1]) : 0;

  const handleOverrideChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    if (!isNaN(val)) {
      updateNodeData(id, {
        neuronDef: {
          ...neuronDef,
          source_code: `def override(x):\n    return ${val}\n`,
        },
      });
    }
  };

  const shortHexId = useMemo(() => {
    let hash = 0;
    for (let i = 0; i < id.length; i++) {
      hash = (hash * 31 + id.charCodeAt(i)) | 0;
    }
    return `0x${Math.abs(hash).toString(16).slice(0, 4).toUpperCase().padStart(4, "0")}`;
  }, [id]);

  const subgraphNodeCount = useMemo(() => {
    if (!isSubgraph || !neuronDef.subgraph?.nodes) return 0;
    return Object.keys(neuronDef.subgraph.nodes).length;
  }, [isSubgraph, neuronDef.subgraph]);

  // ── VOLUMETRIC SCULPTURE MODE ("particle_nebula") ──
  // Pure 3D floating rotating/twisting sculpture with multi-plane Z depth
  if (visualMode === "particle_nebula") {
    return (
      <div
        onMouseEnter={() => setIsHovered(true)}
        onMouseMove={onMouseMove}
        onMouseLeave={onMouseLeave}
        className="relative flex items-center justify-between px-3 py-3 min-h-[104px] select-none group"
        style={{
          minWidth: archetype.minWidth - 10,
          perspective: "850px",
        }}
      >
        <div
          className="relative flex w-full items-center justify-between transition-transform duration-200 ease-out"
          style={{
            transformStyle: "preserve-3d",
            transform: `rotateX(${tilt.rx}deg) rotateY(${tilt.ry}deg) scale(${
              selected ? 1.07 : isHovered ? 1.03 : 1
            })`,
          }}
        >
          {/* Plane -2 (Z = -18px): Extruded 3D Backplate */}
          <div
            className="pointer-events-none absolute inset-1"
            style={{
              clipPath: archetype.clipPath,
              transform: "translateZ(-18px)",
              backgroundColor: `${accentColor}20`,
              border: `1px solid ${accentColor}55`,
              boxShadow: `0 18px 36px rgba(0,0,0,0.9), 0 0 24px ${accentColor}33`,
            }}
          />

          {/* Plane 0 (Z = 0px): Main Shaped Cyber Frame */}
          <div
            className="pointer-events-none absolute inset-1 transition-all duration-300"
            style={{
              clipPath: archetype.clipPath,
              transform: "translateZ(0px)",
              background: `linear-gradient(135deg, ${accentColor}26 0%, rgba(3, 10, 24, 0.85) 55%, ${secondaryColor}22 100%)`,
              border: `1.5px solid ${selected ? accentColor : `${accentColor}88`}`,
              boxShadow: `inset 0 1px 0 rgba(255,255,255,0.22), 0 0 18px ${accentColor}40`,
            }}
          />

          {/* Plane +2 (Z = +22px): Floating Top Laser Callout Banner */}
          <div
            className="pointer-events-none absolute -top-9 left-1/2 flex flex-col items-center z-20"
            style={{
              transform: "translateX(-50%) translateZ(22px)",
            }}
          >
            <div
              className="flex items-center gap-1.5 whitespace-nowrap px-2.5 py-0.5 backdrop-blur-md"
              style={{
                clipPath:
                  "polygon(6px 0%, calc(100% - 6px) 0%, 100% 50%, calc(100% - 6px) 100%, 6px 100%, 0% 50%)",
                border: `1px solid ${selected ? accentColor : `${accentColor}99`}`,
                backgroundColor: "rgba(2, 8, 22, 0.94)",
                boxShadow: `0 6px 16px rgba(0,0,0,0.85), 0 0 14px ${accentColor}55`,
              }}
            >
              <span
                className="h-1.5 w-1.5 rounded-full"
                style={{
                  backgroundColor: accentColor,
                  boxShadow: `0 0 6px ${accentColor}`,
                }}
              />
              <span
                className="font-mono text-[10px] font-bold tracking-wider uppercase"
                style={{ color: "#f8fafc", textShadow: `0 0 8px ${accentColor}` }}
              >
                {label}
              </span>
              <span className="font-mono text-[8px]" style={{ color: secondaryColor }}>
                [{archetype.badgeText.split(" // ")[0]}]
              </span>
            </div>
            <div
              className="h-2.5 w-[1.5px]"
              style={{
                background: `linear-gradient(to bottom, ${accentColor}, transparent)`,
              }}
            />
          </div>

          {/* Left Input Ports (Z = +16px) */}
          <div
            className="relative z-10 flex flex-col gap-1.5"
            style={{ transform: "translateZ(16px)" }}
          >
            {inputs.map((p, i) => (
              <div key={p.name} className="relative flex items-center">
                <Handle
                  type="target"
                  position={Position.Left}
                  id={`in-${i}`}
                  className="!w-3 !h-3 !rounded-sm transition-transform duration-150 hover:!scale-125"
                  style={{
                    top: "auto",
                    position: "relative",
                    left: -7,
                    transform: "rotate(45deg)",
                    backgroundColor: accentColor,
                    borderColor: "#ffffff",
                    borderWidth: 1.5,
                    boxShadow: `0 0 10px ${accentColor}`,
                  }}
                />
                <span className="ml-1 font-mono text-[9px] font-semibold text-cyan-50 drop-shadow-[0_1px_2px_rgba(0,0,0,0.95)]">
                  {p.name}
                </span>
              </div>
            ))}
          </div>

          {/* Center 3D Rotating/Twisting Volumetric Sculpture (Z = +14px) */}
          <div
            className="pointer-events-none relative z-10 mx-2 flex flex-col items-center justify-center"
            style={{ transform: "translateZ(14px)" }}
          >
            <SemanticNodeGlyph
              archetypeId={archetype.id}
              neuronName={neuronDef.name}
              primary={accentColor}
              secondary={secondaryColor}
              size={62}
              selected={selected}
            />
          </div>

          {/* Right Output Ports (Z = +16px) */}
          <div
            className="relative z-10 flex flex-col gap-1.5 items-end"
            style={{ transform: "translateZ(16px)" }}
          >
            {outputs.map((p, i) => (
              <div key={p.name} className="relative flex items-center">
                <span className="mr-1 font-mono text-[9px] font-semibold text-amber-50 drop-shadow-[0_1px_2px_rgba(0,0,0,0.95)]">
                  {p.name}
                </span>
                <Handle
                  type="source"
                  position={Position.Right}
                  id={`out-${i}`}
                  className="!w-3 !h-3 !rounded-sm transition-transform duration-150 hover:!scale-125"
                  style={{
                    top: "auto",
                    position: "relative",
                    right: -7,
                    transform: "rotate(45deg)",
                    backgroundColor: secondaryColor,
                    borderColor: "#ffffff",
                    borderWidth: 1.5,
                    boxShadow: `0 0 10px ${secondaryColor}`,
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // ── FULL 3D VOLUMETRIC CYBER NODE MODE ("hybrid_hud") ──
  // Multi-plane CSS 3D `preserve-3d` chassis with 5 distinct Z-depth planes,
  // rotating outer turbine rings, and a live 60 FPS 3D twisting/rotating mesh core.
  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      className="relative select-none"
      style={{
        minWidth: archetype.minWidth + 12,
        perspective: "900px",
      }}
    >
      <div
        className="relative transition-transform duration-200 ease-out"
        style={{
          transformStyle: "preserve-3d",
          transform: `rotateX(${tilt.rx}deg) rotateY(${tilt.ry}deg) scale(${
            selected ? 1.04 : isHovered ? 1.02 : 1
          })`,
        }}
      >
        {/* ── PLANE -2 (Z = -20px): Deep 3D Shadow & Rear Rotating Reactor Ring ── */}
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            clipPath: archetype.clipPath,
            transform: "translateZ(-20px) translateX(4px) translateY(6px)",
            backgroundColor: "rgba(1, 4, 12, 0.92)",
            border: `1.5px solid ${accentColor}55`,
            boxShadow: selected
              ? `0 20px 40px rgba(0,0,0,0.95), 0 0 30px ${accentColor}66`
              : `0 14px 28px rgba(0,0,0,0.9), 0 0 16px ${accentColor}30`,
          }}
        />

        {/* Rear Counter-Rotating Cyber Gear peeking behind the chassis (Z = -14px) */}
        <div
          className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 opacity-40"
          style={{ transform: "translate(-50%, -50%) translateZ(-14px)" }}
        >
          <svg
            width={136}
            height={136}
            viewBox="0 0 136 136"
            className="animate-[spin_22s_linear_infinite_reverse]"
          >
            <circle
              cx="68"
              cy="68"
              r="62"
              fill="none"
              stroke={accentColor}
              strokeWidth="1.5"
              strokeDasharray="10 8 3 8"
            />
            <circle
              cx="68"
              cy="68"
              r="54"
              fill="none"
              stroke={secondaryColor}
              strokeWidth="1"
              strokeDasharray="24 12"
            />
          </svg>
        </div>

        {/* ── PLANE -1 (Z = -9px): Extruded Metallic-Glass Bevel Plate ── */}
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            clipPath: archetype.clipPath,
            transform: "translateZ(-9px) translateX(2px) translateY(3px)",
            background: `linear-gradient(135deg, ${accentColor}44, rgba(8, 18, 38, 0.92), ${secondaryColor}44)`,
          }}
        />

        {/* ── PLANE 0 (Z = 0px): Main Shaped Cyber Armor Shell ── */}
        <div
          className="relative p-[1.5px]"
          style={{
            clipPath: archetype.clipPath,
            transform: "translateZ(0px)",
            background: selected
              ? `linear-gradient(135deg, ${accentColor}, ${secondaryColor}, ${accentColor})`
              : isHovered
              ? `linear-gradient(135deg, ${accentColor}dd, ${secondaryColor}88, ${accentColor}bb)`
              : `linear-gradient(135deg, ${accentColor}99, rgba(15, 23, 42, 0.92), ${secondaryColor}77)`,
          }}
        >
          <div
            className="relative px-3.5 py-2.5 backdrop-blur-md"
            style={{
              clipPath: archetype.clipPath,
              background: selected
                ? `linear-gradient(160deg, rgba(6, 16, 36, 0.94) 0%, rgba(2, 7, 19, 0.96) 60%, ${accentColor}22 100%)`
                : `linear-gradient(160deg, rgba(4, 11, 26, 0.92) 0%, rgba(2, 6, 16, 0.96) 70%, ${accentColor}14 100%)`,
              boxShadow: `inset 0 1px 0 rgba(255,255,255,0.18), inset 0 0 26px ${accentColor}22`,
            }}
          >
            {/* Top Archetype Telemetry Bar */}
            <div
              className="mb-1.5 flex items-center justify-between border-b pb-1 font-mono text-[8px] tracking-widest"
              style={{
                borderColor: `${accentColor}38`,
                color: secondaryColor,
              }}
            >
              <span className="flex items-center gap-1.5 font-bold">
                <span
                  className={`inline-block h-1.5 w-1.5 rotate-45 ${
                    isTraining || selected ? "animate-ping" : ""
                  }`}
                  style={{
                    backgroundColor: accentColor,
                    boxShadow: `0 0 6px ${accentColor}`,
                  }}
                />
                {archetype.badgeText}
              </span>
              <span className="opacity-75">{shortHexId}</span>
            </div>

            {/* Node Title & Semantic Badges */}
            <div className="mb-2 flex items-center justify-between gap-2">
              <span
                className="truncate font-mono text-xs font-extrabold tracking-wide uppercase"
                style={{
                  color: "#f8fafc",
                  textShadow: `0 0 12px ${accentColor}99`,
                }}
              >
                {label}
              </span>

              <div className="flex items-center gap-1 shrink-0">
                {isModule && neuronDef.module_type !== label && (
                  <span
                    className="rounded-sm border px-1 py-0.2 font-mono text-[8px] uppercase"
                    style={{
                      borderColor: `${accentColor}66`,
                      backgroundColor: `${accentColor}1a`,
                      color: accentColor,
                    }}
                  >
                    {neuronDef.module_type}
                  </span>
                )}
                {trainingMethod && (
                  <span
                    className="rounded-sm border px-1 font-mono text-[8px] uppercase"
                    style={{
                      borderColor: `${secondaryColor}66`,
                      backgroundColor: `${secondaryColor}1a`,
                      color: secondaryColor,
                    }}
                  >
                    {trainingMethod.slice(0, 3)}
                  </span>
                )}
                {variantRef && (
                  <span className="max-w-[84px] truncate rounded-sm border border-amber-400/50 bg-amber-950/60 px-1 font-mono text-[8px] uppercase text-amber-200">
                    {String(variantRef.family)}@{String(variantRef.version)}
                  </span>
                )}
                {roleTag && (
                  <span
                    className="rounded-sm px-1.5 font-mono text-[8px] font-bold"
                    style={{
                      backgroundColor: `${accentColor}30`,
                      color: accentColor,
                      border: `1px solid ${accentColor}`,
                    }}
                  >
                    {roleTag}
                  </span>
                )}
                {typeof neuronDef.module_config?.num_heads === "number" &&
                  typeof neuronDef.module_config?.num_kv_heads === "number" && (
                    <span className="rounded-sm border border-cyan-400/40 bg-cyan-950/70 px-1 font-mono text-[8px] text-cyan-200">
                      {neuronDef.module_config.num_heads ===
                      neuronDef.module_config.num_kv_heads
                        ? "MHA"
                        : neuronDef.module_config.num_kv_heads === 1
                        ? "MQA"
                        : "GQA"}
                    </span>
                  )}
              </div>
            </div>

            {/* Center Body: Left Input Ports + Center 3D Volumetric Hologram Chamber + Right Output Ports */}
            <div className="flex items-center justify-between gap-3 font-mono text-[10px]">
              {/* Input Ports */}
              <div className="flex flex-col gap-1.5 min-w-[44px]">
                {inputs.map((p, i) => {
                  let shape = "";
                  if (p.name === "q" || p.name === "v" || p.name === "heads")
                    shape = "[B,H,T,D]";
                  else if (p.name === "k") shape = "[B,KV,T,D]";
                  else if (
                    p.dtype === "tensor" &&
                    p.name !== "k" &&
                    p.name !== "q" &&
                    p.name !== "v"
                  )
                    shape = "[B,T,C]";
                  else if (p.dtype === "tokens") shape = "[B,T]";

                  return (
                    <div key={p.name} className="relative flex items-center group/port">
                      <Handle
                        type="target"
                        position={Position.Left}
                        id={`in-${i}`}
                        className="!w-3 !h-3 !rounded-sm transition-all duration-150 hover:!scale-125"
                        style={{
                          top: "auto",
                          position: "relative",
                          left: -8,
                          transform: "rotate(45deg)",
                          backgroundColor: accentColor,
                          borderColor: "#ffffff",
                          borderWidth: 1.5,
                          boxShadow: `0 0 8px ${accentColor}`,
                        }}
                      />
                      <span className="flex items-center gap-1 text-gray-200">
                        {p.name}
                        {shape && (
                          <span
                            className="pointer-events-none rounded-sm border px-1 text-[7px] opacity-75 group-hover/port:opacity-100"
                            style={{
                              borderColor: `${accentColor}44`,
                              backgroundColor: `${accentColor}14`,
                              color: accentColor,
                            }}
                          >
                            {shape}
                          </span>
                        )}
                      </span>
                    </div>
                  );
                })}
              </div>

              {/* ── PLANE +1: 3D Volumetric Rotating & Twisting Core Pedestal ── */}
              <div
                className="pointer-events-none relative flex flex-col items-center justify-center rounded-lg border px-1.5 py-1"
                style={{
                  borderColor: `${accentColor}40`,
                  background: `radial-gradient(circle at 50% 50%, ${accentColor}22 0%, rgba(2, 6, 18, 0.85) 75%)`,
                  boxShadow: `0 8px 20px rgba(0,0,0,0.85), inset 0 0 16px ${accentColor}25, 0 0 14px ${accentColor}20`,
                }}
              >
                <SemanticNodeGlyph
                  archetypeId={archetype.id}
                  neuronName={neuronDef.name}
                  primary={accentColor}
                  secondary={secondaryColor}
                  size={54}
                  selected={selected}
                />
              </div>

              {/* Output Ports */}
              <div className="flex flex-col items-end gap-1.5 min-w-[44px]">
                {outputs.map((p, i) => {
                  let shape = "";
                  if (p.name === "q_rot" || p.name === "k_rot" || p.name === "heads")
                    shape = "[B,H,T,D]";
                  else if (
                    p.dtype === "tensor" &&
                    p.name !== "heads" &&
                    p.name !== "q_rot" &&
                    p.name !== "k_rot"
                  )
                    shape = "[B,T,C]";

                  return (
                    <div key={p.name} className="relative flex items-center group/port">
                      <span className="mr-1 flex flex-row-reverse items-center gap-1 text-gray-200">
                        {p.name}
                        {shape && (
                          <span
                            className="pointer-events-none rounded-sm border px-1 text-[7px] opacity-75 group-hover/port:opacity-100"
                            style={{
                              borderColor: `${secondaryColor}44`,
                              backgroundColor: `${secondaryColor}14`,
                              color: secondaryColor,
                            }}
                          >
                            {shape}
                          </span>
                        )}
                      </span>
                      <Handle
                        type="source"
                        position={Position.Right}
                        id={`out-${i}`}
                        className="!w-3 !h-3 !rounded-sm transition-all duration-150 hover:!scale-125"
                        style={{
                          top: "auto",
                          position: "relative",
                          right: -8,
                          transform: "rotate(45deg)",
                          backgroundColor: secondaryColor,
                          borderColor: "#ffffff",
                          borderWidth: 1.5,
                          boxShadow: `0 0 8px ${secondaryColor}`,
                        }}
                      />
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Subgraph / Variant Footer */}
            {isSubgraph && (
              <div
                className="mt-2 flex items-center justify-between rounded-sm border px-2 py-0.5 font-mono text-[8px]"
                style={{
                  borderColor: `${accentColor}44`,
                  backgroundColor: `${accentColor}14`,
                  color: accentColor,
                }}
              >
                <span>
                  ◈{" "}
                  {variantRef
                    ? `VARIANT: ${String(variantRef.family)}`
                    : `${subgraphNodeCount} INTERNAL NODES`}
                </span>
                <span className="uppercase tracking-wider opacity-85">2×CLICK WARP ▸</span>
              </div>
            )}

            {/* Scalar Override Input */}
            {isOverride && (
              <div className="mt-2 flex items-center justify-between rounded border border-cyan-500/30 bg-slate-950/90 px-2 py-1 font-mono text-[10px] text-cyan-200">
                <span>OVERRIDE:</span>
                <input
                  type="number"
                  value={overrideValue}
                  onChange={handleOverrideChange}
                  className="w-16 rounded border border-cyan-500/40 bg-slate-900 px-1 py-0.5 text-right text-cyan-100"
                  step="0.1"
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Holographic Tensor Trace Preview Tooltip */}
      {isHovered && traceStats.length > 0 && (
        <div
          className="pointer-events-none absolute left-0 top-full z-50 mt-2.5 min-w-[230px] rounded-lg border bg-slate-950/95 px-3 py-2 shadow-2xl backdrop-blur-md"
          style={{
            borderColor: accentColor,
            boxShadow: `0 0 24px ${accentColor}44`,
          }}
        >
          <div className="mb-1 flex items-center justify-between gap-2 border-b border-cyan-500/20 pb-1">
            <span className="font-mono text-[10px] font-bold" style={{ color: accentColor }}>
              TELEMETRY // {label}
            </span>
            <span className="font-mono text-[8px] uppercase tracking-wider text-emerald-300">
              {torchTraceSource === "dataset" ? "DATASET" : "MANUAL"}
            </span>
          </div>
          {traceStats.slice(0, 2).map((stat, idx) => (
            <div
              key={`${id}-${idx}`}
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
      )}
    </div>
  );
}

export const ParticleNeuronNode = memo(ParticleNeuronNodeInner);
export default ParticleNeuronNode;
