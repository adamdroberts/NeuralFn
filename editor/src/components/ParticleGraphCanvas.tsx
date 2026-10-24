import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  Controls,
  applyNodeChanges,
  type Connection,
  type EdgeChange,
  type Node,
  type NodeChange,
  type NodeTypes,
  type OnConnect,
  type Viewport,
  useReactFlow,
} from "@xyflow/react";
import { ParticleNeuronNode } from "./ParticleNeuronNode";
import ParticleEdge from "./ParticleEdge";
import WebGLCyberScene, { type WebGLNodeAnchor } from "./WebGLCyberScene";
import {
  FullGraphZonesLayer,
  SubgraphExpansionOverlay,
  useSubgraphExpansion,
} from "./SubgraphExpansionOverlay";
import {
  type ParticleSwarmGeometry,
  resolveNodeArchetype,
} from "./nodeArchetypes";
import {
  selectBreadcrumbs,
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
import {
  PARTICLE_PALETTES,
  type ParticleColorPalette,
  useSettingsStore,
} from "../store/settingsStore";

const particleNodeTypes: NodeTypes = { neuron: ParticleNeuronNode as any };
const particleEdgeTypes = { default: ParticleEdge };

interface AmbientParticle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  alpha: number;
  depth: number;
  hueShift: number;
}

interface OrbitalParticle {
  id: number;
  nodeId: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  prevX: number;
  prevY: number;
  theta: number;
  phi: number;
  orbitSpeed: number;
  phiSpeed: number;
  shellRadius: number;
  size: number;
  rgb: [number, number, number];
  alpha: number;
}

interface EdgePhoton {
  edgeId: string;
  t: number;
  speed: number;
  phase: number;
  strand: number; // 0 = core, 1 = helix A, -1 = helix B
  size: number;
  history: Array<{ x: number; y: number }>;
}

interface BurstParticle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  size: number;
  rgb: [number, number, number];
}

interface ShockwaveRing {
  x: number;
  y: number;
  radius: number;
  maxRadius: number;
  life: number;
  maxLife: number;
  rgb: [number, number, number];
  lineWidth: number;
  dashed?: boolean;
}

interface NodeAnchor {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  rgb: [number, number, number];
  secondaryRgb: [number, number, number];
  swarmGeometry: ParticleSwarmGeometry;
  selected: boolean;
  isSubgraph: boolean;
}

function cubicBezierPoint(
  p0x: number,
  p0y: number,
  p1x: number,
  p1y: number,
  p2x: number,
  p2y: number,
  p3x: number,
  p3y: number,
  t: number,
): { x: number; y: number; nx: number; ny: number } {
  const u = 1 - t;
  const tt = t * t;
  const uu = u * u;
  const uuu = uu * u;
  const ttt = tt * t;

  const x = uuu * p0x + 3 * uu * t * p1x + 3 * u * tt * p2x + ttt * p3x;
  const y = uuu * p0y + 3 * uu * t * p1y + 3 * u * tt * p2y + ttt * p3y;

  // Derivative for normal vector
  const dx =
    3 * uu * (p1x - p0x) + 6 * u * t * (p2x - p1x) + 3 * tt * (p3x - p2x);
  const dy =
    3 * uu * (p1y - p0y) + 6 * u * t * (p2y - p1y) + 3 * tt * (p3y - p2y);
  const len = Math.hypot(dx, dy) || 1;

  return {
    x,
    y,
    nx: -dy / len,
    ny: dx / len,
  };
}

function hexToRgbTuple(hex: string): [number, number, number] {
  const clean = hex.replace("#", "");
  if (clean.length !== 6) return [0, 240, 255];
  const num = parseInt(clean, 16);
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

export default function ParticleGraphCanvas() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { screenToFlowPosition, fitView, setCenter, getViewport, setViewport } =
    useReactFlow();

  const activeNodes = useGraphStore(selectFlowNodes);
  const activeEdges = useGraphStore(selectFlowEdges);
  const rootGraph = useGraphStore((state) => state.rootGraph);
  const breadcrumbs = useGraphStore(selectBreadcrumbs);
  const applyActiveNodeChanges = useGraphStore((state) => state.applyActiveNodeChanges);
  const applyActiveEdgeChanges = useGraphStore((state) => state.applyActiveEdgeChanges);
  const connectActiveGraph = useGraphStore((state) => state.connectActiveGraph);
  const selectNode = useGraphStore((state) => state.selectNode);
  const openSubgraph = useGraphStore((state) => state.openSubgraph);
  const setPath = useGraphStore((state) => state.setPath);
  const currentPath = useGraphStore((state) => state.currentPath);
  const setPreferredInsertPosition = useGraphStore(
    (state) => state.setPreferredInsertPosition,
  );
  const isTraining = useGraphStore((state) => state.isTraining);
  const selectedNodeId = useGraphStore((state) => state.selectedNodeId);

  // Settings store
  const visualMode = useSettingsStore((state) => state.particleVisualMode);
  const setVisualMode = useSettingsStore((state) => state.setParticleVisualMode);
  const particleDensity = useSettingsStore((state) => state.particleDensity);
  const colorPalette = useSettingsStore((state) => state.colorPalette);
  const setColorPalette = useSettingsStore((state) => state.setColorPalette);
  const holoPerspective3D = useSettingsStore((state) => state.holoPerspective3D);
  const setHoloPerspective3D = useSettingsStore((state) => state.setHoloPerspective3D);
  const motionIntensity = useSettingsStore((state) => state.motionIntensity);
  const hudTelemetryOverlay = useSettingsStore((state) => state.hudTelemetryOverlay);
  const viewFullGraph = useSettingsStore((state) => state.viewFullGraph);
  const setViewFullGraph = useSettingsStore((state) => state.setViewFullGraph);
  const toggleViewFullGraph = useSettingsStore((state) => state.toggleViewFullGraph);
  const setSettingsOpen = useSettingsStore((state) => state.setSettingsOpen);
  const pulseSequence = useSettingsStore((state) => state.pulseSequence);
  const triggerPulse = useSettingsStore((state) => state.triggerPulse);
  const setParticleGraphEditorEnabled = useSettingsStore(
    (state) => state.setParticleGraphEditorEnabled,
  );

  // Full Graph Mode expansion
  const fullExpanded = useMemo(
    () => (viewFullGraph ? buildFullExpandedGraph(rootGraph, []) : null),
    [viewFullGraph, rootGraph],
  );
  const [fullGraphNodes, setFullGraphNodes] = useState<Node<NeuronNodeData>[]>([]);

  useEffect(() => {
    if (!fullExpanded) {
      setFullGraphNodes([]);
      return;
    }
    setFullGraphNodes((prev) => {
      const prevById = new Map(prev.map((n) => [n.id, n]));
      return fullExpanded.nodes.map((n) => {
        const existing = prevById.get(n.id);
        return {
          ...n,
          measured: existing?.measured ?? n.measured,
          selected: n.id === selectedNodeId,
        };
      });
    });
  }, [fullExpanded, selectedNodeId]);

  const palette = PARTICLE_PALETTES[colorPalette];

  // Viewport tracking ref for 60fps Canvas sync + state for WebGL 3D scene
  const viewportRef = useRef<Viewport>({ x: 0, y: 0, zoom: 1 });
  const [viewportState, setViewportState] = useState<Viewport>({ x: 0, y: 0, zoom: 1 });
  const mouseRef = useRef<{
    x: number;
    y: number;
    vx: number;
    vy: number;
    active: boolean;
    normX: number;
    normY: number;
  }>({
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    active: false,
    normX: 0,
    normY: 0,
  });

  // Transition state banner
  const [transitionStatus, setTransitionStatus] = useState<string>(
    "VOLUMETRIC 3D CORE // COALESCING",
  );
  const [warpOverlayAlpha, setWarpOverlayAlpha] = useState(0);

  // Target particle counts based on density setting
  const densityConfig = useMemo(() => {
    if (particleDensity === "balanced") {
      return { ambient: 180, perNode: 26, perEdge: 7 };
    }
    if (particleDensity === "ultra") {
      return { ambient: 520, perNode: 72, perEdge: 18 };
    }
    return { ambient: 320, perNode: 44, perEdge: 12 };
  }, [particleDensity]);
  const densityConfigRef = useRef(densityConfig);
  densityConfigRef.current = densityConfig;

  const paletteRef = useRef(palette);
  paletteRef.current = palette;

  const ambientParticlesRef = useRef<AmbientParticle[]>([]);
  const orbitalParticlesRef = useRef<OrbitalParticle[]>([]);
  const edgePhotonsRef = useRef<EdgePhoton[]>([]);
  const burstParticlesRef = useRef<BurstParticle[]>([]);
  const shockwavesRef = useRef<ShockwaveRing[]>([]);
  const warpTimerRef = useRef<number>(1.0); // 1.0 -> 0.0 during transition

  // Helper to spawn a supernova particle burst + shockwave in flow coordinates
  const spawnFlowBurst = useCallback(
    (
      flowX: number,
      flowY: number,
      rgb: [number, number, number],
      count = 55,
      maxRadius = 220,
    ) => {
      for (let i = 0; i < count; i++) {
        const angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5) * 0.35;
        const speed = 1.5 + Math.random() * 8.5;
        burstParticlesRef.current.push({
          x: flowX,
          y: flowY,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed,
          life: 0,
          maxLife: 28 + Math.random() * 26,
          size: 1.4 + Math.random() * 2.6,
          rgb,
        });
      }
      shockwavesRef.current.push({
        x: flowX,
        y: flowY,
        radius: 8,
        maxRadius,
        life: 0,
        maxLife: 42,
        rgb,
        lineWidth: 2.5,
        dashed: false,
      });
      shockwavesRef.current.push({
        x: flowX,
        y: flowY,
        radius: 4,
        maxRadius: maxRadius * 0.7,
        life: 0,
        maxLife: 34,
        rgb: paletteRef.current.accentRgb,
        lineWidth: 1.5,
        dashed: true,
      });
    },
    [],
  );

  const syncViewportFromAnim = useCallback((vp: Viewport) => {
    viewportRef.current = vp;
    setViewportState(vp);
  }, []);

  // Subgraph Click Expansion & Breadcrumb Collapse Animation Controller
  const {
    expansionState,
    isExpanding,
    expansionMode,
    triggerSubgraphExpansion,
  } = useSubgraphExpansion({
    containerRef,
    rootGraph,
    currentPath,
    activeNodes,
    activeEdges,
    viewFullGraph,
    openSubgraph,
    setViewport,
    getViewport,
    onViewportSync: syncViewportFromAnim,
    onSpawnBurst: spawnFlowBurst,
  });
  const isExpandingRef = useRef(isExpanding);
  isExpandingRef.current = isExpanding;
  const expansionModeRef = useRef(expansionMode);
  expansionModeRef.current = expansionMode;

  // Compute displayed nodes and edges across Full Graph Mode, Subgraph Expansion Animation, or Standard Mode
  const nodes = useMemo(() => {
    if (viewFullGraph && fullExpanded) {
      return fullGraphNodes.length > 0 ? fullGraphNodes : fullExpanded.nodes;
    }
    if (expansionState) {
      const measuredMap = new Map(activeNodes.map((n) => [n.id, n.measured]));
      return expansionState.animatedNodes.map((n) => ({
        ...n,
        measured: measuredMap.get(n.id) ?? n.measured,
      }));
    }
    return activeNodes;
  }, [viewFullGraph, fullExpanded, fullGraphNodes, expansionState, activeNodes]);

  const edges = useMemo(() => {
    if (viewFullGraph && fullExpanded) {
      return fullExpanded.edges;
    }
    if (expansionState) {
      return expansionState.animatedEdges;
    }
    return activeEdges;
  }, [viewFullGraph, fullExpanded, expansionState, activeEdges]);

  // Refs for live simulation data
  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  const edgesRef = useRef(edges);
  edgesRef.current = edges;
  const visualModeRef = useRef(visualMode);
  visualModeRef.current = visualMode;
  const isTrainingRef = useRef(isTraining);
  isTrainingRef.current = isTraining;
  const motionIntensityRef = useRef(motionIntensity);
  motionIntensityRef.current = motionIntensity;
  const hudOverlayRef = useRef(hudTelemetryOverlay);
  hudOverlayRef.current = hudTelemetryOverlay;

  const prevNodeCountRef = useRef<number>(nodes.length);
  const prevEdgeCountRef = useRef<number>(edges.length);

  // Trigger full-graph warp / coalescence when subgraph path changes, fullGraph toggles, or visualMode changes
  const pathKey = useMemo(
    () =>
      currentPath
        .map((seg) =>
          seg.kind === "node" ? `n:${seg.nodeId}` : `v:${seg.family}@${seg.version}`,
        )
        .join("/"),
    [currentPath],
  );

  useEffect(() => {
    warpTimerRef.current = 1.0;
    setWarpOverlayAlpha(isExpandingRef.current ? 0.35 : 1);
    setTransitionStatus(
      viewFullGraph
        ? "FULL ARCHITECTURE VIEW // ALL SUBGRAPHS EXPANDED"
        : expansionModeRef.current === "collapse"
        ? "SUBGRAPH COLLAPSING // RETURNING TO PARENT"
        : isExpandingRef.current
        ? "SUBGRAPH EXPANDING // MATERIALIZING COMPONENTS"
        : currentPath.length > 0
        ? `SUB-CORE WARP // DEPTH ${currentPath.length}`
        : "HOLO-MATRIX SYNCHRONIZED",
    );

    // Clear old graph orbital particles & edge photons when switching subgraphs so particles
    // never fly in from out-of-bounds coordinates outside the canvas; new particles spawn
    // directly with each node as it is summoned from the portal.
    orbitalParticlesRef.current.length = 0;
    edgePhotonsRef.current.length = 0;

    const fadeTimer = window.setTimeout(() => {
      setWarpOverlayAlpha(0);
    }, 420);
    const statusTimer = window.setTimeout(() => {
      setTransitionStatus(
        viewFullGraph ? "FULL GRAPH // LOCKED" : "MATRIX LOCKED // 60 FPS",
      );
      if (!isExpandingRef.current) {
        fitView({ duration: 550, padding: viewFullGraph ? 0.14 : 0.22 }).catch(
          () => undefined,
        );
      }
    }, 650);

    return () => {
      window.clearTimeout(fadeTimer);
      window.clearTimeout(statusTimer);
    };
  }, [pathKey, visualMode, viewFullGraph, spawnFlowBurst, fitView]);

  // Pulse wave trigger (from HUD button or settings)
  useEffect(() => {
    if (pulseSequence === 0) return;
    setTransitionStatus("ELECTROMAGNETIC PULSE // ACTIVE");
    for (const n of nodesRef.current) {
      const cx = n.position.x + (n.measured?.width ?? 170) / 2;
      const cy = n.position.y + (n.measured?.height ?? 84) / 2;
      spawnFlowBurst(cx, cy, paletteRef.current.primaryRgb, 28, 200);
    }
    for (const p of orbitalParticlesRef.current) {
      p.vx += (Math.random() - 0.5) * 14;
      p.vy += (Math.random() - 0.5) * 14;
    }
    const timer = window.setTimeout(() => {
      setTransitionStatus("MATRIX LOCKED // 60 FPS");
    }, 900);
    return () => window.clearTimeout(timer);
  }, [pulseSequence, spawnFlowBurst]);

  // Detect node or edge additions to fire localized particle bursts
  useEffect(() => {
    if (nodes.length > prevNodeCountRef.current) {
      const newest = nodes[nodes.length - 1];
      if (newest) {
        const cx = newest.position.x + 85;
        const cy = newest.position.y + 42;
        spawnFlowBurst(cx, cy, paletteRef.current.accentRgb, 65, 240);
      }
    }
    prevNodeCountRef.current = nodes.length;
  }, [nodes, spawnFlowBurst]);

  useEffect(() => {
    if (edges.length > prevEdgeCountRef.current) {
      const newestEdge = edges[edges.length - 1];
      if (newestEdge) {
        const targetNode = nodes.find((n) => n.id === newestEdge.target);
        if (targetNode) {
          spawnFlowBurst(
            targetNode.position.x + 10,
            targetNode.position.y + 40,
            paletteRef.current.primaryRgb,
            50,
            180,
          );
        }
      }
    }
    prevEdgeCountRef.current = edges.length;
  }, [edges, nodes, spawnFlowBurst]);

  // Selection burst
  useEffect(() => {
    if (!selectedNodeId) return;
    const sel = nodesRef.current.find((n) => n.id === selectedNodeId);
    if (sel) {
      const cx = sel.position.x + (sel.measured?.width ?? 170) / 2;
      const cy = sel.position.y + (sel.measured?.height ?? 84) / 2;
      spawnFlowBurst(cx, cy, paletteRef.current.accentRgb, 36, 165);
    }
  }, [selectedNodeId, spawnFlowBurst]);

  const updatePreferredInsertPosition = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
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

  // ── MAIN 60 FPS PARTICLE & HOLOGRAPHIC SIMULATION LOOP ──
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let width = 0;
    let height = 0;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);

    const resizeCanvas = () => {
      const rect = container.getBoundingClientRect();
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    };

    resizeCanvas();
    const ro =
      typeof ResizeObserver !== "undefined" ? new ResizeObserver(resizeCanvas) : null;
    ro?.observe(container);

    let time = 0;
    let particleIdCounter = 1;

    const renderFrame = () => {
      time += 0.016;
      const pal = paletteRef.current;
      const cfg = densityConfigRef.current;
      const mode = visualModeRef.current;
      const currentNodes = nodesRef.current;
      const currentEdges = edgesRef.current;

      // Sync latest viewport from React Flow
      try {
        const vp = getViewport();
        if (vp && Number.isFinite(vp.zoom) && vp.zoom > 0) {
          viewportRef.current = vp;
        }
      } catch {
        // Fallback to last known viewportRef
      }
      const { x: vpX, y: vpY, zoom: vpZoom } = viewportRef.current;

      if (warpTimerRef.current > 0) {
        warpTimerRef.current = Math.max(0, warpTimerRef.current - 0.022);
      }

      ctx.save();
      ctx.scale(dpr, dpr);

      // 1. Deep Holographic Void Background
      const bgGrad = ctx.createRadialGradient(
        width * 0.5,
        height * 0.5,
        20,
        width * 0.5,
        height * 0.5,
        Math.max(width, height) * 0.78,
      );
      bgGrad.addColorStop(0, pal.bgCore);
      bgGrad.addColorStop(0.65, "#020611");
      bgGrad.addColorStop(1, "#010308");
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, width, height);

      // 2. Holographic Perspective Grid & Telemetry Schematic
      const gridStep = Math.max(24, 52 * vpZoom);
      const offsetX = ((vpX % gridStep) + gridStep) % gridStep;
      const offsetY = ((vpY % gridStep) + gridStep) % gridStep;

      ctx.strokeStyle = pal.gridColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let gx = offsetX; gx < width; gx += gridStep) {
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, height);
      }
      for (let gy = offsetY; gy < height; gy += gridStep) {
        ctx.moveTo(0, gy);
        ctx.lineTo(width, gy);
      }
      ctx.stroke();

      // Background Cyber Telemetry Rings
      if (hudOverlayRef.current) {
        const arcCx = width * 0.5 + (vpX - width * 0.35) * 0.08;
        const arcCy = height * 0.5 + (vpY - height * 0.35) * 0.08;
        const baseR = Math.min(width, height) * 0.34;
        const [pr, pg, pb] = pal.primaryRgb;
        const [ar, ag, ab] = pal.accentRgb;

        ctx.save();
        ctx.translate(arcCx, arcCy);

        // Outer degree tick ring
        ctx.strokeStyle = `rgba(${pr}, ${pg}, ${pb}, 0.11)`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(0, 0, baseR, 0, Math.PI * 2);
        ctx.stroke();

        // Degree ticks
        const tickCount = 72;
        ctx.beginPath();
        for (let i = 0; i < tickCount; i++) {
          const ang = (i / tickCount) * Math.PI * 2 + time * 0.04;
          const inner = i % 6 === 0 ? baseR - 10 : baseR - 4;
          ctx.moveTo(Math.cos(ang) * inner, Math.sin(ang) * inner);
          ctx.lineTo(Math.cos(ang) * baseR, Math.sin(ang) * baseR);
        }
        ctx.strokeStyle = `rgba(${pr}, ${pg}, ${pb}, 0.14)`;
        ctx.stroke();

        // Counter-rotating segmented arc ring
        ctx.save();
        ctx.rotate(-time * 0.12);
        ctx.strokeStyle = `rgba(${ar}, ${ag}, ${ab}, 0.12)`;
        ctx.lineWidth = 2.5;
        ctx.setLineDash([28, 18, 8, 18]);
        ctx.beginPath();
        ctx.arc(0, 0, baseR * 0.78, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();

        // Inner reactor ring
        ctx.save();
        ctx.rotate(time * 0.2);
        ctx.strokeStyle = `rgba(${pr}, ${pg}, ${pb}, 0.15)`;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([12, 10]);
        ctx.beginPath();
        ctx.arc(0, 0, baseR * 0.48, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();

        ctx.restore();
      }

      // 3. Ambient Curl-Noise Field Particles (Screen Space with Depth Parallax)
      const ambients = ambientParticlesRef.current;
      while (ambients.length < cfg.ambient) {
        ambients.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5,
          size: 0.7 + Math.random() * 1.9,
          alpha: 0.18 + Math.random() * 0.55,
          depth: 0.3 + Math.random() * 0.7,
          hueShift: Math.random(),
        });
      }
      if (ambients.length > cfg.ambient) {
        ambients.length = cfg.ambient;
      }

      ctx.save();
      ctx.globalCompositeOperation = "lighter";

      const m = mouseRef.current;
      for (let i = 0; i < ambients.length; i++) {
        const p = ambients[i];
        // Trigonometric curl-noise velocity field
        const angle =
          Math.sin(p.x * 0.004 + time * 0.6) * Math.cos(p.y * 0.004 - time * 0.4) * Math.PI * 2;
        p.vx += Math.cos(angle) * 0.03 * p.depth;
        p.vy += Math.sin(angle) * 0.03 * p.depth;

        // Subtle mouse vortex interaction
        if (m.active) {
          const mdx = p.x - m.x;
          const mdy = p.y - m.y;
          const mdistSq = mdx * mdx + mdy * mdy;
          if (mdistSq < 22000 && mdistSq > 4) {
            const mdist = Math.sqrt(mdistSq);
            const force = (1 - mdist / 148) * 0.45;
            // Swirl + gentle push
            p.vx += (-mdy / mdist) * force + (mdx / mdist) * force * 0.35;
            p.vy += (mdx / mdist) * force + (mdy / mdist) * force * 0.35;
          }
        }

        p.vx *= 0.94;
        p.vy *= 0.94;
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < -10) p.x = width + 10;
        if (p.x > width + 10) p.x = -10;
        if (p.y < -10) p.y = height + 10;
        if (p.y > height + 10) p.y = -10;

        const rgb =
          p.hueShift > 0.72
            ? pal.accentRgb
            : p.hueShift > 0.35
            ? pal.primaryRgb
            : pal.secondaryRgb;

        ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${p.alpha})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      }

      // 4. Build Node Anchors in Flow Coordinates with Semantic Archetype Geometry
      const nodeMap = new Map<string, NodeAnchor>();
      for (const n of currentNodes) {
        if (Number(n.style?.opacity ?? 1) <= 0.05) {
          continue;
        }
        const archetype = resolveNodeArchetype(
          n.data.label,
          n.data.neuronDef,
          n.data.isInput,
          n.data.isOutput,
        );
        const w =
          mode === "particle_nebula"
            ? n.measured?.width ?? archetype.minWidth - 16
            : n.measured?.width ?? archetype.minWidth;
        const h =
          mode === "particle_nebula"
            ? n.measured?.height ?? 86
            : n.measured?.height ?? 96;
        const cx = n.position.x + w * 0.5;
        const cy = n.position.y + h * 0.5;

        nodeMap.set(n.id, {
          id: n.id,
          x: cx,
          y: cy,
          width: w,
          height: h,
          rgb: [...archetype.primaryRgb] as [number, number, number],
          secondaryRgb: [...archetype.secondaryRgb] as [number, number, number],
          swarmGeometry: archetype.swarmGeometry,
          selected: Boolean(n.selected),
          isSubgraph: n.data.neuronDef.kind === "subgraph",
        });
      }

      // 5. Reconcile Orbital Node Swarm Particles
      const orbitals = orbitalParticlesRef.current;
      const targetPerNode = cfg.perNode;
      const activeNodeIds = Array.from(nodeMap.keys());

      if (activeNodeIds.length > 0) {
        const countsByNode = new Map<string, number>();
        for (const id of activeNodeIds) countsByNode.set(id, 0);

        // Remove particles belonging to despawned nodes so they never fly in from outside the canvas
        for (let i = orbitals.length - 1; i >= 0; i--) {
          const p = orbitals[i];
          if (!nodeMap.has(p.nodeId)) {
            orbitals.splice(i, 1);
            continue;
          }
          const currentCount = countsByNode.get(p.nodeId) ?? 0;
          if (currentCount >= targetPerNode) {
            orbitals.splice(i, 1);
          } else {
            countsByNode.set(p.nodeId, currentCount + 1);
          }
        }

        // Spawn missing orbital particles directly inside each node's core as it is summoned
        for (const id of activeNodeIds) {
          const anchor = nodeMap.get(id)!;
          let count = countsByNode.get(id) ?? 0;
          while (count < targetPerNode) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            orbitals.push({
              id: particleIdCounter++,
              nodeId: id,
              x: anchor.x + (Math.random() - 0.5) * 14,
              y: anchor.y + (Math.random() - 0.5) * 14,
              vx: (Math.random() - 0.5) * 1.2,
              vy: (Math.random() - 0.5) * 1.2,
              prevX: anchor.x,
              prevY: anchor.y,
              theta,
              phi,
              orbitSpeed: (0.015 + Math.random() * 0.028) * (Math.random() > 0.5 ? 1 : -1),
              phiSpeed: (0.008 + Math.random() * 0.016) * (Math.random() > 0.5 ? 1 : -1),
              shellRadius: 0.4 + Math.random() * 0.85,
              size: 1.2 + Math.random() * 2.2,
              rgb: anchor.rgb,
              alpha: 0.45 + Math.random() * 0.5,
            });
            count++;
          }
        }
      } else {
        orbitals.length = 0;
      }

      // 6. Simulate & Render Node Orbital Swarms (in Flow -> Screen Space)
      const trainingBoost = isTrainingRef.current ? 1.85 : 1.0;
      const springK = motionIntensityRef.current === "cinematic" ? 0.095 : 0.13;
      const damping = motionIntensityRef.current === "cinematic" ? 0.84 : 0.8;

      // Draw soft radial nebula glow behind each node
      for (const anchor of nodeMap.values()) {
        const sx = anchor.x * vpZoom + vpX;
        const sy = anchor.y * vpZoom + vpY;
        const glowR =
          (mode === "particle_nebula" ? 78 : Math.max(anchor.width, anchor.height) * 0.68) *
          vpZoom;
        if (sx + glowR < 0 || sx - glowR > width || sy + glowR < 0 || sy - glowR > height) {
          continue;
        }
        const [r, g, b] = anchor.rgb;
        const grad = ctx.createRadialGradient(sx, sy, 2, sx, sy, glowR);
        grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${anchor.selected ? 0.34 : 0.18})`);
        grad.addColorStop(0.5, `rgba(${r}, ${g}, ${b}, ${anchor.selected ? 0.12 : 0.06})`);
        grad.addColorStop(1, "rgba(0, 0, 0, 0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(sx, sy, glowR, 0, Math.PI * 2);
        ctx.fill();
      }

      // Update & draw each orbital particle
      for (let i = 0; i < orbitals.length; i++) {
        const p = orbitals[i];
        const anchor = nodeMap.get(p.nodeId);
        if (!anchor) continue;

        p.prevX = p.x;
        p.prevY = p.y;

        const speedMult = (anchor.selected ? 1.65 : 1.0) * trainingBoost;
        p.theta += p.orbitSpeed * speedMult;
        p.phi += p.phiSpeed * speedMult;

        // Semantic 3D / Geometric Particle Swarm Projection per Node Archetype
        let rx: number;
        let ry: number;
        const scale = anchor.selected ? 1.15 : 1.0;
        const halfW = anchor.width * 0.52 * scale;
        const halfH = anchor.height * 0.54 * scale;

        switch (anchor.swarmGeometry) {
          case "tri_prism": {
            // Triangular / Diamond prism lattice
            const lobe = p.id % 3;
            const lobeAngle = (lobe * Math.PI * 2) / 3 + time * 0.25;
            const r = (24 + p.shellRadius * 36) * scale;
            rx =
              anchor.x +
              Math.cos(lobeAngle) * (halfW * 0.48) +
              Math.cos(p.theta) * r * 0.65;
            ry =
              anchor.y +
              Math.sin(lobeAngle) * (halfH * 0.48) +
              Math.sin(p.theta) * r * 0.55;
            break;
          }
          case "lissajous_wave": {
            // Horizontal Figure-8 / State-Space Standing Wave
            rx = anchor.x + Math.sin(p.theta) * (halfW + 10 * p.shellRadius);
            ry =
              anchor.y +
              Math.sin(p.theta * 2 + p.phi) * (halfH * 0.72 + 8 * p.shellRadius);
            break;
          }
          case "vortex_funnel": {
            // Hyperbolic Gravity-Well Funnel
            const band = (Math.sin(p.phi) + 1) * 0.5; // 0 (bottom) to 1 (top)
            const funnelSpread = (0.35 + band * 0.85) * halfW;
            rx = anchor.x + Math.cos(p.theta * 1.4) * funnelSpread;
            ry = anchor.y + (band - 0.5) * (halfH * 1.65);
            break;
          }
          case "satellite_clusters": {
            // 4 Discrete Orbiting Expert Satellite Clusters
            const clusterIdx = p.id % 4;
            const clusterAng = (clusterIdx * Math.PI) / 2 + time * 0.35;
            const cxOff = Math.cos(clusterAng) * (halfW * 0.85);
            const cyOff = Math.sin(clusterAng) * (halfH * 0.85);
            const localR = 10 + p.shellRadius * 14;
            rx = anchor.x + cxOff + Math.cos(p.theta * 2) * localR;
            ry = anchor.y + cyOff + Math.sin(p.theta * 2) * localR;
            break;
          }
          case "matrix_lattice": {
            // Rectangular / Diamond Tensor Grid Lattice
            const col = (p.id % 5) - 2;
            const row = ((p.id >> 2) % 3) - 1;
            const jitterX = Math.cos(p.theta) * 9;
            const jitterY = Math.sin(p.theta) * 9;
            rx = anchor.x + col * (halfW * 0.42) + jitterX;
            ry = anchor.y + row * (halfH * 0.55) + jitterY;
            break;
          }
          case "stabilizer_rings": {
            // Two Orthogonal Counter-Rotating Gyroscopic Rings
            if (p.id % 2 === 0) {
              rx = anchor.x + Math.cos(p.theta) * (halfW + 10);
              ry = anchor.y + Math.sin(p.theta) * (halfH * 0.32);
            } else {
              rx = anchor.x + Math.cos(p.theta) * (halfW * 0.35);
              ry = anchor.y + Math.sin(p.theta) * (halfH + 8);
            }
            break;
          }
          case "cylinder_stack": {
            // 3 Stacked Horizontal Memory Register Discs
            const layer = (p.id % 3) - 1; // -1, 0, 1
            rx = anchor.x + Math.cos(p.theta) * (halfW * 0.95);
            ry =
              anchor.y +
              layer * (halfH * 0.58) +
              Math.sin(p.theta) * (halfH * 0.22);
            break;
          }
          case "radial_pulsar": {
            // 8-Ray Tactical Starburst Pulsar
            const ray = p.id % 8;
            const rayAng = (ray * Math.PI) / 4 + time * 0.15;
            const pulseR =
              (18 + ((Math.sin(p.theta * 2) + 1) * 0.5) * (halfW * 0.85)) * scale;
            rx = anchor.x + Math.cos(rayAng) * pulseR;
            ry = anchor.y + Math.sin(rayAng) * pulseR * 0.82;
            break;
          }
          case "hex_shield": {
            // 6-Sided Hexagonal Citadel Perimeter Shield
            const seg = p.id % 6;
            const a1 = (seg * Math.PI) / 3 + time * 0.12;
            const a2 = ((seg + 1) * Math.PI) / 3 + time * 0.12;
            const frac = (Math.sin(p.theta) + 1) * 0.5;
            const rX = halfW + 10 * p.shellRadius;
            const rY = halfH + 10 * p.shellRadius;
            rx =
              anchor.x +
              (Math.cos(a1) * (1 - frac) + Math.cos(a2) * frac) * rX;
            ry =
              anchor.y +
              (Math.sin(a1) * (1 - frac) + Math.sin(a2) * frac) * rY;
            break;
          }
          case "emitter_beam":
          case "harmonic_orbit":
          default: {
            const radX = halfW + 10 * p.shellRadius;
            const radY = halfH + 10 * p.shellRadius;
            rx = anchor.x + Math.cos(p.theta) * radX;
            ry = anchor.y + Math.sin(p.theta) * radY * Math.sin(p.phi);
            break;
          }
        }

        // Spring physics toward target orbital coordinate
        p.vx = (p.vx + (rx - p.x) * springK) * damping;
        p.vy = (p.vy + (ry - p.y) * springK) * damping;
        p.x += p.vx;
        p.y += p.vy;

        // Smooth color interpolation toward current node's role color
        p.rgb[0] += (anchor.rgb[0] - p.rgb[0]) * 0.08;
        p.rgb[1] += (anchor.rgb[1] - p.rgb[1]) * 0.08;
        p.rgb[2] += (anchor.rgb[2] - p.rgb[2]) * 0.08;

        const sx = p.x * vpZoom + vpX;
        const sy = p.y * vpZoom + vpY;
        if (sx < -20 || sx > width + 20 || sy < -20 || sy > height + 20) continue;

        const psx = p.prevX * vpZoom + vpX;
        const psy = p.prevY * vpZoom + vpY;
        const r = Math.round(p.rgb[0]);
        const g = Math.round(p.rgb[1]);
        const b = Math.round(p.rgb[2]);

        // Draw velocity comet streak when moving fast (e.g. during warp morph or node drag!)
        const velMag = Math.hypot(sx - psx, sy - psy);
        if (velMag > 1.8) {
          ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${Math.min(0.75, p.alpha * 0.85)})`;
          ctx.lineWidth = Math.max(1, p.size * vpZoom * 0.85);
          ctx.beginPath();
          ctx.moveTo(psx, psy);
          ctx.lineTo(sx, sy);
          ctx.stroke();
        }

        const radiusPx = Math.max(1, p.size * vpZoom);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${p.alpha})`;
        ctx.beginPath();
        ctx.arc(sx, sy, radiusPx, 0, Math.PI * 2);
        ctx.fill();
      }

      // 7. Synaptic Double-Helix Photon Streams along Edges
      const photons = edgePhotonsRef.current;
      const targetPerEdge = cfg.perEdge;
      const activeEdgeIds = new Set(currentEdges.map((e) => e.id));

      for (let i = photons.length - 1; i >= 0; i--) {
        if (!activeEdgeIds.has(photons[i].edgeId)) {
          photons.splice(i, 1);
        }
      }

      const edgeCounts = new Map<string, number>();
      for (const ph of photons) {
        edgeCounts.set(ph.edgeId, (edgeCounts.get(ph.edgeId) ?? 0) + 1);
      }

      for (const edge of currentEdges) {
        let count = edgeCounts.get(edge.id) ?? 0;
        while (count < targetPerEdge) {
          const strandCycle = count % 3 === 0 ? 0 : count % 3 === 1 ? 1 : -1;
          photons.push({
            edgeId: edge.id,
            t: Math.random(),
            speed: 0.0055 + Math.random() * 0.0065,
            phase: Math.random() * Math.PI * 2,
            strand: strandCycle,
            size: strandCycle === 0 ? 2.4 : 1.5,
            history: [],
          });
          count++;
        }
      }

      // Map edge geometry and step photons
      const edgeById = new Map(currentEdges.map((e) => [e.id, e]));
      for (let i = 0; i < photons.length; i++) {
        const ph = photons[i];
        const edge = edgeById.get(ph.edgeId);
        if (!edge) continue;

        const srcNode = currentNodes.find((n) => n.id === edge.source);
        const dstNode = currentNodes.find((n) => n.id === edge.target);
        if (!srcNode || !dstNode) continue;

        const srcW =
          mode === "particle_nebula"
            ? srcNode.measured?.width ?? 148
            : srcNode.measured?.width ?? 174;
        const srcH =
          mode === "particle_nebula"
            ? srcNode.measured?.height ?? 76
            : srcNode.measured?.height ?? 86;
        const dstH =
          mode === "particle_nebula"
            ? dstNode.measured?.height ?? 76
            : dstNode.measured?.height ?? 86;

        const srcPortIdx = edge.sourceHandle
          ? parseInt(edge.sourceHandle.replace("out-", ""), 10) || 0
          : 0;
        const dstPortIdx = edge.targetHandle
          ? parseInt(edge.targetHandle.replace("in-", ""), 10) || 0
          : 0;

        const srcPortCount = Math.max(1, srcNode.data.neuronDef.output_ports.length);
        const dstPortCount = Math.max(1, dstNode.data.neuronDef.input_ports.length);

        const p0x = srcNode.position.x + srcW;
        const p0y =
          srcNode.position.y +
          srcH * 0.54 +
          (srcPortIdx - (srcPortCount - 1) * 0.5) * 18;
        const p3x = dstNode.position.x;
        const p3y =
          dstNode.position.y +
          dstH * 0.54 +
          (dstPortIdx - (dstPortCount - 1) * 0.5) * 18;

        const ctrlDist = Math.max(48, Math.abs(p3x - p0x) * 0.45);
        const p1x = p0x + ctrlDist;
        const p1y = p0y;
        const p2x = p3x - ctrlDist;
        const p2y = p3y;

        ph.t += ph.speed * trainingBoost;
        if (ph.t >= 1) {
          ph.t -= 1;
          ph.history.length = 0;
        }

        const pt = cubicBezierPoint(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, ph.t);

        // Double-helix sinusoidal offset perpendicular to curve tangent
        const envelope = Math.sin(ph.t * Math.PI); // 0 at ends, 1 in middle
        const helixOffset =
          ph.strand === 0
            ? 0
            : Math.sin(ph.t * 16 - time * 4 + ph.phase) * 9 * ph.strand * envelope;

        const fx = pt.x + pt.nx * helixOffset;
        const fy = pt.y + pt.ny * helixOffset;

        ph.history.unshift({ x: fx, y: fy });
        if (ph.history.length > 6) {
          ph.history.pop();
        }

        const sx = fx * vpZoom + vpX;
        const sy = fy * vpZoom + vpY;
        if (sx < -30 || sx > width + 30 || sy < -30 || sy > height + 30) continue;

        const rgb = ph.strand === 0 ? pal.primaryRgb : pal.accentRgb;

        // Draw comet tail
        if (ph.history.length > 1) {
          ctx.beginPath();
          ctx.moveTo(sx, sy);
          for (let hIdx = 1; hIdx < ph.history.length; hIdx++) {
            const hp = ph.history[hIdx];
            ctx.lineTo(hp.x * vpZoom + vpX, hp.y * vpZoom + vpY);
          }
          ctx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.55)`;
          ctx.lineWidth = Math.max(1, ph.size * vpZoom * 0.75);
          ctx.stroke();
        }

        // Photon head
        ctx.fillStyle =
          ph.strand === 0
            ? "#ffffff"
            : `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.95)`;
        ctx.beginPath();
        ctx.arc(sx, sy, Math.max(1.2, ph.size * vpZoom), 0, Math.PI * 2);
        ctx.fill();
      }

      // 8. Supernova Burst Particles & Shockwave Rings
      const bursts = burstParticlesRef.current;
      for (let i = bursts.length - 1; i >= 0; i--) {
        const bp = bursts[i];
        bp.life += 1;
        if (bp.life >= bp.maxLife) {
          bursts.splice(i, 1);
          continue;
        }
        bp.vx *= 0.93;
        bp.vy *= 0.93;
        bp.x += bp.vx;
        bp.y += bp.vy;

        const progress = bp.life / bp.maxLife;
        const alpha = (1 - progress) * 0.95;
        const sx = bp.x * vpZoom + vpX;
        const sy = bp.y * vpZoom + vpY;
        const [r, g, b] = bp.rgb;

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
        ctx.beginPath();
        ctx.arc(sx, sy, Math.max(1, bp.size * (1 - progress * 0.5) * vpZoom), 0, Math.PI * 2);
        ctx.fill();
      }

      const waves = shockwavesRef.current;
      for (let i = waves.length - 1; i >= 0; i--) {
        const sw = waves[i];
        sw.life += 1;
        if (sw.life >= sw.maxLife) {
          waves.splice(i, 1);
          continue;
        }
        const progress = sw.life / sw.maxLife;
        // Ease-out cubic expansion
        const eased = 1 - Math.pow(1 - progress, 3);
        const currentR = (sw.radius + (sw.maxRadius - sw.radius) * eased) * vpZoom;
        const sx = sw.x * vpZoom + vpX;
        const sy = sw.y * vpZoom + vpY;
        const [r, g, b] = sw.rgb;

        ctx.save();
        ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${(1 - progress) * 0.75})`;
        ctx.lineWidth = sw.lineWidth;
        if (sw.dashed) {
          ctx.setLineDash([10, 8]);
        }
        ctx.beginPath();
        ctx.arc(sx, sy, currentR, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // 9. Interactive Mouse Cursor Holographic Reticle
      if (m.active) {
        const [pr, pg, pb] = pal.primaryRgb;
        ctx.strokeStyle = `rgba(${pr}, ${pg}, ${pb}, 0.28)`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(m.x, m.y, 18, time * 2, time * 2 + Math.PI * 1.35);
        ctx.stroke();
      }

      ctx.restore();
      animationFrameId = window.requestAnimationFrame(renderFrame);
    };

    animationFrameId = window.requestAnimationFrame(renderFrame);
    return () => {
      window.cancelAnimationFrame(animationFrameId);
      ro?.disconnect();
    };
  }, [getViewport]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      if (viewFullGraph) {
        setFullGraphNodes((prev) =>
          applyNodeChanges(changes, prev) as Node<NeuronNodeData>[],
        );
        return;
      }
      if (isExpandingRef.current) {
        // Allow dimension measurement updates while ignoring transient animated positions
        const dimOnly = changes.filter((c) => c.type === "dimensions");
        if (dimOnly.length > 0) {
          applyActiveNodeChanges(dimOnly);
        }
        return;
      }
      applyActiveNodeChanges(changes);
    },
    [applyActiveNodeChanges, viewFullGraph],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      if (viewFullGraph || isExpandingRef.current) {
        return;
      }
      applyActiveEdgeChanges(changes);
    },
    [applyActiveEdgeChanges, viewFullGraph],
  );

  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      if (viewFullGraph) {
        return;
      }
      connectActiveGraph(connection);
    },
    [connectActiveGraph, viewFullGraph],
  );

  const onNodeClick = useCallback(
    (event: React.MouseEvent, node: Node<NeuronNodeData>) => {
      selectNode(node.id);
      if (
        !viewFullGraph &&
        !event.shiftKey &&
        node.data.neuronDef.kind === "subgraph" &&
        node.data.neuronDef.subgraph
      ) {
        triggerSubgraphExpansion(node);
      }
    },
    [selectNode, triggerSubgraphExpansion, viewFullGraph],
  );

  const onNodeDoubleClick = useCallback(
    (_: React.MouseEvent, node: Node<NeuronNodeData>) => {
      if (
        !viewFullGraph &&
        !isExpandingRef.current &&
        node.data.neuronDef.kind === "subgraph" &&
        node.data.neuronDef.subgraph
      ) {
        triggerSubgraphExpansion(node);
      }
    },
    [triggerSubgraphExpansion, viewFullGraph],
  );

  const handleIsolateZone = useCallback(
    (path: GraphPathSegment[]) => {
      setViewFullGraph(false);
      setPath(path);
    },
    [setPath, setViewFullGraph],
  );

  const handleFitZone = useCallback(
    (zone: SubgraphZone) => {
      const container = containerRef.current;
      const rect = container?.getBoundingClientRect();
      const cw = Math.max(480, rect?.width ?? 960);
      const ch = Math.max(360, rect?.height ?? 640);
      const targetZoom = Math.min(
        1.2,
        Math.max(
          0.18,
          Math.min((cw * 0.78) / zone.width, (ch * 0.74) / zone.height),
        ),
      );
      const cx = zone.x + zone.width * 0.5;
      const cy = zone.y + zone.height * 0.5;
      setCenter(cx, cy, { zoom: targetZoom, duration: 520 });
    },
    [setCenter],
  );

  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  const onMove = useCallback((_: any, viewport: Viewport) => {
    viewportRef.current = viewport;
    setViewportState(viewport);
  }, []);

  const onMoveEnd = useCallback(
    (_: any, viewport: Viewport) => {
      viewportRef.current = viewport;
      setViewportState(viewport);
      updatePreferredInsertPosition();
    },
    [updatePreferredInsertPosition],
  );

  // Build WebGL 3D node anchors for the GPU wireframe/shader scene
  const webglNodes = useMemo<WebGLNodeAnchor[]>(() => {
    return nodes
      .filter((n) => Number(n.style?.opacity ?? 1) > 0.05)
      .map((n, idx) => {
        const archetype = resolveNodeArchetype(
          n.data.label,
          n.data.neuronDef,
          n.data.isInput,
          n.data.isOutput,
        );
        const w = n.measured?.width ?? archetype.minWidth;
        const h = n.measured?.height ?? 100;
        return {
          id: n.id,
          x: n.position.x + w * 0.5,
          y: n.position.y + h * 0.5,
          rgb: [...archetype.primaryRgb] as [number, number, number],
          secondaryRgb: [...archetype.secondaryRgb] as [number, number, number],
          selected: Boolean(n.selected),
          seed: idx * 1.37,
        };
      });
  }, [nodes]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const prev = mouseRef.current;
    mouseRef.current = {
      x,
      y,
      vx: x - prev.x,
      vy: y - prev.y,
      active: true,
      normX: (x / Math.max(1, rect.width) - 0.5) * 2,
      normY: (y / Math.max(1, rect.height) - 0.5) * 2,
    };
  }, []);

  const handleMouseLeave = useCallback(() => {
    mouseRef.current.active = false;
    mouseRef.current.normX = 0;
    mouseRef.current.normY = 0;
  }, []);

  const cyclePalette = useCallback(() => {
    const keys = Object.keys(PARTICLE_PALETTES) as ParticleColorPalette[];
    const idx = keys.indexOf(colorPalette);
    setColorPalette(keys[(idx + 1) % keys.length]);
  }, [colorPalette, setColorPalette]);

  // Compute bounding box for the Tactical Circular Radar Scanner
  const radarData = useMemo(() => {
    if (nodes.length === 0) {
      return { minX: -200, maxX: 200, minY: -200, maxY: 200, blips: [] };
    }
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const n of nodes) {
      minX = Math.min(minX, n.position.x);
      maxX = Math.max(maxX, n.position.x + 170);
      minY = Math.min(minY, n.position.y);
      maxY = Math.max(maxY, n.position.y + 85);
    }
    const padX = Math.max(120, (maxX - minX) * 0.25);
    const padY = Math.max(120, (maxY - minY) * 0.25);
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;
    const spanX = Math.max(1, maxX - minX);
    const spanY = Math.max(1, maxY - minY);

    const blips = nodes.map((n) => {
      const nx = ((n.position.x + 85 - minX) / spanX - 0.5) * 2; // -1 to 1
      const ny = ((n.position.y + 42 - minY) / spanY - 0.5) * 2;
      return {
        id: n.id,
        rx: nx * 48,
        ry: ny * 48,
        flowX: n.position.x + 85,
        flowY: n.position.y + 42,
        selected: Boolean(n.selected),
        isSubgraph: n.data.neuronDef.kind === "subgraph",
        isModule: n.data.neuronDef.kind === "module",
      };
    });

    return { minX, maxX, minY, maxY, blips };
  }, [nodes]);

  const totalParticleEstimate =
    densityConfig.ambient +
    nodes.length * densityConfig.perNode +
    edges.length * densityConfig.perEdge;

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className="relative flex-1 h-full min-h-0 overflow-hidden bg-slate-950 select-none"
      style={{
        perspective: holoPerspective3D ? "1400px" : undefined,
      }}
    >
      {/* 1. 60 FPS Hardware-Accelerated Particle & Holographic Canvas Layer */}
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 z-0"
      />

      {/* 1b. Hardware-Accelerated WebGL 3D Shader & Rotating Wireframe Scene */}
      <WebGLCyberScene
        nodes={webglNodes}
        viewport={viewportState}
        primaryRgb={palette.primaryRgb}
        secondaryRgb={palette.secondaryRgb}
        accentRgb={palette.accentRgb}
        isTraining={isTraining}
      />

      {/* 2. Holographic Scanline & Vignette Overlay */}
      <div
        className="pointer-events-none absolute inset-0 z-[1]"
        style={{
          background:
            "repeating-linear-gradient(0deg, rgba(0, 240, 255, 0.018) 0px, rgba(0, 240, 255, 0.018) 1px, transparent 1px, transparent 3px)",
          boxShadow: "inset 0 0 90px rgba(0, 0, 0, 0.75)",
        }}
      />

      {/* 3. Cinematic Subgraph / Template Warp Flash Overlay */}
      <div
        className="pointer-events-none absolute inset-0 z-30 transition-opacity duration-500"
        style={{
          opacity: warpOverlayAlpha * 0.45,
          background: `radial-gradient(circle at center, ${palette.primary}55 0%, ${palette.accent}22 45%, transparent 75%)`,
        }}
      />

      {/* 3b. Full Graph Mode Nested Subgraph Zones Layer */}
      <FullGraphZonesLayer
        zones={viewFullGraph && fullExpanded ? fullExpanded.zones : []}
        viewport={viewportState}
        cyberMode={true}
        onIsolateZone={handleIsolateZone}
        onFitZone={handleFitZone}
      />

      {/* 3c. Animated Subgraph Expansion Chassis Overlay */}
      <SubgraphExpansionOverlay
        expansionState={expansionState}
        cyberMode={true}
      />

      {/* 4. Interactive React Flow Layer with 3D Holo-Perspective Support */}
      <div
        className="relative z-10 h-full w-full transition-transform duration-300 ease-out"
        style={{
          transform: holoPerspective3D
            ? "rotateX(11deg) rotateY(-2deg) scale(0.985)"
            : "none",
          transformStyle: holoPerspective3D ? "preserve-3d" : undefined,
        }}
      >
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
          nodeTypes={particleNodeTypes}
          edgeTypes={particleEdgeTypes}
          nodeDragThreshold={5}
          fitView
          style={{ background: "transparent" }}
          className="!bg-transparent"
          defaultEdgeOptions={{
            animated: false,
          }}
        >
          <Controls className="!bg-slate-950/90 !border-cyan-500/40 !text-cyan-200 !shadow-[0_0_20px_rgba(0,240,255,0.2)] [&>button]:!bg-slate-950/90 [&>button]:!border-cyan-500/30 [&>button]:!text-cyan-200 hover:[&>button]:!bg-cyan-950" />
        </ReactFlow>
      </div>

      {/* 5. Top Holographic Command & Telemetry HUD Overlay */}
      <div className="pointer-events-none absolute left-3 right-3 top-3 z-20 flex flex-wrap items-center justify-between gap-2">
        {/* Left Telemetry Badge & Subgraph Breadcrumb Matrix */}
        <div className="pointer-events-auto flex items-center gap-2 rounded-lg border border-cyan-500/35 bg-slate-950/85 px-3 py-1.5 backdrop-blur-md shadow-[0_0_25px_rgba(0,240,255,0.14)]">
          <div
            className="flex h-6 w-6 items-center justify-center rounded-full border"
            style={{
              borderColor: palette.primary,
              boxShadow: `0 0 10px ${palette.primary}66`,
            }}
          >
            <div
              className="h-2.5 w-2.5 animate-pulse rounded-full"
              style={{ backgroundColor: palette.primary }}
            />
          </div>

          <div className="flex flex-col">
            <div className="flex items-center gap-2 font-mono text-[10px] font-bold tracking-widest text-cyan-300">
              <span style={{ color: palette.primary }}>CYBER 3D // WEBGL ENGINE</span>
              <span className="text-gray-600">|</span>
              <span className="text-amber-300">{transitionStatus}</span>
              {viewFullGraph && (
                <>
                  <span className="text-gray-600">|</span>
                  <span className="rounded border border-emerald-500/50 bg-emerald-950/70 px-1.5 py-0.2 text-[9px] text-emerald-300">
                    ⊞ FULL GRAPH ({fullExpanded?.zones.length ?? 0} SUBGRAPHS)
                  </span>
                </>
              )}
            </div>
            <div className="flex items-center gap-2 font-mono text-[9px] text-gray-400">
              <span>PARTICLES: {totalParticleEstimate.toLocaleString()}</span>
              <span>•</span>
              <span>NODES: {nodes.length}</span>
              <span>•</span>
              <span>SYNAPSES: {edges.length}</span>
              {breadcrumbs.length > 1 && (
                <>
                  <span>•</span>
                  <div className="flex items-center gap-1 text-amber-300">
                    {breadcrumbs.map((crumb, idx) => (
                      <React.Fragment key={`${crumb.id}-${idx}`}>
                        {idx > 0 && <span>▸</span>}
                        <button
                          onClick={() => {
                            if (viewFullGraph) setViewFullGraph(false);
                            setPath(crumb.path);
                          }}
                          className="hover:underline"
                        >
                          {crumb.label}
                        </button>
                      </React.Fragment>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Right Interactive Holographic Mode & FX Controls */}
        <div className="pointer-events-auto flex flex-wrap items-center gap-1.5 rounded-lg border border-cyan-500/35 bg-slate-950/85 px-2.5 py-1.5 font-mono text-[10px] backdrop-blur-md shadow-[0_0_25px_rgba(0,240,255,0.14)]">
          {/* View Full Graph Toggle */}
          <button
            type="button"
            onClick={() => setViewFullGraph(!viewFullGraph)}
            className={`flex items-center gap-1.5 rounded border px-2.5 py-1 font-bold transition-all ${
              viewFullGraph
                ? "border-emerald-400/90 bg-emerald-500/25 text-emerald-200 shadow-[0_0_16px_rgba(16,185,129,0.4)]"
                : "border-cyan-500/45 bg-cyan-950/45 text-cyan-200 hover:border-cyan-400 hover:bg-cyan-900/50"
            }`}
            title="Expand all nested subgraphs into a single unified full-architecture graph view"
          >
            <span>⊞</span>
            <span>FULL GRAPH: {viewFullGraph ? "ON" : "OFF"}</span>
          </button>

          {/* Switch between HUD + Swarm vs Pure Particle Nebula */}
          <button
            type="button"
            onClick={() =>
              setVisualMode(
                visualMode === "hybrid_hud" ? "particle_nebula" : "hybrid_hud",
              )
            }
            className="flex items-center gap-1.5 rounded border px-2.5 py-1 font-bold transition-all"
            style={{
              borderColor: `${palette.primary}77`,
              backgroundColor: `${palette.primary}18`,
              color: palette.primary,
            }}
            title="Switch between Full 3D Volumetric Chassis and Compact 3D Sculpture Mode"
          >
            <span>✦</span>
            <span>
              {visualMode === "hybrid_hud" ? "MODE: 3D CHASSIS" : "MODE: 3D SCULPTURE"}
            </span>
          </button>

          {/* 3D Holographic Tilt Toggle */}
          <button
            type="button"
            onClick={() => setHoloPerspective3D(!holoPerspective3D)}
            className={`rounded border px-2 py-1 transition-all ${
              holoPerspective3D
                ? "border-amber-400/80 bg-amber-500/20 text-amber-200 shadow-[0_0_12px_rgba(251,191,36,0.3)]"
                : "border-gray-700 bg-gray-900/80 text-gray-300 hover:border-gray-600"
            }`}
            title="Toggle 3D Holographic Table Projection Tilt"
          >
            3D TILT: {holoPerspective3D ? "ON" : "OFF"}
          </button>

          {/* Palette Cycler */}
          <button
            type="button"
            onClick={cyclePalette}
            className="flex items-center gap-1.5 rounded border border-gray-700 bg-gray-900/80 px-2 py-1 text-gray-200 hover:border-cyan-500/50"
            title="Cycle Holographic Color Matrix"
          >
            <span
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: palette.primary }}
            />
            <span>{palette.name.split(" ")[0].toUpperCase()}</span>
          </button>

          {/* Electromagnetic Pulse Wave Button */}
          <button
            type="button"
            onClick={() => {
              triggerPulse();
            }}
            className="rounded border border-amber-500/50 bg-amber-950/50 px-2 py-1 font-bold text-amber-200 transition-all hover:bg-amber-900/60 hover:shadow-[0_0_14px_rgba(245,158,11,0.4)]"
            title="Fire an Electromagnetic Shockwave & Particle Surge across the graph"
          >
            ⚡ PULSE
          </button>

          {/* Fit View / Recenter */}
          <button
            type="button"
            onClick={() => fitView({ duration: 500, padding: 0.22 })}
            className="rounded border border-gray-700 bg-gray-900/80 px-2 py-1 text-gray-300 hover:border-gray-600 hover:text-white"
            title="Smoothly Recenter Holographic Projection"
          >
            ⌖ ALIGN
          </button>

          {/* Open Full Settings Modal */}
          <button
            type="button"
            onClick={() => setSettingsOpen(true)}
            className="rounded border border-cyan-500/40 bg-cyan-950/50 px-2 py-1 text-cyan-200 hover:bg-cyan-900/60"
            title="Open Holographic Particle Engine Settings"
          >
            ⚙ CFG
          </button>

          {/* Quick Exit to Standard 2D Canvas */}
          <button
            type="button"
            onClick={() => setParticleGraphEditorEnabled(false)}
            className="rounded border border-gray-800 bg-gray-900/70 px-2 py-1 text-gray-400 hover:border-rose-500/50 hover:text-rose-200"
            title="Switch back to standard 2D editor"
          >
            2D
          </button>
        </div>
      </div>

      {/* 6. Bottom-Right Circular Tactical Radar Scanner HUD */}
      {hudTelemetryOverlay && (
        <div className="pointer-events-auto absolute bottom-4 right-4 z-20 flex flex-col items-center">
          <div
            className="relative flex h-32 w-32 items-center justify-center rounded-full border bg-slate-950/85 backdrop-blur-md shadow-[0_0_30px_rgba(0,240,255,0.2)]"
            style={{ borderColor: `${palette.primary}66` }}
            title="Tactical Radar — Click any blip to warp camera"
          >
            {/* Radar Concentric Rings & Crosshairs */}
            <svg viewBox="-60 -60 120 120" className="h-full w-full">
              <circle
                cx="0"
                cy="0"
                r="54"
                fill="none"
                stroke={palette.primary}
                strokeWidth="0.8"
                strokeOpacity="0.4"
                strokeDasharray="6 4"
              />
              <circle
                cx="0"
                cy="0"
                r="36"
                fill="none"
                stroke={palette.primary}
                strokeWidth="0.6"
                strokeOpacity="0.25"
              />
              <circle
                cx="0"
                cy="0"
                r="18"
                fill="none"
                stroke={palette.primary}
                strokeWidth="0.6"
                strokeOpacity="0.2"
              />
              <line
                x1="-54"
                y1="0"
                x2="54"
                y2="0"
                stroke={palette.primary}
                strokeWidth="0.5"
                strokeOpacity="0.25"
              />
              <line
                x1="0"
                y1="-54"
                x2="0"
                y2="54"
                stroke={palette.primary}
                strokeWidth="0.5"
                strokeOpacity="0.25"
              />

              {/* Rotating Radar Sweep Line */}
              <g className="animate-[spin_4s_linear_infinite]">
                <line
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="-54"
                  stroke={palette.primary}
                  strokeWidth="1.5"
                  strokeOpacity="0.8"
                />
                <path
                  d="M 0 0 L 0 -54 A 54 54 0 0 0 -38 -38 Z"
                  fill={palette.primary}
                  fillOpacity="0.14"
                />
              </g>

              {/* Node Blips on Radar */}
              {radarData.blips.map((blip) => {
                const blipColor = blip.selected
                  ? "#ffffff"
                  : blip.isSubgraph
                  ? palette.subgraph
                  : blip.isModule
                  ? palette.module
                  : palette.primary;
                return (
                  <g
                    key={blip.id}
                    className="cursor-pointer"
                    onClick={() => {
                      selectNode(blip.id);
                      setCenter(blip.flowX, blip.flowY, {
                        zoom: Math.max(viewportRef.current.zoom, 0.95),
                        duration: 500,
                      }).catch(() => undefined);
                    }}
                  >
                    <circle
                      cx={blip.rx}
                      cy={blip.ry}
                      r={blip.selected ? 3.6 : 2.3}
                      fill={blipColor}
                    />
                    {blip.selected && (
                      <circle
                        cx={blip.rx}
                        cy={blip.ry}
                        r={6.5}
                        fill="none"
                        stroke={palette.accent}
                        strokeWidth="1"
                      />
                    )}
                  </g>
                );
              })}
            </svg>

            <div
              className="pointer-events-none absolute bottom-1.5 font-mono text-[7px] uppercase tracking-widest"
              style={{ color: palette.primary }}
            >
              RADAR // {nodes.length} TARGETS
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
