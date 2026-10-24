import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import type { Edge as RFEdge, Node, Viewport } from "@xyflow/react";
import { type NodeArchetypeId, resolveNodeArchetype } from "./nodeArchetypes";
import { SemanticNodeGlyph } from "./ParticleNeuronNode";
import {
  type CoreMorphSource,
  type FlowNodeData,
  type GraphPathSegment,
  type SubgraphZone,
  getGraphAtPath,
  graphToFlowEdges,
  graphToFlowNodes,
} from "../store/graphUtils";
import type { GraphData } from "../api/client";

export interface ScreenRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface SiblingGhost {
  id: string;
  label: string;
  rect: ScreenRect;
  dx: number;
  dy: number;
  color: string;
}

export interface CollapsingChildGhost {
  id: string;
  archetypeId: NodeArchetypeId;
  neuronName: string;
  clipPath: string;
  label: string;
  badgeText: string;
  color: string;
  secondaryColor: string;
  startRect: ScreenRect;
  startCenterX: number;
  startCenterY: number;
  burstOffsetX: number;
  burstOffsetY: number;
  recallStartAt: number;
  recallEndAt: number;
}

export interface CollapsingEdgeGhost {
  id: string;
  sourceId: string;
  targetId: string;
  color: string;
  despawnAt: number;
}

export interface EmergingNodeTether {
  id: string;
  label: string;
  screenX: number;
  screenY: number;
  color: string;
  progress: number;
  alpha: number;
}

export interface SubgraphExpansionAnimState {
  mode: "expand" | "collapse";
  nodeId: string;
  neuronName: string;
  archetypeId: NodeArchetypeId;
  label: string;
  badgeText: string;
  primaryColor: string;
  secondaryColor: string;
  primaryRgb: [number, number, number];
  /** Screen position of the central animated 3D core / dimensional rift portal */
  singularityX: number;
  singularityY: number;
  /** Pulse intensity (0..1) whenever a node breaches the portal mouth */
  portalPulse: number;
  /** Screen position where the node's title/name sits before detaching (or after re-attaching) */
  titleAnchorX: number;
  titleAnchorY: number;
  progress: number;
  spawnedCount: number;
  totalCount: number;
  siblingGhosts: SiblingGhost[];
  collapsingChildren: CollapsingChildGhost[];
  collapsingEdges: CollapsingEdgeGhost[];
  emergingTethers: EmergingNodeTether[];
  animatedNodes: Node<FlowNodeData>[];
  animatedEdges: RFEdge[];
}

interface GraphViewSnapshot {
  path: GraphPathSegment[];
  nodes: Node<FlowNodeData>[];
  edges: RFEdge[];
  viewport: Viewport;
}

interface UseSubgraphExpansionParams {
  containerRef: React.RefObject<HTMLDivElement | null>;
  rootGraph?: GraphData;
  currentPath?: GraphPathSegment[];
  activeNodes: Node<FlowNodeData>[];
  activeEdges?: RFEdge[];
  viewFullGraph?: boolean;
  openSubgraph: (id: string) => void;
  setViewport: (viewport: Viewport, options?: { duration?: number }) => void;
  getViewport: () => Viewport;
  onViewportSync?: (viewport: Viewport) => void;
  onSpawnBurst?: (
    flowX: number,
    flowY: number,
    rgb: [number, number, number],
    count?: number,
    maxRadius?: number,
  ) => void;
}

function serializePathKey(path: GraphPathSegment[]): string {
  return path
    .map((seg) =>
      seg.kind === "node"
        ? `n:${seg.nodeId}`
        : `v:${seg.family}@${seg.version}`,
    )
    .join("/");
}

function isStrictPrefixPath(
  candidatePrefix: GraphPathSegment[],
  fullPath: GraphPathSegment[],
): boolean {
  if (candidatePrefix.length >= fullPath.length) {
    return false;
  }
  for (let i = 0; i < candidatePrefix.length; i++) {
    const a = candidatePrefix[i];
    const b = fullPath[i];
    if (a.kind !== b.kind) return false;
    if (a.kind === "node" && b.kind === "node" && a.nodeId !== b.nodeId) {
      return false;
    }
    if (
      a.kind === "variant" &&
      b.kind === "variant" &&
      (a.family !== b.family || a.version !== b.version)
    ) {
      return false;
    }
  }
  return true;
}

export function useSubgraphExpansion({
  containerRef,
  rootGraph,
  currentPath = [],
  activeNodes,
  activeEdges = [],
  viewFullGraph = false,
  openSubgraph,
  setViewport,
  getViewport,
  onViewportSync,
  onSpawnBurst,
}: UseSubgraphExpansionParams) {
  const [expansionState, setExpansionState] =
    useState<SubgraphExpansionAnimState | null>(null);
  const rafRef = useRef<number | null>(null);
  const activeNodesRef = useRef(activeNodes);
  activeNodesRef.current = activeNodes;
  const activeEdgesRef = useRef(activeEdges);
  activeEdgesRef.current = activeEdges;
  const currentPathRef = useRef(currentPath);
  currentPathRef.current = currentPath;

  // Remembers which subgraph node ID was clicked at each depth so jumping back via breadcrumbs
  // collapses back into the exact instance node that was expanded.
  const enteredNodeByDepthRef = useRef<Map<number, string>>(new Map());
  const forwardAnimStartedForPathRef = useRef<string | null>(null);

  const safeGetViewport = useCallback((): Viewport => {
    try {
      const currentVp = getViewport();
      if (currentVp && Number.isFinite(currentVp.zoom) && currentVp.zoom > 0) {
        return currentVp;
      }
    } catch {
      // Fallback
    }
    return { x: 0, y: 0, zoom: 1 };
  }, [getViewport]);

  const lastSnapshotRef = useRef<GraphViewSnapshot>({
    path: currentPath,
    nodes: activeNodes,
    edges: activeEdges,
    viewport: { x: 0, y: 0, zoom: 1 },
  });

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  const triggerSubgraphExpansion = useCallback(
    (clickedNode: Node<FlowNodeData>) => {
      const subgraph = clickedNode.data.neuronDef.subgraph;
      if (clickedNode.data.neuronDef.kind !== "subgraph" || !subgraph) {
        return false;
      }

      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }

      const container = containerRef.current;
      const rect = container?.getBoundingClientRect();
      const canvasW = Math.max(480, rect?.width ?? 960);
      const canvasH = Math.max(360, rect?.height ?? 640);

      const vp = safeGetViewport();

      const archetype = resolveNodeArchetype(
        clickedNode.data.label,
        clickedNode.data.neuronDef,
        clickedNode.data.isInput,
        clickedNode.data.isOutput,
      );

      const nodeW = clickedNode.measured?.width ?? archetype.minWidth + 12;
      const nodeH = clickedNode.measured?.height ?? 112;

      const startScreenRect: ScreenRect = {
        x: clickedNode.position.x * vp.zoom + vp.x,
        y: clickedNode.position.y * vp.zoom + vp.y,
        width: Math.max(120, nodeW * vp.zoom),
        height: Math.max(68, nodeH * vp.zoom),
      };

      // The central animated part of the node (3D Volumetric Core) in screen coordinates
      const startCoreX = startScreenRect.x + startScreenRect.width * 0.5;
      const startCoreY = startScreenRect.y + startScreenRect.height * 0.54;
      const startTitleX = startCoreX;
      const startTitleY = startScreenRect.y + 16;

      // Record the entered node ID at this depth so reverse breadcrumb navigation knows the exact parent node
      enteredNodeByDepthRef.current.set(
        currentPathRef.current.length,
        clickedNode.id,
      );

      // Capture outgoing sibling nodes so they smoothly recede outward as we zoom into the core
      const siblingGhosts: SiblingGhost[] = [];
      for (const sibling of activeNodesRef.current) {
        if (sibling.id === clickedNode.id) continue;
        const sibArch = resolveNodeArchetype(
          sibling.data.label,
          sibling.data.neuronDef,
          sibling.data.isInput,
          sibling.data.isOutput,
        );
        const sw = (sibling.measured?.width ?? sibArch.minWidth) * vp.zoom;
        const sh = (sibling.measured?.height ?? 100) * vp.zoom;
        const sx = sibling.position.x * vp.zoom + vp.x;
        const sy = sibling.position.y * vp.zoom + vp.y;
        const scx = sx + sw * 0.5;
        const scy = sy + sh * 0.5;
        const vx = scx - startCoreX;
        const vy = scy - startCoreY;
        const dist = Math.hypot(vx, vy) || 1;
        siblingGhosts.push({
          id: sibling.id,
          label: sibling.data.label,
          rect: { x: sx, y: sy, width: sw, height: sh },
          dx: (vx / dist) * 210,
          dy: (vy / dist) * 210,
          color: sibArch.primary,
        });
      }

      // Prepare target subgraph child components sorted in topological / left-to-right order
      const rawChildNodes = graphToFlowNodes(subgraph);
      const rawChildEdges = graphToFlowEdges(subgraph);

      const sortedChildNodes = [...rawChildNodes].sort((a, b) =>
        Math.abs(a.position.x - b.position.x) > 14
          ? a.position.x - b.position.x
          : a.position.y - b.position.y,
      );

      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      for (const cn of sortedChildNodes) {
        minX = Math.min(minX, cn.position.x);
        maxX = Math.max(maxX, cn.position.x + 225);
        minY = Math.min(minY, cn.position.y);
        maxY = Math.max(maxY, cn.position.y + 115);
      }
      if (!Number.isFinite(minX)) {
        minX = 0;
        maxX = 400;
        minY = 0;
        maxY = 260;
      }

      const centerX = (minX + maxX) * 0.5;
      const centerY = (minY + maxY) * 0.5;
      const spanX = Math.max(380, maxX - minX);
      const spanY = Math.max(250, maxY - minY);

      // Final zoomed-out viewport framing all child components cleanly
      const endZoom = Math.min(
        1.08,
        Math.max(
          0.22,
          Math.min((canvasW * 0.72) / spanX, (canvasH * 0.68) / spanY),
        ),
      );
      const endVpX = canvasW * 0.5 - centerX * endZoom;
      const endVpY = canvasH * 0.52 - centerY * endZoom;

      // Starting zoomed-in viewport anchored at the central animated part of the clicked node
      const startZoom = Math.min(2.35, Math.max(endZoom * 2.15, vp.zoom * 1.4));
      const startVpX = startCoreX - centerX * startZoom;
      const startVpY = startCoreY - centerY * startZoom;

      const totalCount = sortedChildNodes.length;

      // Radial burst layout all around the parent node's central core:
      // Stage 1 (p = 0.08 -> 0.34): Pull each cloned 3D core out of the parent into a ring all around the parent.
      // Stage 2 (p = 0.34 -> 0.82): All emitted cores slowly transition into their target places at once while
      //                             visibly morphing their 3D vector wireframe animations into the new node's wireframe.
      // Stage 3 (p = 0.82 -> 1.00): Once in place with their new 3D wireframes formed, the outer boxes, ports,
      //                             and labels grow outward from the outer edges of each central 3D core.
      const burstRadius = Math.min(200, Math.max(125, 95 + totalCount * 7));
      const emitWindowStart = 0.08;
      const emitWindowEnd = 0.32;
      const emitGap =
        totalCount <= 1
          ? 0
          : (emitWindowEnd - emitWindowStart - 0.12) /
            Math.max(1, totalCount - 1);

      const nodeBurstMetaById = new Map<
        string,
        {
          emitStart: number;
          emitEnd: number;
          burstNodeX: number;
          burstNodeY: number;
        }
      >();

      sortedChildNodes.forEach((cn, idx) => {
        const angle =
          (idx / Math.max(1, totalCount)) * Math.PI * 2 - Math.PI * 0.5;
        const burstNodeX = centerX - 95 + Math.cos(angle) * burstRadius;
        const burstNodeY =
          centerY - 48 + Math.sin(angle) * (burstRadius * 0.78);
        const emitStart = emitWindowStart + idx * emitGap;
        const emitEnd = Math.min(0.34, emitStart + 0.14);
        nodeBurstMetaById.set(cn.id, {
          emitStart,
          emitEnd,
          burstNodeX,
          burstNodeY,
        });
      });

      const parentCoreMorphSource: CoreMorphSource = {
        archetypeId: archetype.id,
        neuronName: clickedNode.data.neuronDef.name,
        primary: archetype.primary,
        secondary: archetype.secondary,
      };

      const cloneDetachedIds = new Set<string>();
      const shellGrownIds = new Set<string>();
      let initialBurstFired = false;

      const nextSegment: GraphPathSegment = clickedNode.data.neuronDef.variant_ref
        ? {
            kind: "variant",
            family: clickedNode.data.neuronDef.variant_ref.family,
            version: clickedNode.data.neuronDef.variant_ref.version,
          }
        : { kind: "node", nodeId: clickedNode.id };
      const nextPath = [...currentPathRef.current, nextSegment];
      forwardAnimStartedForPathRef.current = serializePathKey(nextPath);

      // Advance the graph store path into the clicked subgraph
      openSubgraph(clickedNode.id);

      // Snappy 500ms duration for the radial burst, 3D vector wireframe metamorphosis, and outer-edge shell growth
      const DURATION_MS = 500;
      const startTime = performance.now();
      let portalPulse = 0;

      const step = (now: number) => {
        const elapsed = now - startTime;
        const p = Math.min(1, Math.max(0, elapsed / DURATION_MS));

        // Phase 0 (p = 0.0 -> 0.12): Detach node title and smoothly frame the subgraph canvas
        const zoomU = Math.min(1, p / 0.12);
        const camEase = 1 - Math.pow(1 - zoomU, 3);

        const currVp: Viewport = {
          x: startVpX + (endVpX - startVpX) * camEase,
          y: startVpY + (endVpY - startVpY) * camEase,
          zoom: startZoom + (endZoom - startZoom) * camEase,
        };
        try {
          setViewport(currVp, { duration: 0 });
          onViewportSync?.(currVp);
        } catch {
          // Ignore if React Flow instance is mid-mount
        }

        // Screen coordinates of the parent node's central animated 3D core
        const singularityX = centerX * currVp.zoom + currVp.x;
        const singularityY = centerY * currVp.zoom + currVp.y;

        portalPulse = Math.max(0, portalPulse * 0.9);

        if (!initialBurstFired && p >= emitWindowStart) {
          initialBurstFired = true;
          portalPulse = 1.0;
          onSpawnBurst?.(
            centerX,
            centerY,
            [...archetype.primaryRgb] as [number, number, number],
            65,
            220,
          );
        }

        const animatedNodes: Node<FlowNodeData>[] = [];
        let spawnedCount = 0;

        const startNodeX = centerX - 95;
        const startNodeY = centerY - 48;

        for (const cn of sortedChildNodes) {
          const meta = nodeBurstMetaById.get(cn.id) ?? {
            emitStart: 0.08,
            emitEnd: 0.24,
            burstNodeX: startNodeX,
            burstNodeY: startNodeY,
          };

          if (p < meta.emitStart) {
            // Not pulled out of the parent core yet — keep hidden at center
            animatedNodes.push({
              ...cn,
              position: { x: startNodeX, y: startNodeY },
              style: {
                ...cn.style,
                opacity: 0,
                pointerEvents: "none",
              },
              data: {
                ...cn.data,
                spawnScale: 1,
                coreMorphProgress: 0,
                coreMorphFrom: parentCoreMorphSource,
                shellExpandProgress: 0,
              },
            });
            continue;
          }
          spawnedCount++;

          const cnArch = resolveNodeArchetype(
            cn.data.label,
            cn.data.neuronDef,
            cn.data.isInput,
            cn.data.isOutput,
          );

          if (!cloneDetachedIds.has(cn.id)) {
            cloneDetachedIds.add(cn.id);
            portalPulse = 1.0;
          }

          let animX: number;
          let animY: number;
          let coreMorphProgress = 0;
          let shellExpandProgress = 0;
          let nodeOpacity = 1;

          if (p < 0.34) {
            // Stage 1: Burst emission — pull each cloned 3D core out from the parent center
            // into its radial ring position all around the parent node.
            // Keep `coreMorphProgress = 0` so all emitted cores around the parent still display
            // the parent's rotating 3D vector wireframe before they all transition together.
            const emitU = Math.min(
              1,
              (p - meta.emitStart) /
                Math.max(0.01, meta.emitEnd - meta.emitStart),
            );
            const emitEase = 1 - Math.pow(1 - emitU, 3);
            // Gentle orbital float while waiting in the halo ring around the parent
            const hoverPhase = (p - meta.emitEnd) * 14 + spawnedCount * 0.9;
            const hoverX = emitU >= 1 ? Math.cos(hoverPhase) * 4 : 0;
            const hoverY = emitU >= 1 ? Math.sin(hoverPhase) * 4 : 0;

            animX =
              startNodeX +
              (meta.burstNodeX - startNodeX) * emitEase +
              hoverX;
            animY =
              startNodeY +
              (meta.burstNodeY - startNodeY) * emitEase +
              hoverY;
            coreMorphProgress = 0;
            shellExpandProgress = 0;
            nodeOpacity = Math.min(1, emitU * 3.5);
          } else if (p < 0.82) {
            // Stage 2: Slow simultaneous transition & 3D vector wireframe metamorphosis (~2.3 seconds).
            // All emitted cores move from their radial burst ring around the parent into their target
            // graph positions at once while `coreMorphProgress` slowly sweeps 0 -> 1 so the user can
            // clearly watch every 3D vector wireframe animation transforming into its new shape!
            const morphU = Math.min(1, (p - 0.34) / 0.48);
            const moveEase =
              morphU < 0.5
                ? 4 * morphU * morphU * morphU
                : 1 - Math.pow(-2 * morphU + 2, 3) / 2;

            animX =
              meta.burstNodeX +
              (cn.position.x - meta.burstNodeX) * moveEase;
            animY =
              meta.burstNodeY +
              (cn.position.y - meta.burstNodeY) * moveEase;
            coreMorphProgress = morphU;
            shellExpandProgress = 0;
            nodeOpacity = 1;
          } else {
            // Stage 3: Outer-edge box & label growth (p = 0.82 -> 1.00).
            // Now that the 3D vector wireframes have finished changing at their target locations,
            // the outer box, ports, and labels grow outward from the outer edges of each 3D core!
            animX = cn.position.x;
            animY = cn.position.y;
            coreMorphProgress = 1;
            shellExpandProgress = Math.min(1, (p - 0.82) / 0.18);
            nodeOpacity = 1;

            if (shellExpandProgress >= 0.12 && !shellGrownIds.has(cn.id)) {
              shellGrownIds.add(cn.id);
              onSpawnBurst?.(
                cn.position.x + 95,
                cn.position.y + 48,
                [...cnArch.primaryRgb] as [number, number, number],
                16,
                105,
              );
            }
          }

          animatedNodes.push({
            ...cn,
            position: { x: animX, y: animY },
            style: {
              ...cn.style,
              opacity: nodeOpacity,
            },
            data: {
              ...cn.data,
              spawnScale: 1,
              coreMorphProgress,
              coreMorphFrom: parentCoreMorphSource,
              shellExpandProgress,
            },
          });
        }

        // Ignite synaptic edges once the outer boxes & ports have grown outward from the cores
        const animatedEdges: RFEdge[] = [];
        if (p >= 0.88) {
          const edgeU = Math.min(1, (p - 0.88) / 0.12);
          for (const edge of rawChildEdges) {
            animatedEdges.push({
              ...edge,
              style: {
                ...edge.style,
                opacity: edgeU,
              },
            });
          }
        }

        setExpansionState({
          mode: "expand",
          nodeId: clickedNode.id,
          neuronName: clickedNode.data.neuronDef.name,
          archetypeId: archetype.id,
          label: clickedNode.data.label,
          badgeText: archetype.badgeText,
          primaryColor: archetype.primary,
          secondaryColor: archetype.secondary,
          primaryRgb: [...archetype.primaryRgb] as [number, number, number],
          singularityX,
          singularityY,
          portalPulse,
          titleAnchorX: startTitleX,
          titleAnchorY: startTitleY,
          progress: p,
          spawnedCount,
          totalCount,
          siblingGhosts,
          collapsingChildren: [],
          collapsingEdges: [],
          emergingTethers: [],
          animatedNodes,
          animatedEdges,
        });

        if (p < 1) {
          rafRef.current = window.requestAnimationFrame(step);
        } else {
          rafRef.current = null;
          setExpansionState(null);
        }
      };

      rafRef.current = window.requestAnimationFrame(step);
      return true;
    },
    [
      containerRef,
      onSpawnBurst,
      onViewportSync,
      openSubgraph,
      safeGetViewport,
      setViewport,
    ],
  );

  // Reverse animation on breadcrumb click:
  // 1) Each child node's labels and outer box shrink inward into the outer edges of its central animated 3D core.
  // 2) The bare animated 3D core moves back toward the central parent core while morphing back into the parent's animated 3D core.
  // 3) As the camera zooms out to the parent graph, the parent node's box and labels grow outward from the outer edges of its central animated 3D core.
  const startSubgraphCollapse = useCallback(
    (
      departingSnapshot: GraphViewSnapshot,
      targetPath: GraphPathSegment[],
      parentNodesInput: Node<FlowNodeData>[],
      parentEdgesInput: RFEdge[],
    ) => {
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }

      const container = containerRef.current;
      const rect = container?.getBoundingClientRect();
      const canvasW = Math.max(480, rect?.width ?? 960);
      const canvasH = Math.max(360, rect?.height ?? 640);

      // 1. Identify which subgraph node in the destination parent graph we are collapsing back into
      const targetDepth = targetPath.length;
      const stepSegment = departingSnapshot.path[targetDepth];
      const rememberedNodeId = enteredNodeByDepthRef.current.get(targetDepth);

      let targetParentNode: Node<FlowNodeData> | undefined;
      if (rememberedNodeId) {
        targetParentNode = parentNodesInput.find(
          (n) =>
            n.id === rememberedNodeId && n.data.neuronDef.kind === "subgraph",
        );
      }
      if (!targetParentNode && stepSegment) {
        if (stepSegment.kind === "node") {
          targetParentNode = parentNodesInput.find(
            (n) => n.id === stepSegment.nodeId,
          );
        } else if (stepSegment.kind === "variant") {
          targetParentNode =
            parentNodesInput.find(
              (n) =>
                n.data.neuronDef.variant_ref?.family === stepSegment.family &&
                n.data.neuronDef.variant_ref?.version === stepSegment.version,
            ) ??
            parentNodesInput.find(
              (n) =>
                n.data.neuronDef.variant_ref?.family === stepSegment.family,
            );
        }
      }
      if (!targetParentNode) {
        targetParentNode =
          parentNodesInput.find((n) => n.data.neuronDef.kind === "subgraph") ??
          parentNodesInput[0];
      }

      const archetype = targetParentNode
        ? resolveNodeArchetype(
            targetParentNode.data.label,
            targetParentNode.data.neuronDef,
            targetParentNode.data.isInput,
            targetParentNode.data.isOutput,
          )
        : resolveNodeArchetype(
            "subgraph",
            {
              id: "subgraph",
              name: "subgraph",
              kind: "subgraph",
              input_ports: [],
              output_ports: [],
              source_code: "",
              subgraph: null,
              module_type: "",
              module_config: {},
              module_state: "",
              input_aliases: [],
              output_aliases: [],
              variant_ref: null,
            },
            false,
            false,
          );

      // 2. Compute parent graph bounding box and destination fitted viewport (endVp)
      let pMinX = Infinity;
      let pMaxX = -Infinity;
      let pMinY = Infinity;
      let pMaxY = -Infinity;
      for (const pn of parentNodesInput) {
        pMinX = Math.min(pMinX, pn.position.x);
        pMaxX = Math.max(pMaxX, pn.position.x + 225);
        pMinY = Math.min(pMinY, pn.position.y);
        pMaxY = Math.max(pMaxY, pn.position.y + 115);
      }
      if (!Number.isFinite(pMinX)) {
        pMinX = 0;
        pMaxX = 400;
        pMinY = 0;
        pMaxY = 260;
      }

      const parentCenterX = (pMinX + pMaxX) * 0.5;
      const parentCenterY = (pMinY + pMaxY) * 0.5;
      const parentSpanX = Math.max(380, pMaxX - pMinX);
      const parentSpanY = Math.max(250, pMaxY - pMinY);

      const endZoom = Math.min(
        1.08,
        Math.max(
          0.22,
          Math.min(
            (canvasW * 0.72) / parentSpanX,
            (canvasH * 0.68) / parentSpanY,
          ),
        ),
      );
      const endVpX = canvasW * 0.5 - parentCenterX * endZoom;
      const endVpY = canvasH * 0.52 - parentCenterY * endZoom;

      const nodeW =
        targetParentNode?.measured?.width ?? archetype.minWidth + 12;
      const nodeH = targetParentNode?.measured?.height ?? 112;
      const targetNodePos = targetParentNode?.position ?? {
        x: parentCenterX - nodeW * 0.5,
        y: parentCenterY - nodeH * 0.5,
      };
      const targetNodeCoreX = targetNodePos.x + nodeW * 0.5;
      const targetNodeCoreY = targetNodePos.y + nodeH * 0.54;

      const startZoom = Math.min(2.3, Math.max(endZoom * 2.1, 1.45));
      const startVpX = canvasW * 0.5 - targetNodeCoreX * startZoom;
      const startVpY = canvasH * 0.52 - targetNodeCoreY * startZoom;

      // 3. Snapshot the departing child components and schedule their reverse box-retract + core-morph-back
      const childVp = departingSnapshot.viewport;
      const reverseSortedChildren = [...departingSnapshot.nodes].sort((a, b) =>
        Math.abs(b.position.x - a.position.x) > 14
          ? b.position.x - a.position.x
          : b.position.y - a.position.y,
      );

      const totalCount = reverseSortedChildren.length;
      const burstRadiusScreen = Math.min(185, Math.max(115, 90 + totalCount * 6));

      const despawnByChildId = new Map<string, number>();
      const collapsingChildren: CollapsingChildGhost[] =
        reverseSortedChildren.map((cn, idx) => {
          const cnArch = resolveNodeArchetype(
            cn.data.label,
            cn.data.neuronDef,
            cn.data.isInput,
            cn.data.isOutput,
          );
          const cw = Math.max(
            120,
            (cn.measured?.width ?? cnArch.minWidth) * childVp.zoom,
          );
          const ch = Math.max(76, (cn.measured?.height ?? 98) * childVp.zoom);
          const cx = cn.position.x * childVp.zoom + childVp.x;
          const cy = cn.position.y * childVp.zoom + childVp.y;
          const centerChildX = cx + cw * 0.5;
          const centerChildY = cy + ch * 0.5;

          const angle =
            (idx / Math.max(1, totalCount)) * Math.PI * 2 - Math.PI * 0.5;
          const burstOffsetX = Math.cos(angle) * burstRadiusScreen;
          const burstOffsetY = Math.sin(angle) * (burstRadiusScreen * 0.78);

          const stagger =
            totalCount <= 1 ? 0 : (idx / Math.max(1, totalCount - 1)) * 0.08;
          const recallStartAt = 0.02 + stagger * 0.5;
          const recallEndAt = 0.76 - stagger * 0.4;
          despawnByChildId.set(cn.id, 0.18);

          return {
            id: cn.id,
            archetypeId: cnArch.id,
            neuronName: cn.data.neuronDef.name,
            clipPath: cnArch.clipPath,
            label: cn.data.label,
            badgeText: cnArch.badgeText,
            color: cnArch.primary,
            secondaryColor: cnArch.secondary,
            startRect: { x: cx, y: cy, width: cw, height: ch },
            startCenterX: centerChildX,
            startCenterY: centerChildY,
            burstOffsetX,
            burstOffsetY,
            recallStartAt,
            recallEndAt,
          };
        });

      const collapsingEdges: CollapsingEdgeGhost[] = [];
      for (const edge of departingSnapshot.edges) {
        const sDespawn = despawnByChildId.get(edge.source) ?? 0.18;
        const tDespawn = despawnByChildId.get(edge.target) ?? 0.18;
        const srcGhost = collapsingChildren.find((c) => c.id === edge.source);
        collapsingEdges.push({
          id: edge.id,
          sourceId: edge.source,
          targetId: edge.target,
          color: srcGhost?.color ?? archetype.primary,
          despawnAt: Math.min(sDespawn, tDespawn),
        });
      }

      const sortedParentNodes = [...parentNodesInput].sort((a, b) =>
        Math.abs(a.position.x - b.position.x) > 14
          ? a.position.x - b.position.x
          : a.position.y - b.position.y,
      );

      let implosionBurstFired = false;

      const DURATION_MS = 500;
      const startTime = performance.now();
      const absorbedChildIds = new Set<string>();
      let portalPulse = 0;

      const step = (now: number) => {
        const elapsed = now - startTime;
        const p = Math.min(1, Math.max(0, elapsed / DURATION_MS));

        // Keep camera steady while child nodes retract their outer boxes/labels into their 3D cores,
        // morph back into a ring around the parent core, and pull into the parent core (p = 0.0 -> 0.74),
        // then zoom out to the parent graph!
        const zoomU = p < 0.74 ? 0 : Math.min(1, (p - 0.74) / 0.26);
        const camEase = 1 - Math.pow(1 - zoomU, 3);

        const currVp: Viewport = {
          x: startVpX + (endVpX - startVpX) * camEase,
          y: startVpY + (endVpY - startVpY) * camEase,
          zoom: startZoom + (endZoom - startZoom) * camEase,
        };
        try {
          setViewport(currVp, { duration: 0 });
          onViewportSync?.(currVp);
        } catch {
          // Ignore if React Flow instance is mid-mount
        }

        const singularityX = targetNodeCoreX * currVp.zoom + currVp.x;
        const singularityY = targetNodeCoreY * currVp.zoom + currVp.y;
        const titleAnchorX = singularityX;
        const titleAnchorY =
          targetNodePos.y * currVp.zoom + currVp.y + 16 * currVp.zoom;

        portalPulse = Math.max(0, portalPulse * 0.88);

        let remainingChildCount = 0;
        for (const child of collapsingChildren) {
          if (p < child.recallEndAt) {
            remainingChildCount++;
          } else if (!absorbedChildIds.has(child.id)) {
            absorbedChildIds.add(child.id);
            portalPulse = 1.0;
          }
        }

        if (!implosionBurstFired && p >= 0.75) {
          implosionBurstFired = true;
          onSpawnBurst?.(
            targetNodeCoreX,
            targetNodeCoreY,
            [...archetype.primaryRgb] as [number, number, number],
            55,
            220,
          );
        }

        // Reveal parent graph nodes: the target parent subgraph node starts as its bare central 3D core
        // and grows its outer box & labels outward from the outer edges of its central 3D core!
        const animatedNodes: Node<FlowNodeData>[] = [];
        for (const pn of sortedParentNodes) {
          const isTargetSubgraphNode =
            targetParentNode && pn.id === targetParentNode.id;

          if (isTargetSubgraphNode) {
            const coreVisible = p >= 0.74 ? 1 : 0;
            const shellGrowU =
              p < 0.78 ? 0 : Math.min(1, (p - 0.78) / 0.22);

            animatedNodes.push({
              ...pn,
              position: pn.position,
              style: {
                ...pn.style,
                opacity: coreVisible,
              },
              data: {
                ...pn.data,
                spawnScale: 1,
                shellExpandProgress: shellGrowU,
                coreMorphProgress: 1,
              },
            });
          } else {
            const sibU = p < 0.78 ? 0 : Math.min(1, (p - 0.78) / 0.22);
            const sibEase = 1 - Math.pow(1 - sibU, 2.5);

            animatedNodes.push({
              ...pn,
              position: pn.position,
              style: {
                ...pn.style,
                opacity: sibEase,
              },
              data: {
                ...pn.data,
                spawnScale: 1,
                shellExpandProgress: sibEase,
                coreMorphProgress: 1,
              },
            });
          }
        }

        const animatedEdges: RFEdge[] = [];
        if (p >= 0.84) {
          const edgeU = Math.min(1, (p - 0.84) / 0.16);
          for (const edge of parentEdgesInput) {
            animatedEdges.push({
              ...edge,
              style: {
                ...edge.style,
                opacity: edgeU,
              },
            });
          }
        }

        setExpansionState({
          mode: "collapse",
          nodeId: targetParentNode?.id ?? "subgraph",
          neuronName: targetParentNode?.data.neuronDef.name ?? "subgraph",
          archetypeId: archetype.id,
          label: targetParentNode?.data.label ?? "Subgraph",
          badgeText: archetype.badgeText,
          primaryColor: archetype.primary,
          secondaryColor: archetype.secondary,
          primaryRgb: [...archetype.primaryRgb] as [number, number, number],
          singularityX,
          singularityY,
          portalPulse,
          titleAnchorX,
          titleAnchorY,
          progress: p,
          spawnedCount: remainingChildCount,
          totalCount,
          siblingGhosts: [],
          collapsingChildren,
          collapsingEdges,
          emergingTethers: [],
          animatedNodes,
          animatedEdges,
        });

        if (p < 1) {
          rafRef.current = window.requestAnimationFrame(step);
        } else {
          rafRef.current = null;
          setExpansionState(null);
        }
      };

      step(startTime);
    },
    [
      containerRef,
      onSpawnBurst,
      onViewportSync,
      setViewport,
    ],
  );

  // Detect breadcrumb jumps back to a parent/ancestor path synchronously before paint
  useLayoutEffect(() => {
    const prevSnapshot = lastSnapshotRef.current;
    const prevKey = serializePathKey(prevSnapshot.path);
    const currKey = serializePathKey(currentPath);

    if (prevKey === currKey) {
      if (!expansionState) {
        lastSnapshotRef.current = {
          path: currentPath,
          nodes: activeNodes,
          edges: activeEdges,
          viewport: safeGetViewport(),
        };
      }
      return;
    }

    if (
      !viewFullGraph &&
      isStrictPrefixPath(currentPath, prevSnapshot.path) &&
      prevSnapshot.nodes.length > 0
    ) {
      const targetGraph = rootGraph
        ? getGraphAtPath(rootGraph, currentPath)
        : null;
      const parentNodes = targetGraph
        ? graphToFlowNodes(targetGraph)
        : activeNodes;
      const parentEdges = targetGraph
        ? graphToFlowEdges(targetGraph)
        : activeEdges;

      startSubgraphCollapse(prevSnapshot, currentPath, parentNodes, parentEdges);
    }

    lastSnapshotRef.current = {
      path: currentPath,
      nodes: activeNodes,
      edges: activeEdges,
      viewport: safeGetViewport(),
    };
    if (forwardAnimStartedForPathRef.current === currKey) {
      forwardAnimStartedForPathRef.current = null;
    }
  }, [
    currentPath,
    activeNodes,
    activeEdges,
    rootGraph,
    viewFullGraph,
    expansionState,
    safeGetViewport,
    startSubgraphCollapse,
  ]);

  return {
    expansionState,
    isExpanding: expansionState !== null,
    expansionMode: expansionState?.mode ?? null,
    triggerSubgraphExpansion,
  };
}

export function SubgraphExpansionOverlay({
  expansionState,
}: {
  expansionState: SubgraphExpansionAnimState | null;
  cyberMode?: boolean;
}) {
  if (!expansionState) {
    return null;
  }

  const {
    mode,
    neuronName,
    archetypeId,
    label,
    badgeText,
    primaryColor,
    secondaryColor,
    singularityX,
    singularityY,
    titleAnchorX,
    titleAnchorY,
    progress,
    portalPulse,
    siblingGhosts,
    collapsingChildren,
    collapsingEdges,
  } = expansionState;

  const isCollapse = mode === "collapse";

  // The parent node's central animated 3D core stays visible at the center while clones detach from it
  // (or while returning child cores merge back into it), then hands off cleanly to the graph nodes.
  const singularityOpacity = isCollapse
    ? progress < 0.76
      ? 1
      : Math.max(0, 1 - (progress - 0.76) / 0.14)
    : progress < 0.42
    ? 1
    : Math.max(0, (0.62 - progress) / 0.2);

  const ghostOpacity =
    !isCollapse && progress < 0.18 ? Math.max(0, 1 - progress / 0.18) : 0;

  const parentCoreMorphSource: CoreMorphSource = {
    archetypeId,
    neuronName,
    primary: primaryColor,
    secondary: secondaryColor,
  };

  // Compute live screen state of each child node during reverse breadcrumb collapse:
  // 1) Stage 1 (p: 0.00 -> 0.20): Outer box and labels shrink inward into the outer edges of its central animated 3D core (`shellU: 1 -> 0`).
  // 2) Stage 2 (p: 0.20 -> 0.64): All bare animated 3D cores transition into the ring around the parent while slowly morphing their 3D vector wireframes back into the parent's 3D wireframe (`morphProgress: 1 -> 0`).
  // 3) Stage 3 (p: 0.64 -> 0.78): The ring of parent-shaped 3D cores pulls into the central parent core (`(singularityX, singularityY)`).
  const liveChildPositions = new Map<
    string,
    {
      cx: number;
      cy: number;
      w: number;
      h: number;
      shellU: number;
      morphProgress: number;
      opacity: number;
    }
  >();

  if (isCollapse) {
    for (const child of collapsingChildren) {
      if (progress >= child.recallEndAt) {
        continue;
      }

      const ringX = singularityX + child.burstOffsetX;
      const ringY = singularityY + child.burstOffsetY;

      if (progress < 0.2) {
        // Stage 1: Retract outer box and labels into the central 3D core
        const shrinkU = Math.max(
          0,
          Math.min(1, (progress - child.recallStartAt) / 0.16),
        );
        const shellU = Math.max(0, 1 - shrinkU);
        liveChildPositions.set(child.id, {
          cx: child.startCenterX,
          cy: child.startCenterY,
          w: child.startRect.width,
          h: child.startRect.height,
          shellU,
          morphProgress: 1,
          opacity: 1,
        });
      } else if (progress < 0.64) {
        // Stage 2: Slowly transition all bare 3D cores into the radial ring around the parent
        // while morphing their 3D vector wireframes back into the parent's 3D wireframe (1 -> 0)
        const morphBackU = Math.min(1, (progress - 0.2) / 0.44);
        const moveEase =
          morphBackU < 0.5
            ? 4 * morphBackU * morphBackU * morphBackU
            : 1 - Math.pow(-2 * morphBackU + 2, 3) / 2;
        const cx =
          child.startCenterX + (ringX - child.startCenterX) * moveEase;
        const cy =
          child.startCenterY + (ringY - child.startCenterY) * moveEase;
        const morphProgress = Math.max(0, 1 - morphBackU);

        liveChildPositions.set(child.id, {
          cx,
          cy,
          w: child.startRect.width,
          h: child.startRect.height,
          shellU: 0,
          morphProgress,
          opacity: 1,
        });
      } else {
        // Stage 3: Pull the ring of parent-shaped 3D cores into the central parent core
        const pullSpan = Math.max(0.04, child.recallEndAt - 0.64);
        const pullU = Math.min(1, Math.max(0, (progress - 0.64) / pullSpan));
        const pullEase = pullU * pullU * (3 - 2 * pullU);
        const cx = ringX + (singularityX - ringX) * pullEase;
        const cy = ringY + (singularityY - ringY) * pullEase;
        const opacity = pullU < 0.78 ? 1 : Math.max(0, (1 - pullU) / 0.22);

        liveChildPositions.set(child.id, {
          cx,
          cy,
          w: child.startRect.width,
          h: child.startRect.height,
          shellU: 0,
          morphProgress: 0,
          opacity,
        });
      }
    }
  }

  // Detached Title / Name Animation at start of expand
  let titleVisible = false;
  let titleX = titleAnchorX;
  let titleY = titleAnchorY;
  let titleScale = 1;
  let titleOpacity = 0;

  if (!isCollapse && progress < 0.32) {
    titleVisible = true;
    const detachU = Math.min(1, progress / 0.32);
    const liftEase = 1 - Math.pow(1 - detachU, 2.6);
    titleX = titleAnchorX;
    titleY = titleAnchorY - liftEase * 72;
    titleScale = 1 + Math.sin(detachU * Math.PI) * 0.1;
    titleOpacity = detachU < 0.5 ? 1 : Math.max(0, (1 - detachU) / 0.5);
  }

  return (
    <div className="pointer-events-none absolute inset-0 z-20 overflow-hidden select-none">
      {/* 1. Receding Parent Sibling Nodes during initial camera lock */}
      {ghostOpacity > 0.01 &&
        siblingGhosts.map((ghost) => {
          const ease = 1 - Math.pow(1 - Math.min(1, progress * 3.5), 2.5);
          const gx = ghost.rect.x + ghost.dx * ease;
          const gy = ghost.rect.y + ghost.dy * ease;
          const scale = Math.max(0.45, 1 - ease * 0.45);
          return (
            <div
              key={ghost.id}
              className="absolute rounded-lg border px-3 py-2 font-mono text-[10px] font-bold uppercase tracking-wider"
              style={{
                left: gx,
                top: gy,
                width: ghost.rect.width,
                height: ghost.rect.height,
                opacity: ghostOpacity * 0.5,
                transform: `scale(${scale})`,
                transformOrigin: "center center",
                borderColor: `${ghost.color}66`,
                backgroundColor: "rgba(2, 8, 22, 0.65)",
                color: ghost.color,
                boxShadow: `0 0 16px ${ghost.color}22`,
              }}
            >
              <div className="truncate">{ghost.label}</div>
            </div>
          );
        })}

      {/* 2. Synaptic edges between remaining child nodes during reverse collapse */}
      {isCollapse && (
        <svg className="pointer-events-none absolute inset-0 h-full w-full">
          {collapsingEdges.map((edge) => {
            if (progress >= edge.despawnAt) return null;
            const src = liveChildPositions.get(edge.sourceId);
            const dst = liveChildPositions.get(edge.targetId);
            if (!src || !dst) return null;
            const edgeOpacity =
              Math.min(src.shellU, dst.shellU) * 0.65;
            if (edgeOpacity <= 0.02) return null;
            const d = `M ${src.cx} ${src.cy} L ${dst.cx} ${dst.cy}`;
            return (
              <path
                key={edge.id}
                d={d}
                fill="none"
                stroke={edge.color}
                strokeWidth="1.5"
                strokeOpacity={edgeOpacity}
              />
            );
          })}
        </svg>
      )}

      {/* 3. Reverse Collapse: Child Nodes retracting their outer box & labels into their 3D Core, then morphing & moving back into the Parent Core */}
      {isCollapse &&
        collapsingChildren.map((child) => {
          const pos = liveChildPositions.get(child.id);
          if (!pos || pos.opacity <= 0.01) return null;

          const boxEase = 1 - Math.pow(1 - pos.shellU, 2.4);
          const boxScaleX = 0.32 + 0.68 * boxEase;
          const boxScaleY = 0.38 + 0.62 * boxEase;
          const boxOpacity =
            pos.shellU <= 0.01 ? 0 : Math.min(1, pos.shellU * 1.6);
          const labelEase = Math.max(0, Math.min(1, (pos.shellU - 0.1) / 0.9));

          return (
            <div
              key={child.id}
              className="absolute flex items-center justify-center font-mono"
              style={{
                left: pos.cx - pos.w * 0.5,
                top: pos.cy - pos.h * 0.5,
                width: pos.w,
                height: pos.h,
                opacity: pos.opacity,
              }}
            >
              {/* Outer Shaped Box / Armor Shell (shrinks inward into the outer edges of the central 3D core) */}
              {boxOpacity > 0.01 && (
                <div
                  className="pointer-events-none absolute inset-0 p-[1.5px]"
                  style={{
                    clipPath: child.clipPath,
                    transformOrigin: "center center",
                    transform: `scaleX(${boxScaleX}) scaleY(${boxScaleY})`,
                    opacity: boxOpacity,
                    background: `linear-gradient(135deg, ${child.color}99, rgba(15, 23, 42, 0.92), ${child.secondaryColor}77)`,
                  }}
                >
                  <div
                    className="h-full w-full"
                    style={{
                      clipPath: child.clipPath,
                      background:
                        "linear-gradient(160deg, rgba(4, 11, 26, 0.92) 0%, rgba(2, 6, 16, 0.96) 70%)",
                    }}
                  />
                </div>
              )}

              {/* Top Label Bar (retracts downward into the top outer edge of the central 3D core) */}
              {labelEase > 0.02 && (
                <div
                  className="pointer-events-none absolute top-2 left-3 right-3 flex items-center justify-between gap-1"
                  style={{
                    transformOrigin: "center bottom",
                    transform: `translateY(${
                      (1 - labelEase) * 20
                    }px) scaleX(${0.5 + 0.5 * labelEase}) scaleY(${labelEase})`,
                    opacity: labelEase,
                  }}
                >
                  <span
                    className="truncate text-[10px] font-bold uppercase"
                    style={{ color: "#f8fafc" }}
                  >
                    {child.label}
                  </span>
                  <span
                    className="shrink-0 rounded border px-1 py-0.2 text-[7px] uppercase"
                    style={{
                      borderColor: `${child.secondaryColor}66`,
                      color: child.secondaryColor,
                    }}
                  >
                    {child.badgeText}
                  </span>
                </div>
              )}

              {/* Central Animated 3D Core (`SemanticNodeGlyph`) — morphs back into the parent's animated 3D core as it moves into the center */}
              <div className="relative z-10 flex items-center justify-center">
                <SemanticNodeGlyph
                  archetypeId={child.archetypeId}
                  neuronName={child.neuronName}
                  primary={child.color}
                  secondary={child.secondaryColor}
                  size={54}
                  selected={true}
                  morphFrom={parentCoreMorphSource}
                  morphProgress={pos.morphProgress}
                />
              </div>
            </div>
          );
        })}

      {/* 4. Central Source Animated 3D Core of the Parent Subgraph Node (from which clones detach and move out) */}
      {singularityOpacity > 0.01 && (
        <div
          className="absolute flex items-center justify-center"
          style={{
            left: singularityX,
            top: singularityY,
            transform: `translate(-50%, -50%) scale(${
              1 + portalPulse * 0.14
            })`,
            opacity: singularityOpacity,
          }}
        >
          <SemanticNodeGlyph
            archetypeId={archetypeId}
            neuronName={neuronName}
            primary={primaryColor}
            secondary={secondaryColor}
            size={56}
            selected={true}
          />
        </div>
      )}

      {/* 5. Detaching Node Title & Name Plate */}
      {titleVisible && titleOpacity > 0.01 && (
        <div
          className="absolute flex flex-col items-center"
          style={{
            left: titleX,
            top: titleY,
            transform: `translate(-50%, -50%) scale(${titleScale})`,
            opacity: titleOpacity,
          }}
        >
          <div
            className="flex items-center gap-2 whitespace-nowrap rounded-full border px-3.5 py-1 font-mono backdrop-blur-md"
            style={{
              borderColor: primaryColor,
              backgroundColor: "rgba(2, 8, 22, 0.94)",
              boxShadow: `0 8px 24px rgba(0,0,0,0.85), 0 0 20px ${primaryColor}66`,
            }}
          >
            <span
              className="h-2 w-2 rounded-full animate-ping"
              style={{ backgroundColor: primaryColor }}
            />
            <span
              className="text-xs font-extrabold uppercase tracking-wider"
              style={{
                color: "#f8fafc",
                textShadow: `0 0 10px ${primaryColor}`,
              }}
            >
              {label}
            </span>
            <span
              className="rounded border px-1.5 py-0.2 text-[8px] uppercase"
              style={{
                borderColor: `${secondaryColor}66`,
                color: secondaryColor,
              }}
            >
              {badgeText}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

const ZONE_DEPTH_STYLES = [
  {
    border: "#f59e0b",
    bg: "rgba(245, 158, 11, 0.045)",
    badgeBg: "rgba(69, 26, 3, 0.88)",
    text: "#fde68a",
  },
  {
    border: "#00f0ff",
    bg: "rgba(0, 240, 255, 0.045)",
    badgeBg: "rgba(4, 47, 73, 0.88)",
    text: "#a5f3fc",
  },
  {
    border: "#a855f7",
    bg: "rgba(168, 85, 247, 0.05)",
    badgeBg: "rgba(59, 7, 100, 0.88)",
    text: "#e9d5ff",
  },
  {
    border: "#10b981",
    bg: "rgba(16, 185, 129, 0.05)",
    badgeBg: "rgba(6, 78, 59, 0.88)",
    text: "#a7f3d0",
  },
];

export function FullGraphZonesLayer({
  zones,
  viewport,
  cyberMode = true,
  onIsolateZone,
  onFitZone,
}: {
  zones: SubgraphZone[];
  viewport: Viewport;
  cyberMode?: boolean;
  onIsolateZone: (path: GraphPathSegment[]) => void;
  onFitZone: (zone: SubgraphZone) => void;
}) {
  if (zones.length === 0) {
    return null;
  }

  return (
    <div
      className="pointer-events-none absolute inset-0 overflow-hidden select-none"
      style={{ zIndex: 2 }}
    >
      <div
        style={{
          transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`,
          transformOrigin: "0 0",
        }}
      >
        {zones.map((zone) => {
          const palette =
            ZONE_DEPTH_STYLES[zone.depth % ZONE_DEPTH_STYLES.length];
          return (
            <div
              key={zone.id}
              className="absolute rounded-xl transition-colors"
              style={{
                left: zone.x,
                top: zone.y,
                width: zone.width,
                height: zone.height,
                border: `1.5px dashed ${palette.border}77`,
                backgroundColor: palette.bg,
                boxShadow: cyberMode
                  ? `inset 0 0 28px ${palette.border}14, 0 0 18px rgba(0,0,0,0.45)`
                  : undefined,
              }}
            >
              {/* Zone Top Header Bar */}
              <div
                className="pointer-events-auto flex items-center justify-between gap-2 rounded-t-[10px] border-b px-3 py-1 font-mono text-[10px]"
                style={{
                  borderColor: `${palette.border}44`,
                  backgroundColor: palette.badgeBg,
                  color: palette.text,
                }}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className="inline-block h-2 w-2 shrink-0 rotate-45"
                    style={{
                      backgroundColor: palette.border,
                      boxShadow: `0 0 6px ${palette.border}`,
                    }}
                  />
                  <span className="truncate font-bold uppercase tracking-wider">
                    {zone.breadcrumb}
                  </span>
                  {zone.variantLabel && (
                    <span
                      className="rounded border px-1.5 py-0.2 text-[8px] opacity-90"
                      style={{
                        borderColor: `${palette.border}66`,
                      }}
                    >
                      {zone.variantLabel}
                    </span>
                  )}
                  <span className="text-[8px] opacity-70">
                    ({zone.nodeCount} nodes)
                  </span>
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onFitZone(zone);
                    }}
                    className="rounded border px-1.5 py-0.5 text-[8px] font-bold uppercase transition-colors hover:bg-white/10"
                    style={{
                      borderColor: `${palette.border}55`,
                      color: palette.text,
                    }}
                    title={`Zoom camera to fit ${zone.label}`}
                  >
                    ⌖ Fit
                  </button>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onIsolateZone(zone.path);
                    }}
                    className="rounded border px-1.5 py-0.5 text-[8px] font-bold uppercase transition-colors hover:bg-white/15"
                    style={{
                      borderColor: `${palette.border}88`,
                      backgroundColor: `${palette.border}22`,
                      color: "#ffffff",
                    }}
                    title={`Exit Full Graph Mode and open ${zone.label}`}
                  >
                    ⤢ Isolate
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
