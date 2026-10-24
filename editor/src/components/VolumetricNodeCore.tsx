import React, { useEffect, useRef } from "react";
import type { NodeArchetypeId } from "./nodeArchetypes";

type Vec3 = [number, number, number];
type EdgePair = [number, number];
type FacePoly = {
  indices: number[];
  color?: string;
  alpha?: number;
};

interface ProjectedVert {
  x: number;
  y: number;
  z: number;
  scale: number;
}

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace("#", "").trim();
  if (clean.length === 3) {
    return [
      parseInt(clean[0] + clean[0], 16) || 0,
      parseInt(clean[1] + clean[1], 16) || 240,
      parseInt(clean[2] + clean[2], 16) || 255,
    ];
  }
  const num = parseInt(clean.slice(0, 6), 16);
  if (Number.isNaN(num)) return [0, 240, 255];
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

function rotateX(v: Vec3, a: number): Vec3 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c];
}

function rotateY(v: Vec3, a: number): Vec3 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c];
}

function rotateZ(v: Vec3, a: number): Vec3 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]];
}

function rotateXYZ(v: Vec3, rx: number, ry: number, rz: number): Vec3 {
  return rotateZ(rotateY(rotateX(v, rx), ry), rz);
}

function twistY(v: Vec3, twistStrength: number): Vec3 {
  const angle = v[1] * twistStrength;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c];
}

function project(v: Vec3, cx: number, cy: number, radius: number, fov = 3.2): ProjectedVert {
  const zCam = fov - v[2];
  const scale = fov / Math.max(0.6, zCam);
  return {
    x: cx + v[0] * radius * scale,
    y: cy + v[1] * radius * scale,
    z: v[2],
    scale,
  };
}

function computeActivationY(name: string, x: number): number {
  const n = name.toLowerCase();
  if (n.includes("relu6")) return Math.min(1, Math.max(0, x));
  if (n.includes("leaky") || n.includes("prelu")) return x >= 0 ? x : 0.22 * x;
  if (n.includes("elu") || n.includes("selu")) return x >= 0 ? x : 0.45 * (Math.exp(x) - 1);
  if (n.includes("gelu") || n.includes("silu") || n.includes("swish") || n.includes("mish")) {
    const sig = 1 / (1 + Math.exp(-1.7 * x));
    return x * sig;
  }
  if (n.includes("relu")) return Math.max(0, x);
  if (n.includes("sigmoid")) return (1 / (1 + Math.exp(-2.2 * x))) * 1.6 - 0.8;
  if (n.includes("tanh") || n.includes("softsign")) return Math.tanh(1.5 * x) * 0.85;
  if (n.includes("gaussian")) return Math.exp(-x * x * 1.8) * 1.4 - 0.5;
  if (n.includes("sin")) return Math.sin(x * 2.2) * 0.8;
  if (n.includes("cos")) return Math.cos(x * 2.2) * 0.8;
  return Math.tanh(x * 1.3) * 0.8;
}

const OCTAHEDRON_VERTS: Vec3[] = [
  [0, -1.08, 0],
  [1.0, 0, 0],
  [0, 0, 1.0],
  [-1.0, 0, 0],
  [0, 0, -1.0],
  [0, 1.08, 0],
];

const OCTAHEDRON_EDGES: EdgePair[] = [
  [0, 1], [0, 2], [0, 3], [0, 4],
  [5, 1], [5, 2], [5, 3], [5, 4],
  [1, 2], [2, 3], [3, 4], [4, 1],
];

const OCTAHEDRON_FACES: FacePoly[] = [
  { indices: [0, 1, 2] },
  { indices: [0, 2, 3] },
  { indices: [0, 3, 4] },
  { indices: [0, 4, 1] },
  { indices: [5, 2, 1] },
  { indices: [5, 3, 2] },
  { indices: [5, 4, 3] },
  { indices: [5, 1, 4] },
];

const TETRA_A_VERTS: Vec3[] = [
  [0, -1.1, 0],
  [0.95, 0.55, 0.55],
  [-0.95, 0.55, 0.55],
  [0, 0.55, -1.05],
];

const TETRA_B_VERTS: Vec3[] = [
  [0, 1.1, 0],
  [0.95, -0.55, -0.55],
  [-0.95, -0.55, -0.55],
  [0, -0.55, 1.05],
];

const TETRA_EDGES: EdgePair[] = [
  [0, 1], [0, 2], [0, 3],
  [1, 2], [2, 3], [3, 1],
];

const TETRA_FACES: FacePoly[] = [
  { indices: [0, 1, 2] },
  { indices: [0, 2, 3] },
  { indices: [0, 3, 1] },
  { indices: [1, 3, 2] },
];

// Precompute 4D hypercube (tesseract) 16 vertices & 32 edges
const TESSERACT_VERTS_4D: [number, number, number, number][] = [];
for (let i = 0; i < 16; i++) {
  TESSERACT_VERTS_4D.push([
    i & 1 ? 1 : -1,
    i & 2 ? 1 : -1,
    i & 4 ? 1 : -1,
    i & 8 ? 1 : -1,
  ]);
}
const TESSERACT_EDGES: EdgePair[] = [];
for (let i = 0; i < 16; i++) {
  for (let bit = 0; bit < 4; bit++) {
    const j = i ^ (1 << bit);
    if (i < j) TESSERACT_EDGES.push([i, j]);
  }
}

export interface VolumetricMorphSource {
  archetypeId: NodeArchetypeId;
  neuronName: string;
  primary: string;
  secondary: string;
}

interface WireframeSegment3D {
  v1: Vec3;
  v2: Vec3;
  rgb: [number, number, number];
  alpha: number;
}

export interface VolumetricNodeCoreProps {
  archetypeId: NodeArchetypeId;
  neuronName: string;
  primaryColor: string;
  secondaryColor: string;
  size?: number;
  speedMultiplier?: number;
  selected?: boolean;
  morphFrom?: VolumetricMorphSource;
  morphProgress?: number;
}

/**
 * Real-time 60 FPS 3D Volumetric Geometry & Twisting Mesh Core.
 * Renders depth-sorted 3D faces, normal-lit surfaces, counter-rotating gimbal rings,
 * torsional twisting lattices, and 4D hypercube projections.
 * Supports true 3D vector wireframe vertex-and-edge metamorphosis from `morphFrom`
 * into the target archetype as `morphProgress` sweeps from 0 to 1.
 */
export const VolumetricNodeCore: React.FC<VolumetricNodeCoreProps> = ({
  archetypeId,
  neuronName,
  primaryColor,
  secondaryColor,
  size = 64,
  speedMultiplier = 1,
  selected = false,
  morphFrom,
  morphProgress = 1,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const phaseOffsetRef = useRef<number>(
    neuronName.split("").reduce((acc, ch, i) => acc + ch.charCodeAt(0) * (i + 1) * 0.37, 0) %
      (Math.PI * 2),
  );
  const morphStateRef = useRef<{
    morphFrom?: VolumetricMorphSource;
    morphProgress: number;
  }>({ morphFrom, morphProgress });
  morphStateRef.current = { morphFrom, morphProgress };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let rafId = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);

    const targetPRgb = hexToRgb(primaryColor);
    const targetSRgb = hexToRgb(secondaryColor);

    let activeCollector: WireframeSegment3D[] | null = null;

    const drawFacesAndEdges = (
      verts3D: Vec3[],
      edges: EdgePair[],
      faces: FacePoly[],
      cx: number,
      cy: number,
      radius: number,
      rgb: [number, number, number],
      edgeAlpha = 0.85,
      faceAlphaScale = 0.24,
      drawVertices = true,
    ) => {
      if (activeCollector) {
        for (const [i, j] of edges) {
          if (verts3D[i] && verts3D[j]) {
            activeCollector.push({
              v1: verts3D[i],
              v2: verts3D[j],
              rgb,
              alpha: edgeAlpha,
            });
          }
        }
        return;
      }

      const projected = verts3D.map((v) => project(v, cx, cy, radius));

      // Depth-sort faces back-to-front (Painter's algorithm + normal lighting)
      if (faces.length > 0) {
        const sortedFaces = faces
          .map((f) => {
            const v0 = verts3D[f.indices[0]];
            const v1 = verts3D[f.indices[1]];
            const v2 = verts3D[f.indices[2]];
            const ux = v1[0] - v0[0];
            const uy = v1[1] - v0[1];
            const uz = v1[2] - v0[2];
            const vx = v2[0] - v0[0];
            const vy = v2[1] - v0[1];
            const vz = v2[2] - v0[2];
            const nx = uy * vz - uz * vy;
            const ny = uz * vx - ux * vz;
            const nz = ux * vy - uy * vx;
            const nLen = Math.hypot(nx, ny, nz) || 1;
            // Light from top-left-front (-0.35, -0.5, 0.78)
            const light = Math.max(0.15, Math.abs((nx * -0.35 + ny * -0.5 + nz * 0.78) / nLen));
            const avgZ =
              f.indices.reduce((sum, idx) => sum + verts3D[idx][2], 0) / f.indices.length;
            return { f, avgZ, light };
          })
          .sort((a, b) => a.avgZ - b.avgZ);

        for (const { f, light } of sortedFaces) {
          ctx.beginPath();
          f.indices.forEach((idx, i) => {
            const p = projected[idx];
            if (i === 0) ctx.moveTo(p.x, p.y);
            else ctx.lineTo(p.x, p.y);
          });
          ctx.closePath();
          const a = (f.alpha ?? faceAlphaScale) * (0.45 + 0.85 * light);
          ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${Math.min(0.65, a)})`;
          ctx.fill();
        }
      }

      // Depth-sorted edges
      const sortedEdges = edges
        .map(([i, j]) => ({
          i,
          j,
          z: (verts3D[i][2] + verts3D[j][2]) * 0.5,
        }))
        .sort((a, b) => a.z - b.z);

      for (const { i, j, z } of sortedEdges) {
        const p1 = projected[i];
        const p2 = projected[j];
        const depthFactor = Math.max(0.25, Math.min(1.15, (z + 1.4) / 2.2));
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${Math.min(
          1,
          edgeAlpha * depthFactor,
        )})`;
        ctx.lineWidth = (0.85 + 0.9 * depthFactor) * (size / 64);
        ctx.stroke();
      }

      if (drawVertices) {
        for (const p of projected) {
          const depthFactor = Math.max(0.3, Math.min(1.2, (p.z + 1.3) / 2.2));
          const r = (1.1 + 1.15 * depthFactor) * (size / 64);
          ctx.beginPath();
          ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
          ctx.fillStyle =
            depthFactor > 0.85
              ? `rgba(255, 255, 255, ${0.85 * depthFactor})`
              : `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${0.75 * depthFactor})`;
          ctx.fill();
        }
      }
    };

    const drawRing3D = (
      cx: number,
      cy: number,
      radius3D: number,
      rx: number,
      ry: number,
      rz: number,
      rgb: [number, number, number],
      segments = 24,
      scalePx = size * 0.36,
      dashed = false,
    ) => {
      const rotPts: Vec3[] = [];
      for (let i = 0; i <= segments; i++) {
        const a = (i / segments) * Math.PI * 2;
        const v: Vec3 = [Math.cos(a) * radius3D, Math.sin(a) * radius3D, 0];
        rotPts.push(rotateXYZ(v, rx, ry, rz));
      }
      if (activeCollector) {
        for (let i = 0; i < segments; i++) {
          if (dashed && i % 2 === 1) continue;
          activeCollector.push({
            v1: rotPts[i],
            v2: rotPts[i + 1],
            rgb,
            alpha: 0.82,
          });
        }
        return;
      }

      const pts = rotPts.map((rot) => project(rot, cx, cy, scalePx));
      for (let i = 0; i < segments; i++) {
        if (dashed && i % 2 === 1) continue;
        const p1 = pts[i];
        const p2 = pts[i + 1];
        const depth = Math.max(0.2, Math.min(1.1, ((p1.z + p2.z) * 0.5 + 1.3) / 2.2));
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${0.78 * depth})`;
        ctx.lineWidth = (0.9 + 0.8 * depth) * (size / 64);
        ctx.stroke();
      }
    };

    const emitArchetypeGeometry = (
      curArchId: NodeArchetypeId,
      curNeuronName: string,
      pRgb: [number, number, number],
      sRgb: [number, number, number],
      t: number,
      cx: number,
      cy: number,
      baseR: number,
    ) => {
      const [pr, pg, pb] = pRgb;
      const [sr, sg, sb] = sRgb;
      switch (curArchId) {
        case "attention_prism": {
          // 2 counter-rotating 3D gimbal rings + inner rotating 3D octahedron crystal + Q/K/V laser vectors
          drawRing3D(cx, cy, 1.15, t * 0.9 + 0.8, -t * 0.7, t * 0.4, [sr, sg, sb], 24, baseR, true);
          drawRing3D(cx, cy, 1.02, -t * 0.6, t * 1.1, -t * 0.5, [pr, pg, pb], 24, baseR, false);

          const crystalVerts = OCTAHEDRON_VERTS.map((v) =>
            rotateXYZ([v[0] * 0.78, v[1] * 0.92, v[2] * 0.78], t * 1.15, t * 1.55, Math.sin(t) * 0.4),
          );
          drawFacesAndEdges(
            crystalVerts,
            OCTAHEDRON_EDGES,
            OCTAHEDRON_FACES,
            cx,
            cy,
            baseR,
            [pr, pg, pb],
            0.95,
            0.28,
          );
          break;
        }

        case "sequence_wave": {
          // 3D Twisting Double-Helix Waveguide
          const strands = 14;
          const vertsA: Vec3[] = [];
          const vertsB: Vec3[] = [];
          const pitchX = Math.sin(t * 0.7) * 0.42;
          const yawY = Math.cos(t * 0.55) * 0.48;

          for (let i = 0; i < strands; i++) {
            const u = (i / (strands - 1)) * 2.2 - 1.1;
            const twist = u * 2.4 - t * 2.6;
            const r = 0.68;
            const va: Vec3 = [u, Math.sin(twist) * r, Math.cos(twist) * r];
            const vb: Vec3 = [u, Math.sin(twist + Math.PI) * r, Math.cos(twist + Math.PI) * r];
            vertsA.push(rotateXYZ(va, pitchX, yawY, 0.25));
            vertsB.push(rotateXYZ(vb, pitchX, yawY, 0.25));
          }

          const allVerts = [...vertsA, ...vertsB];
          const edges: EdgePair[] = [];
          for (let i = 0; i < strands - 1; i++) {
            edges.push([i, i + 1]);
            edges.push([strands + i, strands + i + 1]);
          }
          // Cross-rung phase bridges
          for (let i = 0; i < strands; i += 2) {
            edges.push([i, strands + i]);
          }
          drawFacesAndEdges(allVerts, edges, [], cx, cy, baseR, [pr, pg, pb], 0.9, 0, true);
          break;
        }

        case "latent_tesseract": {
          // True 4D-to-3D Stereographic Rotating Hypercube (Tesseract)
          const aXW = t * 1.05;
          const aYW = t * 0.72;
          const c1 = Math.cos(aXW);
          const s1 = Math.sin(aXW);
          const c2 = Math.cos(aYW);
          const s2 = Math.sin(aYW);

          const verts3D: Vec3[] = TESSERACT_VERTS_4D.map(([x, y, z, w]) => {
            // Rotate in XW
            const x1 = x * c1 - w * s1;
            const w1 = x * s1 + w * c1;
            // Rotate in YW
            const y2 = y * c2 - w1 * s2;
            const w2 = y * s2 + w1 * c2;
            // Stereographic projection from 4D to 3D
            const d = 2.65 / (2.65 - w2 * 0.72);
            const v3: Vec3 = [x1 * 0.45 * d, y2 * 0.45 * d, z * 0.45 * d];
            return rotateXYZ(v3, 0.55 + Math.sin(t * 0.4) * 0.25, t * 0.65, 0.3);
          });

          drawFacesAndEdges(
            verts3D,
            TESSERACT_EDGES,
            [],
            cx,
            cy,
            baseR * 1.04,
            [pr, pg, pb],
            0.85,
            0,
            true,
          );
          break;
        }

        case "moe_router": {
          // Central spinning 3D core + 4 orbiting 3D expert satellites with dynamic laser tethers
          const coreVerts = OCTAHEDRON_VERTS.map((v) =>
            rotateXYZ([v[0] * 0.46, v[1] * 0.46, v[2] * 0.46], -t * 1.4, t * 1.8, 0),
          );
          drawFacesAndEdges(
            coreVerts,
            OCTAHEDRON_EDGES,
            OCTAHEDRON_FACES,
            cx,
            cy,
            baseR,
            [sr, sg, sb],
            0.9,
            0.3,
            false,
          );

          const activeExpert = Math.floor((t * 1.8) % 4);
          for (let e = 0; e < 4; e++) {
            const orbitAngle = t * 1.15 + (e * Math.PI) / 2;
            const satCenter = rotateXYZ(
              [Math.cos(orbitAngle) * 1.05, Math.sin(orbitAngle) * 0.38, Math.sin(orbitAngle) * 0.95],
              0.45,
              0.3,
              e % 2 === 0 ? 0.35 : -0.35,
            );
            if (activeCollector) {
              activeCollector.push({
                v1: [0, 0, 0],
                v2: satCenter,
                rgb: [pr, pg, pb],
                alpha: 0.75,
              });
            } else {
              const pCenter = project(satCenter, cx, cy, baseR);

              // Laser routing beam from origin to satellite
              ctx.beginPath();
              ctx.moveTo(cx, cy);
              ctx.lineTo(pCenter.x, pCenter.y);
              ctx.strokeStyle =
                e === activeExpert
                  ? `rgba(255, 255, 255, 0.92)`
                  : `rgba(${pr}, ${pg}, ${pb}, 0.38)`;
              ctx.lineWidth = (e === activeExpert ? 1.8 : 0.9) * (size / 64);
              ctx.stroke();
            }

            // Mini 3D octahedron at satellite position
            const satVerts = OCTAHEDRON_VERTS.map((v) => {
              const local = rotateXYZ([v[0] * 0.25, v[1] * 0.25, v[2] * 0.25], t * 2.2, -t * 1.9, e);
              return [
                satCenter[0] + local[0],
                satCenter[1] + local[1],
                satCenter[2] + local[2],
              ] as Vec3;
            });
            drawFacesAndEdges(
              satVerts,
              OCTAHEDRON_EDGES,
              OCTAHEDRON_FACES,
              cx,
              cy,
              baseR,
              e === activeExpert ? [255, 240, 180] : [pr, pg, pb],
              0.9,
              e === activeExpert ? 0.45 : 0.2,
              false,
            );
          }
          break;
        }

        case "tensor_matrix": {
          // 3D Torsional Twisting Tensor Lattice (3 stacked 3x3 slices that twist around Y)
          const verts: Vec3[] = [];
          const edges: EdgePair[] = [];
          const twistAmount = Math.sin(t * 1.5) * 1.05;
          const coords = [-0.68, 0, 0.68];

          for (let yi = 0; yi < 3; yi++) {
            for (let zi = 0; zi < 3; zi++) {
              for (let xi = 0; xi < 3; xi++) {
                const raw: Vec3 = [coords[xi], coords[yi], coords[zi]];
                const twisted = twistY(raw, twistAmount);
                verts.push(rotateXYZ(twisted, 0.52, t * 0.75, 0.18));
              }
            }
          }

          const idx = (x: number, y: number, z: number) => y * 9 + z * 3 + x;
          for (let y = 0; y < 3; y++) {
            for (let z = 0; z < 3; z++) {
              for (let x = 0; x < 3; x++) {
                // Connect perimeter & internal spine
                const isPerimeter =
                  x === 0 || x === 2 || y === 0 || y === 2 || z === 0 || z === 2;
                if (!isPerimeter) continue;
                if (x < 2) edges.push([idx(x, y, z), idx(x + 1, y, z)]);
                if (y < 2) edges.push([idx(x, y, z), idx(x, y + 1, z)]);
                if (z < 2) edges.push([idx(x, y, z), idx(x, y, z + 1)]);
              }
            }
          }

          drawFacesAndEdges(verts, edges, [], cx, cy, baseR, [pr, pg, pb], 0.72, 0, true);
          break;
        }

        case "lora_adapter": {
          // Counter-twisting 3D Hyperbolic Hourglass (top & bottom hex manifolds pinching through rank-r waist)
          const N = 6;
          const verts: Vec3[] = [];
          const edges: EdgePair[] = [];
          const twist = Math.sin(t * 1.4) * 1.2 + t * 0.9;

          // Top ring (0..5), Waist ring (6..11), Bottom ring (12..17)
          for (let i = 0; i < N; i++) {
            const aTop = (i / N) * Math.PI * 2 + twist;
            verts.push(
              rotateXYZ([Math.cos(aTop) * 0.92, -0.88, Math.sin(aTop) * 0.92], 0.48, t * 0.5, 0.15),
            );
          }
          for (let i = 0; i < N; i++) {
            const aMid = (i / N) * Math.PI * 2;
            const waistR = 0.26 + 0.06 * Math.sin(t * 3);
            verts.push(
              rotateXYZ([Math.cos(aMid) * waistR, 0, Math.sin(aMid) * waistR], 0.48, t * 0.5, 0.15),
            );
          }
          for (let i = 0; i < N; i++) {
            const aBot = (i / N) * Math.PI * 2 - twist;
            verts.push(
              rotateXYZ([Math.cos(aBot) * 0.92, 0.88, Math.sin(aBot) * 0.92], 0.48, t * 0.5, 0.15),
            );
          }

          for (let i = 0; i < N; i++) {
            const next = (i + 1) % N;
            edges.push([i, next]);
            edges.push([N + i, N + next]);
            edges.push([2 * N + i, 2 * N + next]);
            edges.push([i, N + i]);
            edges.push([N + i, 2 * N + i]);
          }

          drawFacesAndEdges(verts, edges, [], cx, cy, baseR, [pr, pg, pb], 0.92, 0, true);
          break;
        }

        case "norm_stabilizer": {
          // 3-Axis Gyroscopic Armillary Sphere + stabilized levitating core
          drawRing3D(cx, cy, 1.12, t * 1.3, t * 0.4, 0, [pr, pg, pb], 28, baseR, false);
          drawRing3D(
            cx,
            cy,
            0.88,
            -t * 0.5,
            t * 1.45,
            Math.PI * 0.25,
            [sr, sg, sb],
            24,
            baseR,
            false,
          );
          drawRing3D(
            cx,
            cy,
            0.64,
            Math.PI * 0.5 + Math.sin(t) * 0.6,
            -t * 1.1,
            t * 1.6,
            [255, 255, 255],
            20,
            baseR,
            true,
          );

          const core = OCTAHEDRON_VERTS.map((v) =>
            rotateXYZ([v[0] * 0.36, v[1] * 0.36, v[2] * 0.36], -t * 0.8, t * 1.2, 0),
          );
          drawFacesAndEdges(core, OCTAHEDRON_EDGES, OCTAHEDRON_FACES, cx, cy, baseR, [pr, pg, pb], 0.9, 0.35, false);
          break;
        }

        case "residual_combiner": {
          // Two interlocking 3D orbital rings fusing into a central synthesis octahedron
          drawRing3D(
            cx,
            cy,
            1.05,
            0.75 + Math.sin(t * 0.9) * 0.3,
            t * 1.2,
            0.6,
            [pr, pg, pb],
            26,
            baseR,
            false,
          );
          drawRing3D(
            cx,
            cy,
            1.05,
            -0.75 + Math.cos(t * 0.9) * 0.3,
            -t * 1.2,
            -0.6,
            [sr, sg, sb],
            26,
            baseR,
            false,
          );
          const core = OCTAHEDRON_VERTS.map((v) =>
            rotateXYZ([v[0] * 0.55, v[1] * 0.55, v[2] * 0.55], t * 1.5, -t * 1.1, t * 0.7),
          );
          drawFacesAndEdges(
            core,
            OCTAHEDRON_EDGES,
            OCTAHEDRON_FACES,
            cx,
            cy,
            baseR,
            [pr, pg, pb],
            0.95,
            0.32,
          );
          break;
        }

        case "memory_vault": {
          // 4 Stacked 3D Octagonal Memory Platters counter-rotating with piston breathing
          const platters = 4;
          const N = 8;
          const verts: Vec3[] = [];
          const edges: EdgePair[] = [];
          const faces: FacePoly[] = [];

          for (let p = 0; p < platters; p++) {
            const baseY = -0.82 + (p / (platters - 1)) * 1.64;
            const y = baseY + Math.sin(t * 2.2 + p * 1.1) * 0.08;
            const dir = p % 2 === 0 ? 1 : -1;
            const rot = t * 0.95 * dir + p * 0.5;
            const baseIdx = p * N;

            for (let i = 0; i < N; i++) {
              const a = (i / N) * Math.PI * 2 + rot;
              const v: Vec3 = [Math.cos(a) * 0.88, y, Math.sin(a) * 0.88];
              verts.push(rotateXYZ(v, 0.52, Math.sin(t * 0.4) * 0.3, 0));
              edges.push([baseIdx + i, baseIdx + ((i + 1) % N)]);
              if (p < platters - 1 && i % 2 === 0) {
                edges.push([baseIdx + i, baseIdx + N + i]);
              }
            }
            faces.push({
              indices: Array.from({ length: N }, (_, i) => baseIdx + i),
              alpha: 0.14,
            });
          }

          drawFacesAndEdges(verts, edges, faces, cx, cy, baseR, [pr, pg, pb], 0.88, 0.16, true);
          break;
        }

        case "objective_beacon": {
          // 3D Stellated Octahedron (Merkaba / Dual Counter-Rotating 3D Tetrahedra)
          const vertsA = TETRA_A_VERTS.map((v) =>
            rotateXYZ(v, 0.45 + Math.sin(t * 0.7) * 0.25, t * 1.35, 0.2),
          );
          const vertsB = TETRA_B_VERTS.map((v) =>
            rotateXYZ(v, 0.45 + Math.sin(t * 0.7) * 0.25, -t * 1.35, -0.2),
          );
          drawFacesAndEdges(
            vertsA,
            TETRA_EDGES,
            TETRA_FACES,
            cx,
            cy,
            baseR,
            [pr, pg, pb],
            0.95,
            0.25,
            true,
          );
          drawFacesAndEdges(
            vertsB,
            TETRA_EDGES,
            TETRA_FACES,
            cx,
            cy,
            baseR,
            [sr, sg, sb],
            0.9,
            0.22,
            true,
          );
          break;
        }

        case "signal_modulator": {
          // 3D Extruded Activation Function Ribbon twisting in space + outer 3D coordinate ring
          drawRing3D(cx, cy, 1.12, 0.55, t * 0.65, 0.2, [sr, sg, sb], 24, baseR, true);

          const steps = 15;
          const verts: Vec3[] = [];
          const edges: EdgePair[] = [];
          const faces: FacePoly[] = [];
          const ry = Math.sin(t * 0.85) * 0.65 + t * 0.45;
          const rx = 0.38 + Math.cos(t * 0.6) * 0.2;

          for (let i = 0; i < steps; i++) {
            const u = (i / (steps - 1)) * 2.0 - 1.0;
            const fnY = -computeActivationY(curNeuronName, u * 1.8) * 0.85;
            const twist = u * 0.65 * Math.sin(t * 1.3);
            const zHalf = 0.26;
            const vFront: Vec3 = [
              u * 0.95,
              fnY + Math.sin(twist) * zHalf,
              Math.cos(twist) * zHalf,
            ];
            const vBack: Vec3 = [
              u * 0.95,
              fnY - Math.sin(twist) * zHalf,
              -Math.cos(twist) * zHalf,
            ];
            verts.push(rotateXYZ(vFront, rx, ry, 0));
            verts.push(rotateXYZ(vBack, rx, ry, 0));

            const idx = i * 2;
            if (i % 2 === 0) edges.push([idx, idx + 1]);
            if (i < steps - 1) {
              edges.push([idx, idx + 2]);
              edges.push([idx + 1, idx + 3]);
              faces.push({ indices: [idx, idx + 2, idx + 3, idx + 1], alpha: 0.28 });
            }
          }

          drawFacesAndEdges(verts, edges, faces, cx, cy, baseR, [pr, pg, pb], 0.95, 0.26, false);
          break;
        }

        case "citadel_subgraph": {
          // Outer 3D Hexagonal Prism Citadel + Counter-Rotating Inner Octahedron Core
          const N = 6;
          const hexVerts: Vec3[] = [];
          const hexEdges: EdgePair[] = [];
          for (let i = 0; i < N; i++) {
            const a = (i / N) * Math.PI * 2 + t * 0.55;
            hexVerts.push(
              rotateXYZ([Math.cos(a) * 1.02, -0.62, Math.sin(a) * 1.02], 0.52, 0.2, 0.1),
            );
          }
          for (let i = 0; i < N; i++) {
            const a = (i / N) * Math.PI * 2 + t * 0.55;
            hexVerts.push(
              rotateXYZ([Math.cos(a) * 1.02, 0.62, Math.sin(a) * 1.02], 0.52, 0.2, 0.1),
            );
          }
          for (let i = 0; i < N; i++) {
            const next = (i + 1) % N;
            hexEdges.push([i, next]);
            hexEdges.push([N + i, N + next]);
            hexEdges.push([i, N + i]);
          }
          drawFacesAndEdges(hexVerts, hexEdges, [], cx, cy, baseR, [pr, pg, pb], 0.8, 0, true);

          const inner = OCTAHEDRON_VERTS.map((v) =>
            rotateXYZ([v[0] * 0.56, v[1] * 0.56, v[2] * 0.56], -t * 1.2, -t * 1.5, 0.3),
          );
          drawFacesAndEdges(
            inner,
            OCTAHEDRON_EDGES,
            OCTAHEDRON_FACES,
            cx,
            cy,
            baseR,
            [sr, sg, sb],
            0.95,
            0.32,
            false,
          );
          break;
        }

        case "portal_input":
        case "portal_output": {
          // 3D Twisting Vortex Tunnel (4 coaxial 3D rings receding/advancing in Z depth)
          const isInput = curArchId === "portal_input";
          const rings = 4;
          const N = 6;
          const verts: Vec3[] = [];
          const edges: EdgePair[] = [];

          for (let r = 0; r < rings; r++) {
            const progress = ((r / rings) + (isInput ? t * 0.35 : -t * 0.35)) % 1;
            const normP = progress < 0 ? progress + 1 : progress;
            const z = -0.9 + normP * 1.8;
            const rad = isInput ? 0.35 + normP * 0.78 : 1.1 - normP * 0.75;
            const twist = normP * 1.4 + t * (isInput ? 1.2 : -1.2);
            const baseIdx = r * N;

            for (let i = 0; i < N; i++) {
              const a = (i / N) * Math.PI * 2 + twist;
              verts.push(
                rotateXYZ(
                  [Math.cos(a) * rad, Math.sin(a) * rad, z],
                  Math.sin(t * 0.7) * 0.32,
                  Math.cos(t * 0.6) * 0.38,
                  0,
                ),
              );
              edges.push([baseIdx + i, baseIdx + ((i + 1) % N)]);
            }
          }
          drawFacesAndEdges(verts, edges, [], cx, cy, baseR, [pr, pg, pb], 0.9, 0, true);
          break;
        }
      }
    };

    const render = (nowMs: number) => {
      const mState = morphStateRef.current;
      const u = Math.max(0, Math.min(1, mState.morphProgress));
      const isMorphing = Boolean(mState.morphFrom && u < 0.995);

      const t =
        nowMs * 0.001 * speedMultiplier * (selected || isMorphing ? 1.35 : 1.0) +
        phaseOffsetRef.current;
      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, size, size);

      const cx = size * 0.5;
      const cy = size * 0.5;
      const baseR = size * 0.34;

      const srcPRgb = mState.morphFrom
        ? hexToRgb(mState.morphFrom.primary)
        : targetPRgb;
      const srcSRgb = mState.morphFrom
        ? hexToRgb(mState.morphFrom.secondary)
        : targetSRgb;

      const blendPRgb: [number, number, number] = isMorphing
        ? [
            Math.round(srcPRgb[0] + (targetPRgb[0] - srcPRgb[0]) * u),
            Math.round(srcPRgb[1] + (targetPRgb[1] - srcPRgb[1]) * u),
            Math.round(srcPRgb[2] + (targetPRgb[2] - srcPRgb[2]) * u),
          ]
        : targetPRgb;
      const blendSRgb: [number, number, number] = isMorphing
        ? [
            Math.round(srcSRgb[0] + (targetSRgb[0] - srcSRgb[0]) * u),
            Math.round(srcSRgb[1] + (targetSRgb[1] - srcSRgb[1]) * u),
            Math.round(srcSRgb[2] + (targetSRgb[2] - srcSRgb[2]) * u),
          ]
        : targetSRgb;

      // Ambient volumetric core glow
      const amb = ctx.createRadialGradient(cx, cy, 1, cx, cy, size * 0.48);
      amb.addColorStop(
        0,
        `rgba(${blendPRgb[0]}, ${blendPRgb[1]}, ${blendPRgb[2]}, ${
          selected || isMorphing ? 0.36 : 0.2
        })`,
      );
      amb.addColorStop(
        0.55,
        `rgba(${blendSRgb[0]}, ${blendSRgb[1]}, ${blendSRgb[2]}, 0.1)`,
      );
      amb.addColorStop(1, "rgba(0, 0, 0, 0)");
      ctx.fillStyle = amb;
      ctx.fillRect(0, 0, size, size);

      ctx.globalCompositeOperation = "lighter";

      if (!isMorphing || !mState.morphFrom) {
        activeCollector = null;
        emitArchetypeGeometry(
          archetypeId,
          neuronName,
          targetPRgb,
          targetSRgb,
          t,
          cx,
          cy,
          baseR,
        );
      } else if (u <= 0.01) {
        // Pure cloned parent 3D wireframe core at start of emission
        activeCollector = null;
        emitArchetypeGeometry(
          mState.morphFrom.archetypeId,
          mState.morphFrom.neuronName,
          srcPRgb,
          srcSRgb,
          t,
          cx,
          cy,
          baseR,
        );
      } else {
        // True 3D Vector Wireframe Metamorphosis:
        // Extract the live 3D rotating wireframe segments of both the parent archetype
        // and the target archetype, then interpolate every 3D vertex & edge in (x, y, z) space!
        const srcSegs: WireframeSegment3D[] = [];
        activeCollector = srcSegs;
        emitArchetypeGeometry(
          mState.morphFrom.archetypeId,
          mState.morphFrom.neuronName,
          srcPRgb,
          srcSRgb,
          t,
          cx,
          cy,
          baseR,
        );

        const dstSegs: WireframeSegment3D[] = [];
        activeCollector = dstSegs;
        emitArchetypeGeometry(
          archetypeId,
          neuronName,
          targetPRgb,
          targetSRgb,
          t,
          cx,
          cy,
          baseR,
        );
        activeCollector = null;

        const maxLen = Math.max(srcSegs.length, dstSegs.length);
        if (maxLen > 0) {
          // Smooth S-curve so you can clearly watch the 3D vector wireframe lines unfold and re-lock
          const morphEase =
            u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
          const midTwist = Math.sin(u * Math.PI) * 0.55;

          const morphedSegs: {
            p1: ProjectedVert;
            p2: ProjectedVert;
            z: number;
            rgb: [number, number, number];
            alpha: number;
          }[] = [];

          for (let k = 0; k < maxLen; k++) {
            const s = srcSegs[k % srcSegs.length];
            const d = dstSegs[k % dstSegs.length];

            const rawV1: Vec3 = [
              s.v1[0] + (d.v1[0] - s.v1[0]) * morphEase,
              s.v1[1] + (d.v1[1] - s.v1[1]) * morphEase,
              s.v1[2] + (d.v1[2] - s.v1[2]) * morphEase,
            ];
            const rawV2: Vec3 = [
              s.v2[0] + (d.v2[0] - s.v2[0]) * morphEase,
              s.v2[1] + (d.v2[1] - s.v2[1]) * morphEase,
              s.v2[2] + (d.v2[2] - s.v2[2]) * morphEase,
            ];

            const v1 = twistY(rawV1, midTwist);
            const v2 = twistY(rawV2, midTwist);

            const p1 = project(v1, cx, cy, baseR);
            const p2 = project(v2, cx, cy, baseR);
            const rgb: [number, number, number] = [
              Math.round(s.rgb[0] + (d.rgb[0] - s.rgb[0]) * morphEase),
              Math.round(s.rgb[1] + (d.rgb[1] - s.rgb[1]) * morphEase),
              Math.round(s.rgb[2] + (d.rgb[2] - s.rgb[2]) * morphEase),
            ];

            morphedSegs.push({
              p1,
              p2,
              z: (v1[2] + v2[2]) * 0.5,
              rgb,
              alpha: s.alpha + (d.alpha - s.alpha) * morphEase,
            });
          }

          morphedSegs.sort((a, b) => a.z - b.z);

          for (const seg of morphedSegs) {
            const depthFactor = Math.max(
              0.3,
              Math.min(1.2, (seg.z + 1.4) / 2.2),
            );
            ctx.beginPath();
            ctx.moveTo(seg.p1.x, seg.p1.y);
            ctx.lineTo(seg.p2.x, seg.p2.y);
            ctx.strokeStyle = `rgba(${seg.rgb[0]}, ${seg.rgb[1]}, ${seg.rgb[2]}, ${Math.min(
              1,
              seg.alpha * depthFactor,
            )})`;
            ctx.lineWidth = (1.0 + 0.95 * depthFactor) * (size / 64);
            ctx.stroke();

            // Draw glowing 3D wireframe vertex joints so the vertex migration is clearly visible
            const vr = (1.15 + 1.0 * depthFactor) * (size / 64);
            ctx.beginPath();
            ctx.arc(seg.p1.x, seg.p1.y, vr, 0, Math.PI * 2);
            ctx.fillStyle =
              depthFactor > 0.82
                ? `rgba(255, 255, 255, ${0.88 * depthFactor})`
                : `rgba(${seg.rgb[0]}, ${seg.rgb[1]}, ${seg.rgb[2]}, ${0.8 * depthFactor})`;
            ctx.fill();
          }
        }
      }

      ctx.restore();
      rafId = window.requestAnimationFrame(render);
    };

    rafId = window.requestAnimationFrame(render);
    return () => window.cancelAnimationFrame(rafId);
  }, [archetypeId, neuronName, primaryColor, secondaryColor, size, speedMultiplier, selected]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size }}
      className="pointer-events-none block select-none"
    />
  );
};

export default VolumetricNodeCore;
