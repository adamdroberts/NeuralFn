import React, { useEffect, useRef } from "react";

export interface WebGLNodeAnchor {
  id: string;
  x: number;
  y: number;
  rgb: [number, number, number];
  secondaryRgb: [number, number, number];
  selected: boolean;
  seed: number;
}

export interface WebGLCyberSceneProps {
  nodes: WebGLNodeAnchor[];
  viewport: { x: number; y: number; zoom: number };
  primaryRgb: [number, number, number];
  secondaryRgb: [number, number, number];
  accentRgb: [number, number, number];
  isTraining: boolean;
}

const BG_VERT_SHADER = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

const BG_FRAG_SHADER = `
precision mediump float;
varying vec2 v_uv;
uniform vec2 u_resolution;
uniform vec3 u_viewport; // x, y, zoom
uniform float u_time;
uniform vec3 u_primary;
uniform vec3 u_secondary;
uniform float u_training;

void main() {
  vec2 fragCoord = vec2(v_uv.x * u_resolution.x, (1.0 - v_uv.y) * u_resolution.y);
  vec2 world = (fragCoord - u_viewport.xy) / max(0.15, u_viewport.z);

  // Subtle 3D perspective depth warp on background field
  float cell = 88.0;
  vec2 gridUv = abs(fract(world / cell - 0.5) - 0.5) / fwidth(world / cell + 0.0001);
  float line = min(gridUv.x, gridUv.y);
  float gridAlpha = 1.0 - min(line, 1.0);

  // Sub-grid
  float subCell = 22.0;
  vec2 subUv = abs(fract(world / subCell - 0.5) - 0.5);
  float subLine = smoothstep(0.035, 0.0, min(subUv.x, subUv.y));

  // Traveling energy waves across the 3D floor
  float wave1 = sin(world.x * 0.006 - u_time * (1.8 + u_training * 1.5)) * 0.5 + 0.5;
  float wave2 = cos(world.y * 0.006 + u_time * 1.3) * 0.5 + 0.5;
  float pulse = pow(wave1 * wave2, 3.0);

  vec3 col = u_primary * (gridAlpha * 0.075 + subLine * 0.022)
           + u_secondary * (pulse * gridAlpha * 0.14);

  float alpha = clamp(gridAlpha * 0.14 + subLine * 0.03 + pulse * 0.04, 0.0, 0.32);
  gl_FragColor = vec4(col, alpha);
}
`;

const MESH_VERT_SHADER = `
attribute vec3 a_localPos;
attribute vec2 a_worldCenter;
attribute vec3 a_color;
attribute float a_seed;
attribute float a_scale;

uniform vec2 u_resolution;
uniform vec3 u_viewport; // panX, panY, zoom
uniform float u_time;

varying vec3 v_color;
varying float v_depthAlpha;

void main() {
  float t = u_time * (0.85 + mod(a_seed, 0.6)) + a_seed;

  // Non-linear torsional twist along local Y axis
  float twistAngle = a_localPos.y * 0.85 * sin(t * 1.2);
  float ct = cos(twistAngle);
  float st = sin(twistAngle);
  vec3 p = vec3(
    a_localPos.x * ct + a_localPos.z * st,
    a_localPos.y,
    -a_localPos.x * st + a_localPos.z * ct
  );

  // 3D rotation (X and Y axes)
  float rx = t * 0.7;
  float ry = t * 1.05;
  float cx = cos(rx), sx = sin(rx);
  float cy = cos(ry), sy = sin(ry);

  vec3 rotX = vec3(p.x, p.y * cx - p.z * sx, p.y * sx + p.z * cx);
  vec3 rotXY = vec3(rotX.x * cy + rotX.z * sy, rotX.y, -rotX.x * sy + rotX.z * cy);

  // Perspective foreshortening
  float fov = 3.4;
  float persp = fov / max(0.9, fov - rotXY.z);
  vec2 worldPos = a_worldCenter + rotXY.xy * a_scale * persp;

  // World to screen to NDC
  vec2 screenPos = worldPos * u_viewport.z + u_viewport.xy;
  vec2 ndc = (screenPos / u_resolution) * 2.0 - 1.0;
  ndc.y = -ndc.y;

  v_color = a_color;
  v_depthAlpha = clamp((rotXY.z + 1.5) / 2.6, 0.18, 1.0);
  gl_Position = vec4(ndc, 0.0, 1.0);
  gl_PointSize = clamp(2.8 * persp * sqrt(max(0.4, u_viewport.z)), 1.5, 6.0);
}
`;

const MESH_FRAG_SHADER = `
precision mediump float;
varying vec3 v_color;
varying float v_depthAlpha;

void main() {
  gl_FragColor = vec4(v_color, v_depthAlpha * 0.34);
}
`;

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(
  gl: WebGLRenderingContext,
  vsSource: string,
  fsSource: string,
): WebGLProgram | null {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  if (!prog) return null;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    gl.deleteProgram(prog);
    return null;
  }
  return prog;
}

// Build unit 3D geodesic / dual-ring orbital cage line segments
function buildUnitOrbitalCage(): Float32Array {
  const segs: number[] = [];
  const N = 16;
  // Ring 1 (XY plane)
  for (let i = 0; i < N; i++) {
    const a1 = (i / N) * Math.PI * 2;
    const a2 = ((i + 1) / N) * Math.PI * 2;
    segs.push(Math.cos(a1), Math.sin(a1), 0, Math.cos(a2), Math.sin(a2), 0);
  }
  // Ring 2 (XZ plane)
  for (let i = 0; i < N; i++) {
    const a1 = (i / N) * Math.PI * 2;
    const a2 = ((i + 1) / N) * Math.PI * 2;
    segs.push(Math.cos(a1), 0, Math.sin(a1), Math.cos(a2), 0, Math.sin(a2));
  }
  // Inner Octahedron struts
  const oct: [number, number, number][] = [
    [0, -0.78, 0],
    [0.78, 0, 0],
    [0, 0, 0.78],
    [-0.78, 0, 0],
    [0, 0, -0.78],
    [0, 0.78, 0],
  ];
  const pairs = [
    [0, 1], [0, 2], [0, 3], [0, 4],
    [5, 1], [5, 2], [5, 3], [5, 4],
    [1, 2], [2, 3], [3, 4], [4, 1],
  ];
  for (const [i, j] of pairs) {
    segs.push(...oct[i], ...oct[j]);
  }
  return new Float32Array(segs);
}

const UNIT_CAGE_VERTS = buildUnitOrbitalCage();
const VERTS_PER_NODE = UNIT_CAGE_VERTS.length / 3;

/**
 * Hardware-accelerated WebGL (GLSL) 3D Volumetric Scene Layer.
 * Uses a single shared WebGL context to render a 3D perspective cyber grid
 * and rotating/twisting 3D wireframe energy cages around every graph node.
 */
export const WebGLCyberScene: React.FC<WebGLCyberSceneProps> = ({
  nodes,
  viewport,
  primaryRgb,
  secondaryRgb,
  isTraining,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stateRef = useRef({
    nodes,
    viewport,
    primaryRgb,
    secondaryRgb,
    isTraining,
  });
  stateRef.current = { nodes, viewport, primaryRgb, secondaryRgb, isTraining };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl =
      (canvas.getContext("webgl2", { alpha: true, antialias: true }) as WebGLRenderingContext | null) ||
      (canvas.getContext("webgl", { alpha: true, antialias: true }) as WebGLRenderingContext | null);
    if (!gl) return;

    // Enable standard derivatives for crisp grid anti-aliasing if WebGL1
    gl.getExtension("OES_standard_derivatives");

    const bgProg = createProgram(gl, BG_VERT_SHADER, BG_FRAG_SHADER);
    const meshProg = createProgram(gl, MESH_VERT_SHADER, MESH_FRAG_SHADER);
    if (!bgProg || !meshProg) return;

    const quadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW,
    );

    const meshBuf = gl.createBuffer();

    let rafId = 0;
    const startTime = performance.now();

    const render = (now: number) => {
      const parent = canvas.parentElement;
      if (parent) {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, Math.floor(parent.clientWidth * dpr));
        const h = Math.max(1, Math.floor(parent.clientHeight * dpr));
        if (canvas.width !== w || canvas.height !== h) {
          canvas.width = w;
          canvas.height = h;
        }
      }

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cssW = canvas.width / dpr;
      const cssH = canvas.height / dpr;

      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

      const {
        nodes: curNodes,
        viewport: vp,
        primaryRgb: pRgb,
        secondaryRgb: sRgb,
        isTraining: training,
      } = stateRef.current;
      const t = (now - startTime) * 0.001;

      // 1. Render 3D WebGL Cyber Grid Shader
      gl.useProgram(bgProg);
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
      const aPos = gl.getAttribLocation(bgProg, "a_pos");
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

      gl.uniform2f(gl.getUniformLocation(bgProg, "u_resolution"), cssW, cssH);
      gl.uniform3f(gl.getUniformLocation(bgProg, "u_viewport"), vp.x, vp.y, vp.zoom);
      gl.uniform1f(gl.getUniformLocation(bgProg, "u_time"), t);
      gl.uniform3f(
        gl.getUniformLocation(bgProg, "u_primary"),
        pRgb[0] / 255,
        pRgb[1] / 255,
        pRgb[2] / 255,
      );
      gl.uniform3f(
        gl.getUniformLocation(bgProg, "u_secondary"),
        sRgb[0] / 255,
        sRgb[1] / 255,
        sRgb[2] / 255,
      );
      gl.uniform1f(gl.getUniformLocation(bgProg, "u_training"), training ? 1.0 : 0.0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      // 2. Render 3D Rotating/Twisting WebGL Wireframe Cages for each Node
      if (curNodes.length > 0 && meshBuf) {
        // 10 floats per vertex: localPos(3), worldCenter(2), color(3), seed(1), scale(1)
        const strideFloats = 10;
        const data = new Float32Array(curNodes.length * VERTS_PER_NODE * strideFloats);
        let offset = 0;

        for (let nIdx = 0; nIdx < curNodes.length; nIdx++) {
          const n = curNodes[nIdx];
          const r = n.rgb[0] / 255;
          const g = n.rgb[1] / 255;
          const b = n.rgb[2] / 255;
          const scale = n.selected ? 34.0 : 26.0;

          for (let v = 0; v < VERTS_PER_NODE; v++) {
            data[offset++] = UNIT_CAGE_VERTS[v * 3];
            data[offset++] = UNIT_CAGE_VERTS[v * 3 + 1];
            data[offset++] = UNIT_CAGE_VERTS[v * 3 + 2];
            data[offset++] = n.x;
            data[offset++] = n.y;
            data[offset++] = r;
            data[offset++] = g;
            data[offset++] = b;
            data[offset++] = n.seed;
            data[offset++] = scale;
          }
        }

        gl.useProgram(meshProg);
        gl.bindBuffer(gl.ARRAY_BUFFER, meshBuf);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);

        const strideBytes = strideFloats * 4;
        const locLocal = gl.getAttribLocation(meshProg, "a_localPos");
        const locCenter = gl.getAttribLocation(meshProg, "a_worldCenter");
        const locColor = gl.getAttribLocation(meshProg, "a_color");
        const locSeed = gl.getAttribLocation(meshProg, "a_seed");
        const locScale = gl.getAttribLocation(meshProg, "a_scale");

        if (locLocal >= 0) {
          gl.enableVertexAttribArray(locLocal);
          gl.vertexAttribPointer(locLocal, 3, gl.FLOAT, false, strideBytes, 0);
        }
        if (locCenter >= 0) {
          gl.enableVertexAttribArray(locCenter);
          gl.vertexAttribPointer(locCenter, 2, gl.FLOAT, false, strideBytes, 12);
        }
        if (locColor >= 0) {
          gl.enableVertexAttribArray(locColor);
          gl.vertexAttribPointer(locColor, 3, gl.FLOAT, false, strideBytes, 20);
        }
        if (locSeed >= 0) {
          gl.enableVertexAttribArray(locSeed);
          gl.vertexAttribPointer(locSeed, 1, gl.FLOAT, false, strideBytes, 32);
        }
        if (locScale >= 0) {
          gl.enableVertexAttribArray(locScale);
          gl.vertexAttribPointer(locScale, 1, gl.FLOAT, false, strideBytes, 36);
        }

        gl.uniform2f(gl.getUniformLocation(meshProg, "u_resolution"), cssW, cssH);
        gl.uniform3f(gl.getUniformLocation(meshProg, "u_viewport"), vp.x, vp.y, vp.zoom);
        gl.uniform1f(gl.getUniformLocation(meshProg, "u_time"), t);

        gl.drawArrays(gl.LINES, 0, curNodes.length * VERTS_PER_NODE);
        gl.drawArrays(gl.POINTS, 0, curNodes.length * VERTS_PER_NODE);
      }

      rafId = window.requestAnimationFrame(render);
    };

    rafId = window.requestAnimationFrame(render);
    return () => {
      window.cancelAnimationFrame(rafId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 z-[1] h-full w-full"
    />
  );
};

export default WebGLCyberScene;
