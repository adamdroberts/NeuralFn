import React, { useEffect } from "react";
import {
  PARTICLE_PALETTES,
  type MotionIntensity,
  type ParticleColorPalette,
  type ParticleDensity,
  type ParticleVisualMode,
  useSettingsStore,
} from "../store/settingsStore";

export default function SettingsModal() {
  const isSettingsOpen = useSettingsStore((state) => state.isSettingsOpen);
  const setSettingsOpen = useSettingsStore((state) => state.setSettingsOpen);

  const particleGraphEditorEnabled = useSettingsStore(
    (state) => state.particleGraphEditorEnabled,
  );
  const setParticleGraphEditorEnabled = useSettingsStore(
    (state) => state.setParticleGraphEditorEnabled,
  );
  const particleVisualMode = useSettingsStore((state) => state.particleVisualMode);
  const setParticleVisualMode = useSettingsStore((state) => state.setParticleVisualMode);
  const particleDensity = useSettingsStore((state) => state.particleDensity);
  const setParticleDensity = useSettingsStore((state) => state.setParticleDensity);
  const colorPalette = useSettingsStore((state) => state.colorPalette);
  const setColorPalette = useSettingsStore((state) => state.setColorPalette);
  const holoPerspective3D = useSettingsStore((state) => state.holoPerspective3D);
  const setHoloPerspective3D = useSettingsStore((state) => state.setHoloPerspective3D);
  const motionIntensity = useSettingsStore((state) => state.motionIntensity);
  const setMotionIntensity = useSettingsStore((state) => state.setMotionIntensity);
  const hudTelemetryOverlay = useSettingsStore((state) => state.hudTelemetryOverlay);
  const setHudTelemetryOverlay = useSettingsStore(
    (state) => state.setHudTelemetryOverlay,
  );
  const resetSettings = useSettingsStore((state) => state.resetSettings);

  useEffect(() => {
    if (!isSettingsOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSettingsOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isSettingsOpen, setSettingsOpen]);

  if (!isSettingsOpen) {
    return null;
  }

  const activePalette = PARTICLE_PALETTES[colorPalette];

  return (
    <div
      className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/75 backdrop-blur-md px-4"
      onClick={() => setSettingsOpen(false)}
    >
      <div
        className="relative w-full max-w-2xl overflow-hidden rounded-2xl border border-cyan-500/30 bg-gray-950/95 text-gray-100 shadow-[0_0_60px_rgba(0,240,255,0.18)]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top Holographic Accent Bar */}
        <div
          className="h-1 w-full"
          style={{
            background: `linear-gradient(90deg, transparent, ${activePalette.primary}, ${activePalette.accent}, ${activePalette.secondary}, transparent)`,
          }}
        />

        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-800/90 px-6 py-4">
          <div className="flex items-center gap-3">
            <div
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-cyan-400/40 bg-cyan-950/50 shadow-[0_0_15px_rgba(0,240,255,0.25)]"
              style={{
                borderColor: `${activePalette.primary}66`,
                color: activePalette.primary,
              }}
            >
              <svg
                viewBox="0 0 24 24"
                className="h-5 w-5 animate-[spin_14s_linear_infinite]"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.75"
              >
                <circle cx="12" cy="12" r="9" strokeDasharray="10 4" />
                <circle cx="12" cy="12" r="5" />
                <circle cx="12" cy="12" r="2" fill="currentColor" />
              </svg>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold tracking-wide text-white">
                  Platform Settings &amp; Feature Flags
                </h2>
                <span className="rounded border border-cyan-500/40 bg-cyan-950/60 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-widest text-cyan-300">
                  SYS.CFG
                </span>
              </div>
              <p className="text-xs text-gray-400">
                Configure experimental visual engines, holographic HUD interfaces, and motion physics.
              </p>
            </div>
          </div>

          <button
            onClick={() => setSettingsOpen(false)}
            className="rounded-lg border border-gray-800 bg-gray-900 px-2.5 py-1.5 text-xs text-gray-400 transition-colors hover:border-gray-700 hover:text-white"
          >
            ✕ ESC
          </button>
        </div>

        {/* Body */}
        <div className="max-h-[75vh] space-y-5 overflow-y-auto p-6">
          {/* Primary Feature Flag Card */}
          <div
            className={`relative overflow-hidden rounded-xl border p-5 transition-all duration-300 ${
              particleGraphEditorEnabled
                ? "border-cyan-400/60 bg-gradient-to-br from-slate-950 via-cyan-950/25 to-slate-950 shadow-[0_0_35px_rgba(0,240,255,0.14)]"
                : "border-gray-800 bg-gray-900/60"
            }`}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-1.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-bold tracking-wide text-white">
                    Volumetric 3D WebGL &amp; Particle Cyber Graph Editor
                  </span>
                  <span className="rounded-full border border-amber-400/50 bg-amber-500/10 px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-amber-300">
                    Feature Flag • Cyber 3D HUD
                  </span>
                  {particleGraphEditorEnabled && (
                    <span className="flex items-center gap-1 rounded-full border border-emerald-400/50 bg-emerald-500/10 px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-emerald-300">
                      <span className="h-1.5 w-1.5 animate-ping rounded-full bg-emerald-400" />
                      ACTIVE
                    </span>
                  )}
                </div>
                <p className="text-xs leading-relaxed text-gray-300">
                  Transforms the entire workspace into a 60 FPS{" "}
                  <span className="font-semibold text-cyan-300">
                    3D volumetric WebGL &amp; particle cybernetic environment
                  </span>
                  . Features semantically distinct 3D rotating and twisting node cores, multi-plane
                  extruded 3D node chassis, synaptic photon streams, and smooth vortex morphing
                  animations when navigating subgraphs or loading templates.
                </p>
              </div>

              {/* Toggle Switch */}
              <button
                type="button"
                role="switch"
                aria-checked={particleGraphEditorEnabled}
                data-testid="toggle-particle-graph-editor"
                onClick={() => setParticleGraphEditorEnabled(!particleGraphEditorEnabled)}
                className={`relative inline-flex h-7 w-14 shrink-0 cursor-pointer items-center rounded-full border transition-all duration-300 ${
                  particleGraphEditorEnabled
                    ? "border-cyan-300 bg-cyan-500/30 shadow-[0_0_16px_rgba(0,240,255,0.5)]"
                    : "border-gray-700 bg-gray-800"
                }`}
              >
                <span
                  className={`inline-block h-5 w-5 transform rounded-full transition-all duration-300 ${
                    particleGraphEditorEnabled
                      ? "translate-x-7 bg-cyan-300 shadow-[0_0_10px_#00f0ff]"
                      : "translate-x-1 bg-gray-400"
                  }`}
                />
              </button>
            </div>

            {/* Feature highlights pills */}
            <div className="mt-3 flex flex-wrap gap-1.5 text-[10px] font-mono">
              <span className="rounded border border-cyan-500/30 bg-cyan-950/40 px-2 py-0.5 text-cyan-200">
                ✦ 60 FPS Curl-Noise &amp; Orbital Swarms
              </span>
              <span className="rounded border border-amber-500/30 bg-amber-950/40 px-2 py-0.5 text-amber-200">
                ✦ Subgraph Vortex Morph Transitions
              </span>
              <span className="rounded border border-blue-500/30 bg-blue-950/40 px-2 py-0.5 text-blue-200">
                ✦ Double-Helix Photon Synapses
              </span>
              <span className="rounded border border-emerald-500/30 bg-emerald-950/40 px-2 py-0.5 text-emerald-200">
                ✦ Interactive Radar &amp; Target Reticles
              </span>
            </div>
          </div>

          {/* Sub-Settings for the Particle Engine */}
          <div
            className={`space-y-4 transition-opacity duration-300 ${
              particleGraphEditorEnabled ? "opacity-100" : "opacity-60"
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-[11px] font-bold uppercase tracking-widest text-cyan-400">
                Holographic Engine Parameters
              </span>
              {!particleGraphEditorEnabled && (
                <span className="text-[11px] text-amber-300/90">
                  Enable the feature flag above to activate in the editor
                </span>
              )}
            </div>

            {/* Projection Mode */}
            <div className="grid gap-3 sm:grid-cols-2">
              {(
                [
                  {
                    id: "hybrid_hud" as ParticleVisualMode,
                    title: "3D Volumetric Chassis + Swarm",
                    desc: "Multi-plane 3D extruded node chassis with rotating & twisting 3D geometry cores, orbital particle halos, and photon conduits.",
                    badge: "RECOMMENDED",
                  },
                  {
                    id: "particle_nebula" as ParticleVisualMode,
                    title: "Pure 3D Sculpture Nebula",
                    desc: "Compact floating 3D rotating & twisting sculptures surrounded by dense particle star-clusters and laser callout banners.",
                    badge: "3D SCULPTURE",
                  },
                ] as const
              ).map((modeOption) => {
                const active = particleVisualMode === modeOption.id;
                return (
                  <button
                    key={modeOption.id}
                    type="button"
                    onClick={() => {
                      setParticleVisualMode(modeOption.id);
                      if (!particleGraphEditorEnabled) {
                        setParticleGraphEditorEnabled(true);
                      }
                    }}
                    className={`flex flex-col items-start rounded-xl border p-3.5 text-left transition-all ${
                      active
                        ? "border-cyan-400/70 bg-cyan-950/30 shadow-[0_0_20px_rgba(0,240,255,0.12)]"
                        : "border-gray-800 bg-gray-900/50 hover:border-gray-700"
                    }`}
                  >
                    <div className="flex w-full items-center justify-between gap-2">
                      <span className="text-xs font-bold text-white">{modeOption.title}</span>
                      <span
                        className={`rounded px-1.5 py-0.5 font-mono text-[9px] uppercase ${
                          active
                            ? "bg-cyan-500/20 text-cyan-300 border border-cyan-400/40"
                            : "bg-gray-800 text-gray-400"
                        }`}
                      >
                        {modeOption.badge}
                      </span>
                    </div>
                    <p className="mt-1.5 text-[11px] leading-relaxed text-gray-400">
                      {modeOption.desc}
                    </p>
                  </button>
                );
              })}
            </div>

            {/* Color Palette Selector */}
            <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-4">
              <div className="mb-2.5 text-xs font-semibold text-gray-200">
                Holographic Color Matrix
              </div>
              <div className="grid gap-2.5 sm:grid-cols-2">
                {(Object.keys(PARTICLE_PALETTES) as ParticleColorPalette[]).map((key) => {
                  const p = PARTICLE_PALETTES[key];
                  const selected = colorPalette === key;
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => setColorPalette(key)}
                      className={`flex items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition-all ${
                        selected
                          ? "border-cyan-400/70 bg-gray-900 shadow-[0_0_15px_rgba(0,240,255,0.12)]"
                          : "border-gray-800 bg-gray-950/60 hover:border-gray-700"
                      }`}
                    >
                      <div className="flex -space-x-1.5">
                        <span
                          className="h-4 w-4 rounded-full border border-black/60"
                          style={{ backgroundColor: p.primary }}
                        />
                        <span
                          className="h-4 w-4 rounded-full border border-black/60"
                          style={{ backgroundColor: p.accent }}
                        />
                        <span
                          className="h-4 w-4 rounded-full border border-black/60"
                          style={{ backgroundColor: p.secondary }}
                        />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-xs font-semibold text-gray-100">
                          {p.name}
                        </div>
                        <div className="truncate text-[10px] text-gray-400">{p.subtitle}</div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Particle Density & Motion Intensity */}
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-4">
                <div className="mb-2 text-xs font-semibold text-gray-200">
                  Particle Swarm Density
                </div>
                <div className="grid grid-cols-3 gap-1.5">
                  {(
                    [
                      { id: "balanced", label: "Balanced", count: "~900" },
                      { id: "high", label: "High", count: "~1,800" },
                      { id: "ultra", label: "Ultra Core", count: "~3,200" },
                    ] as { id: ParticleDensity; label: string; count: string }[]
                  ).map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => setParticleDensity(item.id)}
                      className={`rounded-lg border px-2 py-2 text-center transition-all ${
                        particleDensity === item.id
                          ? "border-cyan-400/70 bg-cyan-950/40 text-cyan-200"
                          : "border-gray-800 bg-gray-950/60 text-gray-400 hover:border-gray-700"
                      }`}
                    >
                      <div className="text-[11px] font-bold">{item.label}</div>
                      <div className="font-mono text-[9px] opacity-75">{item.count}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-4">
                <div className="mb-2 text-xs font-semibold text-gray-200">
                  Transition &amp; Warp Physics
                </div>
                <div className="grid grid-cols-2 gap-1.5">
                  {(
                    [
                      {
                        id: "smooth",
                        label: "Smooth Damped",
                        sub: "Gentle spring glide",
                      },
                      {
                        id: "cinematic",
                        label: "Cinematic Warp",
                        sub: "Vortex & shockwaves",
                      },
                    ] as { id: MotionIntensity; label: string; sub: string }[]
                  ).map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => setMotionIntensity(item.id)}
                      className={`rounded-lg border px-2.5 py-2 text-left transition-all ${
                        motionIntensity === item.id
                          ? "border-cyan-400/70 bg-cyan-950/40 text-cyan-200"
                          : "border-gray-800 bg-gray-950/60 text-gray-400 hover:border-gray-700"
                      }`}
                    >
                      <div className="text-[11px] font-bold">{item.label}</div>
                      <div className="text-[9px] opacity-75">{item.sub}</div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Additional HUD Toggles */}
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="flex cursor-pointer items-center justify-between rounded-xl border border-gray-800 bg-gray-900/50 px-4 py-3">
                <div>
                  <div className="text-xs font-semibold text-gray-200">
                    3D Holographic Tilt &amp; Parallax
                  </div>
                  <div className="text-[10px] text-gray-400">
                    Interactive 3D table projection that tilts with cursor movement
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={holoPerspective3D}
                  onChange={(e) => setHoloPerspective3D(e.target.checked)}
                  className="h-4 w-4 accent-cyan-400"
                />
              </label>

              <label className="flex cursor-pointer items-center justify-between rounded-xl border border-gray-800 bg-gray-900/50 px-4 py-3">
                <div>
                  <div className="text-xs font-semibold text-gray-200">
                    Cyber Diagnostic HUD &amp; Radar
                  </div>
                  <div className="text-[10px] text-gray-400">
                    Volumetric schematic core, radar sweep scanner &amp; live telemetry
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={hudTelemetryOverlay}
                  onChange={(e) => setHudTelemetryOverlay(e.target.checked)}
                  className="h-4 w-4 accent-cyan-400"
                />
              </label>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-gray-800/90 bg-gray-900/60 px-6 py-3.5">
          <button
            type="button"
            onClick={resetSettings}
            className="rounded border border-gray-700 bg-gray-800/80 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-700"
          >
            Reset Defaults
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSettingsOpen(false)}
              className="rounded-lg bg-cyan-500 px-4 py-1.5 text-xs font-bold text-slate-950 shadow-[0_0_20px_rgba(0,240,255,0.35)] transition-all hover:bg-cyan-400"
            >
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
