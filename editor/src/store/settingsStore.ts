import { create } from "zustand";

export type ParticleVisualMode = "hybrid_hud" | "particle_nebula";
export type ParticleDensity = "balanced" | "high" | "ultra";
export type ParticleColorPalette =
  | "cyber_plasma"
  | "solaris_gold"
  | "quantum_violet"
  | "crimson_protocol";
export type MotionIntensity = "smooth" | "cinematic";

export interface PaletteColors {
  id: ParticleColorPalette;
  name: string;
  subtitle: string;
  primary: string;
  primaryRgb: [number, number, number];
  secondary: string;
  secondaryRgb: [number, number, number];
  accent: string;
  accentRgb: [number, number, number];
  subgraph: string;
  subgraphRgb: [number, number, number];
  module: string;
  moduleRgb: [number, number, number];
  terminalIn: string;
  terminalOut: string;
  bgCore: string;
  gridColor: string;
}

export const PARTICLE_PALETTES: Record<ParticleColorPalette, PaletteColors> = {
  cyber_plasma: {
    id: "cyber_plasma",
    name: "Cyber Plasma Core",
    subtitle: "Electric Cyan, Plasma Blue & Solar Gold",
    primary: "#00f0ff",
    primaryRgb: [0, 240, 255],
    secondary: "#38bdf8",
    secondaryRgb: [56, 189, 248],
    accent: "#f59e0b",
    accentRgb: [245, 158, 11],
    subgraph: "#fbbf24",
    subgraphRgb: [251, 191, 36],
    module: "#10b981",
    moduleRgb: [16, 185, 129],
    terminalIn: "#4ade80",
    terminalOut: "#fb923c",
    bgCore: "#020b18",
    gridColor: "rgba(0, 240, 255, 0.09)",
  },
  solaris_gold: {
    id: "solaris_gold",
    name: "Solaris Nanotech",
    subtitle: "Titanium Gold, Solar Amber & Cyber Cyan",
    primary: "#fbbf24",
    primaryRgb: [251, 191, 36],
    secondary: "#f97316",
    secondaryRgb: [249, 115, 22],
    accent: "#00f0ff",
    accentRgb: [0, 240, 255],
    subgraph: "#38bdf8",
    subgraphRgb: [56, 189, 248],
    module: "#f59e0b",
    moduleRgb: [245, 158, 11],
    terminalIn: "#34d399",
    terminalOut: "#f43f5e",
    bgCore: "#0f0903",
    gridColor: "rgba(251, 191, 36, 0.09)",
  },
  quantum_violet: {
    id: "quantum_violet",
    name: "Quantum Nebula",
    subtitle: "Bioluminescent Violet, Electric Indigo & Cyan",
    primary: "#a855f7",
    primaryRgb: [168, 85, 247],
    secondary: "#06b6d4",
    secondaryRgb: [6, 182, 212],
    accent: "#f472b6",
    accentRgb: [244, 114, 182],
    subgraph: "#fbbf24",
    subgraphRgb: [251, 191, 36],
    module: "#2dd4bf",
    moduleRgb: [45, 212, 191],
    terminalIn: "#4ade80",
    terminalOut: "#fb923c",
    bgCore: "#080314",
    gridColor: "rgba(168, 85, 247, 0.09)",
  },
  crimson_protocol: {
    id: "crimson_protocol",
    name: "Crimson Tactical HUD",
    subtitle: "High-Alert Crimson, Solar Orange & Laser Gold",
    primary: "#fb7185",
    primaryRgb: [251, 113, 133],
    secondary: "#f97316",
    secondaryRgb: [249, 115, 22],
    accent: "#facc15",
    accentRgb: [250, 204, 21],
    subgraph: "#fbbf24",
    subgraphRgb: [251, 191, 36],
    module: "#38bdf8",
    moduleRgb: [56, 189, 248],
    terminalIn: "#4ade80",
    terminalOut: "#f43f5e",
    bgCore: "#140408",
    gridColor: "rgba(251, 113, 133, 0.09)",
  },
};

export interface PersistedEditorSettings {
  particleGraphEditorEnabled: boolean;
  particleVisualMode: ParticleVisualMode;
  particleDensity: ParticleDensity;
  colorPalette: ParticleColorPalette;
  holoPerspective3D: boolean;
  motionIntensity: MotionIntensity;
  hudTelemetryOverlay: boolean;
}

export interface SettingsState extends PersistedEditorSettings {
  isSettingsOpen: boolean;
  pulseSequence: number;

  setParticleGraphEditorEnabled: (enabled: boolean) => void;
  toggleParticleGraphEditor: () => void;
  setParticleVisualMode: (mode: ParticleVisualMode) => void;
  setParticleDensity: (density: ParticleDensity) => void;
  setColorPalette: (palette: ParticleColorPalette) => void;
  setHoloPerspective3D: (enabled: boolean) => void;
  setMotionIntensity: (intensity: MotionIntensity) => void;
  setHudTelemetryOverlay: (enabled: boolean) => void;
  setSettingsOpen: (open: boolean) => void;
  triggerPulse: () => void;
  resetSettings: () => void;
}

const STORAGE_KEY = "neuralfn:editor_settings:v1";

const DEFAULT_SETTINGS: PersistedEditorSettings = {
  particleGraphEditorEnabled: false,
  particleVisualMode: "hybrid_hud",
  particleDensity: "high",
  colorPalette: "cyber_plasma",
  holoPerspective3D: false,
  motionIntensity: "cinematic",
  hudTelemetryOverlay: true,
};

function loadPersistedSettings(): PersistedEditorSettings {
  if (typeof window === "undefined") {
    return DEFAULT_SETTINGS;
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return DEFAULT_SETTINGS;
    }
    const parsed = JSON.parse(raw) as Partial<PersistedEditorSettings>;
    return {
      particleGraphEditorEnabled:
        typeof parsed.particleGraphEditorEnabled === "boolean"
          ? parsed.particleGraphEditorEnabled
          : DEFAULT_SETTINGS.particleGraphEditorEnabled,
      particleVisualMode:
        parsed.particleVisualMode === "hybrid_hud" ||
        parsed.particleVisualMode === "particle_nebula"
          ? parsed.particleVisualMode
          : DEFAULT_SETTINGS.particleVisualMode,
      particleDensity:
        parsed.particleDensity === "balanced" ||
        parsed.particleDensity === "high" ||
        parsed.particleDensity === "ultra"
          ? parsed.particleDensity
          : DEFAULT_SETTINGS.particleDensity,
      colorPalette:
        parsed.colorPalette && parsed.colorPalette in PARTICLE_PALETTES
          ? parsed.colorPalette
          : DEFAULT_SETTINGS.colorPalette,
      holoPerspective3D:
        typeof parsed.holoPerspective3D === "boolean"
          ? parsed.holoPerspective3D
          : DEFAULT_SETTINGS.holoPerspective3D,
      motionIntensity:
        parsed.motionIntensity === "smooth" || parsed.motionIntensity === "cinematic"
          ? parsed.motionIntensity
          : DEFAULT_SETTINGS.motionIntensity,
      hudTelemetryOverlay:
        typeof parsed.hudTelemetryOverlay === "boolean"
          ? parsed.hudTelemetryOverlay
          : DEFAULT_SETTINGS.hudTelemetryOverlay,
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function savePersistedSettings(state: PersistedEditorSettings) {
  if (typeof window === "undefined") {
    return;
  }
  try {
    const payload: PersistedEditorSettings = {
      particleGraphEditorEnabled: state.particleGraphEditorEnabled,
      particleVisualMode: state.particleVisualMode,
      particleDensity: state.particleDensity,
      colorPalette: state.colorPalette,
      holoPerspective3D: state.holoPerspective3D,
      motionIntensity: state.motionIntensity,
      hudTelemetryOverlay: state.hudTelemetryOverlay,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Ignore localStorage quota/privacy errors.
  }
}

export const useSettingsStore = create<SettingsState>((set) => ({
  ...loadPersistedSettings(),
  isSettingsOpen: false,
  pulseSequence: 0,

  setParticleGraphEditorEnabled: (particleGraphEditorEnabled) =>
    set((state) => {
      const next = { ...state, particleGraphEditorEnabled };
      savePersistedSettings(next);
      return next;
    }),

  toggleParticleGraphEditor: () =>
    set((state) => {
      const next = {
        ...state,
        particleGraphEditorEnabled: !state.particleGraphEditorEnabled,
      };
      savePersistedSettings(next);
      return next;
    }),

  setParticleVisualMode: (particleVisualMode) =>
    set((state) => {
      const next = { ...state, particleVisualMode };
      savePersistedSettings(next);
      return next;
    }),

  setParticleDensity: (particleDensity) =>
    set((state) => {
      const next = { ...state, particleDensity };
      savePersistedSettings(next);
      return next;
    }),

  setColorPalette: (colorPalette) =>
    set((state) => {
      const next = { ...state, colorPalette };
      savePersistedSettings(next);
      return next;
    }),

  setHoloPerspective3D: (holoPerspective3D) =>
    set((state) => {
      const next = { ...state, holoPerspective3D };
      savePersistedSettings(next);
      return next;
    }),

  setMotionIntensity: (motionIntensity) =>
    set((state) => {
      const next = { ...state, motionIntensity };
      savePersistedSettings(next);
      return next;
    }),

  setHudTelemetryOverlay: (hudTelemetryOverlay) =>
    set((state) => {
      const next = { ...state, hudTelemetryOverlay };
      savePersistedSettings(next);
      return next;
    }),

  setSettingsOpen: (isSettingsOpen) => set((state) => ({ ...state, isSettingsOpen })),

  triggerPulse: () =>
    set((state) => ({ ...state, pulseSequence: state.pulseSequence + 1 })),

  resetSettings: () =>
    set((state) => {
      const next = { ...state, ...DEFAULT_SETTINGS };
      savePersistedSettings(next);
      return next;
    }),
}));
