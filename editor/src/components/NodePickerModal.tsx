import React, { useEffect, useMemo, useRef, useState } from "react";
import type { NeuronDefData } from "../api/client";
import {
  NODE_ARCHETYPES,
  type NodeArchetypeId,
  resolveNodeArchetype,
} from "./nodeArchetypes";
import { SemanticNodeGlyph } from "./ParticleNeuronNode";

export interface CategoryMeta {
  id: string;
  title: string;
  subtitle: string;
  accent: string;
  archetypeId: NodeArchetypeId;
}

export const PICKER_CATEGORIES: CategoryMeta[] = [
  {
    id: "attention",
    title: "Attention & Sequence Mixers",
    subtitle: "SDPA, MLA, Sparse, Mamba SSM, TTT & Head Ops",
    accent: "#00f0ff",
    archetypeId: "attention_prism",
  },
  {
    id: "linear_mlp",
    title: "Linear, MLP & Adapters",
    subtitle: "Linear, SwiGLU, GeGLU, FP8, BitLinear, LoRA & qLoRA",
    accent: "#818cf8",
    archetypeId: "tensor_matrix",
  },
  {
    id: "moe",
    title: "MoE Routing & Experts",
    subtitle: "Top-K Routers, Expert Dispatch, Aux-Free & Semantic MoE",
    accent: "#f59e0b",
    archetypeId: "moe_router",
  },
  {
    id: "jepa_latent",
    title: "JEPA, Latent & World Models",
    subtitle: "Latent Pool, JEPA Predictors, Semantic Chunks & Vision",
    accent: "#c084fc",
    archetypeId: "latent_tesseract",
  },
  {
    id: "embeddings_kv",
    title: "Embeddings, RoPE & KV Memory",
    subtitle: "Token/Pos Embeddings, Rotary Phase & KV Cache/PCA/Quant",
    accent: "#2dd4bf",
    archetypeId: "memory_vault",
  },
  {
    id: "norms_residuals",
    title: "Norms, Residuals & Gating",
    subtitle: "RMSNorm, QK-Norm, DyT, Residual Mix/Add & Softcap",
    accent: "#22d3ee",
    archetypeId: "norm_stabilizer",
  },
  {
    id: "rlhf_losses",
    title: "Fine-Tuning, RLHF & Losses",
    subtitle: "Cross-Entropy, SFT, DPO, PPO, KL, GAE & Reward/Value Heads",
    accent: "#fb7185",
    archetypeId: "objective_beacon",
  },
  {
    id: "activations",
    title: "Activations & Math Primitives",
    subtitle: "ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax & Scalar Math",
    accent: "#60a5fa",
    archetypeId: "signal_modulator",
  },
  {
    id: "terminals_custom",
    title: "Subgraphs, I/O & Data Sources",
    subtitle: "Input/Output Terminals, Dataset Sources, Custom & Subgraphs",
    accent: "#4ade80",
    archetypeId: "portal_input",
  },
];

export default function NodePickerModal({
  open,
  onClose,
  builtins,
  descriptions,
  onSelectBuiltin,
  onAddCustom,
  onAddOverride,
  onAddSubgraph,
}: {
  open: boolean;
  onClose: () => void;
  builtins: NeuronDefData[];
  descriptions: Record<string, string>;
  onSelectBuiltin: (ndef: NeuronDefData) => void;
  onAddCustom: () => void;
  onAddOverride: () => void;
  onAddSubgraph: () => void;
}) {
  const [activeCategory, setActiveCategory] = useState<string>("attention");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [hoveredItem, setHoveredItem] = useState<NeuronDefData | null>(null);
  const [keepOpen, setKeepOpen] = useState<boolean>(false);
  const [animWaveKey, setAnimWaveKey] = useState<number>(0);
  const searchInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (open) {
      setSearchQuery("");
      setAnimWaveKey((k) => k + 1);
      window.setTimeout(() => {
        searchInputRef.current?.focus();
      }, 60);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  const categorizedBuiltins = useMemo(() => {
    const groups: Record<string, NeuronDefData[]> = {};
    for (const cat of PICKER_CATEGORIES) {
      groups[cat.id] = [];
    }
    for (const b of builtins) {
      const arch = resolveNodeArchetype(b.name, b, b.name === "input", b.name === "output");
      if (!groups[arch.categoryId]) {
        groups[arch.categoryId] = [];
      }
      groups[arch.categoryId].push(b);
    }
    return groups;
  }, [builtins]);

  const displayedNodes = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (q.length > 0) {
      return builtins.filter((b) => {
        const desc = (descriptions[b.name] ?? "").toLowerCase();
        return (
          b.name.toLowerCase().includes(q) ||
          (b.module_type ?? "").toLowerCase().includes(q) ||
          desc.includes(q)
        );
      });
    }
    return categorizedBuiltins[activeCategory] ?? [];
  }, [builtins, categorizedBuiltins, activeCategory, searchQuery, descriptions]);

  useEffect(() => {
    if (displayedNodes.length > 0) {
      setHoveredItem(displayedNodes[0]);
    } else {
      setHoveredItem(null);
    }
  }, [displayedNodes]);

  if (!open) return null;

  const currentCatMeta =
    PICKER_CATEGORIES.find((c) => c.id === activeCategory) ?? PICKER_CATEGORIES[0];

  const inspectedNode = hoveredItem ?? displayedNodes[0] ?? null;
  const inspectedArchetype = inspectedNode
    ? resolveNodeArchetype(
        inspectedNode.name,
        inspectedNode,
        inspectedNode.name === "input",
        inspectedNode.name === "output",
      )
    : NODE_ARCHETYPES[currentCatMeta.archetypeId];

  const handlePickNode = (ndef: NeuronDefData) => {
    onSelectBuiltin(ndef);
    if (!keepOpen) {
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/80 backdrop-blur-md px-4 py-6"
      onClick={onClose}
    >
      {/* 3D Perspective Container */}
      <div
        className="relative flex h-[84vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-cyan-400/40 bg-slate-950/95 text-gray-100 shadow-[0_25px_90px_rgba(0,0,0,0.95),0_0_50px_rgba(0,240,255,0.18)]"
        style={{
          clipPath:
            "polygon(18px 0%, calc(100% - 18px) 0%, 100% 18px, 100% calc(100% - 18px), calc(100% - 18px) 100%, 18px 100%, 0% calc(100% - 18px), 0% 18px)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top Laser Rim Accent */}
        <div
          className="h-1 w-full"
          style={{
            background: `linear-gradient(90deg, transparent, ${currentCatMeta.accent}, #f59e0b, #00f0ff, transparent)`,
          }}
        />

        {/* Header Bar with Search & Structural Node Quick-Spawners */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-cyan-500/25 bg-slate-900/75 px-6 py-3.5">
          <div className="flex items-center gap-3">
            <div
              className="flex h-10 w-10 items-center justify-center rounded-lg border bg-slate-950 shadow-inner"
              style={{
                borderColor: `${currentCatMeta.accent}88`,
                boxShadow: `0 0 18px ${currentCatMeta.accent}33`,
              }}
            >
              <SemanticNodeGlyph
                archetypeId={currentCatMeta.archetypeId}
                neuronName="attention"
                primary={currentCatMeta.accent}
                secondary="#f8fafc"
                size={28}
              />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="font-mono text-sm font-extrabold uppercase tracking-wider text-white">
                  Holographic Node Arsenal
                </h2>
                <span className="rounded border border-cyan-400/40 bg-cyan-950/60 px-2 py-0.5 font-mono text-[9px] font-bold uppercase tracking-widest text-cyan-300">
                  {builtins.length} MODULES ONLINE
                </span>
              </div>
              <p className="text-[11px] text-gray-400">
                Select a semantic subsystem on the left or search to materialize neural stages.
              </p>
            </div>
          </div>

          {/* Search Input */}
          <div className="relative flex-1 max-w-md min-w-[220px]">
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setAnimWaveKey((k) => k + 1);
              }}
              placeholder="Search neuron, module, or architecture (e.g. mamba, lora, jepa, attention)..."
              className="w-full rounded-lg border border-cyan-500/40 bg-slate-950/90 px-3.5 py-2 font-mono text-xs text-cyan-100 placeholder-gray-500 shadow-[inset_0_0_12px_rgba(0,240,255,0.1)] outline-none focus:border-cyan-300 focus:shadow-[0_0_18px_rgba(0,240,255,0.28)]"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-gray-400 hover:text-white"
              >
                ✕
              </button>
            )}
          </div>

          {/* Right Controls: Multi-drop pin & Close */}
          <div className="flex items-center gap-2">
            <label className="flex cursor-pointer items-center gap-1.5 rounded-lg border border-gray-800 bg-slate-950/80 px-2.5 py-1.5 font-mono text-[10px] text-gray-300 hover:border-cyan-500/40">
              <input
                type="checkbox"
                checked={keepOpen}
                onChange={(e) => setKeepOpen(e.target.checked)}
                className="accent-cyan-400"
              />
              <span>MULTI-DROP</span>
            </label>
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-gray-700 bg-slate-900 px-3 py-1.5 font-mono text-xs text-gray-300 hover:border-rose-500/60 hover:text-rose-200"
            >
              ✕ ESC
            </button>
          </div>
        </div>

        {/* Structural Builders Quick Ribbon */}
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-cyan-500/15 bg-slate-950/90 px-6 py-2">
          <div className="flex items-center gap-2">
            <span className="font-mono text-[9px] font-bold uppercase tracking-widest text-gray-500">
              STRUCTURAL CONSTRUCTS:
            </span>
            <button
              type="button"
              onClick={() => {
                onAddCustom();
                if (!keepOpen) onClose();
              }}
              className="flex items-center gap-1.5 rounded border border-sky-500/40 bg-sky-950/40 px-2.5 py-1 font-mono text-[10px] font-bold text-sky-200 transition-all hover:border-sky-400 hover:bg-sky-900/60 hover:shadow-[0_0_14px_rgba(56,189,248,0.3)]"
            >
              <span>⚡ + Custom Python Node</span>
            </button>
            <button
              type="button"
              onClick={() => {
                onAddOverride();
                if (!keepOpen) onClose();
              }}
              className="flex items-center gap-1.5 rounded border border-purple-500/40 bg-purple-950/40 px-2.5 py-1 font-mono text-[10px] font-bold text-purple-200 transition-all hover:border-purple-400 hover:bg-purple-900/60 hover:shadow-[0_0_14px_rgba(168,85,247,0.3)]"
            >
              <span>🎛 + Scalar Override</span>
            </button>
            <button
              type="button"
              onClick={() => {
                onAddSubgraph();
                if (!keepOpen) onClose();
              }}
              className="flex items-center gap-1.5 rounded border border-amber-500/40 bg-amber-950/40 px-2.5 py-1 font-mono text-[10px] font-bold text-amber-200 transition-all hover:border-amber-400 hover:bg-amber-900/60 hover:shadow-[0_0_14px_rgba(245,158,11,0.3)]"
            >
              <span>◈ + Subgraph Citadel</span>
            </button>
          </div>
          <div className="font-mono text-[10px] text-cyan-300/75">
            Click any module pod below to deploy directly onto the active graph
          </div>
        </div>

        {/* Main 3-Column Cyber Deck: Left Category Pillars | Center Animated Node Fan-Out | Right 3D Blueprint Inspector */}
        <div className="flex flex-1 min-h-0 overflow-hidden">
          {/* Column 1: Semantic Category Pods */}
          <div className="w-72 shrink-0 overflow-y-auto border-r border-cyan-500/20 bg-slate-950/80 p-3 space-y-1.5">
            <div className="px-2 pb-1 font-mono text-[9px] font-bold uppercase tracking-widest text-cyan-400/80">
              SUBSYSTEM MATRIX
            </div>
            {PICKER_CATEGORIES.map((cat) => {
              const count = categorizedBuiltins[cat.id]?.length ?? 0;
              const isSelected = !searchQuery && activeCategory === cat.id;
              return (
                <button
                  key={cat.id}
                  type="button"
                  onClick={() => {
                    setSearchQuery("");
                    setActiveCategory(cat.id);
                    setAnimWaveKey((k) => k + 1);
                  }}
                  className="group relative flex w-full items-center gap-3 rounded-xl border p-2.5 text-left transition-all duration-200"
                  style={{
                    borderColor: isSelected ? cat.accent : "rgba(30, 41, 59, 0.8)",
                    background: isSelected
                      ? `linear-gradient(135deg, ${cat.accent}22 0%, rgba(15,23,42,0.95) 100%)`
                      : "rgba(15, 23, 42, 0.45)",
                    boxShadow: isSelected
                      ? `0 6px 20px rgba(0,0,0,0.6), 0 0 18px ${cat.accent}28, inset 0 1px 0 rgba(255,255,255,0.12)`
                      : "none",
                    transform: isSelected ? "translateX(4px)" : "none",
                  }}
                >
                  <div
                    className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border bg-slate-950/90"
                    style={{
                      borderColor: `${cat.accent}66`,
                    }}
                  >
                    <SemanticNodeGlyph
                      archetypeId={cat.archetypeId}
                      neuronName="relu"
                      primary={cat.accent}
                      secondary="#ffffff"
                      size={24}
                    />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-1">
                      <span
                        className="truncate font-mono text-xs font-bold"
                        style={{ color: isSelected ? "#ffffff" : "#e2e8f0" }}
                      >
                        {cat.title}
                      </span>
                      <span
                        className="rounded px-1.5 py-0.2 font-mono text-[9px] font-bold"
                        style={{
                          backgroundColor: `${cat.accent}22`,
                          color: cat.accent,
                        }}
                      >
                        {count}
                      </span>
                    </div>
                    <div className="truncate text-[10px] text-gray-400">
                      {cat.subtitle}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Column 2: Staggered Animated Node Pods Grid */}
          <div className="flex-1 overflow-y-auto p-5 bg-[radial-gradient(circle_at_top,rgba(0,240,255,0.06),transparent_70%)]">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span
                  className="h-2 w-2 rotate-45"
                  style={{ backgroundColor: currentCatMeta.accent }}
                />
                <h3 className="font-mono text-xs font-bold uppercase tracking-widest text-white">
                  {searchQuery
                    ? `SEARCH RESULTS FOR "${searchQuery.toUpperCase()}"`
                    : currentCatMeta.title}
                </h3>
              </div>
              <span className="font-mono text-[10px] text-gray-400">
                {displayedNodes.length} {displayedNodes.length === 1 ? "NODE" : "NODES"} READY
              </span>
            </div>

            {displayedNodes.length === 0 ? (
              <div className="flex h-64 flex-col items-center justify-center rounded-xl border border-dashed border-gray-800 text-center">
                <div className="font-mono text-xs text-gray-400">
                  No neural modules matched &ldquo;{searchQuery}&rdquo;
                </div>
                <button
                  type="button"
                  onClick={() => setSearchQuery("")}
                  className="mt-2 rounded border border-cyan-500/40 bg-cyan-950/40 px-3 py-1 font-mono text-xs text-cyan-300"
                >
                  Reset Filter
                </button>
              </div>
            ) : (
              <div
                key={animWaveKey}
                className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3"
              >
                {displayedNodes.map((b, idx) => {
                  const arch = resolveNodeArchetype(
                    b.name,
                    b,
                    b.name === "input",
                    b.name === "output",
                  );
                  const isHovered = inspectedNode?.id === b.id;
                  const desc = descriptions[b.name] || "Specialized neural operator stage";

                  return (
                    <button
                      key={b.id}
                      type="button"
                      onClick={() => handlePickNode(b)}
                      onMouseEnter={() => setHoveredItem(b)}
                      className="cyber-node-card-enter group relative flex flex-col justify-between p-3.5 text-left transition-all duration-200"
                      style={{
                        animationDelay: `${Math.min(idx * 22, 360)}ms`,
                        clipPath: arch.clipPath,
                        background: isHovered
                          ? `linear-gradient(145deg, rgba(10, 22, 46, 0.96) 0%, ${arch.primary}24 100%)`
                          : "linear-gradient(145deg, rgba(7, 14, 30, 0.92) 0%, rgba(3, 8, 20, 0.96) 100%)",
                        border: `1px solid ${isHovered ? arch.primary : `${arch.primary}44`}`,
                        boxShadow: isHovered
                          ? `0 12px 26px rgba(0,0,0,0.85), 0 0 20px ${arch.primary}40, inset 0 1px 0 rgba(255,255,255,0.18)`
                          : "0 6px 16px rgba(0,0,0,0.65)",
                        transform: isHovered ? "translateY(-3px) scale(1.02)" : "none",
                      }}
                    >
                      <div>
                        {/* Card Top Header */}
                        <div className="mb-2 flex items-center justify-between gap-2">
                          <span
                            className="font-mono text-[8px] font-bold uppercase tracking-widest"
                            style={{ color: arch.secondary }}
                          >
                            {arch.badgeText}
                          </span>
                          <span
                            className="rounded-sm border px-1.5 py-0.2 font-mono text-[8px]"
                            style={{
                              borderColor: `${arch.primary}44`,
                              backgroundColor: `${arch.primary}15`,
                              color: arch.primary,
                            }}
                          >
                            {b.input_ports.length} IN ▸ {b.output_ports.length} OUT
                          </span>
                        </div>

                        {/* Center Row: Glyph + Name */}
                        <div className="flex items-center gap-3">
                          <div
                            className="flex h-11 w-11 shrink-0 items-center justify-center rounded border bg-slate-950/80"
                            style={{ borderColor: `${arch.primary}33` }}
                          >
                            <SemanticNodeGlyph
                              archetypeId={arch.id}
                              neuronName={b.name}
                              primary={arch.primary}
                              secondary={arch.secondary}
                              size={32}
                            />
                          </div>
                          <div className="min-w-0 flex-1">
                            <div
                              className="truncate font-mono text-xs font-extrabold uppercase tracking-wide text-white"
                              style={{
                                textShadow: isHovered ? `0 0 10px ${arch.primary}` : "none",
                              }}
                            >
                              {b.name}
                            </div>
                            <div className="line-clamp-2 mt-0.5 text-[10px] leading-snug text-gray-400">
                              {desc}
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Port Pill Preview Footer */}
                      <div className="mt-2.5 flex items-center justify-between border-t border-slate-800/80 pt-1.5 font-mono text-[9px] text-gray-400">
                        <span className="truncate max-w-[45%] text-cyan-200/80">
                          in: {b.input_ports.map((p) => p.name).join(", ") || "—"}
                        </span>
                        <span className="truncate max-w-[45%] text-amber-200/80">
                          out: {b.output_ports.map((p) => p.name).join(", ") || "—"}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Column 3: Right-Side 3D Holographic Node Blueprint Inspector */}
          {inspectedNode && (
            <div className="hidden w-72 shrink-0 flex-col justify-between border-l border-cyan-500/20 bg-slate-950/90 p-4 xl:flex">
              <div className="space-y-4">
                <div className="flex items-center justify-between border-b border-cyan-500/20 pb-2">
                  <span className="font-mono text-[9px] font-bold uppercase tracking-widest text-cyan-400">
                    BLUEPRINT INSPECTOR
                  </span>
                  <span
                    className="font-mono text-[9px] font-bold"
                    style={{ color: inspectedArchetype.primary }}
                  >
                    {inspectedArchetype.badgeText}
                  </span>
                </div>

                {/* 3D Floating Hologram Pedestal */}
                <div
                  className="relative flex flex-col items-center justify-center rounded-xl border p-5"
                  style={{
                    clipPath: inspectedArchetype.clipPath,
                    borderColor: `${inspectedArchetype.primary}66`,
                    background: `radial-gradient(circle at center, ${inspectedArchetype.primary}24 0%, rgba(2,8,20,0.95) 75%)`,
                    boxShadow: `inset 0 0 25px ${inspectedArchetype.primary}22`,
                  }}
                >
                  <SemanticNodeGlyph
                    archetypeId={inspectedArchetype.id}
                    neuronName={inspectedNode.name}
                    primary={inspectedArchetype.primary}
                    secondary={inspectedArchetype.secondary}
                    size={64}
                  />
                  <div
                    className="mt-3 font-mono text-sm font-extrabold uppercase tracking-wider text-white"
                    style={{ textShadow: `0 0 12px ${inspectedArchetype.primary}` }}
                  >
                    {inspectedNode.name}
                  </div>
                  <div className="mt-1 text-center text-[11px] leading-relaxed text-gray-300">
                    {descriptions[inspectedNode.name] || "Neural operator stage"}
                  </div>
                </div>

                {/* Port Specifications */}
                <div className="space-y-2.5 font-mono text-[10px]">
                  <div>
                    <div className="mb-1 text-[9px] font-bold uppercase tracking-wider text-gray-400">
                      INPUT CHANNELS ({inspectedNode.input_ports.length})
                    </div>
                    <div className="space-y-1">
                      {inspectedNode.input_ports.map((p) => (
                        <div
                          key={p.name}
                          className="flex items-center justify-between rounded border border-slate-800 bg-slate-900/80 px-2 py-1"
                        >
                          <span className="font-bold text-cyan-300">{p.name}</span>
                          <span className="text-[9px] text-gray-400">{p.dtype}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <div className="mb-1 text-[9px] font-bold uppercase tracking-wider text-gray-400">
                      OUTPUT CHANNELS ({inspectedNode.output_ports.length})
                    </div>
                    <div className="space-y-1">
                      {inspectedNode.output_ports.map((p) => (
                        <div
                          key={p.name}
                          className="flex items-center justify-between rounded border border-slate-800 bg-slate-900/80 px-2 py-1"
                        >
                          <span className="font-bold text-amber-300">{p.name}</span>
                          <span className="text-[9px] text-gray-400">{p.dtype}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <button
                type="button"
                onClick={() => handlePickNode(inspectedNode)}
                className="w-full rounded-lg py-2.5 font-mono text-xs font-extrabold uppercase tracking-wider text-slate-950 transition-all hover:brightness-110"
                style={{
                  backgroundColor: inspectedArchetype.primary,
                  boxShadow: `0 0 24px ${inspectedArchetype.primary}66`,
                }}
              >
                + Deploy {inspectedNode.name}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
