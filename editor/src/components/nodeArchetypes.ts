import type { NeuronDefData } from "../api/client";

export type NodeArchetypeId =
  | "attention_prism"
  | "sequence_wave"
  | "latent_tesseract"
  | "moe_router"
  | "tensor_matrix"
  | "lora_adapter"
  | "norm_stabilizer"
  | "residual_combiner"
  | "memory_vault"
  | "objective_beacon"
  | "signal_modulator"
  | "citadel_subgraph"
  | "portal_input"
  | "portal_output";

export type ParticleSwarmGeometry =
  | "tri_prism"
  | "lissajous_wave"
  | "vortex_funnel"
  | "satellite_clusters"
  | "matrix_lattice"
  | "stabilizer_rings"
  | "cylinder_stack"
  | "radial_pulsar"
  | "harmonic_orbit"
  | "hex_shield"
  | "emitter_beam";

export interface NodeArchetypeSpec {
  id: NodeArchetypeId;
  categoryId: string;
  categoryLabel: string;
  badgeText: string;
  /** Primary neon hex color */
  primary: string;
  primaryRgb: [number, number, number];
  /** Secondary neon hex color */
  secondary: string;
  secondaryRgb: [number, number, number];
  /** CSS clip-path for the node silhouette */
  clipPath: string;
  /** Border radius fallback when clipPath is subtle */
  borderRadius: string;
  /** Particle formation geometry rendered on the 60fps canvas */
  swarmGeometry: ParticleSwarmGeometry;
  /** Aspect / width hint */
  minWidth: number;
}

export const NODE_ARCHETYPES: Record<NodeArchetypeId, NodeArchetypeSpec> = {
  attention_prism: {
    id: "attention_prism",
    categoryId: "attention",
    categoryLabel: "Attention & Sequence Mixers",
    badgeText: "ATTN // PRISM",
    primary: "#00f0ff",
    primaryRgb: [0, 240, 255],
    secondary: "#38bdf8",
    secondaryRgb: [56, 189, 248],
    clipPath:
      "polygon(14px 0%, calc(100% - 14px) 0%, 100% 18px, 100% calc(100% - 18px), calc(100% - 14px) 100%, 14px 100%, 0% calc(100% - 18px), 0% 18px)",
    borderRadius: "10px",
    swarmGeometry: "tri_prism",
    minWidth: 196,
  },
  sequence_wave: {
    id: "sequence_wave",
    categoryId: "attention",
    categoryLabel: "Attention & Sequence Mixers",
    badgeText: "SSM // WAVEGUIDE",
    primary: "#10b981",
    primaryRgb: [16, 185, 129],
    secondary: "#34d399",
    secondaryRgb: [52, 211, 153],
    clipPath:
      "polygon(18px 0%, calc(100% - 8px) 0%, 100% 12px, 100% calc(100% - 12px), calc(100% - 18px) 100%, 8px 100%, 0% calc(100% - 12px), 0% 12px)",
    borderRadius: "22px",
    swarmGeometry: "lissajous_wave",
    minWidth: 190,
  },
  latent_tesseract: {
    id: "latent_tesseract",
    categoryId: "jepa_latent",
    categoryLabel: "JEPA, Latent & World Models",
    badgeText: "LATENT // TESSERACT",
    primary: "#c084fc",
    primaryRgb: [192, 132, 252],
    secondary: "#e879f9",
    secondaryRgb: [232, 121, 249],
    clipPath:
      "polygon(10px 0%, calc(100% - 10px) 0%, 100% 22%, calc(100% - 12px) 100%, 12px 100%, 0% 22%)",
    borderRadius: "8px",
    swarmGeometry: "vortex_funnel",
    minWidth: 188,
  },
  moe_router: {
    id: "moe_router",
    categoryId: "moe",
    categoryLabel: "MoE Routing & Experts",
    badgeText: "MOE // MULTIPLEX",
    primary: "#f59e0b",
    primaryRgb: [245, 158, 11],
    secondary: "#fb923c",
    secondaryRgb: [251, 146, 60],
    clipPath:
      "polygon(0% 14px, 14px 0%, calc(100% - 18px) 0%, 100% 24px, 100% calc(100% - 24px), calc(100% - 18px) 100%, 14px 100%, 0% calc(100% - 14px))",
    borderRadius: "8px",
    swarmGeometry: "satellite_clusters",
    minWidth: 194,
  },
  tensor_matrix: {
    id: "tensor_matrix",
    categoryId: "linear_mlp",
    categoryLabel: "Linear, MLP & Quantized",
    badgeText: "TENSOR // LATTICE",
    primary: "#818cf8",
    primaryRgb: [129, 140, 248],
    secondary: "#38bdf8",
    secondaryRgb: [56, 189, 248],
    clipPath:
      "polygon(12px 0%, 100% 0%, 100% calc(100% - 12px), calc(100% - 12px) 100%, 0% 100%, 0% 12px)",
    borderRadius: "6px",
    swarmGeometry: "matrix_lattice",
    minWidth: 180,
  },
  lora_adapter: {
    id: "lora_adapter",
    categoryId: "linear_mlp",
    categoryLabel: "Linear, MLP & Quantized",
    badgeText: "ADAPTER // LOW-RANK",
    primary: "#f472b6",
    primaryRgb: [244, 114, 182],
    secondary: "#fb7185",
    secondaryRgb: [251, 113, 133],
    clipPath:
      "polygon(0% 10px, 10px 0%, calc(100% - 10px) 0%, 100% 10px, 100% calc(100% - 10px), calc(100% - 10px) 100%, 10px 100%, 0% calc(100% - 10px))",
    borderRadius: "8px",
    swarmGeometry: "matrix_lattice",
    minWidth: 184,
  },
  norm_stabilizer: {
    id: "norm_stabilizer",
    categoryId: "norms_residuals",
    categoryLabel: "Norms, Residuals & Gating",
    badgeText: "NORM // GYRO-GATE",
    primary: "#22d3ee",
    primaryRgb: [34, 211, 238],
    secondary: "#a5f3fc",
    secondaryRgb: [165, 243, 252],
    clipPath:
      "polygon(16px 0%, calc(100% - 16px) 0%, 100% 50%, calc(100% - 16px) 100%, 16px 100%, 0% 50%)",
    borderRadius: "14px",
    swarmGeometry: "stabilizer_rings",
    minWidth: 172,
  },
  residual_combiner: {
    id: "residual_combiner",
    categoryId: "norms_residuals",
    categoryLabel: "Norms, Residuals & Gating",
    badgeText: "STREAM // JUNCTION",
    primary: "#38bdf8",
    primaryRgb: [56, 189, 248],
    secondary: "#2dd4bf",
    secondaryRgb: [45, 212, 191],
    clipPath:
      "polygon(0% 12px, 12px 0%, calc(100% - 16px) 0%, 100% 50%, calc(100% - 16px) 100%, 12px 100%, 0% calc(100% - 12px))",
    borderRadius: "10px",
    swarmGeometry: "stabilizer_rings",
    minWidth: 178,
  },
  memory_vault: {
    id: "memory_vault",
    categoryId: "embeddings_kv",
    categoryLabel: "Embeddings, RoPE & KV Memory",
    badgeText: "MEM // VAULT-BANK",
    primary: "#2dd4bf",
    primaryRgb: [45, 212, 191],
    secondary: "#14b8a6",
    secondaryRgb: [20, 184, 166],
    clipPath:
      "polygon(8px 0%, calc(100% - 8px) 0%, 100% 8px, 100% calc(100% - 8px), calc(100% - 8px) 100%, 8px 100%, 0% calc(100% - 8px), 0% 8px)",
    borderRadius: "6px",
    swarmGeometry: "cylinder_stack",
    minWidth: 186,
  },
  objective_beacon: {
    id: "objective_beacon",
    categoryId: "rlhf_losses",
    categoryLabel: "Fine-Tuning, RLHF & Losses",
    badgeText: "OBJ // CRITIC-CORE",
    primary: "#fb7185",
    primaryRgb: [251, 113, 133],
    secondary: "#f43f5e",
    secondaryRgb: [244, 63, 94],
    clipPath:
      "polygon(16px 0%, calc(100% - 16px) 0%, 100% 16px, 100% calc(100% - 16px), calc(100% - 16px) 100%, 16px 100%, 0% calc(100% - 16px), 0% 16px)",
    borderRadius: "8px",
    swarmGeometry: "radial_pulsar",
    minWidth: 190,
  },
  signal_modulator: {
    id: "signal_modulator",
    categoryId: "activations",
    categoryLabel: "Activations & Math Primitives",
    badgeText: "ACT // WAVE-SHAPER",
    primary: "#60a5fa",
    primaryRgb: [96, 165, 250],
    secondary: "#a78bfa",
    secondaryRgb: [167, 139, 250],
    clipPath:
      "polygon(0% 10px, 10px 0%, 100% 0%, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0% 100%)",
    borderRadius: "8px",
    swarmGeometry: "harmonic_orbit",
    minWidth: 168,
  },
  citadel_subgraph: {
    id: "citadel_subgraph",
    categoryId: "terminals_custom",
    categoryLabel: "Subgraphs, I/O & Data Sources",
    badgeText: "MACRO // CITADEL",
    primary: "#fbbf24",
    primaryRgb: [251, 191, 36],
    secondary: "#f59e0b",
    secondaryRgb: [245, 158, 11],
    clipPath:
      "polygon(16px 0%, calc(100% - 16px) 0%, 100% 16px, 100% calc(100% - 16px), calc(100% - 16px) 100%, 16px 100%, 0% calc(100% - 16px), 0% 16px)",
    borderRadius: "10px",
    swarmGeometry: "hex_shield",
    minWidth: 204,
  },
  portal_input: {
    id: "portal_input",
    categoryId: "terminals_custom",
    categoryLabel: "Subgraphs, I/O & Data Sources",
    badgeText: "SOURCE // EMITTER",
    primary: "#4ade80",
    primaryRgb: [74, 222, 128],
    secondary: "#22c55e",
    secondaryRgb: [34, 197, 94],
    clipPath:
      "polygon(0% 0%, calc(100% - 16px) 0%, 100% 50%, calc(100% - 16px) 100%, 0% 100%, 8px 50%)",
    borderRadius: "6px",
    swarmGeometry: "emitter_beam",
    minWidth: 168,
  },
  portal_output: {
    id: "portal_output",
    categoryId: "terminals_custom",
    categoryLabel: "Subgraphs, I/O & Data Sources",
    badgeText: "SINK // NEXUS",
    primary: "#fb923c",
    primaryRgb: [251, 146, 60],
    secondary: "#f97316",
    secondaryRgb: [249, 115, 22],
    clipPath:
      "polygon(14px 0%, 100% 0%, calc(100% - 8px) 50%, 100% 100%, 14px 100%, 0% 50%)",
    borderRadius: "6px",
    swarmGeometry: "emitter_beam",
    minWidth: 172,
  },
};

export function resolveNodeArchetype(
  label: string,
  neuronDef: NeuronDefData,
  isInput?: boolean,
  isOutput?: boolean,
): NodeArchetypeSpec {
  const key = `${label} ${neuronDef.name} ${neuronDef.module_type ?? ""}`.toLowerCase();

  if (neuronDef.kind === "subgraph") {
    return NODE_ARCHETYPES.citadel_subgraph;
  }

  if (
    isInput ||
    neuronDef.name === "input" ||
    key.includes("dataset_source") ||
    key.includes("rollout_source") ||
    key.includes("data_source")
  ) {
    return NODE_ARCHETYPES.portal_input;
  }

  if (
    isOutput ||
    neuronDef.name === "output" ||
    key.includes("lm_head") ||
    key.includes("denoise_head")
  ) {
    return NODE_ARCHETYPES.portal_output;
  }

  if (
    key.includes("mamba") ||
    key.includes("ssm") ||
    key.includes("causal_chunk_state") ||
    key.includes("ttt_") ||
    key.includes("universal_transformer") ||
    key.includes("act_halt") ||
    key.includes("act_weighted")
  ) {
    return NODE_ARCHETYPES.sequence_wave;
  }

  if (
    key.includes("attention") ||
    key.includes("sdpa") ||
    key.includes("reshape_heads") ||
    key.includes("merge_heads") ||
    key.includes("repeat_kv") ||
    key.includes("qk_gain")
  ) {
    return NODE_ARCHETYPES.attention_prism;
  }

  if (
    key.includes("latent") ||
    key.includes("jepa") ||
    key.includes("semantic_projector") ||
    key.includes("semantic_hasher") ||
    key.includes("semantic_chunk") ||
    key.includes("byte_patch") ||
    key.includes("mask_scheduler") ||
    key.includes("random_timesteps") ||
    key.includes("muse_glimmer")
  ) {
    return NODE_ARCHETYPES.latent_tesseract;
  }

  if (
    key.includes("moe") ||
    key.includes("router") ||
    key.includes("topk_route") ||
    key.includes("expert_") ||
    key.includes("auxfree") ||
    key.includes("broadcast_chunk_routes") ||
    key.includes("broadcast_expert_routes")
  ) {
    return NODE_ARCHETYPES.moe_router;
  }

  if (
    key.includes("lora") ||
    key.includes("nf4") ||
    key.includes("randmap")
  ) {
    return NODE_ARCHETYPES.lora_adapter;
  }

  if (
    key.includes("linear") ||
    key.includes("mlp") ||
    key.includes("swiglu") ||
    key.includes("geglu") ||
    key.includes("reglu") ||
    key.includes("solu")
  ) {
    return NODE_ARCHETYPES.tensor_matrix;
  }

  if (
    key.includes("residual") ||
    key.includes("manifold_hyper") ||
    key.includes("pair_batch")
  ) {
    return NODE_ARCHETYPES.residual_combiner;
  }

  if (
    key.includes("norm") ||
    key.includes("dyt") ||
    key.includes("softcap") ||
    key.includes("dropout") ||
    key.includes("tensor_scale") ||
    key.includes("loss_scale")
  ) {
    return NODE_ARCHETYPES.norm_stabilizer;
  }

  if (
    key.includes("embedding") ||
    key.includes("rotary") ||
    key.includes("kv_cache") ||
    key.includes("kv_pca") ||
    key.includes("kv_quant")
  ) {
    return NODE_ARCHETYPES.memory_vault;
  }

  if (
    key.includes("loss") ||
    key.includes("cross_entropy") ||
    key.includes("dpo") ||
    key.includes("ppo") ||
    key.includes("kl_penalty") ||
    key.includes("reward") ||
    key.includes("value_head") ||
    key.includes("policy_logits") ||
    key.includes("gae") ||
    key.includes("logp") ||
    key.includes("reference_forward")
  ) {
    return NODE_ARCHETYPES.objective_beacon;
  }

  return NODE_ARCHETYPES.signal_modulator;
}
