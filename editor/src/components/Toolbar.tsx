import React, { useCallback, useEffect, useState } from "react";
import { api, type NeuronDefData } from "../api/client";
import { selectBreadcrumbs, useGraphStore } from "../store/graphStore";
import { normalizeGraph } from "../store/graphUtils";
import { useSettingsStore } from "../store/settingsStore";
import NodePickerModal from "./NodePickerModal";

const DESCRIPTIONS: Record<string, string> = {
  sigmoid: "Sigmoid activation function",
  relu: "Rectified Linear Unit activation",
  tanh_neuron: "Hyperbolic tangent activation",
  threshold: "Binary threshold function (0 or 1)",
  identity: "Passes input unchanged",
  negate: "Negates the input",
  add: "Adds two inputs",
  multiply: "Multiplies two inputs",
  gaussian: "Gaussian activation function",
  log: "Natural logarithm function",
  leaky_relu: "Rectified Linear Unit with a small slope for negative inputs",
  prelu: "Parametric Rectified Linear Unit",
  relu6: "ReLU capped at 6.0",
  elu: "Exponential Linear Unit",
  selu: "Scaled Exponential Linear Unit",
  gelu: "Gaussian Error Linear Unit",
  silu: "Sigmoid Linear Unit (also known as Swish)",
  mish: "Self Regularized Non-Monotonic Activation Function",
  softplus: "Smooth approximation to the ReLU function",
  softsign: "Continuous alternative to tanh: x / (1 + |x|)",
  hard_sigmoid: "Piece-wise linear approximation of sigmoid",
  hard_tanh: "Piece-wise linear approximation of tanh",
  hard_swish: "Piece-wise linear approximation of swish",
  softmax_2: "Softmax probabilities for 2 inputs",
  logsoftmax_2: "Log-Softmax values for 2 inputs",
  input: "Graph input terminal",
  output: "Graph output terminal",
  token_embedding: "Token embedding stage with tied-weight output",
  pos_embedding: "Learned absolute positional embedding stage",
  embedding_add: "Elementwise sum of token and positional embeddings",
  tensor_add: "Elementwise tensor residual addition stage",
  tensor_mul: "Elementwise tensor gating / multiplication stage",
  tensor_scale: "Constant scalar multiplier stage for tensor streams",
  tensor_relu: "Tensor-level ReLU activation stage",
  tensor_relu2: "Tensor-level squared ReLU (ReLU²) activation stage",
  tensor_gelu: "Tensor-level Gaussian Error Linear Unit stage",
  tensor_silu: "Tensor-level SiLU / Swish activation stage",
  tensor_sigmoid: "Tensor-level Sigmoid gating activation stage",
  linear: "Trainable linear projection stage",
  fp8_linear: "FP8 (E4M3) quantized linear projection stage",
  mx_linear: "Microscaled MXFP4 block-scaled linear projection stage",
  bit_linear_158: "Ternary b1.58 ({-1, 0, +1}) quantized linear projection",
  rms_norm: "Tensor RMSNorm stage",
  layer_norm: "Standard LayerNorm normalization stage",
  qk_norm: "Per-head Query/Key normalization stage for attention stability",
  dyt: "Dynamic Tanh (DyT) norm-free stabilization stage",
  reshape_heads: "Reshape hidden states into attention heads",
  merge_heads: "Merge attention heads back into model width",
  repeat_kv: "Repeat grouped KV heads to match query head count",
  rotary_embedding: "Apply rotary positional embedding to Q and K tensors",
  qk_gain: "Learned per-head scale for the query stream",
  scaled_dot_product_attention: "Causal scaled dot-product attention primitive",
  sliding_window_attention: "Local sliding-window causal attention primitive",
  differential_attention: "Differential attention stage subtracting noise-canceling dual heads",
  multi_head_latent_attention: "DeepSeek-style Multi-Head Latent Attention (MLA) with compressed KV",
  native_sparse_attention: "Hardware-aligned Native Sparse Attention (NSA) stage",
  dflash_attention: "Muse Glimmer DFlash speculative cross/self-attention stage",
  cross_attention: "Encoder-decoder multi-head cross-attention stage",
  bidirectional_attention: "Full bidirectional self-attention stage (diffusion / encoder)",
  mamba_ssm: "Selective State-Space Model (Mamba SSM) sequence mixer",
  ttt_layer: "Test-Time Training (TTT) inner-loop fast-weight sequence stage",
  hnet_chunk: "H-Net dynamic hierarchical sequence chunking stage",
  hnet_dechunk: "H-Net hierarchical sequence dechunking reconstruction stage",
  residual_mix: "Learns a mix of the current stream and the original embedding stream",
  causal_self_attention: "Causal self-attention stage with RoPE and grouped KV heads",
  residual_add: "Residual add with a learned per-channel scale",
  hyper_connection: "Multi-stream Hyper-Connection (mHC) residual router",
  mlp_relu2: "ReLU-squared MLP stage",
  tied_lm_head: "Language-model head that reuses embedding weights",
  lm_head: "Untied language-model head",
  logit_softcap: "Softcap the logits before loss",
  token_cross_entropy: "Token cross-entropy loss stage",
  z_loss_cross_entropy: "Cross-entropy loss with router/logit z-loss regularization",
  moe_gate: "Top-K Mixture-of-Experts router gate with load-balance auxiliary loss",
  auxfree_moe_gate: "Auxiliary-loss-free MoE router with dynamic expert bias updates",
  expert_dispatch: "Sparse expert MLP dispatch and weighted combine stage",
  semantic_moe_gate: "Semantic-space MoE router conditioned on latent chunk representations",
  jepa_latent_pool: "JEPA context/target latent pooling stage",
  jepa_predictor: "Latent world-model predictor mapping context latents to target latents",
  jepa_loss: "Joint-Embedding Predictive Architecture latent representation loss",
  jepa_ema_stop_grad: "EMA target-encoder stop-gradient projection stage",
  jepa_mask: "Span/block masking generator for JEPA latent pretraining",
  kv_pca_compress: "Low-rank PCA key/value cache compression stage",
  kv_quant_compress: "Sub-byte quantized key/value cache compression stage",
  dataset_source: "Project dataset token stream source ([batch, seq_len])",
  semantic_data_source: "Semantic multi-scale token & span dataset source",
  muse_glimmer_vision_tower: "Muse Glimmer 3D-patch multimodal vision encoder tower",
  muse_glimmer_media_projector: "Multimodal vision-to-language MLP projector stage",
  muse_glimmer_media_fusion: "In-place image/video placeholder embedding fusion stage",
  // ── Fine-tuning operators ─────────────────────────────────────────
  lora_linear: "LoRA linear: frozen base weight + trainable low-rank delta",
  nf4_linear: "qLoRA linear: nf4-packed base weight + trainable LoRA delta",
  masked_token_cross_entropy: "Response-only cross-entropy (SFT loss with prompt mask)",
  reference_forward: "Frozen reference model forward (DPO / RLHF ref logits)",
  sft_dataset_source: "SFT data source: (tokens, targets, loss_mask) triples",
  sequence_logp: "Sum of target logprobs under logits, masked per-token",
  dpo_pairwise_loss: "Direct Preference Optimization loss (sigmoid/hinge/ipo)",
  dpo_dataset_source: "DPO pair source: (chosen, rejected) × (tokens, targets, mask)",
  reward_head: "Scalar reward head (linear-to-1 pooled over the sequence)",
  preference_bce_loss: "Bradley-Terry preference loss for reward-model training",
  value_head: "Per-token value head for PPO critic",
  ppo_clipped_loss: "PPO clipped policy-and-value loss",
  kl_penalty: "Shapes rewards with a KL penalty against the reference model",
  reward_forward: "Frozen reward-model forward (PPO)",
  ppo_rollout_source: "Rollout buffer source emitted by PPOTrainer",
  gae_compute: "Generalized Advantage Estimation over a rollout",
};

export default function Toolbar() {
  const projectId = useGraphStore((state) => state.projectId);
  const sessionId = useGraphStore((state) => state.sessionId);
  const hydrateSession = useGraphStore((state) => state.hydrateSession);
  const builtins = useGraphStore((state) => state.builtins);
  const setBuiltins = useGraphStore((state) => state.setBuiltins);
  const addBuiltinNode = useGraphStore((state) => state.addBuiltinNode);
  const addCustomNode = useGraphStore((state) => state.addCustomNode);
  const addSubgraphNode = useGraphStore((state) => state.addSubgraphNode);
  const rootGraph = useGraphStore((state) => state.rootGraph);
  const breadcrumbs = useGraphStore(selectBreadcrumbs);
  const setPath = useGraphStore((state) => state.setPath);
  const particleGraphEditorEnabled = useSettingsStore(
    (state) => state.particleGraphEditorEnabled,
  );
  const toggleParticleGraphEditor = useSettingsStore(
    (state) => state.toggleParticleGraphEditor,
  );
  const viewFullGraph = useSettingsStore((state) => state.viewFullGraph);
  const setViewFullGraph = useSettingsStore((state) => state.setViewFullGraph);
  const toggleViewFullGraph = useSettingsStore(
    (state) => state.toggleViewFullGraph,
  );
  const setSettingsOpen = useSettingsStore((state) => state.setSettingsOpen);
  const activePanelDialog = useSettingsStore(
    (state) => state.activePanelDialog,
  );
  const setActivePanelDialog = useSettingsStore(
    (state) => state.setActivePanelDialog,
  );
  const openRightPanelDialog = useSettingsStore(
    (state) => state.openRightPanelDialog,
  );
  const isTraining = useGraphStore((state) => state.isTraining);
  const [showNodePicker, setShowNodePicker] = useState(false);

  useEffect(() => {
    api.getBuiltins().then(setBuiltins).catch(() => {});
  }, [setBuiltins]);

  const onAddBuiltin = useCallback(
    (ndef: NeuronDefData) => {
      addBuiltinNode(ndef);
    },
    [addBuiltinNode]
  );

  const onAddCustom = useCallback(() => {
    addCustomNode();
  }, [addCustomNode]);

  const onAddOverride = useCallback(() => {
    const ndef: NeuronDefData = {
        id: "override-" + Date.now().toString(36),
        name: "override",
        kind: "function",
        input_ports: [{ name: "x", range: null, precision: null, dtype: "float" }],
        output_ports: [{ name: "y", range: null, precision: null, dtype: "float" }],
        source_code: "def override(x):\n    return 0.0\n",
        subgraph: null,
        module_type: "",
        module_config: {},
        module_state: "",
        input_aliases: [],
        output_aliases: [],
        variant_ref: null,
    };
    addBuiltinNode(ndef);
  }, [addBuiltinNode]);

  const onAddSubgraph = useCallback(() => {
    addSubgraphNode();
  }, [addSubgraphNode]);

  const [gptType, setGptType] = useState("nanogpt");

  const onAddGPT = useCallback(() => {
    if (projectId && sessionId) {
      api.applyGPTTemplate(projectId, sessionId, { name: "gpt", config: { preset: gptType } })
        .then((detail) => {
          hydrateSession({
            projectId,
            sessionId,
            graph: detail.graph,
            revision: detail.revision,
          });
        })
        .catch(() => {});
      return;
    }

    // Fallback for non-session contexts: preview payload + local insertion.
    api.buildGPTTemplate({ name: "gpt", config: { preset: gptType } }).then((template) => {
      addBuiltinNode(template.node_def);
      if (template.extra_nodes?.length) {
        for (const node of template.extra_nodes) {
          addBuiltinNode(node.neuron_def, {
            x: node.position?.[0] ?? 0,
            y: node.position?.[1] ?? 0,
          });
        }
      }
    }).catch(() => {});
  }, [addBuiltinNode, gptType, hydrateSession, projectId, sessionId]);

  const onSave = useCallback(() => {
    const data = JSON.stringify(rootGraph, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "neuralfn-network.json";
    a.click();
    URL.revokeObjectURL(url);
  }, [rootGraph]);

  const onLoad = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      const text = await file.text();
      try {
        const data = JSON.parse(text);
        useGraphStore.getState().setRootGraph(normalizeGraph(data, data.name ?? "Loaded graph"));
      } catch {
        alert("Invalid graph file");
      }
    };
    input.click();
  }, []);

  return (
    <div
      className={
        particleGraphEditorEnabled
          ? "cyber-toolbar-deck"
          : "border-b border-gray-800 bg-gray-900"
      }
    >
      <div className="flex flex-wrap items-center gap-2 px-3 py-2">
        <span
          className={`mr-2 text-sm font-bold tracking-wide ${
            particleGraphEditorEnabled
              ? "font-mono uppercase text-cyan-300 drop-shadow-[0_0_8px_rgba(0,240,255,0.5)]"
              : "text-blue-400"
          }`}
        >
          NeuralFn
        </span>

        {/* Primary Add Node Button -> Launches the Animated Categorized Holographic Node Picker */}
        <button
          type="button"
          onClick={() => setShowNodePicker(true)}
          className={`flex items-center gap-1.5 rounded px-3 py-1 text-xs font-bold transition-all ${
            particleGraphEditorEnabled
              ? "border border-cyan-400/70 bg-cyan-500/20 font-mono uppercase tracking-wider text-cyan-100 shadow-[0_0_16px_rgba(0,240,255,0.3)] hover:bg-cyan-400/30 hover:shadow-[0_0_22px_rgba(0,240,255,0.5)]"
              : "border border-blue-500/50 bg-blue-600 text-white hover:bg-blue-500"
          }`}
          title="Open categorized Holographic Node Arsenal"
        >
          <span>✦ + Add Node</span>
          <span className="rounded bg-black/40 px-1.5 py-0.2 font-mono text-[9px] text-cyan-300">
            {builtins.length}
          </span>
        </button>

        <button
          onClick={onAddCustom}
          className={
            particleGraphEditorEnabled
              ? "rounded border border-sky-500/35 bg-slate-900/90 px-2.5 py-1 font-mono text-xs text-sky-200 hover:border-sky-400/70 hover:bg-sky-950/60"
              : "bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs px-2 py-1 rounded"
          }
        >
          + Custom Node
        </button>
        <button
          onClick={onAddOverride}
          className={
            particleGraphEditorEnabled
              ? "rounded border border-purple-500/40 bg-purple-950/50 px-2.5 py-1 font-mono text-xs text-purple-200 hover:border-purple-400/70 hover:bg-purple-900/60"
              : "bg-purple-900 hover:bg-purple-800 text-purple-100 text-xs px-2 py-1 rounded"
          }
        >
          + Override
        </button>
        <button
          onClick={onAddSubgraph}
          className={
            particleGraphEditorEnabled
              ? "rounded border border-amber-500/45 bg-amber-950/50 px-2.5 py-1 font-mono text-xs text-amber-200 hover:border-amber-400/80 hover:bg-amber-900/60"
              : "bg-amber-900 hover:bg-amber-800 text-amber-100 text-xs px-2 py-1 rounded"
          }
        >
          + Subgraph
        </button>
        <div
          className={
            particleGraphEditorEnabled
              ? "flex items-center rounded border border-emerald-500/45 bg-emerald-950/60 shadow-[0_0_14px_rgba(16,185,129,0.18)]"
              : "flex bg-emerald-900 rounded items-center"
          }
        >
          <button
            onClick={onAddGPT}
            className="hover:bg-emerald-800/80 text-emerald-100 text-xs px-2.5 py-1 rounded-l font-mono"
          >
            + GPT Template
          </button>
          <select
            className="bg-emerald-950 text-emerald-100 text-[10px] uppercase font-bold tracking-wider py-1 px-1 rounded-r border-l border-emerald-800 outline-none cursor-pointer"
            value={gptType}
            onChange={(e) => setGptType(e.target.value)}
          >
            <option value="nanogpt">NanoGPT</option>
            <option value="gpt2">GPT-2</option>
            <option value="llama">LLaMA</option>
            <option value="muse_glimmer">Muse Glimmer 30B</option>
            <option value="moe">MoE</option>
            <option value="llama_fast">LLaMA Fast</option>
            <option value="mixllama_fast">MixLLaMA Fast</option>
            <option value="jamba">Jamba Hybrid</option>
            <option value="ternary_b158">Ternary b1.58</option>
            <option value="seq2seq">Seq2Seq MoE</option>
            <option value="diffusion">DiffLLaMA</option>
            <option value="ttt_llama">TTT LLaMA</option>
            <option value="llm_jepa">LLM JEPA</option>
            <option value="dense_jepa_evo">Dense JEPA Evo</option>
            <option value="moe_jepa_evo">MoE JEPA Evo</option>
            <option value="hnet_lm">H-Net LM</option>
            <option value="universal_llama">Universal LLaMA</option>
            <option value="llama_megakernel">LLaMA Megakernel</option>
            <option value="kv_pca_llama">KV PCA LLaMA</option>
            <option value="jepa_semantic_hybrid">JEPA Semantic Hybrid</option>
            <option value="semantic_router_moe">Semantic Router MoE</option>
            <option value="semantic_dense_jepa_evo">Semantic Dense JEPA Evo</option>
            <option value="semantic_moe_jepa_evo">Semantic MoE JEPA Evo</option>
            <optgroup label="Dense GPT-2 pretraining variants">
              <option value="gpt2_zloss">GPT-2 + z-loss</option>
              <option value="gpt2_softcap">GPT-2 + logit softcap</option>
              <option value="gpt2_qknorm">GPT-2 + QK-norm</option>
              <option value="gpt2_diff">GPT-2 + differential attention</option>
              <option value="gpt2_stable">GPT-2 Stable (z-loss + QK-norm)</option>
            </optgroup>
            <optgroup label="Frontier (recognizable SOTA)">
              <option value="deepseek_v3">DeepSeek-V3 (MLA + auxfree MoE)</option>
              <option value="deepseek_v4">DeepSeek-V4 (CSA/NSA + mHC + FP8)</option>
              <option value="gemma3">Gemma-3 (sliding window + GeGLU)</option>
              <option value="diff_transformer">Differential Transformer</option>
              <option value="qwen3_longctx">Qwen3 Long-Context (YaRN)</option>
              <option value="longctx_sparse_llama">Long-Context Sparse LLaMA (NSA)</option>
              <option value="modern_norms_llama">Modern-Norms LLaMA (DyT + QK-norm + GeGLU)</option>
            </optgroup>
            <optgroup label="Blackwell precision">
              <option value="fp8_llama">FP8 LLaMA (E4M3)</option>
              <option value="mxfp4_llama">MXFP4 LLaMA (microscaled)</option>
            </optgroup>
            <optgroup label="NeuralFn crosses">
              <option value="auxfree_moe_jepa_evo">Auxfree MoE JEPA Evo</option>
              <option value="diff_semantic_moe_jepa_evo">Differential Semantic MoE JEPA Evo</option>
              <option value="dyt_geglu_semantic_dense_jepa_evo">DyT+GeGLU Semantic Dense JEPA Evo</option>
            </optgroup>
            <optgroup label="Modernized (+QK-norm +RoPE/YaRN +GeGLU +auxfree)">
              <option value="nanogpt_modern">NanoGPT (modern)</option>
              <option value="gpt2_modern">GPT-2 (modern)</option>
              <option value="llama_modern">LLaMA (modern)</option>
              <option value="moe_modern">MoE (modern)</option>
              <option value="jamba_modern">Jamba Hybrid (modern)</option>
              <option value="ternary_b158_modern">Ternary b1.58 (modern)</option>
              <option value="seq2seq_modern">Seq2Seq MoE (modern)</option>
              <option value="diffusion_modern">DiffLLaMA (modern)</option>
              <option value="ttt_llama_modern">TTT LLaMA (modern)</option>
              <option value="llm_jepa_modern">LLM JEPA (modern)</option>
              <option value="dense_jepa_evo_modern">Dense JEPA Evo (modern)</option>
              <option value="moe_jepa_evo_modern">MoE JEPA Evo (modern)</option>
              <option value="hnet_lm_modern">H-Net LM (modern)</option>
              <option value="universal_llama_modern">Universal LLaMA (modern)</option>
              <option value="kv_pca_llama_modern">KV PCA LLaMA (modern)</option>
              <option value="jepa_semantic_hybrid_modern">JEPA Semantic Hybrid (modern)</option>
              <option value="semantic_router_moe_modern">Semantic Router MoE (modern)</option>
              <option value="semantic_dense_jepa_evo_modern">Semantic Dense JEPA Evo (modern)</option>
              <option value="semantic_moe_jepa_evo_modern">Semantic MoE JEPA Evo (modern)</option>
            </optgroup>
          </select>
        </div>

        <div className="flex items-center gap-1 ml-2 text-[10px] text-gray-400 overflow-x-auto font-mono">
          {breadcrumbs.map((crumb, idx) => {
            const isParent = idx < breadcrumbs.length - 1;
            return (
              <React.Fragment key={`${crumb.id}-${idx}`}>
                {idx > 0 && <span className="text-cyan-500/60">/</span>}
                <button
                  onClick={() => {
                    if (viewFullGraph) setViewFullGraph(false);
                    setPath(crumb.path);
                  }}
                  title={
                    isParent
                      ? `Collapse subgraph back to ${crumb.label} ⤡`
                      : `Current graph: ${crumb.label}`
                  }
                  className={`flex items-center gap-1 whitespace-nowrap rounded px-1.5 py-0.5 transition-colors ${
                    !isParent
                      ? particleGraphEditorEnabled
                        ? "border border-cyan-500/40 bg-cyan-950/60 text-cyan-200"
                        : "bg-gray-800 text-gray-200"
                      : particleGraphEditorEnabled
                      ? "border border-amber-500/30 bg-amber-950/30 text-amber-200/90 hover:border-amber-400/70 hover:bg-amber-900/50 hover:text-amber-100"
                      : "hover:bg-gray-800 hover:text-gray-200"
                  }`}
                >
                  {isParent && <span className="text-[9px] opacity-80">⤡</span>}
                  <span>{crumb.label}</span>
                </button>
              </React.Fragment>
            );
          })}
        </div>

        <div className="ml-auto flex items-center gap-1.5 flex-wrap">
          {/* Dialog Launchers for Left (Variants), Bottom (Training), and Right (Inspector / Chat) Panels */}
          <button
            type="button"
            onClick={() =>
              setActivePanelDialog(activePanelDialog === "left" ? null : "left")
            }
            className={
              particleGraphEditorEnabled
                ? `flex items-center gap-1 rounded border px-2 py-1 font-mono text-xs transition-all ${
                    activePanelDialog === "left"
                      ? "border-amber-400/90 bg-amber-500/25 text-amber-100 shadow-[0_0_14px_rgba(251,191,36,0.3)]"
                      : "border-amber-500/35 bg-slate-950/90 text-amber-200/90 hover:border-amber-400/70 hover:bg-amber-950/50"
                  }`
                : "rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-amber-200 hover:bg-gray-700"
            }
            title="Open Variant Matrix / Library Dialog (Left Panel)"
          >
            <span>◈ Variants</span>
            <span className="rounded bg-black/40 px-1 py-0.2 font-mono text-[9px] text-amber-300">
              {Object.keys(rootGraph.variant_library ?? {}).length}
            </span>
          </button>

          <button
            type="button"
            onClick={() =>
              setActivePanelDialog(
                activePanelDialog === "bottom" ? null : "bottom",
              )
            }
            className={
              particleGraphEditorEnabled
                ? `flex items-center gap-1 rounded border px-2 py-1 font-mono text-xs transition-all ${
                    activePanelDialog === "bottom" || isTraining
                      ? "border-cyan-400/90 bg-cyan-500/25 text-cyan-100 shadow-[0_0_14px_rgba(0,240,255,0.35)]"
                      : "border-cyan-500/35 bg-slate-950/90 text-cyan-200/90 hover:border-cyan-400/70 hover:bg-cyan-950/50"
                  }`
                : "rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-cyan-200 hover:bg-gray-700"
            }
            title="Open Training & Telemetry Dialog (Bottom Panel)"
          >
            <span className={isTraining ? "animate-pulse text-emerald-300" : ""}>
              ⚡ Training
            </span>
          </button>

          <button
            type="button"
            onClick={() => openRightPanelDialog("inspector")}
            className={
              particleGraphEditorEnabled
                ? `flex items-center gap-1 rounded border px-2 py-1 font-mono text-xs transition-all ${
                    activePanelDialog === "right"
                      ? "border-sky-400/90 bg-sky-500/25 text-sky-100 shadow-[0_0_14px_rgba(56,189,248,0.3)]"
                      : "border-sky-500/35 bg-slate-950/90 text-sky-200/90 hover:border-sky-400/70 hover:bg-sky-950/50"
                  }`
                : "rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-sky-200 hover:bg-gray-700"
            }
            title="Open Node Inspector, Graph Config & Code Dialog (Right Panel)"
          >
            <span>◧ Inspector</span>
          </button>

          <button
            type="button"
            onClick={() => openRightPanelDialog("chat")}
            className={
              particleGraphEditorEnabled
                ? "flex items-center gap-1 rounded border border-indigo-400/50 bg-indigo-950/70 px-2 py-1 font-mono text-xs text-indigo-200 hover:border-indigo-400/90 hover:bg-indigo-900/70"
                : "rounded border border-indigo-700/60 bg-indigo-900/70 px-2 py-1 text-xs text-indigo-100 hover:bg-indigo-800"
            }
            title="Open Model Chat Dialog (Right Panel)"
          >
            <span>💬 Chat</span>
          </button>

          <button
            type="button"
            onClick={toggleViewFullGraph}
            className={
              particleGraphEditorEnabled
                ? `flex items-center gap-1.5 rounded border px-2.5 py-1 font-mono text-xs font-bold transition-all ${
                    viewFullGraph
                      ? "border-emerald-400/90 bg-emerald-500/25 text-emerald-200 shadow-[0_0_16px_rgba(16,185,129,0.35)]"
                      : "border-cyan-500/35 bg-slate-950/90 text-cyan-200/90 hover:border-cyan-400/70 hover:bg-cyan-950/60"
                  }`
                : `flex items-center gap-1.5 rounded border px-2.5 py-1 text-xs font-medium transition-all ${
                    viewFullGraph
                      ? "border-emerald-500/70 bg-emerald-900/70 text-emerald-100"
                      : "border-gray-700 bg-gray-800 text-gray-200 hover:bg-gray-700"
                  }`
            }
            title="Expand all nested subgraphs into a single unified full-architecture graph view"
          >
            <span>⊞</span>
            <span>{viewFullGraph ? "Full Graph: ON" : "View Full Graph"}</span>
          </button>

          <div className="flex items-center rounded border border-cyan-500/35 bg-slate-950/90">
            <button
              type="button"
              onClick={toggleParticleGraphEditor}
              className={`flex items-center gap-1.5 rounded-l px-2.5 py-1 text-xs font-medium transition-all ${
                particleGraphEditorEnabled
                  ? "bg-cyan-500/25 text-cyan-200 shadow-[0_0_14px_rgba(0,240,255,0.3)]"
                  : "text-cyan-300/80 hover:bg-cyan-950/60 hover:text-cyan-200"
              }`}
              title="Toggle Volumetric 3D Cyber Mode"
            >
              <span>✦</span>
              <span>{particleGraphEditorEnabled ? "Cyber 3D: ON" : "Cyber 3D"}</span>
            </button>
            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              className="border-l border-cyan-500/30 px-2 py-1 text-xs text-cyan-300/80 hover:bg-cyan-950/60 hover:text-cyan-100 rounded-r"
              title="Open Holographic Particle Engine & Feature Flag Settings"
            >
              ⚙
            </button>
          </div>
          <button
            onClick={onSave}
            className={
              particleGraphEditorEnabled
                ? "rounded border border-cyan-500/30 bg-slate-900/90 px-2.5 py-1 font-mono text-xs text-cyan-100 hover:border-cyan-400/60 hover:bg-cyan-950/60"
                : "bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs px-2 py-1 rounded"
            }
          >
            Save
          </button>
          <button
            onClick={onLoad}
            className={
              particleGraphEditorEnabled
                ? "rounded border border-cyan-500/30 bg-slate-900/90 px-2.5 py-1 font-mono text-xs text-cyan-100 hover:border-cyan-400/60 hover:bg-cyan-950/60"
                : "bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs px-2 py-1 rounded"
            }
          >
            Load
          </button>
        </div>
      </div>

      {/* Categorized Animated 3D Holographic Node Picker Modal */}
      <NodePickerModal
        open={showNodePicker}
        onClose={() => setShowNodePicker(false)}
        builtins={builtins}
        descriptions={DESCRIPTIONS}
        onSelectBuiltin={onAddBuiltin}
        onAddCustom={onAddCustom}
        onAddOverride={onAddOverride}
        onAddSubgraph={onAddSubgraph}
      />
    </div>
  );
}
