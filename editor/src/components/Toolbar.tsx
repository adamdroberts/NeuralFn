import React, { useCallback, useEffect, useState } from "react";
import { useReactFlow } from "@xyflow/react";
import { api, type NeuronDefData } from "../api/client";
import { selectBreadcrumbs, useGraphStore } from "../store/graphStore";
import { normalizeGraph } from "../store/graphUtils";
import NeuronIcon from "./NeuronIcon";

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
  linear: "Trainable linear projection stage",
  rms_norm: "Tensor RMSNorm stage",
  reshape_heads: "Reshape hidden states into attention heads",
  merge_heads: "Merge attention heads back into model width",
  repeat_kv: "Repeat grouped KV heads to match query head count",
  rotary_embedding: "Apply rotary positional embedding to Q and K tensors",
  qk_gain: "Learned per-head scale for the query stream",
  scaled_dot_product_attention: "Causal scaled dot-product attention primitive",
  residual_mix: "Learns a mix of the current stream and the original embedding stream",
  causal_self_attention: "Causal self-attention stage with RoPE and grouped KV heads",
  residual_add: "Residual add with a learned per-channel scale",
  mlp_relu2: "ReLU-squared MLP stage",
  tied_lm_head: "Language-model head that reuses embedding weights",
  lm_head: "Untied language-model head",
  logit_softcap: "Softcap the logits before loss",
  token_cross_entropy: "Token cross-entropy loss stage",
};

export default function Toolbar() {
  const { screenToFlowPosition } = useReactFlow();

  const builtins = useGraphStore((state) => state.builtins);
  const setBuiltins = useGraphStore((state) => state.setBuiltins);
  const addBuiltinNode = useGraphStore((state) => state.addBuiltinNode);
  const addCustomNode = useGraphStore((state) => state.addCustomNode);
  const addSubgraphNode = useGraphStore((state) => state.addSubgraphNode);
  const updateActiveGraphSettings = useGraphStore((state) => state.updateActiveGraphSettings);
  const mergeVariantLibrary = useGraphStore((state) => state.mergeVariantLibrary);
  const rootGraph = useGraphStore((state) => state.rootGraph);
  const breadcrumbs = useGraphStore(selectBreadcrumbs);
  const setPath = useGraphStore((state) => state.setPath);
  const [showLibrary, setShowLibrary] = useState(true);
  const [hoveredNeuron, setHoveredNeuron] = useState<{ builtin: NeuronDefData, rect: DOMRect } | null>(null);

  useEffect(() => {
    api.getBuiltins().then(setBuiltins).catch(() => {});
  }, [setBuiltins]);

  const onAddBuiltin = useCallback(
    (ndef: NeuronDefData, e: React.MouseEvent) => {
      const pos = screenToFlowPosition({ x: e.clientX, y: e.clientY + 100 });
      addBuiltinNode(ndef, pos);
    },
    [addBuiltinNode, screenToFlowPosition]
  );

  const onAddCustom = useCallback((e: React.MouseEvent) => {
    const pos = screenToFlowPosition({ x: e.clientX, y: e.clientY + 100 });
    addCustomNode(pos);
  }, [addCustomNode, screenToFlowPosition]);

  const onAddOverride = useCallback((e: React.MouseEvent) => {
    const pos = screenToFlowPosition({ x: e.clientX, y: e.clientY + 100 });
    const ndef: NeuronDefData = {
        id: "override-" + Date.now().toString(36),
        name: "override",
        kind: "function",
        input_ports: [{ name: "x", range: [-1, 1], precision: 0.1, dtype: "float" }],
        output_ports: [{ name: "y", range: [-1, 1], precision: 0.1, dtype: "float" }],
        source_code: "def override(x):\n    return 0.0\n",
        subgraph: null,
        module_type: "",
        module_config: {},
        module_state: "",
        input_aliases: [],
        output_aliases: [],
        variant_ref: null,
    };
    addBuiltinNode(ndef, pos);
  }, [addBuiltinNode, screenToFlowPosition]);

  const onAddSubgraph = useCallback((e: React.MouseEvent) => {
    const pos = screenToFlowPosition({ x: e.clientX, y: e.clientY + 100 });
    addSubgraphNode(pos);
  }, [addSubgraphNode, screenToFlowPosition]);

  const onAddGPT = useCallback((e: React.MouseEvent) => {
    const pos = screenToFlowPosition({ x: e.clientX, y: e.clientY + 100 });
    api.buildGPTTemplate({ name: "gpt" }).then((template) => {
      mergeVariantLibrary(template.variant_library);
      updateActiveGraphSettings(template.graph_settings);
      addBuiltinNode(template.node_def, pos);
    }).catch(() => {});
  }, [addBuiltinNode, mergeVariantLibrary, screenToFlowPosition, updateActiveGraphSettings]);

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
    <div className="border-b border-gray-800 bg-gray-900">
      <div className="flex items-center gap-2 px-3 py-2">
        <span className="text-sm font-bold text-blue-400 mr-2">NeuralFn</span>

        <button
          onClick={() => setShowLibrary(!showLibrary)}
          className="bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs px-2 py-1 rounded"
        >
          {showLibrary ? "Hide Library" : "Show Library"}
        </button>
        <button
          onClick={onAddCustom}
          className="bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs px-2 py-1 rounded"
        >
          + Custom Node
        </button>
        <button
          onClick={onAddOverride}
          className="bg-purple-900 hover:bg-purple-800 text-purple-100 text-xs px-2 py-1 rounded"
        >
          + Override
        </button>
        <button
          onClick={onAddSubgraph}
          className="bg-amber-900 hover:bg-amber-800 text-amber-100 text-xs px-2 py-1 rounded"
        >
          + Subgraph
        </button>
        <button
          onClick={onAddGPT}
          className="bg-emerald-900 hover:bg-emerald-800 text-emerald-100 text-xs px-2 py-1 rounded"
        >
          + GPT Template
        </button>

        <div className="flex items-center gap-1 ml-2 text-[10px] text-gray-400 overflow-x-auto">
          {breadcrumbs.map((crumb, idx) => (
            <React.Fragment key={`${crumb.id}-${idx}`}>
              {idx > 0 && <span>/</span>}
              <button
                onClick={() => setPath(crumb.path)}
                className={`whitespace-nowrap rounded px-1 py-0.5 ${
                  idx === breadcrumbs.length - 1
                    ? "bg-gray-800 text-gray-200"
                    : "hover:bg-gray-800 hover:text-gray-200"
                }`}
              >
                {crumb.label}
              </button>
            </React.Fragment>
          ))}
        </div>

        <div className="ml-auto flex gap-2">
          <button
            onClick={onSave}
            className="bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs px-2 py-1 rounded"
          >
            Save
          </button>
          <button
            onClick={onLoad}
            className="bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs px-2 py-1 rounded"
          >
            Load
          </button>
        </div>
      </div>

      {showLibrary && (
        <div className="flex gap-1 px-3 pb-2 flex-wrap relative">
          {builtins.map((b) => {
            const desc = DESCRIPTIONS[b.name] ? `${DESCRIPTIONS[b.name]}\n` : "";
            
            return (
              <button
                key={b.id}
                onClick={(e) => {
                  onAddBuiltin(b, e);
                  setHoveredNeuron(null);
                }}
                onMouseEnter={(e) => {
                  setHoveredNeuron({ builtin: b, rect: e.currentTarget.getBoundingClientRect() });
                }}
                onMouseLeave={() => setHoveredNeuron(null)}
                className="bg-gray-800 hover:bg-blue-900 border border-gray-700 text-gray-300 text-[10px] px-2 py-0.5 rounded flex items-center justify-center whitespace-nowrap transition-colors"
                title={`${desc}(${b.input_ports.length} in / ${b.output_ports.length} out)`}
              >
                <span>{b.name}</span>
                <NeuronIcon name={b.name} />
              </button>
            );
          })}
        </div>
      )}

      {/* Floating Tooltip */}
      {hoveredNeuron && (
        <div 
          className="fixed flex flex-col items-center bg-gray-950 border border-gray-700 px-4 py-3 rounded-lg shadow-2xl z-[9999] pointer-events-none"
          style={{
            top: hoveredNeuron.rect.bottom + 8,
            left: Math.max(10, Math.min(window.innerWidth - 180, hoveredNeuron.rect.left + hoveredNeuron.rect.width / 2 - 85)),
            width: 170
          }}
        >
          <div className="text-xs font-bold text-blue-300 mb-2">{hoveredNeuron.builtin.name}</div>
          <div className="bg-gray-900 rounded p-1">
            <NeuronIcon name={hoveredNeuron.builtin.name} expanded={true} />
          </div>
          <div className="text-[9px] text-gray-500 mt-2 text-center whitespace-normal leading-snug">
            {DESCRIPTIONS[hoveredNeuron.builtin.name] || 'Custom function'}
          </div>
        </div>
      )}
    </div>
  );
}
