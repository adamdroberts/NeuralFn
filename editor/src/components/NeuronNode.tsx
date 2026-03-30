import React, { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { type NeuronNodeData, useGraphStore } from "../store/graphStore";

function NeuronNodeInner({ id, data, selected }: NodeProps & { data: NeuronNodeData }) {
  const { label, neuronDef, isInput, isOutput } = data;
  const inputs = neuronDef.input_ports;
  const outputs = neuronDef.output_ports;
  const isSubgraph = neuronDef.kind === "subgraph";
  const isModule = neuronDef.kind === "module";
  const trainingMethod = neuronDef.subgraph?.training_method;
  const variantRef = neuronDef.variant_ref;
  const updateNodeData = useGraphStore((state) => state.updateNodeData);

  const roleTag = isInput ? "IN" : isOutput ? "OUT" : null;
  const isOverride = neuronDef.source_code?.includes("def override(x):") ?? false;
  const overrideMatch = neuronDef.source_code?.match(/return\s+([-\d.]+)/);
  const overrideValue = overrideMatch ? parseFloat(overrideMatch[1]) : 0;

  const handleOverrideChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    if (!isNaN(val)) {
      updateNodeData(id, {
        neuronDef: {
          ...neuronDef,
          source_code: `def override(x):\n    return ${val}\n`
        }
      });
    }
  };

  return (
    <div
      className={`rounded-lg border px-3 py-2 min-w-[140px] shadow-lg transition-colors ${
        selected
          ? isSubgraph
            ? "border-amber-400 bg-gray-800"
            : isModule
            ? "border-emerald-400 bg-gray-800"
            : "border-blue-400 bg-gray-800"
          : isSubgraph
            ? "border-amber-700 bg-gray-900"
            : isModule
            ? "border-emerald-700 bg-gray-900"
            : "border-gray-700 bg-gray-900"
      }`}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <span
          className={`text-xs font-bold truncate ${
            isSubgraph ? "text-amber-300" : isModule ? "text-emerald-300" : "text-blue-300"
          }`}
        >
          {label}
        </span>
        <div className="flex items-center gap-1">
          {isModule && (
            <span className="text-[10px] font-mono px-1 rounded bg-emerald-950 text-emerald-200 uppercase">
              {neuronDef.module_type}
            </span>
          )}
          {trainingMethod && (
            <span className="text-[10px] font-mono px-1 rounded bg-amber-950 text-amber-200 uppercase">
              {trainingMethod.slice(0, 3)}
            </span>
          )}
          {variantRef && (
            <span className="text-[10px] font-mono px-1 rounded bg-gray-950 text-amber-200 uppercase overflow-hidden text-ellipsis max-w-[80px]">
              {variantRef.family === "moe" ? "MoE" : 
               variantRef.family === "llama" ? "LLaMA" : 
               variantRef.family === "gpt2" ? "GPT2" : 
               variantRef.family === "nanogpt" ? "NanoGPT" :
               String(variantRef.family)}@{String(variantRef.version)}
            </span>
          )}
          {roleTag && (
            <span className="text-[10px] font-mono px-1 rounded bg-blue-900 text-blue-200">
              {roleTag}
            </span>
          )}
          {(typeof neuronDef.module_config?.num_heads === 'number' && typeof neuronDef.module_config?.num_kv_heads === 'number') && (
            <span className="text-[10px] font-mono px-1 rounded bg-purple-950 text-purple-200">
               {neuronDef.module_config.num_heads === neuronDef.module_config.num_kv_heads ? 'MHA' : (neuronDef.module_config.num_kv_heads === 1 ? 'MQA' : 'GQA')}
            </span>
          )}
          {neuronDef.module_type?.includes("kv_cache") && (
            <span className="text-[10px] font-mono px-1 rounded bg-cyan-950 text-cyan-200">
               cache:on
            </span>
          )}
        </div>
      </div>

      {/* Preset Diff View Snippet */}
      {isSubgraph && variantRef && (
        <div className="mb-2 text-[10px] text-amber-200/80 bg-black/20 p-1 rounded border border-amber-900/30 flex justify-between">
            <span>Preset: {String(variantRef.family)}</span>
            <span className="text-gray-500 hover:text-white cursor-pointer transition-colors" title="Diff View">D</span>
        </div>
      )}
      
      {/* MoE Telemetry Mini-Chart */}
      {(neuronDef.module_type === "expert_dispatch" || variantRef?.family === "moe") && (
         <div className="mb-2 p-1 bg-black/40 rounded flex flex-col gap-0.5" title="Expert Utilization Telemetry">
            <span className="text-[8px] text-gray-500 uppercase tracking-wider">Expert Telemetry</span>
            <div className="flex gap-0.5 h-3 items-end">
               {[0.8, 0.4, 0.9, 0.2, 0.6, 0.3, 0.7, 0.5].map((val, i) => (
                  <div key={i} className="flex-1 bg-purple-500/50 hover:bg-purple-400 rounded-t-sm transition-all" style={{height: `${val * 100}%`}}></div>
               ))}
            </div>
         </div>
      )}

      {isSubgraph && (
        <div className="mb-2 text-[10px] text-amber-200/80">
          {variantRef ? "Double-click to open shared variant" : "Double-click to open subgraph"}
        </div>
      )}

      {isOverride && (
        <div className="mb-2 flex items-center justify-between text-[10px] text-gray-400 bg-gray-950 px-2 py-1 rounded">
          <span>Value:</span>
          <input
            type="number"
            value={overrideValue}
            onChange={handleOverrideChange}
            className="w-16 bg-gray-800 text-gray-200 px-1 py-0.5 rounded border border-gray-700 text-right no-spinners"
            step="0.1"
          />
        </div>
      )}

      <div className="flex justify-between gap-4 text-[10px] text-gray-400">
        <div className="flex flex-col gap-1">
          {inputs.map((p, i) => {
            let shape = "";
            if (p.name === "q" || p.name === "v" || p.name === "heads") shape = "[B,H,T,D]";
            else if (p.name === "k") shape = "[B,KV_H,T,D]";
            else if (p.dtype === "tensor" && p.name !== "k" && p.name !== "q" && p.name !== "v") shape = "[B,T,C]";
            else if (p.dtype === "tokens") shape = "[B,T]";

            return (
            <div key={p.name} className="relative flex items-center group">
              <Handle
                type="target"
                position={Position.Left}
                id={`in-${i}`}
                className="!w-2.5 !h-2.5 !bg-green-400 !border-green-600 shadow-[0_0_5px_rgba(74,222,128,0.5)]"
                style={{ top: "auto", position: "relative", left: -6 }}
              />
              <span className="ml-1 flex items-center gap-1" title={`[${p.range[0]}, ${p.range[1]}] p=${p.precision}`}>
                {p.name}
                {shape && <span className="text-[8px] opacity-0 group-hover:opacity-100 transition-opacity bg-black/60 px-1 rounded text-cyan-300 pointer-events-none">{shape}</span>}
              </span>
            </div>
            );
          })}
        </div>

        <div className="flex flex-col gap-1 items-end">
          {outputs.map((p, i) => {
            let shape = "";
            if (p.name === "q_rot" || p.name === "k_rot" || p.name === "heads") shape = "[B,H,T,D]";
            else if (p.dtype === "tensor" && p.name !== "heads" && p.name !== "q_rot" && p.name !== "k_rot") shape = "[B,T,C]";

            return (
            <div key={p.name} className="relative flex items-center group">
              <span className="mr-1 flex items-center gap-1 flex-row-reverse" title={`[${p.range[0]}, ${p.range[1]}] p=${p.precision}`}>
                {p.name}
                {shape && <span className="text-[8px] opacity-0 group-hover:opacity-100 transition-opacity bg-black/60 px-1 rounded text-orange-300 pointer-events-none">{shape}</span>}
              </span>
              <Handle
                type="source"
                position={Position.Right}
                id={`out-${i}`}
                className="!w-2.5 !h-2.5 !bg-orange-400 !border-orange-600 shadow-[0_0_5px_rgba(251,146,60,0.5)]"
                style={{ top: "auto", position: "relative", right: -6 }}
              />
            </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export const NeuronNode = memo(NeuronNodeInner);
