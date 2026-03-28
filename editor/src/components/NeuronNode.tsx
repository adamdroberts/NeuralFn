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
            <span className="text-[10px] font-mono px-1 rounded bg-gray-950 text-amber-200">
              {variantRef.family}@{variantRef.version}
            </span>
          )}
          {roleTag && (
            <span className="text-[10px] font-mono px-1 rounded bg-blue-900 text-blue-200">
              {roleTag}
            </span>
          )}
        </div>
      </div>

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
          {inputs.map((p, i) => (
            <div key={p.name} className="relative flex items-center">
              <Handle
                type="target"
                position={Position.Left}
                id={`in-${i}`}
                className="!w-2.5 !h-2.5 !bg-green-400 !border-green-600"
                style={{ top: "auto", position: "relative", left: -6 }}
              />
              <span className="ml-1" title={`[${p.range[0]}, ${p.range[1]}] p=${p.precision}`}>
                {p.name}
              </span>
            </div>
          ))}
        </div>

        <div className="flex flex-col gap-1 items-end">
          {outputs.map((p, i) => (
            <div key={p.name} className="relative flex items-center">
              <span className="mr-1" title={`[${p.range[0]}, ${p.range[1]}] p=${p.precision}`}>
                {p.name}
              </span>
              <Handle
                type="source"
                position={Position.Right}
                id={`out-${i}`}
                className="!w-2.5 !h-2.5 !bg-orange-400 !border-orange-600"
                style={{ top: "auto", position: "relative", right: -6 }}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export const NeuronNode = memo(NeuronNodeInner);
