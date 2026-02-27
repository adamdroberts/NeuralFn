import React, { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { NeuronNodeData } from "../store/graphStore";

function NeuronNodeInner({ id, data, selected }: NodeProps & { data: NeuronNodeData }) {
  const { label, neuronDef, isInput, isOutput } = data;
  const inputs = neuronDef.input_ports;
  const outputs = neuronDef.output_ports;

  const roleTag = isInput ? "IN" : isOutput ? "OUT" : null;

  return (
    <div
      className={`rounded-lg border px-3 py-2 min-w-[140px] shadow-lg transition-colors ${
        selected
          ? "border-blue-400 bg-gray-800"
          : "border-gray-700 bg-gray-900"
      }`}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className="text-xs font-bold text-blue-300 truncate">
          {label}
        </span>
        {roleTag && (
          <span className="text-[10px] font-mono px-1 rounded bg-blue-900 text-blue-200">
            {roleTag}
          </span>
        )}
      </div>

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
