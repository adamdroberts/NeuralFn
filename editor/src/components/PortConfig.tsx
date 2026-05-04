import React, { useCallback } from "react";
import type { Node } from "@xyflow/react";
import type { NeuronNodeData } from "../store/graphStore";
import { useGraphStore } from "../store/graphStore";
import type { PortData } from "../api/client";

interface Props {
  node: Node<NeuronNodeData>;
}

export default function PortConfig({ node }: Props) {
  const { updateNodeData, toggleInput, toggleOutput } = useGraphStore();

  const updatePort = useCallback(
    (kind: "input_ports" | "output_ports", idx: number, patch: Partial<PortData>) => {
      const ports = [...node.data.neuronDef[kind]];
      ports[idx] = { ...ports[idx], ...patch };
      updateNodeData(node.id, {
        neuronDef: { ...node.data.neuronDef, [kind]: ports },
      });
    },
    [node, updateNodeData]
  );

  const renderPorts = (kind: "input_ports" | "output_ports", label: string) => {
    const ports = node.data.neuronDef[kind];
    return (
      <div className="mb-2">
        <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-1">
          {label}
        </div>
        {ports.map((p, i) => (
          <div key={i} className="flex gap-1 items-center mb-1 text-[11px]">
            <span className="w-12 truncate text-gray-300" title={p.name}>
              {p.name}
            </span>
            <input
              type="number"
              className="w-14 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200"
              value={p.range ? p.range[0] : ""}
              step={0.1}
              placeholder="lo"
              title="lower bound (blank = unbounded)"
              onChange={(e) => {
                const raw = e.target.value;
                if (raw === "") {
                  updatePort(kind, i, { range: null });
                } else {
                  const lo = parseFloat(raw) || 0;
                  const hi = p.range ? p.range[1] : lo + 1;
                  updatePort(kind, i, { range: [lo, hi] });
                }
              }}
            />
            <span className="text-gray-600">..</span>
            <input
              type="number"
              className="w-14 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200"
              value={p.range ? p.range[1] : ""}
              step={0.1}
              placeholder="hi"
              title="upper bound (blank = unbounded)"
              onChange={(e) => {
                const raw = e.target.value;
                if (raw === "") {
                  updatePort(kind, i, { range: null });
                } else {
                  const hi = parseFloat(raw) || 0;
                  const lo = p.range ? p.range[0] : hi - 1;
                  updatePort(kind, i, { range: [lo, hi] });
                }
              }}
            />
            <input
              type="number"
              className="w-14 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200"
              value={p.precision ?? ""}
              step={0.001}
              title="precision (blank = unquantized)"
              placeholder="none"
              onChange={(e) => {
                const raw = e.target.value;
                updatePort(kind, i, {
                  precision: raw === "" ? null : parseFloat(raw),
                });
              }}
            />
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="p-3 border-t border-gray-800 overflow-auto max-h-60">
      {renderPorts("input_ports", "Inputs")}
      {renderPorts("output_ports", "Outputs")}

      <div className="flex gap-2 mt-2">
        <button
          onClick={() => toggleInput(node.id)}
          className={`text-[10px] px-2 py-1 rounded ${
            node.data.isInput
              ? "bg-green-800 text-green-200"
              : "bg-gray-800 text-gray-400"
          }`}
        >
          {node.data.isInput ? "Graph Input" : "Set as Input"}
        </button>
        <button
          onClick={() => toggleOutput(node.id)}
          className={`text-[10px] px-2 py-1 rounded ${
            node.data.isOutput
              ? "bg-orange-800 text-orange-200"
              : "bg-gray-800 text-gray-400"
          }`}
        >
          {node.data.isOutput ? "Graph Output" : "Set as Output"}
        </button>
      </div>
    </div>
  );
}
