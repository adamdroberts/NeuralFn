import React, { useCallback, useEffect, useState } from "react";
import { api, type NeuronDefData } from "../api/client";
import { useGraphStore, type NeuronNodeData } from "../store/graphStore";
import type { Node } from "@xyflow/react";

export default function Toolbar() {
  const { builtins, setBuiltins, addNode, nodes } = useGraphStore();
  const [showLibrary, setShowLibrary] = useState(true);

  useEffect(() => {
    api.getBuiltins().then(setBuiltins).catch(() => {});
  }, [setBuiltins]);

  const onAddBuiltin = useCallback(
    (ndef: NeuronDefData) => {
      const id = `n-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      const node: Node<NeuronNodeData> = {
        id,
        type: "neuron",
        position: { x: 100 + Math.random() * 300, y: 100 + Math.random() * 300 },
        data: {
          label: ndef.name,
          neuronDef: { ...ndef, id },
          isInput: ndef.name === "input",
          isOutput: ndef.name === "output",
        },
      };
      addNode(node);
    },
    [addNode]
  );

  const onAddCustom = useCallback(() => {
    const id = `n-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    const ndef: NeuronDefData = {
      id,
      name: "custom",
      input_ports: [{ name: "x", range: [-10, 10], precision: 0.001, dtype: "float" }],
      output_ports: [{ name: "y", range: [-10, 10], precision: 0.001, dtype: "float" }],
      source_code: "def custom(x):\n    return x\n",
    };
    const node: Node<NeuronNodeData> = {
      id,
      type: "neuron",
      position: { x: 200 + Math.random() * 200, y: 200 + Math.random() * 200 },
      data: {
        label: "custom",
        neuronDef: ndef,
        isInput: false,
        isOutput: false,
      },
    };
    addNode(node);
  }, [addNode]);

  const onSave = useCallback(() => {
    const data = JSON.stringify({ nodes, edges: useGraphStore.getState().edges }, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "neuralfn-graph.json";
    a.click();
    URL.revokeObjectURL(url);
  }, [nodes]);

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
        if (data.nodes) useGraphStore.getState().setNodes(data.nodes);
        if (data.edges) useGraphStore.getState().setEdges(data.edges);
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
        <div className="flex gap-1 px-3 pb-2 flex-wrap">
          {builtins.map((b) => (
            <button
              key={b.id}
              onClick={() => onAddBuiltin(b)}
              className="bg-gray-800 hover:bg-blue-900 border border-gray-700 text-gray-300 text-[10px] px-2 py-0.5 rounded"
              title={`${b.input_ports.length} in / ${b.output_ports.length} out`}
            >
              {b.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
