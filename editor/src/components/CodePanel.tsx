import React, { useCallback, useEffect, useState } from "react";
import Editor from "@monaco-editor/react";
import { selectActiveGraph, selectSelectedNode, useGraphStore } from "../store/graphStore";
import PortConfig from "./PortConfig";
import type { TrainingMethod } from "../api/client";

export default function CodePanel() {
  const activeGraph = useGraphStore(selectActiveGraph);
  const node = useGraphStore(selectSelectedNode);
  const updateNodeData = useGraphStore((state) => state.updateNodeData);
  const updateActiveGraphSettings = useGraphStore((state) => state.updateActiveGraphSettings);
  const openSubgraph = useGraphStore((state) => state.openSubgraph);

  const [code, setCode] = useState("");

  useEffect(() => {
    if (node && node.data.neuronDef.kind === "function") {
      setCode(node.data.neuronDef.source_code || `def ${node.data.label}(x):\n    return x\n`);
    }
  }, [node]);

  const onCodeChange = useCallback(
    (value: string | undefined) => {
      if (!node || node.data.neuronDef.kind !== "function" || value === undefined) return;
      setCode(value);
      updateNodeData(node.id, {
        neuronDef: { ...node.data.neuronDef, source_code: value },
      });
    },
    [node, updateNodeData]
  );

  if (!node) {
    return (
      <div className="w-80 border-l border-gray-800 bg-gray-900 flex flex-col p-4 gap-3">
        <div>
          <div className="text-sm font-bold text-blue-300">{activeGraph.name}</div>
          <div className="text-[11px] text-gray-500">Active graph settings</div>
        </div>

        <label className="text-[11px] text-gray-400">
          Graph Name
          <input
            type="text"
            value={activeGraph.name}
            onChange={(e) => updateActiveGraphSettings({ name: e.target.value })}
            className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
          />
        </label>

        <label className="text-[11px] text-gray-400">
          Training Method
          <select
            value={activeGraph.training_method}
            onChange={(e) =>
              updateActiveGraphSettings({ training_method: e.target.value as TrainingMethod })
            }
            className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
          >
            <option value="surrogate">surrogate</option>
            <option value="evolutionary">evolutionary</option>
            <option value="frozen">frozen</option>
          </select>
        </label>

        <div className="text-[11px] text-gray-500 leading-5">
          Select a function node to edit code, or select a subgraph node to edit aliases and its nested graph settings.
        </div>
      </div>
    );
  }

  if (node.data.neuronDef.kind === "subgraph") {
    const subgraph = node.data.neuronDef.subgraph;
    const updateAlias = (kind: "input_aliases" | "output_aliases", idx: number, value: string) => {
      if (!subgraph) return;
      const aliases = [...node.data.neuronDef[kind]];
      aliases[idx] = value;
      updateNodeData(node.id, {
        neuronDef: {
          ...node.data.neuronDef,
          [kind]: aliases,
        },
      });
    };

    const updateTrainingMethod = (method: TrainingMethod) => {
      if (!subgraph) return;
      updateNodeData(node.id, {
        neuronDef: {
          ...node.data.neuronDef,
          subgraph: { ...subgraph, training_method: method },
        },
      });
    };

    return (
      <div className="w-80 border-l border-gray-800 bg-gray-900 flex flex-col overflow-hidden">
        <div className="px-3 py-2 border-b border-gray-800 flex items-center justify-between">
          <span className="text-sm font-bold text-amber-300">{node.data.label}</span>
          <span className="text-[10px] text-gray-500 font-mono">{node.id}</span>
        </div>

        <div className="p-3 flex flex-col gap-3 overflow-auto">
          <label className="text-[11px] text-gray-400">
            Node Name
            <input
              type="text"
              value={node.data.neuronDef.name}
              onChange={(e) =>
                updateNodeData(node.id, {
                  neuronDef: { ...node.data.neuronDef, name: e.target.value },
                })
              }
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
            />
          </label>

          <label className="text-[11px] text-gray-400">
            Nested Training Method
            <select
              value={subgraph?.training_method ?? "surrogate"}
              onChange={(e) => updateTrainingMethod(e.target.value as TrainingMethod)}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
            >
              <option value="surrogate">surrogate</option>
              <option value="evolutionary">evolutionary</option>
              <option value="frozen">frozen</option>
            </select>
          </label>

          <button
            onClick={() => openSubgraph(node.id)}
            className="bg-amber-800 hover:bg-amber-700 text-amber-100 text-xs px-3 py-1.5 rounded font-medium"
          >
            Open Subgraph
          </button>

          <div>
            <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">
              Input Aliases
            </div>
            <div className="flex flex-col gap-2">
              {node.data.neuronDef.input_ports.map((port, idx) => (
                <input
                  key={`in-${idx}`}
                  type="text"
                  value={node.data.neuronDef.input_aliases[idx] ?? port.name}
                  onChange={(e) => updateAlias("input_aliases", idx, e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-[11px] text-gray-200"
                />
              ))}
              {node.data.neuronDef.input_ports.length === 0 && (
                <div className="text-[11px] text-gray-500">No exposed inputs yet.</div>
              )}
            </div>
          </div>

          <div>
            <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">
              Output Aliases
            </div>
            <div className="flex flex-col gap-2">
              {node.data.neuronDef.output_ports.map((port, idx) => (
                <input
                  key={`out-${idx}`}
                  type="text"
                  value={node.data.neuronDef.output_aliases[idx] ?? port.name}
                  onChange={(e) => updateAlias("output_aliases", idx, e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-[11px] text-gray-200"
                />
              ))}
              {node.data.neuronDef.output_ports.length === 0 && (
                <div className="text-[11px] text-gray-500">No exposed outputs yet.</div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 border-l border-gray-800 bg-gray-900 flex flex-col overflow-hidden">
      <div className="px-3 py-2 border-b border-gray-800 flex items-center justify-between">
        <span className="text-sm font-bold text-blue-300">{node.data.label}</span>
        <span className="text-[10px] text-gray-500 font-mono">{node.id}</span>
      </div>

      <div className="flex-1 min-h-0">
        <Editor
          height="100%"
          language="python"
          theme="vs-dark"
          value={code}
          onChange={onCodeChange}
          options={{
            minimap: { enabled: false },
            fontSize: 12,
            lineNumbers: "on",
            scrollBeyondLastLine: false,
            wordWrap: "on",
            padding: { top: 8 },
          }}
        />
      </div>

      <PortConfig node={node} />
    </div>
  );
}
