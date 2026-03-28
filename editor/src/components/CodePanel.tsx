import React, { useCallback, useEffect, useMemo, useState } from "react";
import Editor from "@monaco-editor/react";
import { selectActiveGraph, selectSelectedNode, useGraphStore } from "../store/graphStore";
import { listCompatibleVariantVersions } from "../store/graphUtils";
import PortConfig from "./PortConfig";
import type { TrainingMethod } from "../api/client";

export default function CodePanel() {
  const activeGraph = useGraphStore(selectActiveGraph);
  const node = useGraphStore(selectSelectedNode);
  const rootGraph = useGraphStore((state) => state.rootGraph);
  const currentPath = useGraphStore((state) => state.currentPath);
  const lastError = useGraphStore((state) => state.lastError);
  const clearError = useGraphStore((state) => state.clearError);
  const updateNodeData = useGraphStore((state) => state.updateNodeData);
  const updateActiveGraphSettings = useGraphStore((state) => state.updateActiveGraphSettings);
  const openSubgraph = useGraphStore((state) => state.openSubgraph);

  const saveNodeAsVariant = useGraphStore((state) => state.saveNodeAsVariant);
  const swapNodeVariant = useGraphStore((state) => state.swapNodeVariant);

  const [code, setCode] = useState("");
  const [familyName, setFamilyName] = useState("");
  const [versionName, setVersionName] = useState("baseline");
  const [forkVersion, setForkVersion] = useState("v2");

  useEffect(() => {
    if (node && node.data.neuronDef.kind === "function") {
      setCode(node.data.neuronDef.source_code || `def ${node.data.label}(x):\n    return x\n`);
    } else if (node && node.data.neuronDef.kind === "module") {
      setCode(JSON.stringify(node.data.neuronDef.module_config ?? {}, null, 2));
    } else {
      setCode("");
    }
  }, [node]);

  useEffect(() => {
    if (node?.data.neuronDef.kind === "subgraph") {
      const variantRef = node.data.neuronDef.variant_ref;
      setFamilyName(variantRef?.family ?? node.data.neuronDef.name);
      setVersionName(variantRef?.version ?? "baseline");
      setForkVersion(
        variantRef?.version === "baseline"
          ? "v2"
          : `${variantRef?.version ?? "variant"}_alt`,
      );
    }
  }, [node]);

  const activeVariant = useMemo(() => {
    const segment = currentPath[currentPath.length - 1];
    return segment?.kind === "variant" ? segment : null;
  }, [currentPath]);

  const onCodeChange = useCallback(
    (value: string | undefined) => {
      if (!node || node.data.neuronDef.kind !== "function" || value === undefined) return;
      setCode(value);
      clearError();
      updateNodeData(node.id, {
        neuronDef: { ...node.data.neuronDef, source_code: value },
      });
    },
    [clearError, node, updateNodeData],
  );

  const renderError = () =>
    lastError ? (
      <div className="mx-3 mt-3 rounded border border-rose-800 bg-rose-950/50 px-3 py-2 text-[11px] text-rose-200">
        {lastError}
      </div>
    ) : null;

  if (!node) {
    const isTorchGraph = activeGraph.training_method === "torch" || activeGraph.runtime === "torch";
    return (
      <div className="w-80 border-l border-gray-800 bg-gray-900 flex flex-col">
        <div className="p-4 flex flex-col gap-3 overflow-auto">
          <div>
            <div className="text-sm font-bold text-blue-300">{activeGraph.name}</div>
            <div className="text-[11px] text-gray-500">
              {activeVariant
                ? `Editing shared variant ${activeVariant.family}@${activeVariant.version}`
                : "Active graph settings"}
            </div>
          </div>

          {renderError()}

          {activeVariant && (
            <div className="rounded border border-amber-800 bg-amber-950/40 px-3 py-2 text-[11px] text-amber-100">
              Changes here update every node linked to this exact version as long as the external interface stays the same.
            </div>
          )}

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
              <option value="torch">torch</option>
            </select>
          </label>

          {isTorchGraph && (
            <>
              <label className="text-[11px] text-gray-400">
                Torch Device
                <input
                  type="text"
                  value={String(activeGraph.torch_config.device ?? "cuda")}
                  onChange={(e) =>
                    updateActiveGraphSettings({
                      torch_config: {
                        ...activeGraph.torch_config,
                        device: e.target.value || "cuda",
                      },
                    })
                  }
                  className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200 font-mono"
                />
              </label>

              <label className="text-[11px] text-gray-400">
                AMP DType
                <select
                  value={String(activeGraph.torch_config.amp_dtype ?? "bfloat16")}
                  onChange={(e) =>
                    updateActiveGraphSettings({
                      torch_config: {
                        ...activeGraph.torch_config,
                        amp_dtype: e.target.value,
                      },
                    })
                  }
                  className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                >
                  <option value="bfloat16">bfloat16</option>
                  <option value="float16">float16</option>
                </select>
              </label>
            </>
          )}

          <div className="text-[11px] text-gray-500 leading-5">
            Select a node to edit it, double-click a subgraph to open it, or open a saved family/version from the variant library.
          </div>
        </div>
      </div>
    );
  }

  if (node.data.neuronDef.kind === "subgraph") {
    const subgraph = node.data.neuronDef.subgraph;
    const variantRef = node.data.neuronDef.variant_ref;
    const compatibleVersions = variantRef
      ? listCompatibleVariantVersions(rootGraph, variantRef.family, subgraph)
      : [];

    const updateAlias = (kind: "input_aliases" | "output_aliases", idx: number, value: string) => {
      if (!subgraph) return;
      const aliases = [...node.data.neuronDef[kind]];
      aliases[idx] = value;
      clearError();
      updateNodeData(node.id, {
        neuronDef: {
          ...node.data.neuronDef,
          [kind]: aliases,
        },
      });
    };

    const updateTrainingMethod = (method: TrainingMethod) => {
      if (!subgraph) return;
      clearError();
      updateNodeData(node.id, {
        neuronDef: {
          ...node.data.neuronDef,
          subgraph: {
            ...subgraph,
            training_method: method,
            runtime: method === "torch" ? "torch" : subgraph.runtime,
          },
        },
      });
    };

    return (
      <div className="w-80 border-l border-gray-800 bg-gray-900 flex flex-col overflow-hidden">
        <div className="px-3 py-2 border-b border-gray-800 flex items-center justify-between">
          <span className="text-sm font-bold text-amber-300">{node.data.label}</span>
          <span className="text-[10px] text-gray-500 font-mono">{node.id}</span>
        </div>

        {renderError()}

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

          {variantRef ? (
            <>
              <div className="rounded border border-amber-800 bg-amber-950/40 px-3 py-2 text-[11px] text-amber-100">
                Linked to <span className="font-mono">{variantRef.family}@{variantRef.version}</span>
              </div>

              <button
                onClick={() => openSubgraph(node.id)}
                className="bg-amber-800 hover:bg-amber-700 text-amber-100 text-xs px-3 py-1.5 rounded font-medium"
              >
                Open Variant
              </button>

              <label className="text-[11px] text-gray-400">
                Swap Version
                <select
                  value={variantRef.version}
                  onChange={(e) => swapNodeVariant(node.id, variantRef.family, e.target.value)}
                  className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                >
                  {compatibleVersions.map((version) => (
                    <option key={version} value={version}>
                      {version}
                    </option>
                  ))}
                </select>
              </label>

              <div className="rounded border border-gray-800 bg-gray-950/60 p-3">
                <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">
                  Save New Version
                </div>
                <label className="block text-[11px] text-gray-400">
                  New Version
                  <input
                    type="text"
                    value={forkVersion}
                    onChange={(e) => setForkVersion(e.target.value)}
                    className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200 font-mono"
                  />
                </label>
                <button
                  onClick={() => saveNodeAsVariant(node.id, variantRef.family, forkVersion, true)}
                  className="mt-2 w-full rounded bg-gray-800 px-3 py-1.5 text-xs text-gray-100 hover:bg-gray-700"
                >
                  Fork And Relink
                </button>
              </div>
            </>
          ) : (
            <>
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
                  <option value="torch">torch</option>
                </select>
              </label>

              <button
                onClick={() => openSubgraph(node.id)}
                className="bg-amber-800 hover:bg-amber-700 text-amber-100 text-xs px-3 py-1.5 rounded font-medium"
              >
                Open Subgraph
              </button>

              <div className="rounded border border-gray-800 bg-gray-950/60 p-3">
                <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">
                  Save As Variant
                </div>
                <label className="block text-[11px] text-gray-400">
                  Family
                  <input
                    type="text"
                    value={familyName}
                    onChange={(e) => setFamilyName(e.target.value)}
                    className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
                  />
                </label>
                <label className="mt-2 block text-[11px] text-gray-400">
                  Version
                  <input
                    type="text"
                    value={versionName}
                    onChange={(e) => setVersionName(e.target.value)}
                    className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200 font-mono"
                  />
                </label>
                <button
                  onClick={() => saveNodeAsVariant(node.id, familyName, versionName, true)}
                  className="mt-2 w-full rounded bg-gray-800 px-3 py-1.5 text-xs text-gray-100 hover:bg-gray-700"
                >
                  Save And Link
                </button>
              </div>
            </>
          )}

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

  if (node.data.neuronDef.kind === "module") {
    const updateModuleConfig = (value: string | undefined) => {
      if (value === undefined) return;
      setCode(value);
      try {
        clearError();
        updateNodeData(node.id, {
          neuronDef: {
            ...node.data.neuronDef,
            module_config: JSON.parse(value),
          },
        });
      } catch {
        // Keep the editor responsive while JSON is mid-edit.
      }
    };

    return (
      <div className="w-80 border-l border-gray-800 bg-gray-900 flex flex-col overflow-hidden">
        <div className="px-3 py-2 border-b border-gray-800 flex items-center justify-between">
          <span className="text-sm font-bold text-emerald-300">{node.data.label}</span>
          <span className="text-[10px] text-gray-500 font-mono">{node.id}</span>
        </div>

        {renderError()}

        <div className="px-3 py-2 border-b border-gray-800 text-[11px] text-gray-400 space-y-2">
          <label className="block">
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
          <div>
            Module Type
            <div className="mt-1 rounded bg-gray-800 border border-gray-700 px-2 py-1 text-gray-200 font-mono">
              {node.data.neuronDef.module_type}
            </div>
          </div>
          <div className="text-[10px] text-gray-500">
            Edit the module config as JSON. Training writes weights into the hidden serialized state field.
          </div>
        </div>

        <div className="flex-1 min-h-0">
          <Editor
            height="100%"
            language="json"
            theme="vs-dark"
            value={code}
            onChange={updateModuleConfig}
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
      </div>
    );
  }

  return (
    <div className="w-80 border-l border-gray-800 bg-gray-900 flex flex-col overflow-hidden">
      <div className="px-3 py-2 border-b border-gray-800 flex items-center justify-between">
        <span className="text-sm font-bold text-blue-300">{node.data.label}</span>
        <span className="text-[10px] text-gray-500 font-mono">{node.id}</span>
      </div>

      {renderError()}

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
