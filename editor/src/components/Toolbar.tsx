import React, { useCallback, useEffect, useState } from "react";
import { api, type NeuronDefData } from "../api/client";
import { selectBreadcrumbs, selectCurrentPath, useGraphStore } from "../store/graphStore";
import { normalizeGraph } from "../store/graphUtils";

export default function Toolbar() {
  const builtins = useGraphStore((state) => state.builtins);
  const setBuiltins = useGraphStore((state) => state.setBuiltins);
  const addBuiltinNode = useGraphStore((state) => state.addBuiltinNode);
  const addCustomNode = useGraphStore((state) => state.addCustomNode);
  const addSubgraphNode = useGraphStore((state) => state.addSubgraphNode);
  const rootGraph = useGraphStore((state) => state.rootGraph);
  const currentPath = useGraphStore(selectCurrentPath);
  const breadcrumbs = useGraphStore(selectBreadcrumbs);
  const setPath = useGraphStore((state) => state.setPath);
  const [showLibrary, setShowLibrary] = useState(true);

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

  const onAddSubgraph = useCallback(() => {
    addSubgraphNode();
  }, [addSubgraphNode]);

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
          onClick={onAddSubgraph}
          className="bg-amber-900 hover:bg-amber-800 text-amber-100 text-xs px-2 py-1 rounded"
        >
          + Subgraph
        </button>

        <div className="flex items-center gap-1 ml-2 text-[10px] text-gray-400 overflow-x-auto">
          {breadcrumbs.map((crumb, idx) => (
            <React.Fragment key={`${crumb.id}-${idx}`}>
              {idx > 0 && <span>/</span>}
              <button
                onClick={() => setPath(currentPath.slice(0, idx))}
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
