import React, { useCallback, useEffect, useState } from "react";
import { api, type DatasetInfo } from "../api/client";

interface DatasetSourcePanelProps {
  node: any;
  renderError: () => React.ReactNode;
  updateNodeData: (id: string, data: any) => void;
  clearError: () => void;
}

export default function DatasetSourcePanel({
  node,
  renderError,
  updateNodeData,
  clearError,
}: DatasetSourcePanelProps) {
  const config = (node.data.neuronDef.module_config ?? {}) as {
    dataset_names?: string[];
    seq_len?: number;
  };
  const selectedDatasets = config.dataset_names ?? [];
  const seqLen = config.seq_len ?? 64;

  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);

  const refreshDatasets = useCallback(async () => {
    try {
      setDatasets(await api.getDatasets());
    } catch {
      setDatasets([]);
    }
  }, []);

  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  const updateConfig = useCallback(
    (patch: Partial<{ dataset_names: string[]; seq_len: number }>) => {
      clearError();
      updateNodeData(node.id, {
        neuronDef: {
          ...node.data.neuronDef,
          module_config: { ...config, ...patch },
        },
      });
    },
    [clearError, config, node, updateNodeData]
  );

  const toggleDataset = (name: string) => {
    const next = selectedDatasets.includes(name)
      ? selectedDatasets.filter((n) => n !== name)
      : [...selectedDatasets, name];
    updateConfig({ dataset_names: next });
  };

  const formatTokens = (n: number | null) => {
    if (n === null) return "?";
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return String(n);
  };

  return (
    <div className="w-80 border-l border-gray-800 bg-gray-900 flex flex-col overflow-hidden">
      <div className="px-3 py-2 border-b border-gray-800 flex items-center justify-between">
        <span className="text-sm font-bold text-violet-300">{node.data.label}</span>
        <span className="text-[10px] text-gray-500 font-mono">{node.id}</span>
      </div>

      {renderError()}

      <div className="p-3 flex flex-col gap-3 overflow-auto flex-1">
        <label className="text-[11px] text-gray-400">
          Sequence Length
          <input
            type="number"
            value={seqLen}
            onChange={(e) => updateConfig({ seq_len: parseInt(e.target.value, 10) || 64 })}
            className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-200"
          />
        </label>

        <div className="text-[10px] text-gray-500">
          {selectedDatasets.length > 0 ? (
            <span className="text-blue-300">
              {selectedDatasets.length} dataset{selectedDatasets.length > 1 ? "s" : ""} selected
            </span>
          ) : (
            "No datasets selected"
          )}
        </div>

        {/* Refresh button */}
        <button
          onClick={refreshDatasets}
          className="text-[10px] text-gray-400 hover:text-gray-200 self-end"
        >
          ↻ Refresh
        </button>

        {/* Dataset list */}
        {datasets.length === 0 ? (
          <div className="text-[10px] text-gray-600 italic text-center py-4">
            No datasets available. Use the Training Panel to download from HuggingFace or upload local files.
          </div>
        ) : (
          <div className="max-h-64 overflow-auto divide-y divide-gray-800/50 border border-gray-800 rounded">
            {datasets.map((ds) => (
              <div
                key={ds.name}
                className={`flex items-center gap-2 px-2 py-1.5 cursor-pointer transition-colors ${
                  selectedDatasets.includes(ds.name)
                    ? "bg-blue-900/30 border-l-2 border-blue-400"
                    : "hover:bg-gray-800/50 border-l-2 border-transparent"
                }`}
                onClick={() => toggleDataset(ds.name)}
              >
                <input
                  type="checkbox"
                  checked={selectedDatasets.includes(ds.name)}
                  onChange={() => toggleDataset(ds.name)}
                  className="accent-blue-500"
                />
                <div className="flex-1 min-w-0">
                  <div className="text-[11px] text-gray-200 font-medium truncate">
                    {ds.name}
                  </div>
                  <div className="text-[9px] text-gray-500">
                    {ds.source === "huggingface" && ds.hf_path
                      ? `HF: ${ds.hf_path}`
                      : ds.source}
                    {ds.num_tokens != null && (
                      <span className="ml-2">• {formatTokens(ds.num_tokens)} tokens</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="text-[9px] text-gray-600 leading-relaxed">
          Connect <span className="text-emerald-400">tokens</span> → token_embedding and{" "}
          <span className="text-emerald-400">targets</span> → cross_entropy targets port.
        </div>
      </div>
    </div>
  );
}
