import React, { useCallback, useEffect, useState } from "react";
import Editor from "@monaco-editor/react";
import { useGraphStore } from "../store/graphStore";
import PortConfig from "./PortConfig";

export default function CodePanel() {
  const { nodes, selectedNodeId, updateNodeData } = useGraphStore();
  const node = nodes.find((n) => n.id === selectedNodeId);

  const [code, setCode] = useState("");

  useEffect(() => {
    if (node) {
      setCode(node.data.neuronDef.source_code || `def ${node.data.label}(x):\n    return x\n`);
    }
  }, [selectedNodeId]);

  const onCodeChange = useCallback(
    (value: string | undefined) => {
      if (!node || value === undefined) return;
      setCode(value);
      updateNodeData(node.id, {
        neuronDef: { ...node.data.neuronDef, source_code: value },
      });
    },
    [node, updateNodeData]
  );

  if (!node) {
    return (
      <div className="w-80 border-l border-gray-800 bg-gray-900 flex items-center justify-center text-gray-500 text-sm p-4">
        Select a node to edit its code and ports
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
