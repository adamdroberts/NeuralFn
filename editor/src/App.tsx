import React from "react";
import { ReactFlowProvider } from "@xyflow/react";
import GraphCanvas from "./components/GraphCanvas";
import CodePanel from "./components/CodePanel";
import LibraryPanel from "./components/LibraryPanel";
import TrainingPanel from "./components/TrainingPanel";
import Toolbar from "./components/Toolbar";

export default function App() {
  return (
    <ReactFlowProvider>
      <div className="h-screen flex flex-col bg-gray-950 text-gray-100">
        <Toolbar />
        <div className="flex flex-1 min-h-0">
          <LibraryPanel />
          <GraphCanvas />
          <CodePanel />
        </div>
        <TrainingPanel />
      </div>
    </ReactFlowProvider>
  );
}
