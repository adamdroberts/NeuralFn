import React, { useEffect, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import GraphCanvas from "./components/GraphCanvas";
import CodePanel from "./components/CodePanel";
import LibraryPanel from "./components/LibraryPanel";
import TrainingPanel from "./components/TrainingPanel";
import Toolbar from "./components/Toolbar";
import { useGraphStore } from "./store/graphStore";
import { api } from "./api/client";

function AgentBanner() {
  const [active, setActive] = useState(false);
  const setRootGraph = useGraphStore(state => state.setRootGraph);

  useEffect(() => {
    let lastActive = false;
    const interval = setInterval(async () => {
      try {
        const res = await fetch("/api/agent/status");
        if (!res.ok) return;
        const data = await res.json();
        setActive(data.active);
        
        // Auto-refresh graph if agent just became inactive
        if (lastActive && !data.active) {
           const graph = await api.getGraph();
           setRootGraph(graph);
        }
        lastActive = data.active;
      } catch (err) {
        console.error("Failed to fetch agent status", err);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [setRootGraph]);

  if (!active) return null;
  return (
    <div className="bg-purple-600 text-white text-center py-1.5 px-4 text-sm font-semibold tracking-wide flex items-center justify-center space-x-2 animate-pulse shadow-md z-50">
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
      <span>AI Agent is controlling the editor...</span>
    </div>
  );
}

export default function App() {
  return (
    <ReactFlowProvider>
      <div className="h-screen flex flex-col bg-gray-950 text-gray-100 relative">
        <AgentBanner />
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
