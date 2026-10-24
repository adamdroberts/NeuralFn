import React, { useEffect, useMemo, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { useParams } from "react-router-dom";
import { api } from "../api/client";
import ChatPanel from "../components/ChatPanel";
import CodePanel from "../components/CodePanel";
import GraphCanvas from "../components/GraphCanvas";
import LibraryPanel from "../components/LibraryPanel";
import Toolbar from "../components/Toolbar";
import TrainingPanel from "../components/TrainingPanel";
import { useGraphStore } from "../store/graphStore";
import { useSettingsStore } from "../store/settingsStore";
import { reloadActiveSessionGraph, syncActiveSessionGraph } from "../routes/sessionSync";

function AgentBanner() {
  const projectId = useGraphStore((state) => state.projectId);
  const sessionId = useGraphStore((state) => state.sessionId);
  const [active, setActive] = useState(false);

  useEffect(() => {
    if (!projectId || !sessionId) {
      setActive(false);
      return;
    }
    let lastActive = false;
    const timer = window.setInterval(async () => {
      try {
        const status = await api.getAgentStatus(projectId, sessionId);
        setActive(status.active);
        if (lastActive && !status.active) {
          await reloadActiveSessionGraph();
        }
        lastActive = status.active;
      } catch {
        setActive(false);
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [projectId, sessionId]);

  if (!active) {
    return null;
  }

  return (
    <div className="bg-purple-600 text-white text-center py-1.5 px-4 text-sm font-semibold tracking-wide flex items-center justify-center space-x-2 animate-pulse shadow-md z-50">
      <span>AI Agent is controlling the active session...</span>
    </div>
  );
}

export default function EditorPage() {
  const params = useParams<{ projectId: string; sessionId: string }>();
  const hydrateSession = useGraphStore((state) => state.hydrateSession);
  const setHydrationState = useGraphStore((state) => state.setHydrationState);
  const rootGraph = useGraphStore((state) => state.rootGraph);
  const hydrationState = useGraphStore((state) => state.hydrationState);
  const isDirty = useGraphStore((state) => state.isDirty);
  const isSaving = useGraphStore((state) => state.isSaving);
  const saveError = useGraphStore((state) => state.saveError);
  const revision = useGraphStore((state) => state.revision);
  const particleGraphEditorEnabled = useSettingsStore(
    (state) => state.particleGraphEditorEnabled,
  );

  useEffect(() => {
    if (!params.projectId || !params.sessionId) {
      return;
    }
    let cancelled = false;
    setHydrationState("loading");
    api
      .getSession(params.projectId, params.sessionId)
      .then((detail) => {
        if (cancelled) return;
        hydrateSession({
          projectId: params.projectId!,
          sessionId: params.sessionId!,
          graph: detail.graph,
          revision: detail.revision,
        });
      })
      .catch(() => {
        if (cancelled) return;
        setHydrationState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [params.projectId, params.sessionId, hydrateSession, setHydrationState]);

  useEffect(() => {
    if (!params.projectId || !params.sessionId || hydrationState !== "ready" || !isDirty) {
      return;
    }
    const timeout = window.setTimeout(() => {
      syncActiveSessionGraph({ skipIfClean: true }).catch(() => undefined);
    }, 600);
    return () => window.clearTimeout(timeout);
  }, [params.projectId, params.sessionId, rootGraph, hydrationState, isDirty]);

  const statusText = useMemo(() => {
    if (hydrationState === "loading") return "Loading session...";
    if (isSaving) return "Saving...";
    if (saveError) return saveError;
    if (isDirty) return "Unsaved changes";
    return `Revision ${revision}`;
  }, [hydrationState, isSaving, saveError, isDirty, revision]);

  if (!params.projectId || !params.sessionId) {
    return <div className="p-6 text-sm text-gray-400">Select a project session to open the editor.</div>;
  }

  if (hydrationState === "loading") {
    return <div className="p-6 text-sm text-gray-400">Loading editor session...</div>;
  }

  return (
    <ReactFlowProvider>
      <div
        className={`flex h-full min-h-0 flex-1 flex-col text-gray-100 relative ${
          particleGraphEditorEnabled ? "cyber-cockpit-shell" : "bg-gray-950"
        }`}
      >
        <AgentBanner />
        <div
          className={
            particleGraphEditorEnabled
              ? "border-b border-cyan-500/20 bg-slate-950/90 px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-cyan-300/80 flex items-center justify-between"
              : "border-b border-gray-800 bg-gray-950/90 px-3 py-1 text-[11px] text-gray-400"
          }
        >
          <span>{statusText}</span>
          {particleGraphEditorEnabled && (
            <span className="text-[9px] text-cyan-400/70">
              SYS.MODE // FULL CYBER 3D COCKPIT
            </span>
          )}
        </div>
        <Toolbar />
        <div
          className={`flex flex-1 min-h-0 relative overflow-hidden ${
            particleGraphEditorEnabled ? "cyber-workspace-stage" : ""
          }`}
        >
          <LibraryPanel />
          <GraphCanvas />
          <CodePanel />
          <ChatPanel />
        </div>
        <TrainingPanel />
      </div>
    </ReactFlowProvider>
  );
}
