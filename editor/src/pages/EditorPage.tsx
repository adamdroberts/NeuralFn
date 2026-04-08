import React, { useEffect, useMemo, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { useParams } from "react-router-dom";
import { api } from "../api/client";
import CodePanel from "../components/CodePanel";
import GraphCanvas from "../components/GraphCanvas";
import LibraryPanel from "../components/LibraryPanel";
import Toolbar from "../components/Toolbar";
import TrainingPanel from "../components/TrainingPanel";
import { useGraphStore } from "../store/graphStore";
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
      <div className="flex h-full min-h-0 flex-1 flex-col bg-gray-950 text-gray-100 relative">
        <AgentBanner />
        <div className="border-b border-gray-800 bg-gray-950/90 px-3 py-1 text-[11px] text-gray-400">
          {statusText}
        </div>
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
