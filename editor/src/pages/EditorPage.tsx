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
import { selectSelectedNode, useGraphStore } from "../store/graphStore";
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
  const selectedNode = useGraphStore(selectSelectedNode);
  const isTraining = useGraphStore((state) => state.isTraining);
  const lossHistory = useGraphStore((state) => state.lossHistory);
  const particleGraphEditorEnabled = useSettingsStore(
    (state) => state.particleGraphEditorEnabled,
  );
  const activePanelDialog = useSettingsStore(
    (state) => state.activePanelDialog,
  );
  const rightPanelTab = useSettingsStore((state) => state.rightPanelTab);
  const setActivePanelDialog = useSettingsStore(
    (state) => state.setActivePanelDialog,
  );
  const openRightPanelDialog = useSettingsStore(
    (state) => state.openRightPanelDialog,
  );

  useEffect(() => {
    if (!activePanelDialog) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setActivePanelDialog(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activePanelDialog, setActivePanelDialog]);

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

  const variantCount = Object.keys(rootGraph.variant_library ?? {}).length;
  const latestLoss =
    lossHistory.length > 0
      ? lossHistory[lossHistory.length - 1].loss.toFixed(4)
      : null;

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

        {/* Full-Canvas Graph Stage with Left, Bottom, and Right Dialog Launcher Buttons */}
        <div
          className={`flex flex-1 min-h-0 relative overflow-hidden ${
            particleGraphEditorEnabled ? "cyber-workspace-stage" : ""
          }`}
        >
          <GraphCanvas />

          {/* LEFT EDGE BUTTON -> Opens Variant Matrix / Library Dialog */}
          <div className="pointer-events-none absolute left-3 top-1/2 z-20 -translate-y-1/2">
            <button
              type="button"
              onClick={() => setActivePanelDialog("left")}
              className={
                particleGraphEditorEnabled
                  ? "pointer-events-auto flex items-center gap-2 rounded-lg border border-amber-400/60 bg-slate-950/90 px-3 py-2 font-mono text-xs font-bold uppercase tracking-wider text-amber-200 shadow-[0_0_20px_rgba(245,158,11,0.28)] backdrop-blur-md transition-all hover:border-amber-300 hover:bg-amber-950/80 hover:shadow-[0_0_26px_rgba(245,158,11,0.45)]"
                  : "pointer-events-auto flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-900/95 px-3 py-2 text-xs font-semibold text-amber-200 shadow-lg hover:bg-gray-800"
              }
              title="Open Variant Matrix / Library Dialog"
            >
              <span>◈ Variants</span>
              <span className="rounded bg-amber-500/20 px-1.5 py-0.5 font-mono text-[10px] text-amber-300">
                {variantCount}
              </span>
            </button>
          </div>

          {/* RIGHT EDGE BUTTONS -> Opens Node/Graph Inspector or Model Chat Dialog */}
          <div className="pointer-events-none absolute right-3 top-1/2 z-20 flex -translate-y-1/2 flex-col items-end gap-2">
            <button
              type="button"
              onClick={() => openRightPanelDialog("inspector")}
              className={
                particleGraphEditorEnabled
                  ? "pointer-events-auto flex items-center gap-2 rounded-lg border border-cyan-400/65 bg-slate-950/90 px-3 py-2 font-mono text-xs font-bold uppercase tracking-wider text-cyan-100 shadow-[0_0_20px_rgba(0,240,255,0.28)] backdrop-blur-md transition-all hover:border-cyan-300 hover:bg-cyan-950/80 hover:shadow-[0_0_26px_rgba(0,240,255,0.45)]"
                  : "pointer-events-auto flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-900/95 px-3 py-2 text-xs font-semibold text-sky-200 shadow-lg hover:bg-gray-800"
              }
              title="Open Node Inspector, Graph Settings & Code Editor Dialog"
            >
              <span>◧ {selectedNode ? selectedNode.data.label : "Inspector"}</span>
              {selectedNode && (
                <span className="rounded bg-cyan-500/20 px-1.5 py-0.5 font-mono text-[9px] text-cyan-300">
                  Selected
                </span>
              )}
            </button>

            <button
              type="button"
              onClick={() => openRightPanelDialog("chat")}
              className={
                particleGraphEditorEnabled
                  ? "pointer-events-auto flex items-center gap-2 rounded-lg border border-indigo-400/65 bg-indigo-950/85 px-3 py-2 font-mono text-xs font-bold uppercase tracking-wider text-indigo-100 shadow-[0_0_20px_rgba(99,102,241,0.3)] backdrop-blur-md transition-all hover:border-indigo-300 hover:bg-indigo-900/85"
                  : "pointer-events-auto flex items-center gap-2 rounded-lg border border-indigo-700 bg-indigo-950/95 px-3 py-2 text-xs font-semibold text-indigo-100 shadow-lg hover:bg-indigo-900"
              }
              title="Open Model Chat Dialog"
            >
              <span>💬 Model Chat</span>
            </button>
          </div>

          {/* BOTTOM EDGE BUTTON -> Opens Training & Telemetry Dialog */}
          <div className="pointer-events-none absolute bottom-3 left-1/2 z-20 -translate-x-1/2">
            <button
              type="button"
              onClick={() => setActivePanelDialog("bottom")}
              className={
                particleGraphEditorEnabled
                  ? "pointer-events-auto flex items-center gap-3 rounded-full border border-cyan-400/70 bg-slate-950/92 px-5 py-2 font-mono text-xs font-bold uppercase tracking-wider text-cyan-100 shadow-[0_0_24px_rgba(0,240,255,0.32)] backdrop-blur-md transition-all hover:border-cyan-300 hover:bg-cyan-950/80 hover:shadow-[0_0_32px_rgba(0,240,255,0.5)]"
                  : "pointer-events-auto flex items-center gap-3 rounded-full border border-gray-700 bg-gray-900/95 px-5 py-2 text-xs font-semibold text-gray-100 shadow-xl hover:bg-gray-800"
              }
              title="Open Training, Loss Telemetry & Torch Trace Dialog"
            >
              <span className={isTraining ? "animate-pulse text-emerald-300" : "text-cyan-300"}>
                ⚡ {isTraining ? "Training Active" : "Training & Trace"}
              </span>
              <span className="rounded border border-cyan-500/30 bg-black/40 px-2 py-0.5 font-mono text-[10px] text-amber-300">
                {rootGraph.training_method} · {rootGraph.runtime}
              </span>
              {latestLoss && (
                <span className="rounded bg-emerald-950/80 px-2 py-0.5 font-mono text-[10px] text-emerald-300">
                  Loss: {latestLoss}
                </span>
              )}
            </button>
          </div>
        </div>

        {/* 1. LEFT PANEL DIALOG: Variant Matrix / Library */}
        {activePanelDialog === "left" && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-md"
            onClick={() => setActivePanelDialog(null)}
          >
            <div
              className={`flex max-h-[82vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border shadow-2xl ${
                particleGraphEditorEnabled
                  ? "border-amber-400/50 bg-[#030917]/95 shadow-[0_0_50px_rgba(245,158,11,0.22)]"
                  : "border-gray-800 bg-gray-950"
              }`}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between border-b border-amber-500/25 bg-slate-950/90 px-5 py-3.5">
                <div className="flex items-center gap-2.5">
                  <span className="font-mono text-sm font-bold uppercase tracking-wider text-amber-300">
                    ◈ Variant Matrix &amp; Subgraph Library
                  </span>
                  <span className="rounded border border-amber-500/30 bg-amber-950/50 px-2 py-0.5 font-mono text-[10px] text-amber-200">
                    {variantCount} families
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setActivePanelDialog(null)}
                  className="rounded-lg border border-gray-700 bg-gray-900 px-3 py-1 font-mono text-xs text-gray-300 hover:border-amber-400/60 hover:text-white"
                >
                  ✕ Close (ESC)
                </button>
              </div>
              <div className="flex-1 min-h-0 overflow-auto">
                <LibraryPanel onDone={() => setActivePanelDialog(null)} />
              </div>
            </div>
          </div>
        )}

        {/* 2. BOTTOM PANEL DIALOG: Training, Loss Curve & Torch Trace (kept mounted so active training & trace polling persist) */}
        <div
          className={
            activePanelDialog === "bottom"
              ? "fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-md"
              : "hidden"
          }
          onClick={() => setActivePanelDialog(null)}
        >
          <div
            className={`flex max-h-[86vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border shadow-2xl ${
              particleGraphEditorEnabled
                ? "border-cyan-400/50 bg-[#030917]/95 shadow-[0_0_55px_rgba(0,240,255,0.25)]"
                : "border-gray-800 bg-gray-950"
            }`}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-cyan-500/25 bg-slate-950/90 px-5 py-3.5">
              <div className="flex items-center gap-3">
                <span className="font-mono text-sm font-bold uppercase tracking-wider text-cyan-300">
                  ⚡ Training, Telemetry &amp; Torch Trace
                </span>
                {isTraining && (
                  <span className="animate-pulse rounded border border-emerald-400/60 bg-emerald-950/70 px-2 py-0.5 font-mono text-[10px] font-bold uppercase text-emerald-300">
                    ● Run Active
                  </span>
                )}
              </div>
              <button
                type="button"
                onClick={() => setActivePanelDialog(null)}
                className="rounded-lg border border-gray-700 bg-gray-900 px-3 py-1 font-mono text-xs text-gray-300 hover:border-cyan-400/60 hover:text-white"
              >
                ✕ Close (ESC)
              </button>
            </div>
            <div className="flex-1 min-h-0 overflow-auto">
              <TrainingPanel />
            </div>
          </div>
        </div>

        {/* 3. RIGHT PANEL DIALOG: Node/Graph Inspector, Code Editor & Model Chat (kept mounted so chat & editor state persist) */}
        <div
          className={
            activePanelDialog === "right"
              ? "fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-md"
              : "hidden"
          }
          onClick={() => setActivePanelDialog(null)}
        >
          <div
            className={`flex h-[82vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border shadow-2xl ${
              particleGraphEditorEnabled
                ? "border-cyan-400/50 bg-[#030917]/95 shadow-[0_0_55px_rgba(0,240,255,0.25)]"
                : "border-gray-800 bg-gray-950"
            }`}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-cyan-500/25 bg-slate-950/90 px-5 py-3">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => openRightPanelDialog("inspector")}
                  className={`rounded-lg border px-3 py-1.5 font-mono text-xs font-bold uppercase tracking-wider transition-all ${
                    rightPanelTab === "inspector"
                      ? "border-cyan-400/80 bg-cyan-500/25 text-cyan-100 shadow-[0_0_16px_rgba(0,240,255,0.3)]"
                      : "border-gray-800 bg-gray-900/70 text-gray-400 hover:text-gray-200"
                  }`}
                >
                  ◧ {selectedNode ? `Inspector: ${selectedNode.data.label}` : "Node & Graph Inspector"}
                </button>
                <button
                  type="button"
                  onClick={() => openRightPanelDialog("chat")}
                  className={`rounded-lg border px-3 py-1.5 font-mono text-xs font-bold uppercase tracking-wider transition-all ${
                    rightPanelTab === "chat"
                      ? "border-indigo-400/80 bg-indigo-500/25 text-indigo-100 shadow-[0_0_16px_rgba(99,102,241,0.35)]"
                      : "border-gray-800 bg-gray-900/70 text-gray-400 hover:text-gray-200"
                  }`}
                >
                  💬 Model Chat
                </button>
              </div>
              <button
                type="button"
                onClick={() => setActivePanelDialog(null)}
                className="rounded-lg border border-gray-700 bg-gray-900 px-3 py-1 font-mono text-xs text-gray-300 hover:border-cyan-400/60 hover:text-white"
              >
                ✕ Close (ESC)
              </button>
            </div>
            <div className="flex flex-1 min-h-0 overflow-hidden">
              <div
                className={
                  rightPanelTab === "inspector"
                    ? "flex flex-1 min-h-0 w-full overflow-hidden"
                    : "hidden"
                }
              >
                <CodePanel />
              </div>
              <div
                className={
                  rightPanelTab === "chat"
                    ? "flex flex-1 min-h-0 w-full overflow-hidden"
                    : "hidden"
                }
              >
                <ChatPanel />
              </div>
            </div>
          </div>
        </div>
      </div>
    </ReactFlowProvider>
  );
}
