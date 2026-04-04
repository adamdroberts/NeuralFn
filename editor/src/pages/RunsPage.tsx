import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, type RunSnapshot } from "../api/client";

export default function RunsPage() {
  const { projectId, sessionId } = useParams<{ projectId: string; sessionId: string }>();
  const [runs, setRuns] = useState<Array<Record<string, unknown>>>([]);
  const [activeRun, setActiveRun] = useState<RunSnapshot | null>(null);

  useEffect(() => {
    if (!projectId || !sessionId) return;
    let cancelled = false;

    const refresh = async () => {
      try {
        const [runList, active] = await Promise.all([
          api.listRuns(projectId, sessionId),
          api.getActiveRun(projectId, sessionId),
        ]);
        if (!cancelled) {
          setRuns(runList);
          setActiveRun(active);
        }
      } catch {
        if (!cancelled) {
          setRuns([]);
          setActiveRun(null);
        }
      }
    };

    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [projectId, sessionId]);

  return (
    <div className="p-6 space-y-4">
      <div>
        <div className="text-xl font-semibold text-gray-100">Runs</div>
        <div className="text-sm text-gray-500">Live and recent training runs for the active session.</div>
      </div>

      {activeRun?.run_id && (
        <div className="rounded-lg border border-blue-900 bg-blue-950/20 p-4">
          <div className="text-sm font-semibold text-blue-300">Active Run</div>
          <div className="mt-2 grid gap-2 text-sm text-gray-300 md:grid-cols-4">
            <div>Status: {activeRun.status ?? "unknown"}</div>
            <div>Method: {activeRun.method ?? activeRun.requested_method ?? "unknown"}</div>
            <div>Loss: {activeRun.last_loss ?? "-"}</div>
            <div>Step: {activeRun.last_step ?? "-"}</div>
          </div>
        </div>
      )}

      <div className="rounded-lg border border-gray-800 bg-gray-900">
        <div className="border-b border-gray-800 px-4 py-3 text-sm font-semibold text-gray-200">
          Recent Runs
        </div>
        <div className="divide-y divide-gray-800">
          {runs.length === 0 ? (
            <div className="px-4 py-6 text-sm text-gray-500">No runs recorded for this session yet.</div>
          ) : (
            runs.map((run) => (
              <div key={String(run.id)} className="px-4 py-3 text-sm text-gray-300 grid gap-2 md:grid-cols-5">
                <div className="font-mono text-gray-400">{String(run.id)}</div>
                <div>{String(run.status ?? "unknown")}</div>
                <div>{String(run.resolved_method ?? run.requested_method ?? "unknown")}</div>
                <div>{run.last_loss == null ? "-" : String(run.last_loss)}</div>
                <div>{String(run.started_at ?? "")}</div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
