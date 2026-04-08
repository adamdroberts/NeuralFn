import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, type ProjectAnalytics } from "../api/client";

export default function AnalyticsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [analytics, setAnalytics] = useState<ProjectAnalytics | null>(null);

  useEffect(() => {
    if (!projectId) return;
    let cancelled = false;
    api.getProjectAnalytics(projectId)
      .then((payload) => {
        if (!cancelled) setAnalytics(payload);
      })
      .catch(() => {
        if (!cancelled) setAnalytics(null);
      });
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  return (
    <div className="p-6 space-y-4">
      <div>
        <div className="text-xl font-semibold text-gray-100">Analytics</div>
        <div className="text-sm text-gray-500">Project-level activity and recent run status trends.</div>
      </div>

      {!analytics ? (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-6 text-sm text-gray-500">
          No analytics available yet.
        </div>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <div className="text-xs uppercase tracking-wider text-gray-500">Sessions</div>
              <div className="mt-2 text-2xl font-semibold text-gray-100">{analytics.session_count}</div>
            </div>
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <div className="text-xs uppercase tracking-wider text-gray-500">Recent Runs</div>
              <div className="mt-2 text-2xl font-semibold text-gray-100">{analytics.recent_run_count}</div>
            </div>
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <div className="text-xs uppercase tracking-wider text-gray-500">Statuses</div>
              <div className="mt-2 text-sm text-gray-300">
                {Object.entries(analytics.status_counts).length === 0
                  ? "No run status data yet."
                  : Object.entries(analytics.status_counts).map(([status, count]) => `${status}: ${count}`).join(", ")}
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-gray-800 bg-gray-900">
            <div className="border-b border-gray-800 px-4 py-3 text-sm font-semibold text-gray-200">
              Latest Runs
            </div>
            <div className="divide-y divide-gray-800">
              {analytics.latest_runs.length === 0 ? (
                <div className="px-4 py-6 text-sm text-gray-500">No recent runs yet.</div>
              ) : (
                analytics.latest_runs.map((run) => (
                  <div key={run.id} className="grid gap-2 px-4 py-3 text-sm text-gray-300 md:grid-cols-5">
                    <div className="font-mono text-gray-400">{run.id}</div>
                    <div>{run.status}</div>
                    <div>{run.resolved_method ?? "-"}</div>
                    <div>{run.last_loss ?? "-"}</div>
                    <div>{run.started_at}</div>
                  </div>
                ))
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
