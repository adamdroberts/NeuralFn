import { ApiError, api } from "../api/client";
import { useGraphStore } from "../store/graphStore";

export async function reloadActiveSessionGraph() {
  const state = useGraphStore.getState();
  if (!state.projectId || !state.sessionId) {
    return null;
  }
  const detail = await api.getSession(state.projectId, state.sessionId);
  useGraphStore.getState().hydrateSession({
    projectId: state.projectId,
    sessionId: state.sessionId,
    graph: detail.graph,
    revision: detail.revision,
  });
  return detail;
}

export async function syncActiveSessionGraph(options?: {
  persistSnapshot?: boolean;
  snapshotReason?: string;
  skipIfClean?: boolean;
}) {
  const state = useGraphStore.getState();
  if (!state.projectId || !state.sessionId) {
    return null;
  }
  if (options?.skipIfClean && !state.isDirty) {
    return null;
  }

  useGraphStore.getState().setSaving(true);
  try {
    const detail = await api.putSessionGraph(state.projectId, state.sessionId, {
      graph: state.rootGraph,
      expected_revision: state.revision,
      persist_snapshot: options?.persistSnapshot ?? false,
      snapshot_reason: options?.snapshotReason ?? "autosave",
    });
    useGraphStore.getState().markSessionSaved(detail.revision);
    return detail;
  } catch (error) {
    if (error instanceof ApiError && error.status === 409) {
      await reloadActiveSessionGraph();
      useGraphStore.getState().setSaveError("Session changed on the server. Latest version reloaded.");
    } else {
      useGraphStore.getState().setSaveError(
        error instanceof Error ? error.message : "Failed to save session",
      );
    }
    throw error;
  }
}
