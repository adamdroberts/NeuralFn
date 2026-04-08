import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type DatasetInfo } from "../api/client";
import { useAppState } from "../routes/AppState";
import { useParams } from "react-router-dom";

function formatTokens(n: number | null | undefined) {
  if (n == null) return "?";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function deriveUploadName(file: File) {
  return file.name.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_");
}

function ensureCurrentProject(projectIds: string[], currentProjectId: string | null) {
  if (!currentProjectId) return projectIds;
  return Array.from(new Set([currentProjectId, ...projectIds]));
}

export default function DatasetsPage() {
  const params = useParams<{ projectId: string; sessionId: string }>();
  const { bootstrap } = useAppState();
  const projectId = params.projectId ?? null;
  const projects = bootstrap?.projects ?? [];

  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [accessDrafts, setAccessDrafts] = useState<Record<string, string[]>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [hfInput, setHfInput] = useState("");
  const [hfSplit, setHfSplit] = useState("train");
  const [hfMaxRows, setHfMaxRows] = useState("");
  const [selectedProjectIds, setSelectedProjectIds] = useState<string[]>(projectId ? [projectId] : []);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadStatus, setDownloadStatus] = useState("Idle");
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [savingDatasetName, setSavingDatasetName] = useState<string | null>(null);
  const [deletingDatasetName, setDeletingDatasetName] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const refreshDatasets = useCallback(async () => {
    if (!projectId) {
      setDatasets([]);
      return;
    }
    setLoading(true);
    try {
      const next = await api.getDatasets(projectId);
      setDatasets(next);
      setError(null);
    } catch (err) {
      setDatasets([]);
      setError(err instanceof Error ? err.message : "Failed to load datasets");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    setSelectedProjectIds(projectId ? [projectId] : []);
  }, [projectId]);

  useEffect(() => {
    void refreshDatasets();
  }, [refreshDatasets]);

  useEffect(() => {
    const drafts: Record<string, string[]> = {};
    for (const dataset of datasets) {
      drafts[dataset.name] = ensureCurrentProject(dataset.project_ids ?? [], projectId);
    }
    setAccessDrafts(drafts);
  }, [datasets, projectId]);

  useEffect(() => {
    if (!isDownloading) return;
    setDownloadProgress(6);
    setDownloadStatus("Contacting Hugging Face...");
    const startedAt = Date.now();
    const interval = window.setInterval(() => {
      const elapsedMs = Date.now() - startedAt;
      setDownloadProgress((prev) => {
        const next = prev < 35 ? prev + 5 : prev < 70 ? prev + 3 : prev < 92 ? prev + 1 : prev;
        return Math.min(next, 92);
      });
      if (elapsedMs > 12000) {
        setDownloadStatus("Tokenizing and indexing dataset...");
      } else if (elapsedMs > 4000) {
        setDownloadStatus("Downloading dataset files...");
      }
    }, 500);
    return () => window.clearInterval(interval);
  }, [isDownloading]);

  const selectedProjectSet = useMemo(() => new Set(selectedProjectIds), [selectedProjectIds]);

  const toggleSelectedProject = (nextProjectId: string) => {
    if (nextProjectId === projectId) return;
    setSelectedProjectIds((prev) => {
      const next = prev.includes(nextProjectId)
        ? prev.filter((candidate) => candidate !== nextProjectId)
        : [...prev, nextProjectId];
      return ensureCurrentProject(next, projectId);
    });
  };

  const toggleDraftProject = (datasetName: string, nextProjectId: string) => {
    if (nextProjectId === projectId) return;
    setAccessDrafts((prev) => {
      const current = prev[datasetName] ?? ensureCurrentProject([], projectId);
      const next = current.includes(nextProjectId)
        ? current.filter((candidate) => candidate !== nextProjectId)
        : [...current, nextProjectId];
      return {
        ...prev,
        [datasetName]: ensureCurrentProject(next, projectId),
      };
    });
  };

  const handleDownload = async () => {
    if (!projectId || !hfInput.trim()) return;
    setIsDownloading(true);
    setDownloadProgress(4);
    setDownloadStatus("Starting download...");
    setDownloadError(null);
    try {
      await api.downloadDataset(projectId, {
        hf_path: hfInput.trim(),
        hf_split: hfSplit,
        max_rows: hfMaxRows ? parseInt(hfMaxRows, 10) : null,
        project_ids: ensureCurrentProject(selectedProjectIds, projectId),
      });
      setDownloadProgress(100);
      setDownloadStatus("Download complete");
      setHfInput("");
      setHfMaxRows("");
      await refreshDatasets();
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : "Download failed");
      setDownloadStatus("Download failed");
    } finally {
      window.setTimeout(() => {
        setIsDownloading(false);
        setDownloadProgress(0);
        setDownloadStatus("Idle");
      }, 350);
    }
  };

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !projectId) return;
    try {
      await api.uploadDataset(
        projectId,
        deriveUploadName(file),
        file,
        ensureCurrentProject(selectedProjectIds, projectId),
      );
      await refreshDatasets();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleSaveAccess = async (dataset: DatasetInfo) => {
    if (!projectId) return;
    const draft = ensureCurrentProject(accessDrafts[dataset.name] ?? [], projectId);
    setSavingDatasetName(dataset.name);
    try {
      const updated = await api.setDatasetAccess(projectId, dataset.name, { project_ids: draft });
      setDatasets((prev) => prev.map((item) => (item.name === updated.name ? updated : item)));
      setAccessDrafts((prev) => ({ ...prev, [dataset.name]: updated.project_ids ?? draft }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update dataset access");
    } finally {
      setSavingDatasetName(null);
    }
  };

  const handleDelete = async (datasetName: string) => {
    if (!projectId) return;
    setDeletingDatasetName(datasetName);
    try {
      await api.deleteDataset(projectId, datasetName);
      await refreshDatasets();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete dataset");
    } finally {
      setDeletingDatasetName(null);
    }
  };

  if (!projectId) {
    return <div className="p-6 text-sm text-gray-500">Select a project session to manage datasets.</div>;
  }

  return (
    <div className="h-full overflow-auto bg-gray-950 px-6 py-5 text-gray-100">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <div>
          <div className="text-xl font-semibold text-blue-300">Datasets</div>
          <div className="mt-1 text-sm text-gray-400">
            Download or upload datasets for this project, then connect them to training through a
            `dataset_source` node in the editor.
          </div>
        </div>

        {error && (
          <div className="rounded-lg border border-rose-800 bg-rose-950/40 px-4 py-3 text-sm text-rose-200">
            {error}
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_18rem]">
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <div className="text-lg font-semibold text-gray-100">Download From Hugging Face</div>
            <div className="mt-1 text-sm text-gray-500">
              The dataset will be visible to the selected projects and available for `dataset_source`
              nodes immediately after download.
            </div>

            <div className="mt-4 flex flex-wrap gap-3">
              <label className="min-w-[18rem] flex-1 text-xs text-gray-400">
                Hugging Face Path
                <input
                  type="text"
                  value={hfInput}
                  onChange={(event) => setHfInput(event.target.value)}
                  placeholder="e.g. karpathy/tiny_shakespeare"
                  className="mt-1 block w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 font-mono text-sm text-gray-100"
                />
              </label>

              <label className="w-28 text-xs text-gray-400">
                Split
                <input
                  type="text"
                  value={hfSplit}
                  onChange={(event) => setHfSplit(event.target.value)}
                  className="mt-1 block w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 font-mono text-sm text-gray-100"
                />
              </label>

              <label className="w-28 text-xs text-gray-400">
                Max Rows
                <input
                  type="text"
                  value={hfMaxRows}
                  onChange={(event) => setHfMaxRows(event.target.value)}
                  placeholder="all"
                  className="mt-1 block w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 font-mono text-sm text-gray-100"
                />
              </label>
            </div>

            {downloadError && (
              <div className="mt-3 text-sm text-rose-300">{downloadError}</div>
            )}

            {isDownloading && (
              <div className="mt-4 rounded border border-emerald-900/70 bg-emerald-950/30 px-3 py-3">
                <div className="mb-2 flex items-center justify-between text-xs">
                  <span className="text-emerald-300">{downloadStatus}</span>
                  <span className="font-mono text-emerald-200">{downloadProgress}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded bg-gray-800">
                  <div
                    className="h-full rounded bg-emerald-500 transition-all duration-500"
                    style={{ width: `${downloadProgress}%` }}
                  />
                </div>
              </div>
            )}

            <div className="mt-4 flex flex-wrap items-center gap-3">
              <button
                onClick={handleDownload}
                disabled={isDownloading || !hfInput.trim()}
                className="rounded bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:bg-gray-700 disabled:text-gray-500"
              >
                {isDownloading ? "Downloading..." : "Download Dataset"}
              </button>

              <button
                onClick={() => fileInputRef.current?.click()}
                className="rounded bg-gray-700 px-4 py-2 text-sm font-medium text-gray-100 hover:bg-gray-600"
              >
                Upload Local File
              </button>

              <input
                ref={fileInputRef}
                type="file"
                onChange={handleUpload}
                accept=".txt,.json,.jsonl,.csv,.parquet"
                className="hidden"
              />

              <span className="text-xs text-gray-500">Supported: .txt, .json, .jsonl, .csv, .parquet</span>
            </div>
          </div>

          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <div className="text-sm font-semibold text-gray-100">Project Access</div>
            <div className="mt-1 text-xs leading-5 text-gray-500">
              New datasets are shared with these projects. The current project stays enabled because
              you are managing the catalog from its scope.
            </div>

            <div className="mt-4 space-y-2">
              {projects.map((project) => (
                <label
                  key={project.id}
                  className={`flex items-center gap-2 rounded border px-3 py-2 text-sm ${
                    selectedProjectSet.has(project.id)
                      ? "border-blue-500 bg-blue-950/20 text-blue-100"
                      : "border-gray-800 bg-gray-950 text-gray-300"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selectedProjectSet.has(project.id)}
                    disabled={project.id === projectId}
                    onChange={() => toggleSelectedProject(project.id)}
                    className="accent-blue-500"
                  />
                  <span className="min-w-0 flex-1 truncate">{project.name}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-lg font-semibold text-gray-100">Visible Datasets</div>
              <div className="mt-1 text-sm text-gray-500">
                These datasets are accessible from the current project and can be attached from the
                `dataset_source` node panel.
              </div>
            </div>
            <button
              onClick={() => void refreshDatasets()}
              className="rounded border border-gray-700 px-3 py-1.5 text-sm text-gray-300 hover:bg-gray-800"
            >
              Refresh
            </button>
          </div>

          {loading ? (
            <div className="mt-4 text-sm text-gray-500">Loading datasets...</div>
          ) : datasets.length === 0 ? (
            <div className="mt-4 rounded border border-dashed border-gray-800 bg-gray-950/40 px-4 py-6 text-sm text-gray-500">
              No datasets are visible to this project yet.
            </div>
          ) : (
            <div className="mt-4 space-y-4">
              {datasets.map((dataset) => {
                const draftProjectIds = ensureCurrentProject(accessDrafts[dataset.name] ?? [], projectId);
                return (
                  <div
                    key={dataset.name}
                    className="rounded-lg border border-gray-800 bg-gray-950/60 p-4"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-semibold text-gray-100">
                          {dataset.name}
                        </div>
                        <div className="mt-1 text-xs text-gray-500">
                          {dataset.source}
                          {dataset.hf_path ? ` • ${dataset.hf_path}` : ""}
                          {dataset.variant ? ` • ${dataset.variant}` : ""}
                          {dataset.num_tokens != null ? ` • ${formatTokens(dataset.num_tokens)} tokens` : ""}
                        </div>
                      </div>

                      <button
                        onClick={() => void handleDelete(dataset.name)}
                        disabled={deletingDatasetName === dataset.name}
                        className="rounded border border-rose-800 px-3 py-1.5 text-xs text-rose-200 hover:bg-rose-950/40 disabled:opacity-60"
                      >
                        {deletingDatasetName === dataset.name ? "Deleting..." : "Delete"}
                      </button>
                    </div>

                    <div className="mt-4 grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                      {projects.map((project) => {
                        const checked = draftProjectIds.includes(project.id);
                        return (
                          <label
                            key={project.id}
                            className={`flex items-center gap-2 rounded border px-3 py-2 text-sm ${
                              checked
                                ? "border-blue-500 bg-blue-950/20 text-blue-100"
                                : "border-gray-800 bg-gray-900 text-gray-300"
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={checked}
                              disabled={project.id === projectId}
                              onChange={() => toggleDraftProject(dataset.name, project.id)}
                              className="accent-blue-500"
                            />
                            <span className="min-w-0 flex-1 truncate">{project.name}</span>
                          </label>
                        );
                      })}
                    </div>

                    <div className="mt-4 flex items-center justify-between gap-3">
                      <div className="text-xs text-gray-500">
                        Current project access is fixed while you edit sharing from this scope.
                      </div>
                      <button
                        onClick={() => void handleSaveAccess(dataset)}
                        disabled={savingDatasetName === dataset.name}
                        className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:bg-gray-700"
                      >
                        {savingDatasetName === dataset.name ? "Saving..." : "Save Access"}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
