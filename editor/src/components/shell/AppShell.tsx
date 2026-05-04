import React, { useEffect, useMemo, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate, useParams } from "react-router-dom";
import { api, type SessionSummary } from "../../api/client";
import { useAppState } from "../../routes/AppState";

function surfaceForPath(pathname: string) {
  if (pathname.includes("/studio")) return "studio";
  if (pathname.includes("/datasets")) return "datasets";
  if (pathname.includes("/runs")) return "runs";
  if (pathname.includes("/analytics")) return "analytics";
  if (pathname.includes("/admin")) return "admin";
  return "editor";
}

function scopedPath(projectId: string, sessionId: string, surface: string) {
  return `/app/projects/${projectId}/sessions/${sessionId}/${surface}`;
}

export default function AppShell() {
  const { bootstrap, logout, refreshBootstrap, setActiveSession } = useAppState();
  const navigate = useNavigate();
  const location = useLocation();
  const params = useParams<{ projectId: string; sessionId: string }>();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [showCreateProject, setShowCreateProject] = useState(false);
  const [creatingProject, setCreatingProject] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [projectDescription, setProjectDescription] = useState("");
  const [projectError, setProjectError] = useState<string | null>(null);

  const projects = bootstrap?.projects ?? [];
  const activeProjectId = params.projectId ?? bootstrap?.active_project_id ?? null;
  const activeSessionId = params.sessionId ?? bootstrap?.active_session_id ?? null;
  const currentSurface = surfaceForPath(location.pathname);

  useEffect(() => {
    if (!activeProjectId) {
      setSessions([]);
      return;
    }
    let cancelled = false;
    setLoadingSessions(true);
    api.listSessions(activeProjectId)
      .then((payload) => {
        if (!cancelled) {
          setSessions(payload);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSessions([]);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingSessions(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [activeProjectId]);

  const currentProject = useMemo(
    () => projects.find((project) => project.id === activeProjectId) ?? null,
    [projects, activeProjectId],
  );

  const handleProjectChange = async (projectId: string) => {
    const nextSessions = await api.listSessions(projectId);
    const nextSessionId = nextSessions[0]?.id ?? null;
    await setActiveSession(projectId, nextSessionId);
    setSessions(nextSessions);
    if (nextSessionId) {
      navigate(scopedPath(projectId, nextSessionId, currentSurface));
    } else {
      navigate("/app");
    }
  };

  const handleSessionChange = async (sessionId: string) => {
    if (!activeProjectId) return;
    await setActiveSession(activeProjectId, sessionId);
    navigate(scopedPath(activeProjectId, sessionId, currentSurface));
  };

  const handleCreateProject = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!projectName.trim()) return;
    setCreatingProject(true);
    setProjectError(null);
    try {
      const created = await api.createProject({
        name: projectName.trim(),
        description: projectDescription.trim() || null,
      });
      await refreshBootstrap();
      setSessions([created.default_session]);
      setProjectName("");
      setProjectDescription("");
      setShowCreateProject(false);
      const nextSurface = currentSurface === "admin" ? "editor" : currentSurface;
      navigate(scopedPath(created.project.id, created.default_session.id, nextSurface));
    } catch (err) {
      setProjectError(err instanceof Error ? err.message : "Failed to create project");
    } finally {
      setCreatingProject(false);
    }
  };

  return (
    <div className="h-screen bg-gray-950 text-gray-100 flex flex-col">
      <header className="border-b border-gray-800 bg-gray-900/95">
        <div className="flex items-center gap-3 px-4 py-3">
          <div>
            <div className="text-lg font-bold text-blue-400">NeuralFn Platform</div>
            <div className="text-[11px] text-gray-500">
              {bootstrap?.user ? `${bootstrap.user.display_name} • ${bootstrap.user.email}` : "Signed out"}
            </div>
          </div>

          <div className="ml-6 flex items-center gap-2">
            <select
              value={activeProjectId ?? ""}
              onChange={(event) => void handleProjectChange(event.target.value)}
              className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm"
            >
              {projects.length === 0 ? (
                <option value="">No projects</option>
              ) : (
                projects.map((project) => (
                  <option key={project.id} value={project.id}>
                    {project.name}
                  </option>
                ))
              )}
            </select>

            <select
              value={activeSessionId ?? ""}
              disabled={!activeProjectId || loadingSessions || sessions.length === 0}
              onChange={(event) => void handleSessionChange(event.target.value)}
              className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm"
            >
              {sessions.length === 0 ? (
                <option value="">{loadingSessions ? "Loading sessions..." : "No sessions"}</option>
              ) : (
                sessions.map((session) => (
                  <option key={session.id} value={session.id}>
                    {session.name}
                  </option>
                ))
              )}
            </select>

            <button
              onClick={() => {
                setProjectError(null);
                setShowCreateProject(true);
              }}
              className="rounded border border-gray-700 px-3 py-1 text-sm text-gray-200 hover:bg-gray-800"
            >
              New Project
            </button>
          </div>

          <nav className="ml-6 flex items-center gap-2 text-sm">
            {activeProjectId && activeSessionId && (
              <>
                <NavLink
                  to={scopedPath(activeProjectId, activeSessionId, "editor")}
                  className={({ isActive }) =>
                    `rounded px-3 py-1 ${isActive ? "bg-blue-600 text-white" : "text-gray-300 hover:bg-gray-800"}`
                  }
                >
                  Editor
                </NavLink>
                <NavLink
                  to={scopedPath(activeProjectId, activeSessionId, "studio")}
                  className={({ isActive }) =>
                    `rounded px-3 py-1 ${isActive ? "bg-indigo-600 text-white" : "text-indigo-200 hover:bg-indigo-900/40"}`
                  }
                >
                  🧪 Studio
                </NavLink>
                <NavLink
                  to={scopedPath(activeProjectId, activeSessionId, "datasets")}
                  className={({ isActive }) =>
                    `rounded px-3 py-1 ${isActive ? "bg-blue-600 text-white" : "text-gray-300 hover:bg-gray-800"}`
                  }
                >
                  Datasets
                </NavLink>
                <NavLink
                  to={scopedPath(activeProjectId, activeSessionId, "runs")}
                  className={({ isActive }) =>
                    `rounded px-3 py-1 ${isActive ? "bg-blue-600 text-white" : "text-gray-300 hover:bg-gray-800"}`
                  }
                >
                  Runs
                </NavLink>
                <NavLink
                  to={scopedPath(activeProjectId, activeSessionId, "analytics")}
                  className={({ isActive }) =>
                    `rounded px-3 py-1 ${isActive ? "bg-blue-600 text-white" : "text-gray-300 hover:bg-gray-800"}`
                  }
                >
                  Analytics
                </NavLink>
              </>
            )}
            {bootstrap?.user?.is_admin && (
              <NavLink
                to="/app/admin"
                className={({ isActive }) =>
                  `rounded px-3 py-1 ${isActive ? "bg-blue-600 text-white" : "text-gray-300 hover:bg-gray-800"}`
                }
              >
                Admin
              </NavLink>
            )}
          </nav>

          <div className="ml-auto flex items-center gap-3 text-xs text-gray-500">
            {currentProject && <span>{currentProject.role}</span>}
            <button
              onClick={() => void logout().then(() => navigate("/login"))}
              className="rounded bg-gray-800 px-3 py-1 text-gray-200 hover:bg-gray-700"
            >
              Log out
            </button>
          </div>
        </div>
      </header>

      <main className="flex flex-1 min-h-0 flex-col">
        <Outlet />
      </main>

      {showCreateProject && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4">
          <div className="w-full max-w-md rounded-xl border border-gray-800 bg-gray-900 p-5 shadow-2xl">
            <div className="text-lg font-semibold text-gray-100">Create Personal Project</div>
            <div className="mt-1 text-sm text-gray-500">
              A new project will be created for you with a seeded `Main session`.
            </div>

            <form className="mt-4 space-y-3" onSubmit={handleCreateProject}>
              <label className="block text-sm text-gray-400">
                Project Name
                <input
                  value={projectName}
                  onChange={(event) => setProjectName(event.target.value)}
                  placeholder="My project"
                  className="mt-1 block w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-100"
                />
              </label>

              <label className="block text-sm text-gray-400">
                Description
                <textarea
                  value={projectDescription}
                  onChange={(event) => setProjectDescription(event.target.value)}
                  placeholder="Optional notes"
                  className="mt-1 block h-24 w-full resize-none rounded border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-100"
                />
              </label>

              {projectError && (
                <div className="rounded border border-rose-800 bg-rose-950/40 px-3 py-2 text-sm text-rose-200">
                  {projectError}
                </div>
              )}

              <div className="flex items-center justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => {
                    setShowCreateProject(false);
                    setProjectError(null);
                  }}
                  className="rounded border border-gray-700 px-4 py-2 text-sm text-gray-300 hover:bg-gray-800"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={creatingProject || !projectName.trim()}
                  className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:bg-gray-700"
                >
                  {creatingProject ? "Creating..." : "Create Project"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
