import React, { useEffect, useState } from "react";
import { api, type MembershipInfo, type ProjectSummary, type UserData } from "../api/client";
import { useAppState } from "../routes/AppState";

export default function AdminPage() {
  const { bootstrap, refreshBootstrap } = useAppState();
  const [users, setUsers] = useState<UserData[]>([]);
  const [memberships, setMemberships] = useState<MembershipInfo[]>([]);
  const [projectName, setProjectName] = useState("");
  const [userEmail, setUserEmail] = useState("");
  const [userPassword, setUserPassword] = useState("");
  const [userDisplayName, setUserDisplayName] = useState("");
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");

  const projects = bootstrap?.projects ?? [];
  const activeProjectId = selectedProjectId || bootstrap?.active_project_id || projects[0]?.id || "";

  useEffect(() => {
    if (!bootstrap?.user?.is_admin) return;
    let cancelled = false;
    Promise.all([
      api.listUsers(),
      activeProjectId ? api.listMemberships(activeProjectId) : Promise.resolve([]),
    ])
      .then(([userList, membershipList]) => {
        if (!cancelled) {
          setUsers(userList);
          setMemberships(membershipList);
          if (!selectedProjectId && activeProjectId) {
            setSelectedProjectId(activeProjectId);
          }
        }
      })
      .catch(() => {
        if (!cancelled) {
          setUsers([]);
          setMemberships([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [bootstrap?.user?.is_admin, activeProjectId, selectedProjectId]);

  if (!bootstrap?.user?.is_admin) {
    return (
      <div className="p-6 text-sm text-gray-500">
        Admin access is required to view this page.
      </div>
    );
  }

  const handleCreateProject = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!projectName.trim()) return;
    await api.createProject({ name: projectName.trim() });
    setProjectName("");
    await refreshBootstrap();
  };

  const handleCreateUser = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!userEmail.trim() || !userPassword.trim() || !userDisplayName.trim()) return;
    await api.createUser({
      email: userEmail.trim(),
      password: userPassword,
      display_name: userDisplayName.trim(),
      is_admin: false,
    });
    setUserEmail("");
    setUserPassword("");
    setUserDisplayName("");
    setUsers(await api.listUsers());
  };

  const currentProject = projects.find((project) => project.id === activeProjectId) ?? null;

  return (
    <div className="p-6 grid gap-6 lg:grid-cols-2">
      <div className="space-y-4">
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <div className="text-lg font-semibold text-gray-100">Projects</div>
          <div className="mt-1 text-sm text-gray-500">Create and inspect project workspaces.</div>
          <form className="mt-4 flex gap-2" onSubmit={handleCreateProject}>
            <input
              value={projectName}
              onChange={(event) => setProjectName(event.target.value)}
              placeholder="New project name"
              className="flex-1 rounded border border-gray-700 bg-gray-800 px-3 py-2 text-sm"
            />
            <button className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500">
              Create
            </button>
          </form>
          <div className="mt-4 space-y-2">
            {projects.map((project: ProjectSummary) => (
              <button
                key={project.id}
                onClick={() => setSelectedProjectId(project.id)}
                className={`w-full rounded border px-3 py-2 text-left text-sm ${
                  activeProjectId === project.id
                    ? "border-blue-500 bg-blue-950/20 text-blue-200"
                    : "border-gray-800 bg-gray-950 text-gray-300"
                }`}
              >
                <div className="font-medium">{project.name}</div>
                <div className="text-xs text-gray-500">{project.slug}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <div className="text-lg font-semibold text-gray-100">Users</div>
          <form className="mt-4 space-y-2" onSubmit={handleCreateUser}>
            <input
              value={userDisplayName}
              onChange={(event) => setUserDisplayName(event.target.value)}
              placeholder="Display name"
              className="w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 text-sm"
            />
            <input
              value={userEmail}
              onChange={(event) => setUserEmail(event.target.value)}
              placeholder="Email"
              className="w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 text-sm"
            />
            <input
              type="password"
              value={userPassword}
              onChange={(event) => setUserPassword(event.target.value)}
              placeholder="Temporary password"
              className="w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 text-sm"
            />
            <button className="rounded bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500">
              Create User
            </button>
          </form>
          <div className="mt-4 divide-y divide-gray-800">
            {users.map((user) => (
              <div key={user.id} className="py-2 text-sm text-gray-300">
                <div>{user.display_name}</div>
                <div className="text-xs text-gray-500">
                  {user.email}
                  {user.is_admin ? " • admin" : ""}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <div className="text-lg font-semibold text-gray-100">Project Memberships</div>
        <div className="mt-1 text-sm text-gray-500">
          {currentProject ? `Current project: ${currentProject.name}` : "Select a project to inspect memberships."}
        </div>
        <div className="mt-4 divide-y divide-gray-800">
          {memberships.length === 0 ? (
            <div className="py-3 text-sm text-gray-500">No memberships found for this project.</div>
          ) : (
            memberships.map((membership) => (
              <div key={membership.id} className="py-3 text-sm text-gray-300">
                <div>{membership.user.display_name}</div>
                <div className="text-xs text-gray-500">
                  {membership.user.email} • {membership.role}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
