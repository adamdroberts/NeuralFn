import React from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/shell/AppShell";
import { AppStateProvider, useAppState } from "./routes/AppState";
import AdminPage from "./pages/AdminPage";
import AnalyticsPage from "./pages/AnalyticsPage";
import DatasetsPage from "./pages/DatasetsPage";
import EditorPage from "./pages/EditorPage";
import LoginPage from "./pages/LoginPage";
import RunsPage from "./pages/RunsPage";

function LoadingScreen() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-300 flex items-center justify-center">
      Loading NeuralFn...
    </div>
  );
}

function RootRedirect() {
  const { bootstrap } = useAppState();
  if (bootstrap?.authenticated && bootstrap.active_project_id && bootstrap.active_session_id) {
    return (
      <Navigate
        replace
        to={`/app/projects/${bootstrap.active_project_id}/sessions/${bootstrap.active_session_id}/editor`}
      />
    );
  }
  return <Navigate replace to={bootstrap?.authenticated ? "/app" : "/login"} />;
}

function ProtectedShell() {
  const { bootstrap } = useAppState();
  if (!bootstrap?.authenticated) {
    return <Navigate replace to="/login" />;
  }
  return <AppShell />;
}

function EmptyWorkspace() {
  return (
    <div className="p-6 text-sm text-gray-500">
      No active session is selected yet. Create a project from the header controls and NeuralFn will seed a `Main session` for you automatically.
    </div>
  );
}

function RoutedApp() {
  const { bootstrapping } = useAppState();
  if (bootstrapping) {
    return <LoadingScreen />;
  }

  return (
    <Routes>
      <Route path="/" element={<RootRedirect />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/app" element={<ProtectedShell />}>
        <Route index element={<EmptyWorkspace />} />
        <Route path="admin" element={<AdminPage />} />
        <Route path="projects/:projectId/sessions/:sessionId/editor" element={<EditorPage />} />
        <Route path="projects/:projectId/sessions/:sessionId/datasets" element={<DatasetsPage />} />
        <Route path="projects/:projectId/sessions/:sessionId/runs" element={<RunsPage />} />
        <Route path="projects/:projectId/sessions/:sessionId/analytics" element={<AnalyticsPage />} />
      </Route>
      <Route path="*" element={<Navigate replace to="/" />} />
    </Routes>
  );
}

export default function App() {
  return (
    <AppStateProvider>
      <BrowserRouter>
        <RoutedApp />
      </BrowserRouter>
    </AppStateProvider>
  );
}
