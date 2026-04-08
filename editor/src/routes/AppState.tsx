import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, type BootstrapResponse } from "../api/client";

interface AppStateValue {
  bootstrapping: boolean;
  bootstrap: BootstrapResponse | null;
  error: string | null;
  refreshBootstrap: () => Promise<BootstrapResponse | null>;
  login: (email: string, password: string) => Promise<void>;
  bootstrapAdmin: (email: string, password: string, displayName: string) => Promise<void>;
  logout: () => Promise<void>;
  setActiveSession: (projectId: string | null, sessionId: string | null) => Promise<void>;
}

const AppStateContext = createContext<AppStateValue | null>(null);

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [bootstrapping, setBootstrapping] = useState(true);
  const [bootstrap, setBootstrap] = useState<BootstrapResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshBootstrap = useCallback(async () => {
    setBootstrapping(true);
    try {
      const payload = await api.getBootstrap();
      setBootstrap(payload);
      setError(null);
      return payload;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load app state");
      setBootstrap(null);
      return null;
    } finally {
      setBootstrapping(false);
    }
  }, []);

  useEffect(() => {
    refreshBootstrap().catch(() => undefined);
  }, [refreshBootstrap]);

  const login = useCallback(
    async (email: string, password: string) => {
      await api.login({ email, password });
      await refreshBootstrap();
    },
    [refreshBootstrap],
  );

  const bootstrapAdmin = useCallback(
    async (email: string, password: string, displayName: string) => {
      await api.bootstrapAdmin({
        email,
        password,
        display_name: displayName,
      });
      await refreshBootstrap();
    },
    [refreshBootstrap],
  );

  const logout = useCallback(async () => {
    await api.logout();
    await refreshBootstrap();
  }, [refreshBootstrap]);

  const setActiveSession = useCallback(
    async (projectId: string | null, sessionId: string | null) => {
      await api.setActiveSession(projectId, sessionId);
      await refreshBootstrap();
    },
    [refreshBootstrap],
  );

  const value = useMemo<AppStateValue>(
    () => ({
      bootstrapping,
      bootstrap,
      error,
      refreshBootstrap,
      login,
      bootstrapAdmin,
      logout,
      setActiveSession,
    }),
    [bootstrapping, bootstrap, error, refreshBootstrap, login, bootstrapAdmin, logout, setActiveSession],
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const value = useContext(AppStateContext);
  if (!value) {
    throw new Error("useAppState must be used inside AppStateProvider");
  }
  return value;
}
