import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAppState } from "../routes/AppState";

export default function LoginPage() {
  const { bootstrap, login, bootstrapAdmin, error } = useAppState();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("Admin");
  const [submitting, setSubmitting] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const requiresSetup = bootstrap?.requires_setup ?? false;

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setLocalError(null);
    try {
      if (requiresSetup) {
        await bootstrapAdmin(email, password, displayName);
      } else {
        await login(email, password);
      }
      navigate("/");
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center p-6">
      <div className="w-full max-w-md rounded-xl border border-gray-800 bg-gray-900 p-6 shadow-2xl">
        <div className="mb-6">
          <div className="text-2xl font-bold text-blue-400">NeuralFn</div>
          <div className="mt-1 text-sm text-gray-400">
            {requiresSetup
              ? "Create the first admin account and workspace."
              : "Sign in to your project-scoped editor session."}
          </div>
        </div>

        <form className="space-y-4" onSubmit={handleSubmit}>
          {requiresSetup && (
            <label className="block text-sm text-gray-300">
              Display Name
              <input
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                className="mt-1 w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 text-gray-100"
              />
            </label>
          )}

          <label className="block text-sm text-gray-300">
            Email
            <input
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              className="mt-1 w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 text-gray-100"
              autoComplete="email"
            />
          </label>

          <label className="block text-sm text-gray-300">
            Password
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="mt-1 w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 text-gray-100"
              autoComplete={requiresSetup ? "new-password" : "current-password"}
            />
          </label>

          {(localError || error) && (
            <div className="rounded border border-rose-800 bg-rose-950/40 px-3 py-2 text-sm text-rose-200">
              {localError || error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting || !email || !password}
            className="w-full rounded bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500"
          >
            {submitting
              ? "Working..."
              : requiresSetup
                ? "Create Admin Workspace"
                : "Sign In"}
          </button>
        </form>
      </div>
    </div>
  );
}
