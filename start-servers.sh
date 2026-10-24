#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
EDITOR_PORT="${EDITOR_PORT:-5173}"

# Activate local .venv if present and no virtual/conda environment is active
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" && -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

# Resolve uvicorn command
if command -v uvicorn >/dev/null 2>&1; then
  UVICORN_CMD=(uvicorn)
elif command -v python3 >/dev/null 2>&1 && python3 -c "import uvicorn" >/dev/null 2>&1; then
  UVICORN_CMD=(python3 -m uvicorn)
elif command -v python >/dev/null 2>&1 && python -c "import uvicorn" >/dev/null 2>&1; then
  UVICORN_CMD=(python -m uvicorn)
else
  echo "Error: uvicorn is not installed in the active Python environment." >&2
  echo "Install server dependencies with: pip install -r requirements-full.txt" >&2
  exit 1
fi

# Resolve frontend package manager and ensure editor dependencies exist
if command -v pnpm >/dev/null 2>&1; then
  PKG_MGR="pnpm"
elif command -v npm >/dev/null 2>&1; then
  PKG_MGR="npm"
else
  echo "Error: Neither pnpm nor npm was found on PATH." >&2
  exit 1
fi

if [[ ! -d "$ROOT_DIR/editor/node_modules" ]]; then
  echo "[start_servers] Installing editor dependencies with $PKG_MGR..."
  (cd "$ROOT_DIR/editor" && "$PKG_MGR" install)
fi

BACKEND_PID=""
EDITOR_PID=""

cleanup() {
  local exit_code=$?
  trap - INT TERM EXIT
  echo ""
  echo "[start_servers] Shutting down backend and editor..."
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$EDITOR_PID" ]] && kill -0 "$EDITOR_PID" 2>/dev/null; then
    kill "$EDITOR_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  exit "$exit_code"
}

trap cleanup INT TERM EXIT

echo "============================================================"
echo " NeuralFn Platform Launcher"
echo "------------------------------------------------------------"
echo " Backend API : http://${BACKEND_HOST}:${BACKEND_PORT}"
echo " Web Editor  : http://localhost:${EDITOR_PORT}"
echo " Press Ctrl+C to stop both servers."
echo "============================================================"

"${UVICORN_CMD[@]}" server.app:app --reload --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
BACKEND_PID=$!

(
  cd "$ROOT_DIR/editor"
  "$PKG_MGR" run dev -- --port "$EDITOR_PORT"
) &
EDITOR_PID=$!

# Monitor both background processes; exit if either terminates
while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$EDITOR_PID" 2>/dev/null; do
  sleep 1
done

if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
  wait "$BACKEND_PID"
elif ! kill -0 "$EDITOR_PID" 2>/dev/null; then
  wait "$EDITOR_PID"
fi
