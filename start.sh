#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="api"
PORT_API=8765
PORT_WEB=3456

usage() {
  cat <<'EOF'
Usage: ./start.sh [--api] [--web] [--all] [--api-port PORT] [--web-port PORT]

Defaults: --api (only backend on port 8765)
Options:
  --api            Start FastAPI backend only (default)
  --web            Start React frontend only
  --all            Start both backend and frontend
  --api-port PORT  Set backend port (default 8765)
  --web-port PORT  Set frontend port (default 3456)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api) MODE="api" ;;
    --web) MODE="web" ;;
    --all) MODE="all" ;;
    --api-port) PORT_API="$2"; shift ;;
    --web-port) PORT_WEB="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
  shift
done

kill_port() {
  local port=$1
  local pids
  pids=$(lsof -ti:"$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "[start.sh] Killing existing process on port $port"
    echo "$pids" | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
}

start_api() {
  echo "[start.sh] Starting backend (port ${PORT_API})"
  kill_port "${PORT_API}"
  cd "$ROOT_DIR"
  uv run uvicorn api:app --port "${PORT_API}" --reload
}

start_web() {
  echo "[start.sh] Starting frontend (port ${PORT_WEB})"
  kill_port "${PORT_WEB}"
  cd "$ROOT_DIR/webapp"
  npm install
  VITE_API_URL="http://localhost:${PORT_API}" PORT="${PORT_WEB}" npm run dev
}

if [[ "${MODE}" == "api" ]]; then
  start_api
elif [[ "${MODE}" == "web" ]]; then
  start_web
else
  # all: run api in background, web in foreground; stop api on exit
  trap 'pkill -P $$ || true' EXIT
  start_api &
  sleep 2
  start_web
fi

