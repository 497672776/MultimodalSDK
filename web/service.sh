#!/bin/bash
# Start/stop the MultimodalSDK web demo on the target board.
#
#   MM_WEB_HOME      deployment root          (default: $HOME/mm-sdk-web)
#   MM_WEB_VENV      python venv              (default: $MM_WEB_HOME/venv)
#   MM_SDK_SOURCE    MultimodalSDK source dir (default: $MM_WEB_HOME/sdk/source)
#   MM_WEB_LLAMA_URL llama-server base url    (default: http://127.0.0.1:18810)
#   MM_WEB_PORT      listen port              (default: 8090)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
APP_HOME="${MM_WEB_HOME:-$HOME/mm-sdk-web}"
VENV="${MM_WEB_VENV:-$APP_HOME/venv}"
SDK_SOURCE="${MM_SDK_SOURCE:-$APP_HOME/sdk/source}"
export MM_WEB_LLAMA_URL="${MM_WEB_LLAMA_URL:-http://127.0.0.1:18810}"
export MM_WEB_PORT="${MM_WEB_PORT:-8090}"
export MM_WEB_HOST="${MM_WEB_HOST:-0.0.0.0}"
export MM_WEB_WORK_DIR="${MM_WEB_WORK_DIR:-$APP_HOME/work}"

LOG_DIR="$APP_HOME/logs"
RUN_DIR="$APP_HOME/run"
LOG_FILE="$LOG_DIR/mm-web.log"
PID_FILE="$RUN_DIR/mm-web.pid"

start() {
  mkdir -p "$LOG_DIR" "$RUN_DIR" "$MM_WEB_WORK_DIR"
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already running (pid $(cat "$PID_FILE"))"
    return 0
  fi
  PYTHONPATH="$SDK_SOURCE" nohup "$VENV/bin/python" "$SCRIPT_DIR/app.py" >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  sleep 3
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "started (pid $(cat "$PID_FILE"), port $MM_WEB_PORT), log: $LOG_FILE"
  else
    echo "failed to start, see $LOG_FILE" >&2
    tail -20 "$LOG_FILE" >&2 || true
    exit 1
  fi
}

stop() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
      kill -9 "$pid" 2>/dev/null || true
      echo "stopped (pid $pid)"
    fi
    rm -f "$PID_FILE"
  else
    echo "not running"
  fi
}

status() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "running (pid $(cat "$PID_FILE"))"
    curl -s -m 10 "http://127.0.0.1:$MM_WEB_PORT/api/status" | head -c 600
    echo
  else
    echo "not running"
  fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
