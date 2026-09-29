#!/bin/bash
# Start the vision llama-server used by the "视觉问答" tab.
#
#   MM_VLM_MODEL    path to the language model gguf
#   MM_VLM_MMPROJ   path to the multimodal projector gguf
#   MM_VLM_PORT     listen port (default 18810)
set -euo pipefail

APP_HOME="${MM_WEB_HOME:-$HOME/mm-sdk-web}"
MODEL="${MM_VLM_MODEL:-/mnt/models/vlm/Qwen3VL/Qwen3VL-4B-Instruct-Q4_K_M.gguf}"
MMPROJ="${MM_VLM_MMPROJ:-/mnt/models/vlm/Qwen3VL/mmproj-Qwen3VL-4B-Instruct-F16.gguf}"
PORT="${MM_VLM_PORT:-18810}"
THREADS="${MM_VLM_THREADS:-8}"
CTX="${MM_VLM_CTX:-8192}"

LOG_DIR="$APP_HOME/logs"
RUN_DIR="$APP_HOME/run"
LOG_FILE="$LOG_DIR/llama-vlm.log"
PID_FILE="$RUN_DIR/llama-vlm.pid"
mkdir -p "$LOG_DIR" "$RUN_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "already running (pid $(cat "$PID_FILE"))"
  exit 0
fi
if [[ ! -f "$MODEL" || ! -f "$MMPROJ" ]]; then
  echo "model or mmproj not found: $MODEL / $MMPROJ" >&2
  exit 1
fi

nohup llama-server -m "$MODEL" --mmproj "$MMPROJ" \
  --host 127.0.0.1 --port "$PORT" --ctx-size "$CTX" --threads "$THREADS" \
  --metrics --jinja >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
echo "starting llama-server pid $(cat "$PID_FILE"), port $PORT, log $LOG_FILE"
echo "wait for: curl -s http://127.0.0.1:$PORT/health"
