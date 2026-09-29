#!/bin/bash
# Launch the MultimodalSDK web demo.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
VENV="${MM_WEB_VENV:-$HOME/mm-sdk-web/venv}"
export PYTHONPATH="${MM_SDK_SOURCE:-$SCRIPT_DIR/../source}:${PYTHONPATH:-}"
export MM_WEB_LLAMA_URL="${MM_WEB_LLAMA_URL:-http://127.0.0.1:18810}"
export MM_WEB_HOST="${MM_WEB_HOST:-0.0.0.0}"
export MM_WEB_PORT="${MM_WEB_PORT:-8090}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "python venv not found at $VENV (set MM_WEB_VENV to override)" >&2
  exit 1
fi

exec "$VENV/bin/python" "$SCRIPT_DIR/app.py"
