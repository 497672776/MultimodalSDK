#!/bin/bash
# Sync the SDK sources and this web app to the target board and restart the service.
#
#   MM_WEB_SSH            ssh target, e.g. bianbu@10.0.91.146   (required)
#   MM_WEB_HOME           remote deployment root (default: /home/<user>/mm-sdk-web)
#   MM_WEB_SSH_PASSWORD   password for sshpass based login (optional)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.."; pwd)
MM_WEB_SSH="${MM_WEB_SSH:-${1:-}}"
if [[ -z "$MM_WEB_SSH" ]]; then
  echo "usage: MM_WEB_SSH=user@host $0   (or: $0 user@host)" >&2
  exit 2
fi
REMOTE_HOME="${MM_WEB_HOME:-/home/${MM_WEB_SSH%@*}/mm-sdk-web}"
RSYNC_SSH="ssh -o StrictHostKeyChecking=no"

# Password authentication can be supplied through MM_WEB_SSH_PASSWORD.
SSH_CMD=(ssh $RSYNC_SSH)
RSYNC_CMD=(rsync $RSYNC_SSH)
if [[ -n "${MM_WEB_SSH_PASSWORD:-}" ]]; then
  RSYNC_SSH="sshpass -p $MM_WEB_SSH_PASSWORD ssh -o StrictHostKeyChecking=no"
  SSH_CMD=(sshpass -p "$MM_WEB_SSH_PASSWORD" ssh -o StrictHostKeyChecking=no)
  RSYNC_CMD=(rsync -e "$RSYNC_SSH")
fi

"${SSH_CMD[@]}" "$MM_WEB_SSH" "mkdir -p $REMOTE_HOME/sdk/source $REMOTE_HOME/web $REMOTE_HOME/work $REMOTE_HOME/logs $REMOTE_HOME/run"
"${RSYNC_CMD[@]}" -az --delete --exclude '__pycache__' "$REPO_DIR/source/" "$MM_WEB_SSH:$REMOTE_HOME/sdk/source/"
"${RSYNC_CMD[@]}" -az --delete --exclude '__pycache__' "$REPO_DIR/web/" "$MM_WEB_SSH:$REMOTE_HOME/web/"
"${SSH_CMD[@]}" "$MM_WEB_SSH" "MM_WEB_HOME=$REMOTE_HOME bash $REMOTE_HOME/web/service.sh restart"
