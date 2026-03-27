#!/usr/bin/env bash
# deploy_trainers.sh — sync and deploy trainer containers to GPU machines.
#
# Configure via .env (gitignored):
#   TRAINER_1_HOST=<ssh-alias>   # SERVER-1 (ROCm)
#   TRAINER_2_HOST=<ssh-alias>   # SERVER-2 (CUDA)
#
# Usage:
#   ./deploy_trainers.sh                # deploy both
#   ./deploy_trainers.sh --server1-only
#   ./deploy_trainers.sh --server2-only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TRAINERS_DIR="$REPO_ROOT/trainers"
REMOTE_PATH="~/0_GENESIS_PROJECT/0_ANAMNESIS"

# Load .env if present
[ -f "$REPO_ROOT/.env" ] && set -a && source "$REPO_ROOT/.env" && set +a

TRAINER_1_HOST="${TRAINER_1_HOST:-}"
TRAINER_2_HOST="${TRAINER_2_HOST:-}"

DEPLOY_1=true
DEPLOY_2=true

for arg in "$@"; do
  case "$arg" in
    --server1-only) DEPLOY_2=false ;;
    --server2-only) DEPLOY_1=false ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

sync_and_deploy() {
  local host="$1"
  local compose_file="$2"
  local label="$3"

  if [ -z "$host" ]; then
    echo "==> [$label] Skipped (host not configured — set TRAINER_${label}_HOST in .env)"
    return
  fi

  echo "==> [$label] Syncing trainers/ ..."
  rsync -avz --delete "$TRAINERS_DIR/" "${host}:${REMOTE_PATH}/trainers/"

  echo "==> [$label] Deploying container ..."
  ssh "$host" "mkdir -p ${REMOTE_PATH}/trainers && cd ${REMOTE_PATH} && docker compose -f trainers/${compose_file} up -d --build"

  echo "==> [$label] Done."
}

if $DEPLOY_1; then
  sync_and_deploy "$TRAINER_1_HOST" "docker-compose.trainer1.yml" "SERVER-1"
fi

if $DEPLOY_2; then
  sync_and_deploy "$TRAINER_2_HOST" "docker-compose.trainer2.yml" "SERVER-2"
fi
