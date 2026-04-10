#!/usr/bin/env bash
# ============================================
# ANAMNESIS TRAINER — Stop
# ============================================
# Usage:
#   ./stop.sh                   # Auto-detect from hostname
#   ./stop.sh --host office     # Force specific compose file
#   ./stop.sh --all             # Stop all trainer containers (any host)
# ============================================

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_R_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR="${SCRIPT_R_PATH%${SCRIPT_NAME}}"

cd "$SCRIPT_DIR" &>/dev/null || true

. ~/.env.colors 2>/dev/null || true

HOST_OVERRIDE=""
DO_ALL=false
for arg in "$@"; do
	case "$arg" in
		--all) DO_ALL=true ;;
		--host) shift; HOST_OVERRIDE="$1" ;;
		--host=*) HOST_OVERRIDE="${arg#--host=}" ;;
	esac
done

if $DO_ALL; then
	echo "Stopping all trainer containers..."
	for f in "${SCRIPT_DIR}"docker-compose.*.yml; do
		[[ -f "$f" ]] || continue
		docker compose -f "$f" down &>/dev/null || true
	done
	echo "All trainers stopped."
	exit 0
fi

HOST="${HOST_OVERRIDE:-$(hostname)}"
COMPOSE_FILE="${SCRIPT_DIR}docker-compose.${HOST}.yml"

if [[ ! -f "$COMPOSE_FILE" ]]; then
	echo -e "${RED:-}No compose file for host '$HOST'. Use --host <name> or --all.${NC:-}"
	exit 1
fi

docker compose -f "$COMPOSE_FILE" down
echo "Trainer stopped: $(basename "$COMPOSE_FILE")"
