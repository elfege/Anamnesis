#!/usr/bin/env bash
# ============================================
# ANAMNESIS TRAINER — Start
# ============================================
# Starts the appropriate trainer compose stack based on hostname
# or explicit --host flag. Used by deploy.sh after a rebuild, or
# standalone to start without rebuilding.
#
# Usage:
#   ./start.sh                   # Auto-detect compose file from hostname
#   ./start.sh --host office     # Force specific compose file
#   ./start.sh --host server
#   ./start.sh --host trainer1
#   ./start.sh --host trainer2
#   ./start.sh --debug           # Verbose output
# ============================================

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_R_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR="${SCRIPT_R_PATH%${SCRIPT_NAME}}"

cd "$SCRIPT_DIR" &>/dev/null || true

. ~/.env.colors 2>/dev/null || true

# ── Flags ────────────────────────────────────────────────────────
DEBUG=false
HOST_OVERRIDE=""
for arg in "$@"; do
	case "$arg" in
		--debug|-d) DEBUG=true ;;
		--host) shift; HOST_OVERRIDE="$1" ;;
		--host=*) HOST_OVERRIDE="${arg#--host=}" ;;
	esac
done

dbg() { [[ "$DEBUG" == "true" ]] && echo -e "${YELLOW:-\033[0;33m}[DEBUG] $*${NC:-\033[0m}" || true; }

# ── Detect compose file ──────────────────────────────────────────
detect_compose_file() {
	local host="${1:-$(hostname)}"
	local candidate="${SCRIPT_DIR}docker-compose.${host}.yml"
	if [[ -f "$candidate" ]]; then
		echo "$candidate"
		return 0
	fi
	return 1
}

if [[ -n "$HOST_OVERRIDE" ]]; then
	COMPOSE_FILE=$(detect_compose_file "$HOST_OVERRIDE")
	if [[ -z "$COMPOSE_FILE" ]]; then
		echo -e "${RED:-}No compose file found for host '$HOST_OVERRIDE'${NC:-}"
		echo "Available:"
		ls -1 "${SCRIPT_DIR}"docker-compose.*.yml 2>/dev/null | sed "s|${SCRIPT_DIR}docker-compose.||; s|.yml$||" | sed 's/^/  /'
		exit 1
	fi
else
	COMPOSE_FILE=$(detect_compose_file)
	if [[ -z "$COMPOSE_FILE" ]]; then
		echo -e "${RED:-}No compose file for hostname '$(hostname)'. Use --host <name>.${NC:-}"
		echo "Available:"
		ls -1 "${SCRIPT_DIR}"docker-compose.*.yml 2>/dev/null | sed "s|${SCRIPT_DIR}docker-compose.||; s|.yml$||" | sed 's/^/  /'
		exit 1
	fi
fi

dbg "Using compose file: $COMPOSE_FILE"

# ── Load parent .env (TRAIN_HOST_DIR, HF_CACHE_DIR, etc.) ────────
PARENT_ENV="${SCRIPT_DIR}../.env"
if [[ -f "$PARENT_ENV" ]]; then
	set -a
	. "$PARENT_ENV"
	set +a
	dbg "Loaded parent .env: $PARENT_ENV"
fi

# ── Start ────────────────────────────────────────────────────────
echo -e "${CYAN:-}Starting trainer: $(basename "$COMPOSE_FILE")...${NC:-}"

if [[ "$DEBUG" == "true" ]]; then
	dbg "docker compose -f $COMPOSE_FILE up -d"
	docker compose -f "$COMPOSE_FILE" up -d || {
		echo -e "${RED:-}Failed to start trainer. Check: docker compose -f $COMPOSE_FILE up -d${NC:-}"
		exit 1
	}
else
	docker compose -f "$COMPOSE_FILE" up -d &>/dev/null || {
		echo -e "${RED:-}Failed to start trainer. Re-run with --debug for details.${NC:-}"
		exit 1
	}
fi

# ── Wait for health ──────────────────────────────────────────────
CONTAINER_NAME=$(docker compose -f "$COMPOSE_FILE" ps --format '{{.Name}}' 2>/dev/null | head -1)
dbg "Container: $CONTAINER_NAME"

echo -e "${CYAN:-}Waiting for trainer API on :3011...${NC:-}"
for i in $(seq 1 60); do
	if curl -sf "http://localhost:3011/" >/dev/null 2>&1 || \
	   curl -sf "http://localhost:3011/docs" >/dev/null 2>&1; then
		echo -e "${GREEN:-}Trainer is LIVE on http://localhost:3011${NC:-}"
		exit 0
	fi
	sleep 2
	printf "."
done

echo ""
echo -e "${YELLOW:-}Trainer API did not respond within 120s. Check logs:${NC:-}"
echo "  docker logs $CONTAINER_NAME"
exit 1
