#!/usr/bin/env bash
# ============================================
# ANAMNESIS TRAINER — Docker Image Build & Deploy
# ============================================
# Rebuilds the trainer Docker image and starts the stack.
# Use ./start.sh if you want to skip rebuild.
#
# Usage:
#   ./deploy.sh                        # Auto-detect host, prompts for prune/no-cache
#   ./deploy.sh --host office          # Force specific compose file
#   ./deploy.sh --prune                # Prune first, skip prompt
#   ./deploy.sh --no-cache             # No-cache build, skip prompt
#   ./deploy.sh --prune --no-cache     # Both, no prompts
#   ./deploy.sh --host office --no-cache
# ============================================

deactivate &>/dev/null || true

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_R_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR="${SCRIPT_R_PATH%${SCRIPT_NAME}}"

cd "$SCRIPT_DIR" &>/dev/null || true

. ~/.env.colors 2>/dev/null || true

# ── Parse flags ──────────────────────────────────────────────────
do_prune=false
do_nocache=false
HOST_OVERRIDE=""
while [[ $# -gt 0 ]]; do
	case "$1" in
	--prune) do_prune=true; shift ;;
	--no-cache) do_nocache=true; shift ;;
	--host) HOST_OVERRIDE="$2"; shift 2 ;;
	--host=*) HOST_OVERRIDE="${1#--host=}"; shift ;;
	*) shift ;;
	esac
done

echo "=========================================="
echo "  ANAMNESIS TRAINER - Docker Image Build"
echo "=========================================="
echo ""

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

echo "Using compose file: $(basename "$COMPOSE_FILE")"
echo ""

# ── Sanity checks ────────────────────────────────────────────────
if [[ ! -f Dockerfile ]]; then
	echo -e "${RED:-}Dockerfile not found!${NC:-}"
	exit 1
fi

# ── Prune prompt ─────────────────────────────────────────────────
if ! $do_prune; then
	prune_answer="no"
	read -t 10 -r -p "Prune Docker system? (yes/no, 10s timeout = no): " prune_answer || true
	[[ "$prune_answer" == "yes" || "$prune_answer" == "YES" ]] && do_prune=true
fi
if $do_prune; then
	echo "Pruning Docker resources..."
	docker system prune -f || true
fi

# ── No-cache prompt ──────────────────────────────────────────────
if ! $do_nocache; then
	nocache_answer=""
	read -t 10 -r -p "No-cache build? (type 'no' to skip, ENTER/timeout = yes): " nocache_answer || true
	[[ "$nocache_answer" == "no" || "$nocache_answer" == "NO" ]] || do_nocache=true
fi

# ── Stop + remove existing trainer container ─────────────────────
echo ""
echo "Stopping trainer container..."
CONTAINER_NAME=$(docker compose -f "$COMPOSE_FILE" ps --format '{{.Name}}' 2>/dev/null | head -1)
if [[ -n "$CONTAINER_NAME" ]]; then
	docker stop "$CONTAINER_NAME" &>/dev/null || true
	docker rm "$CONTAINER_NAME" &>/dev/null || true
	echo -e "${GREEN:-}Container stopped and removed${NC:-}"
else
	# Fallback — try to derive name from compose
	docker compose -f "$COMPOSE_FILE" down &>/dev/null || true
fi
echo ""

# ── Build ────────────────────────────────────────────────────────
if $do_nocache; then
	echo "Building Docker image (--no-cache, full rebuild)..."
	docker compose -f "$COMPOSE_FILE" build --no-cache
else
	echo "Building Docker image (cached)..."
	docker compose -f "$COMPOSE_FILE" build
fi

BUILD_RC=$?
if [[ $BUILD_RC -ne 0 ]]; then
	echo -e "${RED:-}Docker build failed (exit $BUILD_RC)${NC:-}"
	exit 1
fi

echo ""
echo -e "${GREEN:-}Docker image built successfully${NC:-}"
echo ""

# ── Hand off to start.sh ─────────────────────────────────────────
if [[ -n "$HOST_OVERRIDE" ]]; then
	./start.sh --host "$HOST_OVERRIDE"
else
	./start.sh
fi
