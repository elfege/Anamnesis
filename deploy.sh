#!/usr/bin/env bash
# ============================================
# ANAMNESIS - Docker Image Build & Deploy
# ============================================
# Rebuilds all Docker images and starts the stack.
# Use ./start.sh if you want to skip rebuild.
#
# Usage:
#   ./deploy.sh                     # Rebuild + start (prompts for prune/no-cache)
#   ./deploy.sh --prune             # Prune first, skip prompt
#   ./deploy.sh --no-cache          # No-cache build, skip prompt
#   ./deploy.sh --prune --no-cache  # Both, no prompts
#
# ============================================

deactivate &>/dev/null || true

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_R_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR="${SCRIPT_R_PATH%${SCRIPT_NAME}}"

cd "$SCRIPT_DIR" &>/dev/null || true

. ~/.env.colors 2>/dev/null || true

# Parse flags
do_prune=false
do_nocache=false
for arg in "$@"; do
	case $arg in
	--prune) do_prune=true ;;
	--no-cache) do_nocache=true ;;
	esac
done

echo "=========================================="
echo "  ANAMNESIS - Docker Image Build"
echo "=========================================="
echo ""

# Check if Dockerfile exists
if [ ! -f Dockerfile ]; then
	echo -e "${RED:-}Dockerfile not found!${NC:-}"
	exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f docker-compose.yml ]; then
	echo -e "${RED:-}docker-compose.yml not found!${NC:-}"
	exit 1
fi

# Prune: flag or prompt (defaults to no on timeout)
if ! $do_prune; then
	prune_answer="no"
	read -t 10 -r -p "Prune Docker system? (yes/no, 10s timeout = no): " prune_answer || true
	[[ "$prune_answer" == "yes" || "$prune_answer" == "YES" ]] && do_prune=true
fi
if $do_prune; then
	echo "Pruning Docker resources..."
	docker system prune -f || true
fi

# No-cache: flag or prompt (defaults to yes on timeout for a clean build)
if ! $do_nocache; then
	nocache_answer=""
	read -t 10 -r -p "No-cache build? (type 'no' to skip, ENTER/timeout = yes): " nocache_answer || true
	[[ "$nocache_answer" == "no" || "$nocache_answer" == "NO" ]] || do_nocache=true
fi

echo ""
echo "Cleaning up containers and images..."
# Stop and remove containers
docker compose down --remove-orphans &>/dev/null || true

echo -e "${GREEN:-}Containers stopped and removed${NC:-}"
echo ""

# Remove old images
docker rmi anamnesis-app &>/dev/null || true
echo -e "${GREEN:-}Cleanup complete${NC:-}"
echo ""

# Build Docker image
if $do_nocache; then
	echo "Building Docker image (--no-cache, full rebuild)..."
	docker compose build --no-cache
else
	echo "Building Docker image (cached)..."
	docker compose build
fi

if [ $? -eq 0 ]; then
	echo ""
	echo -e "${GREEN:-}Docker image built successfully${NC:-}"
	echo ""
	./start.sh
else
	echo -e "${RED:-}Docker build failed${NC:-}"
	exit 1
fi
