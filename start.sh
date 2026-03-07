#!/usr/bin/env bash
# ============================================
# ANAMNESIS - Start
# ============================================

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_R_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR="${SCRIPT_R_PATH%${SCRIPT_NAME}}"

cd "$SCRIPT_DIR" &>/dev/null || true

. ~/.env.colors 2>/dev/null || true
. ~/logger.sh --no-exec &>/dev/null || true
. ~/.bash_utils --no-exec &>/dev/null || true

# ── Wait for internet / AWS connectivity (post-power-loss guard) ─────────────
_AWS_WAIT_URL="https://sts.amazonaws.com"
_LOG_FILE="${LOG_FILE:-$HOME/0_LOGS/log.log}"
mkdir -p "$(dirname "$_LOG_FILE")"
if ! curl -sf --max-time 5 "$_AWS_WAIT_URL" -o /dev/null 2>&1; then
    _msg="[$(date '+%H:%M:%S')] Waiting for internet/AWS (${_AWS_WAIT_URL}) — logging every 5s to: $_LOG_FILE"
    echo -e "${FLASH_ACCENT_YELLOW:-\033[5;33m}${_msg}${NC:-\033[0m}"
    echo "$_msg" >> "$_LOG_FILE"
    until curl -sf --max-time 5 "$_AWS_WAIT_URL" -o /dev/null 2>&1; do
        _msg="[$(date '+%H:%M:%S')] Still waiting for internet/AWS — retrying in 5s"
        echo -e "${FLASH_ACCENT_YELLOW:-\033[5;33m}${_msg}${NC:-\033[0m}"
        echo "$_msg" >> "$_LOG_FILE"
        sleep 5
    done
fi
echo -e "${GREEN:-\033[0;32m}[$(date '+%H:%M:%S')] Internet/AWS connectivity confirmed — proceeding${NC:-\033[0m}"
echo "[$(date '+%H:%M:%S')] Internet/AWS connectivity confirmed" >> "$_LOG_FILE"
# ─────────────────────────────────────────────────────────────────────────────

# ── Ollama check ─────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    echo -e "${CYAN:-}Ollama not found — running install script...${NC:-}"
    bash "$SCRIPT_DIR/install_ollama.sh"
elif ! systemctl is-active --quiet ollama 2>/dev/null; then
    echo -e "${CYAN:-}Ollama installed but not running — starting service...${NC:-}"
    sudo systemctl start ollama
    sleep 2
fi
echo -e "${GREEN:-}Ollama ready: http://localhost:11434${NC:-}"
# ─────────────────────────────────────────────────────────────────

# ── Pull ANTHROPIC_API_KEY from AWS ──────────────────────────────
echo -e "${CYAN:-}Pulling ELFEGE-secrets from AWS...${NC:-}"
pull_aws_secrets ELFEGE-secrets 1
export ANTHROPIC_API_KEY
# ─────────────────────────────────────────────────────────────────

# ── MongoDB health guard ──────────────────────────────────────────
# If anamnesis-mongo exists but is unhealthy, recreate it so the RS re-initializes
_MONGO_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' anamnesis-mongo 2>/dev/null || echo "missing")
if [[ "$_MONGO_HEALTH" == "unhealthy" ]]; then
    echo -e "${FLASH_ACCENT_YELLOW:-\033[5;33m}MongoDB unhealthy — recreating container...${NC:-\033[0m}"
    docker stop anamnesis-mongo &>/dev/null || true
    docker rm anamnesis-mongo &>/dev/null || true
elif [[ "$_MONGO_HEALTH" == "missing" ]]; then
    echo -e "${CYAN:-}MongoDB container not found — will create fresh.${NC:-}"
fi
# ─────────────────────────────────────────────────────────────────

echo -e "${CYAN:-}Starting Anamnesis...${NC:-}"
docker compose up -d

# ── Wait for MongoDB to be healthy first (RS init can take 2-3 min fresh) ──
echo -n "Waiting for MongoDB RS to be ready"
for i in $(seq 1 60); do
    _status=$(docker inspect --format='{{.State.Health.Status}}' anamnesis-mongo 2>/dev/null || echo "missing")
    if [[ "$_status" == "healthy" ]]; then
        echo " healthy."
        break
    elif [[ "$_status" == "unhealthy" ]]; then
        echo ""
        echo -e "${RED:-}MongoDB is unhealthy. Check: docker logs anamnesis-mongo${NC:-}"
        exit 1
    fi
    sleep 5
    printf "."
done

# ── Wait for app health (embedding model load can take 10-30s) ─────────────
echo -n "Waiting for Anamnesis app to be ready"
for i in $(seq 1 120); do
	if curl -sf "http://localhost:3010/health" >/dev/null 2>&1; then
		echo ""
		HOST_IP=$(hostname -I | awk '{print $1}')
		echo -e "${GREEN:-}==========================================${NC:-}"
		echo -e "${GREEN:-}  Anamnesis is LIVE${NC:-}"
		echo -e "  API:       http://${HOST_IP}:3010/docs"
		echo -e "  Dashboard: http://${HOST_IP}:3010/dashboard"
		echo -e "  Health:    http://${HOST_IP}:3010/health"
		echo -e "  MongoDB:   ${HOST_IP}:5438"
		echo -e "${GREEN:-}==========================================${NC:-}"
		exit 0
	fi
	sleep 2
	printf "."
done

echo ""
echo -e "${RED:-}App health check timed out after 240s.${NC:-}"
echo "Check logs: docker logs anamnesis-app"
exit 1
