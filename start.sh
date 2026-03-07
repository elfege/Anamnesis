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

echo -e "${CYAN:-}Starting Anamnesis...${NC:-}"
docker compose up -d

# Wait for health (embedding model load can take 10-30s)
echo -n "Waiting for Anamnesis to be ready"
for i in $(seq 1 90); do
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
	sleep 1
	printf "."
done

echo ""
echo -e "${RED:-}Health check timed out after 90s.${NC:-}"
echo "Check logs: docker compose logs -f"
exit 1
