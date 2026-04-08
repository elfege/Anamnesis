#!/usr/bin/env bash
# ============================================
# ANAMNESIS - Start
# ============================================

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_R_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR="${SCRIPT_R_PATH%${SCRIPT_NAME}}"

cd "$SCRIPT_DIR" &>/dev/null || true

# ── Debug mode ────────────────────────────────────────────────
DEBUG=false
for arg in "$@"; do
	case "$arg" in
		--debug|-d) DEBUG=true ;;
	esac
done

dbg() { [[ "$DEBUG" == "true" ]] && echo -e "${YELLOW:-\033[0;33m}[DEBUG] $*${NC:-\033[0m}" || true; }

if [[ "$DEBUG" == "true" ]]; then
	echo -e "${YELLOW:-\033[0;33m}━━━ DEBUG MODE ━━━${NC:-\033[0m}"
fi

cleanup() {
	local exit_code=$?

	stop_spinner
	trap - SIGINT SIGTERM EXIT ERR
	if [[ $exit_code -ne 0 ]]; then
		echo -e "${RED:-}Startup interrupted. Cleaning up...${NC:-}"
		docker compose down &>/dev/null &
		disown
	fi
	exit $exit_code
}
trap cleanup SIGINT SIGTERM EXIT ERR

type source_global_env &>/dev/null || {
	. ~/.bash_utils --no-exec &>/dev/null
}
source_global_env >/dev/null 2>&1 || {
	echo -e "${RED:-}Failed to load general environment${NC:-}"
	exit 1
}

display_block "ANAMNESIS - The AI Memory Palace For LLMs"

start_spinner "" "${CYAN:-}Initializing...${NC:-}"
sleep 1

start_spinner "" "${CYAN:-}Stopping any existing Anamnesis containers...${NC:-}"
if [[ "$DEBUG" == "true" ]]; then
	dbg "docker compose down"
	docker compose down || {
		echo -e "${RED:-}Failed to stop existing containers. Check: docker compose down${NC:-}"
		exit 1
	}
else
	docker compose down &>/dev/null || {
		echo -e "${RED:-}Failed to stop existing containers. Check: docker compose down${NC:-}"
		exit 1
	}
fi
sleep 1

# ── Wait for internet / AWS connectivity (post-power-loss guard) ─────────────
_AWS_WAIT_URL="https://sts.amazonaws.com"
_LOG_FILE="${LOG_FILE:-$HOME/0_LOGS/log.log}"
mkdir -p "$(dirname "$_LOG_FILE")"
if ! curl -sf --max-time 5 "$_AWS_WAIT_URL" -o /dev/null 2>&1; then
	_msg="[$(date '+%H:%M:%S')] Waiting for internet/AWS (${_AWS_WAIT_URL}) — logging every 5s to: $_LOG_FILE"
	start_spinner "" "${FLASH_CYAN:-\033[5;33m}${_msg}${NC:-\033[0m}"
	start_spinner "" "$_msg"
	until curl -sf --max-time 5 "$_AWS_WAIT_URL" -o /dev/null 2>&1; do
		_msg="[$(date '+%H:%M:%S')] Still waiting for internet/AWS — retrying in 5s"
		start_spinner "" "${FLASH_CYAN:-\033[5;33m}${_msg}${NC:-\033[0m}"
		start_spinner "" "$_msg"
		sleep 5
	done
fi
start_spinner "" "${CYAN:-\033[0;32m}[$(date '+%H:%M:%S')] Internet/AWS connectivity confirmed — proceeding${NC:-\033[0m}"
start_spinner "" "[$(date '+%H:%M:%S')] Internet/AWS connectivity confirmed"
# ─────────────────────────────────────────────────────────────────────────────

# ── Ollama check ─────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
	start_spinner "" "${CYAN:-}Ollama not found — running install script...${NC:-}"
	if [[ "$DEBUG" == "true" ]]; then
		dbg "install_ollama.sh"
		bash "$SCRIPT_DIR/install_ollama.sh" || {
			echo -e "${RED:-}Failed to install Ollama. Check install_ollama.sh${NC:-}"
			exit 1
		}
	else
		bash "$SCRIPT_DIR/install_ollama.sh" &>/dev/null || {
			echo -e "${RED:-}Failed to install Ollama. Check install_ollama.sh${NC:-}"
			exit 1
		}
	fi
elif ! systemctl is-active --quiet ollama 2>/dev/null; then
	start_spinner "" "${CYAN:-}Ollama installed but not running — starting service...${NC:-}"
	dbg "systemctl start ollama"
	sudo systemctl start ollama &>/dev/null || {
		echo -e "${RED:-}Failed to start Ollama service. Check: sudo systemctl status ollama${NC:-}"
		exit 1
	}
	sleep 2
fi
dbg "Ollama version: $(ollama --version 2>/dev/null)"
dbg "Ollama models: $(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | tr '\n' ', ')"
start_spinner "" "${CYAN:-}Ollama ready: http://localhost:11434${NC:-}"
# ─────────────────────────────────────────────────────────────────

# ── Pull secrets from AWS (optional — skipped if .env already exists) ──
if command -v aws &>/dev/null && [[ -z "${SKIP_AWS_PULL:-}" ]]; then
	start_spinner "" "${CYAN:-}Pulling secrets from AWS${NC:-}"

	# Deployment config → .env (docker-compose reads this)
	if [[ "$DEBUG" == "true" ]]; then
		dbg "pull_env.sh 1"
		"$SCRIPT_DIR/pull_env.sh" 1 || {
			echo -e "${YELLOW:-}Could not pull ANAMNESIS-Secrets — using existing .env${NC:-}"
			sleep 3
		}
	else
		"$SCRIPT_DIR/pull_env.sh" 1 &>/dev/null || {
			stop_spinner
			echo -e "${YELLOW:-}⚠ Could not pull ANAMNESIS-Secrets — using existing .env${NC:-}"
			sleep 2
		}
	fi

	# Anthropic API key (from personal secrets)
	if [[ "$DEBUG" == "true" ]]; then
		dbg "Pulling ELFEGE-secrets for ANTHROPIC_API_KEY"
		pull_aws_secrets ELFEGE-secrets 1 && export ANTHROPIC_API_KEY || {
			dbg "Failed to pull ELFEGE-secrets"
		}
	else
		pull_aws_secrets ELFEGE-secrets 1 &>/dev/null && export ANTHROPIC_API_KEY || true
	fi
else
	start_spinner "" "${CYAN:-}Skipping AWS pull — using existing .env${NC:-}"
fi

# Bail if no .env at all
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
	echo -e "${RED:-}No .env found. Copy .env.example → .env and fill in your values.${NC:-}"
	exit 1
fi
# ─────────────────────────────────────────────────────────────────

# ── Ensure SSH config has host.docker.internal entry ──────────────
# The container uses host.docker.internal to SSH back to the host.
# paramiko reads ~/.ssh/config to find the right IdentityFile, so we
# need a matching Host entry.
_SSH_CFG="${SSH_DIR:-$HOME/.ssh}/config"
if [[ -f "$_SSH_CFG" ]] && ! grep -qE '^Host\b.*\bhost\.docker\.internal\b' "$_SSH_CFG"; then
	# Find the IdentityFile from the dellserver entry (same machine)
	_ID_FILE=$(awk '/^Host[[:space:]]+dellserver[[:space:]]*$/{found=1;next} found && /^Host[[:space:]]/{exit} found && /IdentityFile/{print $2;exit}' "$_SSH_CFG")
	_ID_FILE="${_ID_FILE:-~/.ssh/id_rsa_server_home_elfege}"
	_SSH_USER_VAL="${SSH_USER:-elfege}"
	start_spinner "" "${CYAN:-}Adding host.docker.internal to SSH config...${NC:-}"
	cat >> "$_SSH_CFG" <<-EOF

	Host host.docker.internal
	   HostName host.docker.internal
	   User $_SSH_USER_VAL
	   IdentityFile $_ID_FILE
	   StrictHostKeyChecking no
	EOF
	dbg "Added host.docker.internal entry to $_SSH_CFG"
fi
# ─────────────────────────────────────────────────────────────────

# ── MongoDB health guard ──────────────────────────────────────────
# If anamnesis-mongo exists but is unhealthy, recreate it so the RS re-initializes
_MONGO_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' anamnesis-mongo 2>/dev/null || echo "missing")
if [[ "$_MONGO_HEALTH" == "unhealthy" ]]; then
	start_spinner "" "${FLASH_CYAN:-\033[5;33m}MongoDB unhealthy — recreating container...${NC:-\033[0m}"
	docker stop anamnesis-mongo &>/dev/null || true
	docker rm anamnesis-mongo &>/dev/null || true
elif [[ "$_MONGO_HEALTH" == "missing" ]]; then
	start_spinner "" "${CYAN:-}MongoDB container not found — will create fresh.${NC:-}"
fi
# ─────────────────────────────────────────────────────────────────

start_spinner "" "${CYAN:-}Composing services...${NC:-}"
if [[ "$DEBUG" == "true" ]]; then
	stop_spinner
	dbg "docker compose up -d --force-recreate"
	dbg ".env contents:"
	grep -v '^#' "$SCRIPT_DIR/.env" | grep -v '^$' | sed 's/^/  /'
	docker compose up -d --force-recreate || {
		echo -e "${RED:-}Failed to start Anamnesis containers. Check: docker compose up -d${NC:-}"
		exit 1
	}
else
	docker compose up -d &>/dev/null || {
		echo -e "${RED:-}Failed to start Anamnesis containers. Check: docker compose up -d${NC:-}"
		exit 1
	}
fi
stop_spinner

# ── Wait for MongoDB to be healthy first (RS init can take 2-3 min fresh) ──

/bin/clear
echo -e "$CYAN" "Waiting for MongoDB RS to be ready. Please be patient, this can take 2-3 minutes on first run..." "$NC"
for i in $(seq 1 60); do
	_status=$(docker inspect --format='{{.State.Health.Status}}' anamnesis-mongo 2>/dev/null || echo "missing")
	if [[ "$_status" == "healthy" ]]; then
		echo -e "${CYAN}MongoDB is ready.${NC:-}"
		break
	elif [[ "$_status" == "unhealthy" ]]; then
		echo ""
		echo -e "${RED:-}MongoDB is unhealthy. Check: docker logs anamnesis-mongo${NC:-}"
		exit 1
	fi
	sleep 1

	printf "."
done

# ── Wait for app health (embedding model load can take 10-30s) ─────────────
/bin/clear
display_block "ANAMNESIS - The AI Memory Palace For LLMs"

echo -e "${CYAN}" "Waiting for Anamnesis app to be ready" "$NC"

for i in $(seq 1 120); do
	if curl -sf "http://localhost:3010/health" >/dev/null 2>&1; then
		/bin/clear
		display_block "ANAMNESIS - The AI Memory Palace For LLMs"

		HOST_IP=$(hostname -I | awk '{print $1}')
		repeat_print "═" "" "$CYAN"
		echo -e "${CYAN:-}  Anamnesis is LIVE${NC:-}"
		echo -e "  API:       http://${HOST_IP}:3010/docs"
		echo -e "  Dashboard: http://${HOST_IP}:3010/dashboard"
		echo -e "  Health:    http://${HOST_IP}:3010/health"
		echo -e "  MongoDB:   ${HOST_IP}:5438"
		repeat_print "═" "" "$CYAN"
		exit 0
	fi
	sleep 2
	printf "."
done

echo ""
echo -e "${RED:-}App health check timed out after 240s.${NC:-}"
echo "Check logs: docker logs anamnesis-app"
exit 1
