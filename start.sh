#!/usr/bin/env bash
# ============================================
# ANAMNESIS — Start (menu-driven)
# ============================================
# Spin up local + optionally remote services (no builds — see deploy.sh).
#
# Usage:
#   ./start.sh                       # interactive menu
#   ./start.sh --test                # dry-run (no docker/ssh commands executed)
#   ./start.sh --action=local        # skip menu: recreate dellserver stack
#   ./start.sh --action=all          # skip menu: recreate local + remote workers
#   ./start.sh --action=status       # skip menu: print status of all services
#   ./start.sh --debug               # verbose
#   ./start.sh --non-interactive     # legacy: equivalent to --action=local
#
# Remote services (auto-detected from ~/.ssh/config or $SSH_ALIAS_*):
#   • office                — avatar-worker-office, anamnesis-trainer-office
#   • server                — anamnesis-trainer-server (if configured)
#
# ============================================

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_R_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR="${SCRIPT_R_PATH%${SCRIPT_NAME}}"

cd "$SCRIPT_DIR" &>/dev/null || true

# ── Flags ─────────────────────────────────────────────────────
DEBUG=false
TEST=false
ACTION=""
MENU_MODE=true
for arg in "$@"; do
	case "$arg" in
		--debug|-d)        DEBUG=true ;;
		--test|-t)         TEST=true ;;
		--non-interactive) MENU_MODE=false; ACTION="${ACTION:-local}" ;;
		--action=*)        ACTION="${arg#--action=}"; MENU_MODE=false ;;
	esac
done

# ── Bash utils / env ──────────────────────────────────────────
type source_global_env &>/dev/null || {
	. ~/.bash_utils --no-exec &>/dev/null
}
source_global_env >/dev/null 2>&1 || true

# Colors (fall back if env.colors missing)
: "${CYAN:=\033[0;36m}"; : "${GREEN:=\033[0;32m}"; : "${YELLOW:=\033[0;33m}"
: "${RED:=\033[0;31m}"; : "${BOLD:=\033[1m}"; : "${DIM:=\033[2m}"; : "${NC:=\033[0m}"
: "${FLASH_CYAN:=\033[5;33m}"

dbg() { [[ "$DEBUG" == "true" ]] && echo -e "${YELLOW}[DEBUG] $*${NC}" || true; }

# ── Cleanup (restores stdout/stderr if any progress UI is active) ───
cleanup() {
	local exit_code=$?
	command -v stop_spinner &>/dev/null && stop_spinner || true
	trap - SIGINT SIGTERM EXIT ERR
	if [[ $exit_code -ne 0 ]]; then
		echo -e "\n${RED}Startup interrupted (exit=$exit_code).${NC}"
	fi
	exit $exit_code
}
trap cleanup SIGINT SIGTERM EXIT ERR

# ── Shims: start_spinner/stop_spinner fallbacks ───────────────
if ! command -v start_spinner &>/dev/null; then
	start_spinner() { shift 2>/dev/null; echo -e "${CYAN}$*${NC}"; }
fi
if ! command -v stop_spinner &>/dev/null; then
	stop_spinner() { :; }
fi
if ! command -v display_block &>/dev/null; then
	display_block() { echo -e "${BOLD}${CYAN}════ $* ════${NC}"; }
fi

# ── Run helper (honors --test) ───────────────────────────────
run() {
	# run [--label=TEXT] -- CMD ARGS ...
	local label=""
	while [[ "$1" == --label=* ]]; do label="${1#--label=}"; shift; done
	[[ "$1" == "--" ]] && shift
	if $TEST; then
		echo -e "${YELLOW}[TEST]${NC} would run: $*${label:+  ${DIM}— $label${NC}}"
		return 0
	fi
	if $DEBUG; then
		dbg "$*"
		"$@"
	else
		"$@" &>/dev/null
	fi
}

run_ssh() {
	# run_ssh HOST REMOTE_CMD
	local host="$1"; shift
	local cmd="$*"
	if $TEST; then
		echo -e "${YELLOW}[TEST]${NC} would ssh ${host}: ${cmd}"
		return 0
	fi
	if $DEBUG; then
		dbg "ssh $host \"$cmd\""
		ssh "$host" "$cmd"
	else
		ssh "$host" "$cmd" &>/dev/null
	fi
}

# ── Service catalog ───────────────────────────────────────────
# Each service: "handle|where|container|compose_file"
#   where: "local" or an ssh alias ("office", "server")
SERVICES=(
	"anamnesis-app|local|anamnesis-app|docker-compose.yml"
	"anamnesis-mongo|local|anamnesis-mongo|docker-compose.yml"
	"avatar-worker-office|office|avatar-worker-office|~/0_GENESIS_PROJECT/0_ANAMNESIS/avatar_worker/docker-compose.office.yml"
	"anamnesis-trainer-office|office|anamnesis-trainer-office|~/0_GENESIS_PROJECT/0_ANAMNESIS/trainers/docker-compose.office.yml"
	"anamnesis-trainer-server|server|anamnesis-trainer-server|~/0_GENESIS_PROJECT/0_ANAMNESIS/trainers/docker-compose.server.yml"
)

svc_field() {
	# svc_field <index> <spec>
	echo "$2" | awk -F'|' -v i="$1" '{print $i}'
}

# Filter services the user actually has configured (ssh-reachable or local)
host_available() {
	local host="$1"
	[[ "$host" == "local" ]] && return 0
	if $TEST; then return 0; fi
	ssh -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no "$host" true &>/dev/null
}

# ── Actions ──────────────────────────────────────────────────

do_env_prep() {
	# Shared pre-flight: AWS secrets pull, .env presence, SSH config, Mongo guard, Ollama
	display_block "ANAMNESIS — startup pre-flight"

	# Ollama service
	if command -v ollama &>/dev/null; then
		if ! systemctl is-active --quiet ollama 2>/dev/null; then
			start_spinner "" "${CYAN}Starting Ollama service…${NC}"
			run -- sudo systemctl start ollama || true
			stop_spinner
		fi
		start_spinner "" "${CYAN}Ollama ready at http://localhost:11434${NC}"
		stop_spinner
	fi

	# AWS secrets
	if command -v aws &>/dev/null && [[ -z "${SKIP_AWS_PULL:-}" ]]; then
		start_spinner "" "${CYAN}Pulling secrets from AWS…${NC}"
		if $TEST; then
			echo -e "${YELLOW}[TEST]${NC} would run: $SCRIPT_DIR/pull_env.sh 1"
		else
			"$SCRIPT_DIR/pull_env.sh" 1 &>/dev/null || {
				stop_spinner
				echo -e "${YELLOW}⚠ Could not pull ANAMNESIS-Secrets — using existing .env${NC}"
			}
			pull_aws_secrets ELFEGE-secrets 1 &>/dev/null && export ANTHROPIC_API_KEY || true
		fi
		stop_spinner
	fi

	if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
		echo -e "${RED}No .env at $SCRIPT_DIR/.env. Copy .env.example → .env.${NC}"
		return 1
	fi

	# SSH host.docker.internal entry
	local ssh_cfg="${SSH_DIR:-$HOME/.ssh}/config"
	if [[ -f "$ssh_cfg" ]] && ! grep -qE '^Host\b.*\bhost\.docker\.internal\b' "$ssh_cfg"; then
		local lookup="${SSH_HOST_DELLSERVER:-dellserver}"
		local id_file
		id_file=$(awk -v h="$lookup" '$1=="Host" && $2==h{f=1;next} f && /^Host[[:space:]]/{exit} f && /IdentityFile/{print $2;exit}' "$ssh_cfg")
		id_file="${id_file:-~/.ssh/id_rsa}"
		start_spinner "" "${CYAN}Adding host.docker.internal to SSH config…${NC}"
		if $TEST; then
			echo -e "${YELLOW}[TEST]${NC} would append host.docker.internal block to $ssh_cfg"
		else
			cat >> "$ssh_cfg" <<-EOF

			Host host.docker.internal
			   HostName host.docker.internal
			   User ${SSH_USER:-elfege}
			   IdentityFile $id_file
			   StrictHostKeyChecking no
			EOF
		fi
		stop_spinner
	fi

	# Mongo health guard — only rebuild if unhealthy
	local mongo_health
	mongo_health=$(docker inspect --format='{{.State.Health.Status}}' anamnesis-mongo 2>/dev/null || echo "missing")
	if [[ "$mongo_health" == "unhealthy" ]]; then
		start_spinner "" "${FLASH_CYAN}MongoDB unhealthy — recreating…${NC}"
		run -- docker stop anamnesis-mongo || true
		run -- docker rm anamnesis-mongo || true
		stop_spinner
	fi

	# Internet/AWS connectivity poll (post-power-loss guard, only when doing remote work)
	if ! curl -sf --max-time 3 https://sts.amazonaws.com -o /dev/null 2>&1; then
		start_spinner "" "${FLASH_CYAN}Waiting for internet / AWS…${NC}"
		if ! $TEST; then
			until curl -sf --max-time 5 https://sts.amazonaws.com -o /dev/null 2>&1; do sleep 5; done
		fi
		stop_spinner
	fi
}

action_local() {
	do_env_prep || return 1
	display_block "Recreating local stack (dellserver)"
	start_spinner "" "${CYAN}docker compose up -d${NC}"
	run -- docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d
	stop_spinner
	wait_local_health
}

action_all() {
	action_local || return 1
	action_remote_workers
}

action_remote_workers() {
	display_block "Recreating remote workers"
	for spec in "${SERVICES[@]}"; do
		local handle where compose
		handle=$(svc_field 1 "$spec"); where=$(svc_field 2 "$spec"); compose=$(svc_field 4 "$spec")
		[[ "$where" == "local" ]] && continue
		if ! host_available "$where"; then
			echo -e "${DIM}  skipping $handle — $where unreachable${NC}"
			continue
		fi
		start_spinner "" "${CYAN}${handle} @ ${where}${NC}"
		run_ssh "$where" "cd $(dirname "$compose") && docker compose -f $(basename "$compose") up -d"
		stop_spinner
	done
}

action_restart_one() {
	local handle="$1"
	local spec where container compose
	for s in "${SERVICES[@]}"; do
		if [[ "$(svc_field 1 "$s")" == "$handle" ]]; then spec="$s"; break; fi
	done
	if [[ -z "$spec" ]]; then
		echo -e "${RED}Unknown service: $handle${NC}"; return 1
	fi
	where=$(svc_field 2 "$spec"); container=$(svc_field 3 "$spec"); compose=$(svc_field 4 "$spec")

	display_block "Restart: $handle (on $where)"

	# Warn if it's a trainer that may be running a job
	if [[ "$handle" == anamnesis-trainer-* ]]; then
		echo -e "${YELLOW}⚠ Restarting a trainer will KILL any in-progress training job.${NC}"
		if $MENU_MODE; then
			read -r -p "Proceed anyway? [y/N]: " ok
			[[ "$ok" =~ ^[yY]$ ]] || { echo "aborted."; return 1; }
		fi
	fi

	start_spinner "" "${CYAN}restart $handle${NC}"
	if [[ "$where" == "local" ]]; then
		run -- docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d --force-recreate "$container"
	else
		run_ssh "$where" "cd $(dirname "$compose") && docker compose -f $(basename "$compose") up -d --force-recreate"
	fi
	stop_spinner
}

action_status() {
	display_block "Service status"
	printf "  %-30s %-12s %s\n" "SERVICE" "WHERE" "STATUS"
	printf "  %-30s %-12s %s\n" "------------------------------" "----------" "--------------------------"
	for spec in "${SERVICES[@]}"; do
		local handle where container
		handle=$(svc_field 1 "$spec"); where=$(svc_field 2 "$spec"); container=$(svc_field 3 "$spec")
		local status="—"
		if [[ "$where" == "local" ]]; then
			status=$(docker inspect --format='{{.State.Status}}{{if .State.Health}} ({{.State.Health.Status}}){{end}}' "$container" 2>/dev/null || echo "missing")
			[[ -z "$status" ]] && status="missing"
		else
			if host_available "$where"; then
				status=$(ssh -o ConnectTimeout=3 "$where" "docker inspect --format='{{.State.Status}}' $container 2>/dev/null || echo missing" 2>/dev/null)
			else
				status="${DIM}host unreachable${NC}"
			fi
		fi
		printf "  %-30s %-12s %b\n" "$handle" "$where" "$status"
	done
}

action_stop_all() {
	display_block "Stop everything"
	if $MENU_MODE; then
		read -r -p "This will stop local + remote services. Continue? [y/N]: " ok
		[[ "$ok" =~ ^[yY]$ ]] || { echo "aborted."; return 1; }
	fi
	start_spinner "" "${CYAN}stopping local…${NC}"
	run -- docker compose -f "$SCRIPT_DIR/docker-compose.yml" down
	stop_spinner
	for spec in "${SERVICES[@]}"; do
		local handle where compose
		handle=$(svc_field 1 "$spec"); where=$(svc_field 2 "$spec"); compose=$(svc_field 4 "$spec")
		[[ "$where" == "local" ]] && continue
		if ! host_available "$where"; then continue; fi
		start_spinner "" "${CYAN}stopping $handle @ $where${NC}"
		run_ssh "$where" "cd $(dirname "$compose") && docker compose -f $(basename "$compose") down"
		stop_spinner
	done
}

wait_local_health() {
	local host_ip=$(hostname -I | awk '{print $1}')
	if $TEST; then
		echo -e "${YELLOW}[TEST]${NC} skipping health wait"
		return 0
	fi
	echo -e "${CYAN}Waiting for MongoDB (up to 2-3 min on first run)…${NC}"
	for i in $(seq 1 60); do
		local s
		s=$(docker inspect --format='{{.State.Health.Status}}' anamnesis-mongo 2>/dev/null || echo "missing")
		[[ "$s" == "healthy" ]] && { echo -e "  ${GREEN}MongoDB healthy${NC}"; break; }
		[[ "$s" == "unhealthy" ]] && { echo -e "${RED}MongoDB unhealthy — docker logs anamnesis-mongo${NC}"; return 1; }
		sleep 1; printf "."
	done
	echo
	echo -e "${CYAN}Waiting for Anamnesis app…${NC}"
	for i in $(seq 1 120); do
		curl -sf "http://localhost:3010/health" &>/dev/null && {
			echo
			display_block "Anamnesis is LIVE"
			echo -e "  API:        http://${host_ip}:3010/docs"
			echo -e "  Dashboard:  http://${host_ip}:3010/dashboard"
			echo -e "  Avatar:     http://${host_ip}:3010/avatar"
			echo -e "  MongoDB:    ${host_ip}:5438"
			return 0
		}
		sleep 2; printf "."
	done
	echo
	echo -e "${RED}App health check timed out.${NC}  docker logs anamnesis-app"
	return 1
}

# ── Menu ──────────────────────────────────────────────────────

menu_main() {
	while true; do
		echo
		display_block "ANAMNESIS — Service Manager"
		$TEST && echo -e "  ${YELLOW}TEST MODE — no commands will be executed${NC}"
		cat <<-EOF

		  1) Start/recreate local stack (dellserver: anamnesis-app + mongo)
		  2) Start/recreate EVERYTHING (local + office + server)
		  3) Restart individual service…
		  4) Show status of all services
		  5) Stop everything (local + remote)
		  0) Exit

		EOF
		read -r -p "  Select [0-5]: " choice
		case "$choice" in
			1) action_local ;;
			2) action_all ;;
			3) menu_services ;;
			4) action_status ;;
			5) action_stop_all ;;
			0|q|Q|"") return 0 ;;
			*) echo -e "  ${YELLOW}?${NC}" ;;
		esac
	done
}

menu_services() {
	echo
	display_block "Pick a service to restart"
	local i=1
	local picks=()
	for spec in "${SERVICES[@]}"; do
		local handle where
		handle=$(svc_field 1 "$spec"); where=$(svc_field 2 "$spec")
		local reach="${GREEN}●${NC}"
		host_available "$where" || reach="${DIM}○${NC}"
		printf "  %2d) %b  %s ${DIM}(%s)${NC}\n" "$i" "$reach" "$handle" "$where"
		picks+=("$handle")
		i=$((i+1))
	done
	echo "   a) All local (anamnesis-app + mongo)"
	echo "   r) All remote workers"
	echo "   A) Avatar stack (anamnesis-app + avatar-worker-office)"
	echo "   T) All trainers"
	echo "   0) Back"
	echo
	read -r -p "  Select: " pick
	case "$pick" in
		[0-9]*)
			local idx=$((pick-1))
			[[ -n "${picks[$idx]}" ]] && action_restart_one "${picks[$idx]}" || echo "?"
			;;
		a|A) [[ "$pick" == "A" ]] && { action_restart_one anamnesis-app; action_restart_one avatar-worker-office; } \
			 || { action_restart_one anamnesis-app; action_restart_one anamnesis-mongo; } ;;
		r) action_remote_workers ;;
		T) action_restart_one anamnesis-trainer-office; action_restart_one anamnesis-trainer-server ;;
		0|q|Q|"") return 0 ;;
		*) echo "?" ;;
	esac
}

# ── Entry point ───────────────────────────────────────────────
if $MENU_MODE; then
	menu_main
else
	case "$ACTION" in
		""|local)  action_local ;;
		all)       action_all ;;
		status)    action_status ;;
		stop)      action_stop_all ;;
		*)
			if [[ "$ACTION" == restart:* ]]; then
				action_restart_one "${ACTION#restart:}"
			else
				echo -e "${RED}Unknown --action=$ACTION${NC}"
				exit 2
			fi
			;;
	esac
fi
