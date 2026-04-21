#!/usr/bin/env bash
# ============================================
# ANAMNESIS — Stop (menu-driven)
# ============================================
# Stop local and/or remote services.
#
# Usage:
#   ./stop.sh                    # interactive menu
#   ./stop.sh --test             # dry-run
#   ./stop.sh --action=local     # stop dellserver stack only
#   ./stop.sh --action=all       # stop local + remote workers
#   ./stop.sh --action=remote    # stop only remote workers
# ============================================

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_R_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR="${SCRIPT_R_PATH%${SCRIPT_NAME}}"

cd "$SCRIPT_DIR" &>/dev/null || true

DEBUG=false
TEST=false
ACTION=""
MENU_MODE=true
for arg in "$@"; do
	case "$arg" in
		--debug|-d)  DEBUG=true ;;
		--test|-t)   TEST=true ;;
		--action=*)  ACTION="${arg#--action=}"; MENU_MODE=false ;;
	esac
done

: "${CYAN:=\033[0;36m}"; : "${GREEN:=\033[0;32m}"; : "${YELLOW:=\033[0;33m}"
: "${RED:=\033[0;31m}"; : "${BOLD:=\033[1m}"; : "${DIM:=\033[2m}"; : "${NC:=\033[0m}"

if ! command -v display_block &>/dev/null; then
	display_block() { echo -e "${BOLD}${CYAN}════ $* ════${NC}"; }
fi

run() {
	if $TEST; then
		echo -e "${YELLOW}[TEST]${NC} would run: $*"
		return 0
	fi
	if $DEBUG; then echo -e "${YELLOW}[DEBUG] $*${NC}"; fi
	"$@"
}

run_ssh() {
	local host="$1"; shift
	local cmd="$*"
	if $TEST; then
		echo -e "${YELLOW}[TEST]${NC} would ssh ${host}: ${cmd}"
		return 0
	fi
	ssh "$host" "$cmd"
}

# handle|where|compose_path
TARGETS=(
	"local|local|$SCRIPT_DIR/docker-compose.yml"
	"avatar-worker-office|office|~/0_GENESIS_PROJECT/0_ANAMNESIS/avatar_worker/docker-compose.office.yml"
	"anamnesis-trainer-office|office|~/0_GENESIS_PROJECT/0_ANAMNESIS/trainers/docker-compose.office.yml"
	"anamnesis-trainer-server|server|~/0_GENESIS_PROJECT/0_ANAMNESIS/trainers/docker-compose.server.yml"
)

tgt_field() { echo "$2" | awk -F'|' -v i="$1" '{print $i}'; }

host_available() {
	local host="$1"
	[[ "$host" == "local" ]] && return 0
	if $TEST; then return 0; fi
	ssh -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no "$host" true &>/dev/null
}

stop_target() {
	local spec="$1"
	local handle=$(tgt_field 1 "$spec")
	local where=$(tgt_field 2 "$spec")
	local compose=$(tgt_field 3 "$spec")
	if [[ "$where" == "local" ]]; then
		run docker compose -f "$compose" down
	else
		if ! host_available "$where"; then
			echo -e "${DIM}  skipping $handle — $where unreachable${NC}"
			return 0
		fi
		run_ssh "$where" "cd $(dirname "$compose") && docker compose -f $(basename "$compose") down"
	fi
}

action_local() {
	display_block "Stop local stack"
	stop_target "${TARGETS[0]}"
}

action_remote() {
	display_block "Stop remote workers"
	for t in "${TARGETS[@]:1}"; do
		echo -e "  ${CYAN}→ $(tgt_field 1 "$t") @ $(tgt_field 2 "$t")${NC}"
		stop_target "$t"
	done
}

action_all() { action_local; action_remote; }

menu_main() {
	while true; do
		echo
		display_block "ANAMNESIS — Stop Services"
		$TEST && echo -e "  ${YELLOW}TEST MODE — no commands will be executed${NC}"
		cat <<-EOF

		  1) Stop LOCAL only (anamnesis-app + mongo)
		  2) Stop REMOTE workers only
		  3) Stop EVERYTHING
		  0) Exit

		EOF
		read -r -p "  Select [0-3]: " choice
		case "$choice" in
			1) action_local ;;
			2) action_remote ;;
			3) action_all ;;
			0|q|Q|"") return 0 ;;
			*) echo -e "  ${YELLOW}?${NC}" ;;
		esac
	done
}

if $MENU_MODE; then
	menu_main
else
	case "$ACTION" in
		local|"") action_local ;;
		remote)   action_remote ;;
		all)      action_all ;;
		*)
			echo -e "${RED}Unknown --action=$ACTION${NC}"
			exit 2
			;;
	esac
fi
