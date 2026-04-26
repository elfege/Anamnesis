#!/usr/bin/env bash
# ============================================
# ANAMNESIS — Deploy (menu-driven build + start)
# ============================================
# Rebuilds Docker images locally and/or remotely, then calls start.sh.
# Unlike start.sh (which never builds), deploy.sh is the only place that
# invokes `docker compose build`.
#
# Usage:
#   ./deploy.sh                         # interactive menu
#   ./deploy.sh --test                  # dry-run (no builds/no docker commands)
#   ./deploy.sh --action=local          # rebuild anamnesis-app only
#   ./deploy.sh --action=all            # rebuild local + all remote workers
#   ./deploy.sh --action=worker-office  # rebuild only office avatar worker
#   ./deploy.sh --action=d2             # rebuild δ² engine on server (default)
#   ./deploy.sh --action=d2:office      # rebuild δ² engine on office (ROCm)
#   ./deploy.sh --action=d2:runpod      # build δ² locally + push to registry
#   ./deploy.sh --action=d2:all         # build δ² on every reachable GPU host
#   ./deploy.sh --no-cache              # force --no-cache on builds (skips prompt)
#   ./deploy.sh --prune                 # prune docker first
#   ./deploy.sh --debug                 # verbose
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
DO_PRUNE=false
DO_NOCACHE=false
for arg in "$@"; do
	case "$arg" in
		--debug|-d)   DEBUG=true ;;
		--test|-t)    TEST=true ;;
		--prune)      DO_PRUNE=true ;;
		--no-cache)   DO_NOCACHE=true ;;
		--action=*)   ACTION="${arg#--action=}"; MENU_MODE=false ;;
	esac
done

# ── Env ───────────────────────────────────────────────────────
type source_global_env &>/dev/null || {
	. ~/.bash_utils --no-exec &>/dev/null
}
source_global_env >/dev/null 2>&1 || true
. ~/.env.colors 2>/dev/null || true

: "${CYAN:=\033[0;36m}"; : "${GREEN:=\033[0;32m}"; : "${YELLOW:=\033[0;33m}"
: "${RED:=\033[0;31m}"; : "${BOLD:=\033[1m}"; : "${DIM:=\033[2m}"; : "${NC:=\033[0m}"

dbg() { [[ "$DEBUG" == "true" ]] && echo -e "${YELLOW}[DEBUG] $*${NC}" || true; }

cleanup() {
	local exit_code=$?
	command -v stop_spinner &>/dev/null && stop_spinner || true
	trap - SIGINT SIGTERM EXIT ERR
	if [[ $exit_code -ne 0 ]]; then
		echo -e "\n${RED}Deploy interrupted (exit=$exit_code).${NC}"
	fi
	exit $exit_code
}
trap cleanup SIGINT SIGTERM EXIT ERR

if ! command -v display_block &>/dev/null; then
	display_block() { echo -e "${BOLD}${CYAN}════ $* ════${NC}"; }
fi

run() {
	if $TEST; then
		echo -e "${YELLOW}[TEST]${NC} would run: $*"
		return 0
	fi
	if $DEBUG; then
		dbg "$*"
		"$@"
	else
		"$@"
	fi
}

run_ssh() {
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
		ssh "$host" "$cmd"
	fi
}

# ── Build targets ─────────────────────────────────────────────
# handle|where|compose_dir|compose_file|service_name
# NOTE: d²-* entries point at d2/docker-compose.yml which uses --profile
# (cuda or rocm). The build_one helper passes the profile via $D2_PROFILE
# resolved per-target below.
TARGETS=(
	"anamnesis-app|local|$SCRIPT_DIR|docker-compose.yml|anamnesis-app"
	"avatar-worker-office|office|~/0_GENESIS_PROJECT/0_ANAMNESIS/avatar_worker|docker-compose.office.yml|avatar-worker"
	"avatar-worker-server|server|~/0_GENESIS_PROJECT/0_ANAMNESIS/avatar_worker|docker-compose.server.yml|avatar-worker"
	"anamnesis-trainer-office|office|~/0_GENESIS_PROJECT/0_ANAMNESIS/trainers|docker-compose.office.yml|anamnesis-trainer"
	"anamnesis-trainer-server|server|~/0_GENESIS_PROJECT/0_ANAMNESIS/trainers|docker-compose.server.yml|anamnesis-trainer"
	"d2-server|server|~/0_GENESIS_PROJECT/0_ANAMNESIS/d2|docker-compose.yml|d2-cuda"
	"d2-office|office|~/0_GENESIS_PROJECT/0_ANAMNESIS/d2|docker-compose.yml|d2-rocm"
	"d2-runpod|runpod|~/0_GENESIS_PROJECT/0_ANAMNESIS/d2|docker-compose.yml|d2-cuda"
)

tgt_field() { echo "$2" | awk -F'|' -v i="$1" '{print $i}'; }

host_available() {
	local host="$1"
	[[ "$host" == "local" ]] && return 0
	if $TEST; then return 0; fi
	# RunPod is a special pseudo-host: reachability = "is a pod up and registered?"
	# We check the worker_registry endpoint on the orchestrator (dellserver:3010).
	# This avoids needing SSH credentials to a pod that may not exist.
	if [[ "$host" == "runpod" ]]; then
		# A RunPod target is "available" if deploy_runpod.sh has registered a
		# pod URL — i.e. there is a doc with kind="runpod" in worker_registry.
		curl -sf "http://192.168.10.20:3010/api/workers/list?kind=runpod" 2>/dev/null \
			| grep -q '"url"' && return 0
		return 1
	fi
	ssh -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no "$host" true &>/dev/null
}

# Resolve the docker-compose profile for d² targets (server=cuda, office=rocm,
# runpod=cuda). Returns empty for non-d² targets. Used by build_one and the
# start.sh equivalent action_d2.
d2_profile_for() {
	local handle="$1"
	case "$handle" in
		d2-server|d2-runpod) echo "cuda" ;;
		d2-office)           echo "rocm" ;;
		*)                    echo "" ;;
	esac
}

# RunPod-specific deploy: instead of building remotely (RunPod doesn't
# typically expose a Docker daemon you can build on), we build locally with
# the cuda profile and push to a registry, then the pod pulls it. Stub
# implementation — relies on $RUNPOD_REGISTRY in .env. Falls back to local
# build only if no registry configured.
runpod_build_and_push() {
	local image_tag="${RUNPOD_REGISTRY:-elfege/anamnesis-d2}:cuda-latest"
	display_block "RunPod: build + push $image_tag"
	if [[ -z "${RUNPOD_REGISTRY:-}" ]]; then
		echo -e "${YELLOW}  RUNPOD_REGISTRY not set in .env — building locally only, not pushing${NC}"
		echo -e "${DIM}  Set RUNPOD_REGISTRY=docker.io/<youruser>/anamnesis-d2 to enable push${NC}"
	fi
	local nocache_flag=""
	$DO_NOCACHE && nocache_flag="--no-cache"
	# Build with cuda profile
	run bash -c "cd '$SCRIPT_DIR/d2' && docker compose --profile cuda -f docker-compose.yml build $nocache_flag d2-cuda"
	if [[ -n "${RUNPOD_REGISTRY:-}" ]]; then
		# Tag the just-built image and push
		run docker tag anamnesis-d2:cuda "$image_tag"
		run docker push "$image_tag"
	fi
}

# ── Prompt: --no-cache, --prune ───────────────────────────────
prompt_prune() {
	$DO_PRUNE && return 0
	local ans=""
	read -t 10 -r -p "  Prune Docker first? (yes/no, 10s timeout = no): " ans || true
	[[ "$ans" =~ ^(y|Y|yes|YES)$ ]] && DO_PRUNE=true
}

prompt_nocache() {
	$DO_NOCACHE && return 0
	local ans=""
	read -t 10 -r -p "  No-cache build? (type 'no' to skip, ENTER/timeout = yes): " ans || true
	[[ "$ans" =~ ^(n|N|no|NO)$ ]] || DO_NOCACHE=true
}

run_prune() {
	$DO_PRUNE || return 0
	display_block "Docker prune"
	run docker system prune -f
}

# ── Actions ──────────────────────────────────────────────────

build_one() {
	local handle="$1"
	local spec where dir compose service
	for t in "${TARGETS[@]}"; do
		if [[ "$(tgt_field 1 "$t")" == "$handle" ]]; then spec="$t"; break; fi
	done
	if [[ -z "$spec" ]]; then
		echo -e "${RED}Unknown target: $handle${NC}"; return 1
	fi
	where=$(tgt_field 2 "$spec")
	dir=$(tgt_field 3 "$spec")
	compose=$(tgt_field 4 "$spec")
	service=$(tgt_field 5 "$spec")

	display_block "Build: $handle (on $where)"
	if ! host_available "$where"; then
		echo -e "${DIM}  skipping — $where unreachable${NC}"
		return 0
	fi

	local nocache_flag=""
	$DO_NOCACHE && nocache_flag="--no-cache"

	# d² targets need a profile flag (cuda or rocm).
	local profile_flag=""
	local d2p
	d2p=$(d2_profile_for "$handle")
	[[ -n "$d2p" ]] && profile_flag="--profile $d2p"

	# RunPod targets: build locally with cuda profile, push to registry. The
	# pod itself doesn't expose a docker daemon for remote builds.
	if [[ "$where" == "runpod" ]]; then
		runpod_build_and_push
		return $?
	fi

	if [[ "$where" == "local" ]]; then
		run bash -c "cd '$dir' && docker compose $profile_flag -f '$compose' build $nocache_flag $service"
	else
		run_ssh "$where" "cd $dir && docker compose $profile_flag -f $compose build $nocache_flag $service"
	fi
}

action_local() {
	prompt_prune; run_prune
	prompt_nocache
	build_one anamnesis-app || return 1
	# After a local build we start locally (no builds in start.sh)
	display_block "Starting local stack"
	run "$SCRIPT_DIR/start.sh" --action=local ${TEST:+--test} ${DEBUG:+--debug}
}

action_all() {
	prompt_prune; run_prune
	prompt_nocache
	build_one anamnesis-app
	for t in "${TARGETS[@]}"; do
		local h=$(tgt_field 1 "$t")
		local w=$(tgt_field 2 "$t")
		[[ "$w" == "local" ]] && continue
		build_one "$h" || true
	done
	display_block "Starting everything"
	run "$SCRIPT_DIR/start.sh" --action=all ${TEST:+--test} ${DEBUG:+--debug}
}

action_remote_workers() {
	prompt_nocache
	for t in "${TARGETS[@]}"; do
		local h=$(tgt_field 1 "$t")
		local w=$(tgt_field 2 "$t")
		[[ "$w" == "local" ]] && continue
		build_one "$h" || true
	done
}

# Build the δ² engine on whichever GPU host the user picks.
# Default: server (CUDA, 1660 SUPER 6GB — sufficient for SmallMLP MNIST benchmarks).
# Office is reachable but unstable (RX 6800 ROCm crashes in MSG-116 / MSG of 2026-04-25).
# RunPod is the cloud option once a pod has been started via deploy_runpod.sh.
action_d2() {
	local target="${1:-server}"  # default to server (most stable)
	prompt_nocache
	case "$target" in
		server|office|runpod)
			build_one "d2-${target}" || return 1
			;;
		all)
			# Build for every reachable GPU host (server, office, runpod if up).
			for h in d2-server d2-office d2-runpod; do
				local w
				w=$(for t in "${TARGETS[@]}"; do
					[[ "$(tgt_field 1 "$t")" == "$h" ]] && tgt_field 2 "$t"
				done)
				if host_available "$w"; then
					build_one "$h" || true
				else
					echo -e "${DIM}  skip $h — $w unreachable${NC}"
				fi
			done
			;;
		*)
			echo -e "${RED}Unknown d² target: $target (use server|office|runpod|all)${NC}"
			return 2
			;;
	esac
	display_block "Starting δ² engine"
	run "$SCRIPT_DIR/start.sh" --action="d2:${target}" ${TEST:+--test} ${DEBUG:+--debug}
}

# ── Menu ──────────────────────────────────────────────────────

menu_main() {
	while true; do
		echo
		display_block "ANAMNESIS — Build & Deploy"
		$TEST && echo -e "  ${YELLOW}TEST MODE — no commands will be executed${NC}"
		cat <<-EOF

		  1) Rebuild + start LOCAL (anamnesis-app) only
		  2) Rebuild + start EVERYTHING (local + remote workers)
		  3) Rebuild a single image…
		  4) Rebuild all REMOTE workers only (no start)
		  5) Just prune Docker resources (local)
		  6) Build + start δ² engine (server / office / runpod / all)
		  0) Exit

		EOF
		read -r -p "  Select [0-6]: " choice
		case "$choice" in
			1) action_local ;;
			2) action_all ;;
			3) menu_targets ;;
			4) action_remote_workers ;;
			5) DO_PRUNE=true; run_prune ;;
			6) menu_d2 ;;
			0|q|Q|"") return 0 ;;
			*) echo -e "  ${YELLOW}?${NC}" ;;
		esac
	done
}

menu_d2() {
	echo
	display_block "δ² engine — pick host"
	cat <<-EOF

	  1) server   ${DIM}(NVIDIA CUDA — recommended, stable)${NC}
	  2) office   ${DIM}(AMD ROCm — unstable, see crash log 2026-04-25)${NC}
	  3) runpod   ${DIM}(cloud — pod must be started via deploy_runpod.sh first)${NC}
	  4) all      ${DIM}(every reachable GPU host)${NC}
	  0) Back

	EOF
	read -r -p "  Select [0-4]: " pick
	case "$pick" in
		1) action_d2 server ;;
		2) action_d2 office ;;
		3) action_d2 runpod ;;
		4) action_d2 all ;;
		0|q|Q|"") return 0 ;;
		*) echo "?" ;;
	esac
}

menu_targets() {
	echo
	display_block "Pick a target to rebuild"
	local i=1
	local picks=()
	for t in "${TARGETS[@]}"; do
		local h=$(tgt_field 1 "$t")
		local w=$(tgt_field 2 "$t")
		local reach="${GREEN}●${NC}"
		host_available "$w" || reach="${DIM}○${NC}"
		printf "  %2d) %b  %s ${DIM}(%s)${NC}\n" "$i" "$reach" "$h" "$w"
		picks+=("$h")
		i=$((i+1))
	done
	echo "   0) Back"
	echo
	read -r -p "  Select: " pick
	case "$pick" in
		[0-9]*)
			local idx=$((pick-1))
			[[ -n "${picks[$idx]}" ]] && { prompt_nocache; build_one "${picks[$idx]}"; } || echo "?"
			;;
		0|q|Q|"") return 0 ;;
		*) echo "?" ;;
	esac
}

# ── Entry point ───────────────────────────────────────────────
if $MENU_MODE; then
	menu_main
else
	case "$ACTION" in
		local|"") action_local ;;
		all)      action_all ;;
		remote)   action_remote_workers ;;
		worker-office) prompt_nocache; build_one avatar-worker-office ;;
		worker-server) prompt_nocache; build_one avatar-worker-server ;;
		trainer-office) prompt_nocache; build_one anamnesis-trainer-office ;;
		trainer-server) prompt_nocache; build_one anamnesis-trainer-server ;;
		d2)              action_d2 server ;;
		d2:server)       action_d2 server ;;
		d2:office)       action_d2 office ;;
		d2:runpod)       action_d2 runpod ;;
		d2:all)          action_d2 all ;;
		*)
			echo -e "${RED}Unknown --action=$ACTION${NC}"
			exit 2
			;;
	esac
fi
