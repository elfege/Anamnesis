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
# Color vars come from ~/.env.colors (exported as literal '\033...' strings).
# Use `echo -e` everywhere — that's the canonical pattern in this codebase.
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
	if [[ "$TEST" == "true" ]]; then
		echo -e "${YELLOW}[TEST]${NC} would run: $*"
		return 0
	fi
	if [[ "$DEBUG" == "true" ]]; then
		dbg "$*"
		"$@"
	else
		"$@"
	fi
}

run_ssh() {
	local host="$1"; shift
	local cmd="$*"
	if [[ "$TEST" == "true" ]]; then
		echo -e "${YELLOW}[TEST]${NC} would ssh ${host}: ${cmd}"
		return 0
	fi
	if [[ "$DEBUG" == "true" ]]; then
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
	"avatar-worker-runpod|local|$SCRIPT_DIR/avatar_worker|docker-compose.runpod.yml|avatar-worker-runpod"
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
	if [[ "$TEST" == "true" ]]; then return 0; fi
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
		# Skip the unstable + cloud targets here (handled by the optional
		# RunPod augmentation prompt below).
		# d2-office: ROCm/RX 6800 unstable per MSG-116. Opt-in via option 6.
		# d2-runpod / avatar-worker-runpod: see prompt_runpod_augment below.
		[[ "$h" == d2-office || "$h" == d2-runpod || "$h" == avatar-worker-runpod ]] && continue
		build_one "$h" || true
	done

	# ── Optional RunPod augmentation (asked AFTER local builds succeed,
	#    BEFORE start.sh fires services). User can skip every prompt — this
	#    is purely additive cloud GPU capacity. Each prompt is independent
	#    so e.g. "yes to avatar, no to d²" works.
	prompt_runpod_augment

	display_block "Starting everything"
	run "$SCRIPT_DIR/start.sh" --action=all ${TEST:+--test} ${DEBUG:+--debug}
}

# Ask whether to also deploy any RunPod-hosted services as part of this
# first-deployment flow. Each is independent. Skip is the default — empty
# input or 'n' / 'skip' continues without spinning anything cloud-side.
# If RunPod itself is unreachable / image push fails, surface
# [r]etry / [s]kip / [a]bort instead of silently failing.
prompt_runpod_augment() {
	# Safety: skip the whole block in non-interactive runs (CI, --action=*).
	[[ "$MENU_MODE" != "true" ]] && return 0
	[[ "$TEST" == "true" ]] && { echo -e "${YELLOW}[TEST]${NC} would prompt for RunPod augmentation"; return 0; }

	# RunPod creds present?
	if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
		echo -e "${DIM}  (RunPod augmentation skipped — RUNPOD_API_KEY not set in .env)${NC}"
		return 0
	fi

	echo
	display_block "Optional: also deploy RunPod-hosted services?"
	echo -e "${DIM}  Each prompt is independent. Skip any with ENTER. Cost shows before any pod is spun.${NC}"
	echo

	# Avatar worker on RunPod: bigger GPU = working voice + animation
	echo -e "  ${BOLD}avatar-worker on RunPod${NC} ${DIM}(XTTS + SadTalker on a 24GB pod, ~\$0.30/hr)${NC}"
	local ans=""
	read -t 30 -r -p "  Deploy now? [y/N/skip] " ans || true
	case "$ans" in
		y|Y|yes|YES) _runpod_augment_one "avatar:runpod" "avatar worker" ;;
		*)           echo -e "${DIM}  Skipped — local server/office workers stay primary.${NC}" ;;
	esac

	echo
	# d² engine on RunPod: only matters for users who don't have a CUDA host
	echo -e "  ${BOLD}δ² engine on RunPod${NC} ${DIM}(continual-learning research, ~\$0.30/hr; server already has CUDA)${NC}"
	read -t 30 -r -p "  Deploy now? [y/N/skip] " ans || true
	case "$ans" in
		y|Y|yes|YES) _runpod_augment_one "d2:runpod" "δ² engine" ;;
		*)           echo -e "${DIM}  Skipped — d²-server (local CUDA) handles personal/bench runs.${NC}" ;;
	esac
}

# Helper: invoke one RunPod-augmentation action with retry/skip/abort UX
# on failure. Args: $1 = --action= value (e.g. avatar:runpod), $2 = label.
_runpod_augment_one() {
	local action="$1"
	local label="$2"
	while true; do
		display_block "Deploying $label to RunPod"
		if run "$SCRIPT_DIR/deploy.sh" --action="$action" ${TEST:+--test} ${DEBUG:+--debug}; then
			echo -e "${GREEN}  $label: deployed.${NC}"
			return 0
		fi
		echo -e "${RED}  $label: deployment failed.${NC}"
		local choice=""
		read -r -p "  [r]etry / [s]kip / [a]bort entire deploy? " choice
		case "$choice" in
			r|R|retry) continue ;;
			a|A|abort) echo -e "${RED}  Aborting full deploy.${NC}"; exit 1 ;;
			*)         echo -e "${DIM}  Skipped — continuing without $label.${NC}"; return 0 ;;
		esac
	done
}

action_remote_workers() {
	prompt_nocache
	for t in "${TARGETS[@]}"; do
		local h=$(tgt_field 1 "$t")
		local w=$(tgt_field 2 "$t")
		[[ "$w" == "local" ]] && continue
		# Skip d²-* (research-mode, opt-in only — see action_d2)
		[[ "$h" == d2-* ]] && continue
		build_one "$h" || true
	done
}

# Build the δ² engine on whichever GPU host the user picks.
# Default: server (CUDA, 1660 SUPER 6GB — sufficient for SmallMLP MNIST benchmarks).
# Office is reachable but unstable (RX 6800 ROCm crashes in MSG-116 / MSG of 2026-04-25).
# RunPod is the cloud option once a pod has been started via deploy_runpod.sh.
# Build the avatar-worker image for RunPod (locally), push to ghcr.io, then
# spin up a pod via deploy_runpod.sh with RUNPOD_PROFILE=avatar.
# Uses the same private-registry auth (RUNPOD_REGISTRY_AUTH_ID) as d2-runpod.
# Cost: rtx3090 community ~$0.30/hr — confirmation prompt enforced inside
# deploy_runpod.sh; no auto-start.
action_avatar_runpod() {
	prompt_nocache
	local image_tag="ghcr.io/elfege/anamnesis-avatar-worker:cuda-runpod"
	display_block "avatar-worker (RunPod): build + push $image_tag"
	local nocache_flag=""
	$DO_NOCACHE && nocache_flag="--no-cache"
	# Build the image (locally — RunPod doesn't expose a build daemon).
	if ! run bash -c "cd '$SCRIPT_DIR/avatar_worker' && docker build $nocache_flag -t '$image_tag' -f Dockerfile.runpod ."; then
		echo -e "${YELLOW}  Local build failed — you can still try building on the pod via docker exec after start (slow).${NC}"
		return 1
	fi
	# Push (best-effort — if auth not set up, deploy_runpod.sh will fail to
	# pull, and the user can rerun after `docker login ghcr.io`).
	if ! run docker push "$image_tag"; then
		echo -e "${YELLOW}  Push failed — make sure 'docker login ghcr.io' is configured. Pod start will fail until pushed.${NC}"
		read -r -p "  Continue to spin up the pod anyway? [y/N] " ans
		[[ ! "$ans" =~ ^(y|Y|yes|YES)$ ]] && return 1
	fi
	display_block "Spinning RunPod pod (profile=avatar, GPU=rtx3090)"
	echo -e "${DIM}  Confirmation prompt enforced inside deploy_runpod.sh; pod adds itself as AVATAR_WORKER_URL_5 in .env on success.${NC}"
	run bash -c "RUNPOD_PROFILE=avatar '$SCRIPT_DIR/deploy_runpod.sh' start --gpu rtx3090"
}

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
		[[ "$TEST" == "true" ]] && echo -e "  ${YELLOW}TEST MODE — no commands will be executed${NC}"
		echo
		# Menu printed via echo -e so ANSI color escapes are interpreted
		# (heredocs print color vars as literal \033 strings).
		echo -e "  ${BOLD}Chat + Avatar pipeline${NC}     ${DIM}(everyday services)${NC}"
		echo -e "   1) Rebuild + start LOCAL only        ${DIM}→ anamnesis-app + mongo${NC}"
		echo -e "   2) Rebuild + start FULL chat stack   ${DIM}→ anamnesis-app + avatar-workers + trainers + d2-server${NC}"
		echo -e "   3) Rebuild a single image…           ${DIM}→ pick any one target${NC}"
		echo -e "   4) Rebuild REMOTE workers only       ${DIM}→ no start, just images on office/server${NC}"
		echo
		echo -e "  ${BOLD}δ² engine${NC}                    ${DIM}(continual-learning research, opt-in)${NC}"
		echo -e "   6) Build + start δ² engine…          ${DIM}→ server / office / runpod / all${NC}"
		echo
		echo -e "  ${BOLD}Avatar (XTTS + SadTalker)${NC}"
		echo -e "   7) Build + start avatar-worker on RunPod  ${DIM}→ rtx3090 ~\$0.30/hr, additive to local workers${NC}"
		echo
		echo -e "  ${BOLD}Maintenance${NC}"
		echo -e "   5) Just prune Docker resources       ${DIM}→ local only${NC}"
		echo
		echo "   0) Exit"
		echo
		read -r -p "  Select [0-7]: " choice
		case "$choice" in
			1) action_local ;;
			2) action_all ;;
			3) menu_targets ;;
			4) action_remote_workers ;;
			5) DO_PRUNE=true; run_prune ;;
			6) menu_d2 ;;
			7) action_avatar_runpod ;;
			0|q|Q|"") return 0 ;;
			*) echo -e "  ${YELLOW}?${NC}" ;;
		esac
	done
}

menu_d2() {
	echo
	display_block "δ² engine — pick host"
	echo
	echo -e "  1) server   ${DIM}(NVIDIA CUDA — recommended, stable)${NC}"
	echo -e "  2) office   ${DIM}(AMD ROCm — unstable, see crash log 2026-04-25)${NC}"
	echo -e "  3) runpod   ${DIM}(cloud — will offer to spin a pod via deploy_runpod.sh, ~\$0.30/hr)${NC}"
	echo -e "  4) all      ${DIM}(every reachable GPU host)${NC}"
	echo    "  0) Back"
	echo
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
if [[ "$MENU_MODE" == "true" ]]; then
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
		avatar:runpod)   action_avatar_runpod ;;
		*)
			echo -e "${RED}Unknown --action=$ACTION${NC}"
			exit 2
			;;
	esac
fi
