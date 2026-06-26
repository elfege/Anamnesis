#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            status.sh                                       ║
# ║  Live status of every Anamnesis service across all hosts.                  ║
# ║  Standalone (default: full table) AND callable by start.sh menu (--compact).║
# ║  TODO-marked entries indicate services not yet deployed/implemented.       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

########################################################################-########################################################################
#                                                SOURCE ENVIRONMENT                                                                             #
########################################################################-########################################################################
# Canonical 2-line bootstrap (matches ~/0_SCRIPTS/0_SYNC/0_ENGINES/sync_linux_homes.sh).
declare -f source_global_env &>/dev/null || . ~/.bash_utils --no-exec >/dev/null 2>&1 || true
if declare -f source_global_env &>/dev/null; then
    source_global_env || true
else
    # Minimal fallback if .bash_utils isn't available — just load colors.
    [[ -f ~/.env.colors ]] && . ~/.env.colors 2>/dev/null
    : "${BOLD:=$'\033[1m'}"
    : "${NC:=$'\033[0m'}"
    : "${CYAN:=$'\033[36m'}"
    : "${GREEN:=$'\033[32m'}"
    : "${YELLOW:=$'\033[33m'}"
    : "${RED:=$'\033[31m'}"
    : "${DIM:=$'\033[2m'}"
fi

########################################################################-########################################################################
#                                                VARIABLES                                                                                      #
########################################################################-########################################################################
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_TIMEOUT=3

# Service catalog. Each entry: handle|where|container|status_tag
#   where:      "local" | ssh-alias ("office","server","rog","runpod")
#   container:  docker container name (or "-" if not deployed)
#   status_tag: "" (implemented) | "TODO" (not yet built/deployed)
#
# Keep parity with start.sh SERVICES list; TODO entries surface gaps the menu
# should show as unfinished work rather than failures.
SERVICES=(
    "anamnesis-app|local|anamnesis-app|"
    "anamnesis-mongo|local|anamnesis-mongo|"
    "avatar-worker-office|office|avatar-worker-office|"
    "anamnesis-trainer-office|office|anamnesis-trainer-office|"
    "anamnesis-trainer-server|server|anamnesis-trainer-server|"
    "d2-inference-engine-server|server|anamnesis-d2|"
    "d2-inference-engine-office|office|anamnesis-d2|"
)
# Intentionally dropped (2026-06-08):
#   - avatar-worker-server: server's 6GB GTX 1660 SUPER is too tight for SadTalker
#     (OOMed in prod test 2026-05-21). Office RX 6800 16GB is the single avatar host.
#   - avatar-worker-rog: rog is dev/testing only, not a serving target.
#   - d2-runpod: cloud burst capacity is a future-phase concern; not currently wired.

ARGS=("$@")
MODE="full"          # full | compact | service
TARGET_SERVICE=""    # set when --service <name> is used
NO_COLOR=false

########################################################################-########################################################################
#                                                HELP                                                                                           #
########################################################################-########################################################################
status__show_help() {
    echo ""
    echo -e "${BOLD}${SCRIPT_NAME}${NC} — live status of every Anamnesis service"
    echo ""
    echo -e "${BOLD}Usage:${NC}"
    echo -e "  bash ${SCRIPT_NAME} [--compact] [--service <handle>] [--no-color]"
    echo ""
    echo -e "${BOLD}Modes:${NC}"
    echo -e "  (default)              full table — every service, status, host, container"
    echo -e "  ${CYAN}--compact${NC}              one line per service (used by start.sh menu inlining)"
    echo -e "  ${CYAN}--service <handle>${NC}     query a single service by handle (e.g. avatar-worker-office)"
    echo -e "  ${CYAN}--no-color${NC}             strip ANSI colors (for log capture / non-TTY consumers)"
    echo ""
    echo -e "${BOLD}Status values:${NC}"
    echo -e "  ${GREEN}● up${NC}            container running (or its host /health returned OK for remote-only services)"
    echo -e "  ${YELLOW}● restarting${NC}    container in restart loop"
    echo -e "  ${RED}● down${NC}          container exists but stopped/exited/dead"
    echo -e "  ${DIM}○ missing${NC}       container not present on host"
    echo -e "  ${DIM}◌ unreachable${NC}   host SSH probe timed out (${SSH_TIMEOUT}s)"
    echo -e "  ${DIM}⊘ TODO${NC}          service not yet deployed/implemented (placeholder in registry)"
    echo ""
    exit 0
}

for arg in "${ARGS[@]}"; do
    case "$arg" in
        --help|-h) status__show_help ;;
        --compact) MODE="compact" ;;
        --no-color) NO_COLOR=true ;;
        --service)
            MODE="service"
            ;;
    esac
done
# --service <handle> — capture the positional that follows
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" == "--service" ]]; then
        TARGET_SERVICE="${ARGS[$((i+1))]:-}"
        [[ -z "$TARGET_SERVICE" ]] && { echo -e "${RED}--service requires a handle${NC}" >&2; exit 2; }
        break
    fi
done

# Honor --no-color
if $NO_COLOR; then
    BOLD=""; NC=""; CYAN=""; GREEN=""; YELLOW=""; RED=""; DIM=""
fi

########################################################################-########################################################################
#                                                PROBE                                                                                          #
########################################################################-########################################################################
# Probe one container's docker state. Returns one of:
#   up | restarting | down | missing | unreachable | TODO
#
# - "local" host probes via local docker daemon
# - remote hosts probe via ssh <host> docker inspect (with timeout)
# - TODO services skip the probe entirely
status__probe() {
    local where="$1" container="$2" tag="$3"
    if [[ "$tag" == "TODO" ]]; then
        echo "TODO"
        return
    fi
    if [[ "$where" == "runpod" ]]; then
        # RunPod presence is a future-Phase concern; mark as missing for now if not in
        # AVATAR_WORKER_URL_5 / D2_RUNPOD env. Cheap heuristic: env var presence.
        echo "missing"
        return
    fi

    local raw
    if [[ "$where" == "local" ]]; then
        raw=$(docker inspect --format '{{.State.Status}}' "$container" 2>/dev/null) || raw=""
    else
        raw=$(ssh -o ConnectTimeout=$SSH_TIMEOUT -o BatchMode=yes "$where" \
              "docker inspect --format '{{.State.Status}}' $container 2>/dev/null" 2>/dev/null) \
              || { echo "unreachable"; return; }
    fi

    case "$raw" in
        running)    echo "up" ;;
        restarting) echo "restarting" ;;
        "")         echo "missing" ;;
        *)          echo "down" ;;  # exited, dead, paused, created
    esac
}

# Symbolic + colored marker for a status value.
status__marker() {
    local s="$1"
    case "$s" in
        up)          printf '%b' "${GREEN}● up${NC}" ;;
        restarting)  printf '%b' "${YELLOW}● restarting${NC}" ;;
        down)        printf '%b' "${RED}● down${NC}" ;;
        missing)     printf '%b' "${DIM}○ missing${NC}" ;;
        unreachable) printf '%b' "${DIM}◌ unreachable${NC}" ;;
        TODO)        printf '%b' "${DIM}⊘ TODO${NC}" ;;
        *)           printf '%b' "${DIM}? $s${NC}" ;;
    esac
}

########################################################################-########################################################################
#                                                RENDER                                                                                         #
########################################################################-########################################################################
status__render_full() {
    local cols=$(tput cols 2>/dev/null || echo 100)
    echo ""
    echo -e "${BOLD}Anamnesis service status${NC} ${DIM}- $(date '+%Y-%m-%d %H:%M:%S %Z')${NC}"
    printf '%*s\n' "$cols" '' | tr ' ' '-'
    # printf doesn't interpret \033 in format strings; use %b for color escapes
    # and feed them as args. Header text stays in format; color separately.
    printf '%b%-26s %-10s %-22s %-30s%b\n' "${BOLD}" "HANDLE" "WHERE" "CONTAINER" "STATUS" "${NC}"
    printf '%*s\n' "$cols" '' | tr ' ' '-'

    local spec handle where container tag s
    for spec in "${SERVICES[@]}"; do
        handle=$(echo "$spec" | awk -F'|' '{print $1}')
        where=$(echo "$spec" | awk -F'|' '{print $2}')
        container=$(echo "$spec" | awk -F'|' '{print $3}')
        tag=$(echo "$spec" | awk -F'|' '{print $4}')
        s=$(status__probe "$where" "$container" "$tag")
        printf "%-26s %-10s %-22s %b\n" "$handle" "$where" "$container" "$(status__marker "$s")"
    done
    printf '%*s\n' "$cols" '' | tr ' ' '-'

    status__render_active_jobs
    echo -e "${DIM}Use ${CYAN}status --help${DIM} for status value definitions.${NC}"
    echo ""
}

# Render an "Active jobs" block after the service table. Currently sources from
# d² engine /health (training_status, current_run_id, bassin_size, active_lora_adapter,
# loaded_lora_count). Trainer jobs would land here too once trainers expose /jobs.
status__render_active_jobs() {
    local d2_url="${D2_ENDPOINT_URL:-http://192.168.10.15:3015}"
    local d2_health
    d2_health=$(curl -s --max-time 2 "${d2_url}/health" 2>/dev/null) || d2_health=""
    echo ""
    echo -e "${BOLD}Active jobs${NC}"
    if [[ -z "$d2_health" ]]; then
        echo -e "  ${DIM}d² engine unreachable at ${d2_url} \xe2\x80\x94 no active-job data${NC}"
        echo ""
        return
    fi

    # Minimal JSON-field extraction without jq dependency. Each line falls through
    # to a default if jq isn't installed.
    local ts ri opt lora bassin loaded loaded_n
    if command -v jq >/dev/null 2>&1; then
        ts=$(echo "$d2_health"     | jq -r '.training_status // "?"')
        ri=$(echo "$d2_health"     | jq -r '.current_run_id // ""')
        opt=$(echo "$d2_health"    | jq -r '.current_optimizer // ""')
        lora=$(echo "$d2_health"   | jq -r '.active_lora_adapter // ""')
        loaded=$(echo "$d2_health" | jq -r '.model_loaded')
        bassin=$(echo "$d2_health" | jq -r '.bassin_size // 0')
        loaded_n=$(echo "$d2_health" | jq -r '.loaded_lora_count // 0')
    else
        ts=$(echo "$d2_health" | grep -oP '"training_status"\s*:\s*"[^"]*"' | sed 's/.*"\([^"]*\)"$/\1/')
        ri=$(echo "$d2_health" | grep -oP '"current_run_id"\s*:\s*"?[^",}]*' | sed 's/.*:\s*"\?//;s/"$//')
        loaded=$(echo "$d2_health" | grep -oP '"model_loaded"\s*:\s*[a-z]+' | grep -oP 'true|false')
        bassin=$(echo "$d2_health" | grep -oP '"bassin_size"\s*:\s*[0-9]+' | grep -oP '[0-9]+$')
        loaded_n=$(echo "$d2_health" | grep -oP '"loaded_lora_count"\s*:\s*[0-9]+' | grep -oP '[0-9]+$')
        lora=$(echo "$d2_health" | grep -oP '"active_lora_adapter"\s*:\s*"[^"]*"' | sed 's/.*"\([^"]*\)"$/\1/')
        opt=$(echo "$d2_health" | grep -oP '"current_optimizer"\s*:\s*"[^"]*"' | sed 's/.*"\([^"]*\)"$/\1/')
    fi

    local any=0
    if [[ -n "$ts" && "$ts" != "idle" && "$ts" != "null" && "$ts" != "?" ]]; then
        printf "  %b d² training%s %s\n" "${YELLOW}●${NC}" "${ri:+ · run=${ri:0:8}}${opt:+ · $opt}" "$ts"
        any=1
    fi
    if [[ -n "$lora" && "$lora" != "null" ]]; then
        printf "  %b d² LoRA loaded · %s\n" "${GREEN}●${NC}" "$lora"
        any=1
    elif [[ "$loaded" == "true" ]]; then
        printf "  %b d² base model loaded\n" "${GREEN}●${NC}"
        any=1
    fi
    if [[ -n "$loaded_n" && "$loaded_n" != "0" ]]; then
        printf "  %b d² LoRA pool · %s adapter(s) resident\n" "${GREEN}●${NC}" "$loaded_n"
        any=1
    fi
    if [[ -n "$bassin" ]]; then
        local marker="${DIM}○${NC}"
        [[ "$bassin" != "0" ]] && marker="${GREEN}●${NC}"
        printf "  %b d² bassin size · %s tensors\n" "$marker" "$bassin"
        any=1
    fi
    if [[ "$any" == "0" ]]; then
        echo -e "  ${DIM}(no active jobs) idle across all services${NC}"
    fi
    echo ""
}

status__render_compact() {
    # One line per service, no header — designed for embedding in start.sh menu rendering.
    # Format: <handle> <marker>   (left-padded handle column for visual alignment)
    local spec handle where container tag s
    for spec in "${SERVICES[@]}"; do
        handle=$(echo "$spec" | awk -F'|' '{print $1}')
        where=$(echo "$spec" | awk -F'|' '{print $2}')
        container=$(echo "$spec" | awk -F'|' '{print $3}')
        tag=$(echo "$spec" | awk -F'|' '{print $4}')
        s=$(status__probe "$where" "$container" "$tag")
        printf "  %-26s %b\n" "$handle" "$(status__marker "$s")"
    done
}

status__render_one() {
    local spec handle where container tag s
    for spec in "${SERVICES[@]}"; do
        handle=$(echo "$spec" | awk -F'|' '{print $1}')
        if [[ "$handle" == "$TARGET_SERVICE" ]]; then
            where=$(echo "$spec" | awk -F'|' '{print $2}')
            container=$(echo "$spec" | awk -F'|' '{print $3}')
            tag=$(echo "$spec" | awk -F'|' '{print $4}')
            s=$(status__probe "$where" "$container" "$tag")
            printf "%-26s %-10s %-22s %b\n" "$handle" "$where" "$container" "$(status__marker "$s")"
            return 0
        fi
    done
    echo -e "${RED}unknown service: $TARGET_SERVICE${NC}" >&2
    echo -e "${DIM}Known handles:${NC}" >&2
    for spec in "${SERVICES[@]}"; do
        echo "  $(echo "$spec" | awk -F'|' '{print $1}')" >&2
    done
    return 2
}

########################################################################-########################################################################
#                                                MAIN                                                                                           #
########################################################################-########################################################################
status__run() {
    case "$MODE" in
        full)    status__render_full ;;
        compact) status__render_compact ;;
        service) status__render_one ;;
    esac
}

status__run
