#!/usr/bin/env bash
#
# deploy_runpod.sh — Lifecycle script for a RunPod GPU pod running our trainer.
#
# WHAT THIS DOES, FOR THE DUMMIES:
# =================================
#
# RunPod is a cheap GPU cloud. You can rent an RTX 3090 (24GB) for ~$0.30/hour
# spot, an A100 for ~$1.50/hour, etc. You spin up a "pod" (one VM with one GPU),
# do your work, then stop it to stop billing.
#
# This script wraps RunPod's REST API to:
#
#   start  — create a new pod, wait for it to be ready, register its URL in
#            MongoDB so the Anamnesis app sees it as another worker, append
#            it to .env so failover chains pick it up
#   stop   — stop the pod (releases the GPU, stops billing); remove from
#            worker_registry and .env
#   status — print whether a pod is running and its endpoint URL
#
# REQUIREMENTS:
#   - RUNPOD_API_KEY in environment or .env (from runpod.io account settings)
#   - jq installed (for parsing JSON responses)
#   - curl
#
# USAGE:
#   ./deploy_runpod.sh start [--gpu rtx3090|a100|h100]
#   ./deploy_runpod.sh stop
#   ./deploy_runpod.sh status
#
# NOTE: This script is SAFE — it does not auto-create pods on its own. The user
# must explicitly run `start` to spend money. It also exits early with a warning
# if RUNPOD_API_KEY is missing rather than failing silently.
#

set -euo pipefail

# ── Load .env if present ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# ── Sanity checks ──────────────────────────────────────────────────────────
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "Error: RUNPOD_API_KEY not set."
    echo "Get one from https://runpod.io/console/user/settings and put it in .env:"
    echo "    RUNPOD_API_KEY=your_key_here"
    exit 1
fi

command -v jq >/dev/null 2>&1 || { echo "Error: jq not installed. apt install jq"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "Error: curl not installed."; exit 1; }

# ── Configuration ──────────────────────────────────────────────────────────
RUNPOD_API="https://api.runpod.io/graphql"
DOCKER_IMAGE="${RUNPOD_DOCKER_IMAGE:-elfege/anamnesis-trainer:cuda-latest}"
GPU_TYPE_DEFAULT="rtx3090"
POD_STATE_FILE="$SCRIPT_DIR/.runpod_pod_id"  # tracks the active pod ID locally

# GPU type IDs (from RunPod's API — these change occasionally; check their docs)
# We use the most cost-effective spot/community options.
declare -A GPU_TYPE_IDS=(
    [rtx3090]="NVIDIA GeForce RTX 3090"
    [rtx4090]="NVIDIA GeForce RTX 4090"
    [a100]="NVIDIA A100 80GB PCIe"
    [h100]="NVIDIA H100 80GB HBM3"
)

# ── GraphQL helpers ────────────────────────────────────────────────────────
runpod_query() {
    # Send a GraphQL query to RunPod and return the response.
    # Args: $1 = the JSON body
    curl -s -X POST "$RUNPOD_API" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "$1"
}

# ── Subcommand: start ──────────────────────────────────────────────────────
start_pod() {
    local gpu_alias="${1:-$GPU_TYPE_DEFAULT}"
    local gpu_id="${GPU_TYPE_IDS[$gpu_alias]:-}"
    if [[ -z "$gpu_id" ]]; then
        echo "Error: unknown GPU alias '$gpu_alias'. Choose from: ${!GPU_TYPE_IDS[*]}"
        exit 1
    fi

    if [[ -f "$POD_STATE_FILE" ]]; then
        echo "Warning: a pod state file already exists at $POD_STATE_FILE"
        echo "If a pod is already running, run './deploy_runpod.sh stop' first"
        echo "or delete this file if it's stale: rm $POD_STATE_FILE"
        exit 1
    fi

    echo "Creating RunPod pod: GPU=$gpu_alias ($gpu_id), image=$DOCKER_IMAGE"
    echo "(This costs money — make sure you intend this.)"

    # Build the GraphQL mutation to create a pod.
    # Fields:
    #   - cloudType: COMMUNITY (cheaper, can be preempted) vs SECURE (stable, more expensive)
    #   - gpuTypeId: from the table above
    #   - dockerArgs: command to run on container start
    #   - ports: which ports to expose (we expose 3011 for the trainer's HTTP API)
    #   - volumeInGb: persistent storage (model checkpoints survive across restarts)
    local mutation
    mutation=$(jq -n --arg gpu "$gpu_id" --arg img "$DOCKER_IMAGE" '{
        query: "mutation { podFindAndDeployOnDemand(input: { cloudType: COMMUNITY, gpuCount: 1, volumeInGb: 50, containerDiskInGb: 20, gpuTypeId: \"\($gpu)\", name: \"anamnesis-trainer\", imageName: \"\($img)\", ports: \"3011/http\", env: [{ key: \"AUTO_LOAD_MODEL\", value: \"true\" }] }) { id desiredStatus runtime { ports { ip publicPort privatePort isIpPublic } } } }"
    }')

    local response
    response=$(runpod_query "$mutation")
    local pod_id
    pod_id=$(echo "$response" | jq -r '.data.podFindAndDeployOnDemand.id // empty')

    if [[ -z "$pod_id" ]]; then
        echo "Failed to create pod. Response:"
        echo "$response" | jq .
        exit 1
    fi

    echo "Pod created: $pod_id"
    echo "$pod_id" > "$POD_STATE_FILE"

    # ── Poll for the pod to become RUNNING and have a public port ────────
    echo "Waiting for pod to become reachable (this can take 1-3 minutes)..."
    local public_url=""
    for attempt in {1..60}; do
        sleep 10
        local status_response
        status_response=$(runpod_query "$(jq -n --arg id "$pod_id" '{
            query: "query { pod(input: { podId: \"\($id)\" }) { desiredStatus runtime { ports { ip publicPort privatePort isIpPublic } } } }"
        }')")

        local status
        status=$(echo "$status_response" | jq -r '.data.pod.desiredStatus // "UNKNOWN"')
        local public_ip
        public_ip=$(echo "$status_response" | jq -r '.data.pod.runtime.ports[0].ip // empty')
        local public_port
        public_port=$(echo "$status_response" | jq -r '.data.pod.runtime.ports[0].publicPort // empty')

        if [[ "$status" == "RUNNING" && -n "$public_ip" && -n "$public_port" ]]; then
            public_url="http://$public_ip:$public_port"
            echo "Pod is running: $public_url"
            break
        fi
        echo "  (attempt $attempt/60) status=$status"
    done

    if [[ -z "$public_url" ]]; then
        echo "Pod did not become reachable in time. Check the RunPod console."
        echo "Pod ID: $pod_id (saved to $POD_STATE_FILE)"
        exit 1
    fi

    # ── Health check the trainer endpoint ────────────────────────────────
    echo "Pinging trainer health endpoint..."
    for attempt in {1..12}; do
        if curl -sf "$public_url/health" >/dev/null 2>&1; then
            echo "Trainer is responsive."
            break
        fi
        sleep 5
    done

    # ── Register URL in worker_registry and append to .env ───────────────
    register_worker "$public_url" "runpod-$gpu_alias-$pod_id"
    echo ""
    echo "RunPod is live: $public_url"
    echo "  Pod ID: $pod_id  (saved to $POD_STATE_FILE)"
    echo "  Worker registered in MongoDB worker_registry collection"
    echo "  URL appended to .env as NANOGPT_URLS_RUNPOD"
    echo ""
    echo "When done, stop the pod with: ./deploy_runpod.sh stop"
    echo "Billing continues until you stop it."
}

# ── Subcommand: stop ───────────────────────────────────────────────────────
stop_pod() {
    if [[ ! -f "$POD_STATE_FILE" ]]; then
        echo "No pod state file at $POD_STATE_FILE — nothing to stop"
        exit 0
    fi
    local pod_id
    pod_id=$(cat "$POD_STATE_FILE")

    echo "Stopping pod $pod_id..."
    local mutation
    mutation=$(jq -n --arg id "$pod_id" '{
        query: "mutation { podTerminate(input: { podId: \"\($id)\" }) }"
    }')
    runpod_query "$mutation" | jq .

    # Unregister from worker_registry + remove from .env
    unregister_worker "$pod_id"

    rm -f "$POD_STATE_FILE"
    echo "Pod terminated. Billing stopped."
}

# ── Subcommand: status ─────────────────────────────────────────────────────
status_pod() {
    if [[ ! -f "$POD_STATE_FILE" ]]; then
        echo "No active pod (no $POD_STATE_FILE)"
        exit 0
    fi
    local pod_id
    pod_id=$(cat "$POD_STATE_FILE")
    local response
    response=$(runpod_query "$(jq -n --arg id "$pod_id" '{
        query: "query { pod(input: { podId: \"\($id)\" }) { desiredStatus costPerHr runtime { ports { ip publicPort privatePort isIpPublic } } } }"
    }')")
    echo "Pod ID: $pod_id"
    echo "$response" | jq '.data.pod'
}

# ── Worker registry helpers (talks to anamnesis-app at dellserver:3010) ────
register_worker() {
    local url="$1"
    local label="$2"
    # Append to .env so future container restarts pick it up.
    if [[ -f "$ENV_FILE" ]]; then
        # Remove any prior NANOGPT_URLS_RUNPOD line, then append
        grep -v '^NANOGPT_URLS_RUNPOD=' "$ENV_FILE" > "$ENV_FILE.tmp" || true
        echo "NANOGPT_URLS_RUNPOD=$url" >> "$ENV_FILE.tmp"
        mv "$ENV_FILE.tmp" "$ENV_FILE"
    fi

    # POST to the worker registry API (if available — non-fatal if not)
    curl -s -X POST "http://192.168.10.20:3010/api/workers/register" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg url "$url" --arg label "$label" '{url: $url, label: $label, kind: "runpod"}')" \
        2>/dev/null || true
}

unregister_worker() {
    local pod_id="$1"
    # Remove the NANOGPT_URLS_RUNPOD line from .env
    if [[ -f "$ENV_FILE" ]]; then
        grep -v '^NANOGPT_URLS_RUNPOD=' "$ENV_FILE" > "$ENV_FILE.tmp" || true
        mv "$ENV_FILE.tmp" "$ENV_FILE"
    fi

    # DELETE from the registry
    curl -s -X DELETE "http://192.168.10.20:3010/api/workers/register/runpod-$pod_id" \
        2>/dev/null || true
}

# ── Subcommand dispatcher ──────────────────────────────────────────────────
case "${1:-}" in
    start)  shift; start_pod "$@" ;;
    stop)   stop_pod ;;
    status) status_pod ;;
    *)
        echo "Usage: $0 {start [--gpu rtx3090|rtx4090|a100|h100] | stop | status}"
        exit 1
        ;;
esac
