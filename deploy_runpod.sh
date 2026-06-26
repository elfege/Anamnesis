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
# Per-profile state files so trainer/d2/avatar pods can co-exist independently.
# Resolved later once PROFILE is known.
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
PROFILE="${RUNPOD_PROFILE:-trainer}"  # "trainer" | "d2" | "avatar"
GPU_TYPE_DEFAULT="rtx3090"
# Per-profile pod state file so multiple pod kinds can run side-by-side.
# Backward compat: trainer keeps the historical .runpod_pod_id name.
case "$PROFILE" in
    trainer) POD_STATE_FILE="$SCRIPT_DIR/.runpod_pod_id" ;;
    *)       POD_STATE_FILE="$SCRIPT_DIR/.runpod_pod_id.${PROFILE}" ;;
esac

# Per-profile defaults (image + port). Override via RUNPOD_DOCKER_IMAGE / RUNPOD_PORT.
case "$PROFILE" in
    trainer)
        DEFAULT_IMAGE="elfege/anamnesis-trainer:cuda-latest"
        DEFAULT_PORT="3011"
        ;;
    d2)
        DEFAULT_IMAGE="ghcr.io/elfege/anamnesis-d2:cuda-runpod"
        DEFAULT_PORT="3015"
        ;;
    avatar)
        # XTTS + SadTalker GPU worker. Image pre-bakes SadTalker checkpoints +
        # XTTS v2 weights so first request avoids cold-start download.
        # Local fallback workers on server/office stay running independently —
        # this is purely additive (slot 5 in the AVATAR_WORKER_URL_N chain).
        DEFAULT_IMAGE="ghcr.io/elfege/anamnesis-avatar-worker:cuda-runpod"
        DEFAULT_PORT="3013"
        ;;
    *)
        echo "Error: unknown RUNPOD_PROFILE='$PROFILE'. Use 'trainer', 'd2', or 'avatar'."
        exit 1
        ;;
esac
DOCKER_IMAGE="${RUNPOD_DOCKER_IMAGE:-$DEFAULT_IMAGE}"
SERVICE_PORT="${RUNPOD_PORT:-$DEFAULT_PORT}"

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
        local stale_pod_id
        # State file is plain text containing just the pod_id (see line ~245),
        # NOT JSON — read it raw.
        stale_pod_id=$(head -1 "$POD_STATE_FILE" 2>/dev/null | tr -d '[:space:]' || echo "")
        [[ -n "$stale_pod_id" ]] && echo "  Recorded pod_id: $stale_pod_id"
        echo "If a pod is still running on RunPod, './deploy_runpod.sh stop' is the safe choice"
        echo "(otherwise you'll orphan a billing pod and only delete the local pointer)."
        read -r -p "Delete this state file and proceed to create a NEW pod? [y/N] " del_conf
        if [[ "$del_conf" =~ ^(y|Y|yes|YES)$ ]]; then
            rm -f "$POD_STATE_FILE"
            echo "Removed $POD_STATE_FILE"
        else
            echo "Aborted — state file kept."
            exit 1
        fi
    fi

    echo "Creating RunPod pod: profile=$PROFILE, GPU=$gpu_alias ($gpu_id), image=$DOCKER_IMAGE"
    echo "(This costs money — make sure you intend this.)"

    # ── Cost-confirmation guard ───────────────────────────────────────────
    # rtx3090 community ~$0.22-0.45/hr. rtx4090 ~$0.40. a100 ~$1.50. h100 ~$2.50.
    # For non-trivial spend (anything above $1/hr) require explicit re-typed
    # confirmation. Below that, a single y/N prompt suffices.
    local approx_hourly=""
    case "$gpu_alias" in
        rtx3090) approx_hourly="0.30" ;;
        rtx4090) approx_hourly="0.40" ;;
        a100)    approx_hourly="1.50" ;;
        h100)    approx_hourly="2.50" ;;
    esac
    if [[ -n "$approx_hourly" ]]; then
        echo "  Approx hourly cost: \$${approx_hourly}/hr (community spot pricing)."
    fi
    if [[ "${RUNPOD_SKIP_CONFIRM:-}" != "1" ]]; then
        if [[ "$gpu_alias" == "a100" || "$gpu_alias" == "h100" ]]; then
            echo "  This GPU exceeds the \$1/hr threshold — please confirm explicitly."
            read -r -p "  Type the GPU alias ('$gpu_alias') to proceed: " conf
            if [[ "$conf" != "$gpu_alias" ]]; then
                echo "  Aborted — confirmation did not match."
                exit 1
            fi
        else
            read -r -p "  Proceed? [y/N] " conf
            if [[ ! "$conf" =~ ^(y|Y|yes|YES)$ ]]; then
                echo "  Aborted."
                exit 1
            fi
        fi
    fi

    # ── Container registry auth (private images, e.g. ghcr.io/elfege/anamnesis-d2) ──
    # If RUNPOD_REGISTRY_AUTH_ID is set, RunPod uses it to authenticate when pulling
    # the image. Created once via the saveRegistryAuth GraphQL mutation; stored in
    # AWS Secrets Manager (ANAMNESIS-Secrets/RUNPOD_REGISTRY_AUTH_ID) and surfaced
    # into .env by start.sh's vault step. If absent, the pod will only be able to
    # pull public images.
    local auth_id="${RUNPOD_REGISTRY_AUTH_ID:-}"
    local auth_clause=""
    if [[ -n "$auth_id" ]]; then
        echo "Using container registry auth: $auth_id (private image pull enabled)"
        auth_clause=", containerRegistryAuthId: \"$auth_id\""
    else
        echo "Warning: RUNPOD_REGISTRY_AUTH_ID not set — pod can only pull PUBLIC images."
        echo "         If $DOCKER_IMAGE is private, the pod will fail to start."
    fi

    # Build the GraphQL mutation to create a pod.
    # Fields:
    #   - cloudType: COMMUNITY (cheaper, can be preempted) vs SECURE (stable, more expensive)
    #   - gpuTypeId: from the table above
    #   - dockerArgs: command to run on container start
    #   - ports: which port to expose (PROFILE-specific: 3011 for trainer, 3015 for d²)
    #   - volumeInGb: persistent storage (model checkpoints survive across restarts)
    #   - containerRegistryAuthId: opaque RunPod-side ID for private-registry creds
    local pod_name="anamnesis-$PROFILE"
    local ports_spec="$SERVICE_PORT/http"
    # Profile-specific env: trainer/d2 expect AUTO_LOAD_MODEL=true (their
    # server bootstraps a default checkpoint). The avatar worker has no such
    # flag — its models are pre-baked into the image; passing irrelevant env
    # is harmless but we keep it honest.
    local env_block
    case "$PROFILE" in
        avatar)
            env_block='env: [{ key: "GPU_TYPE", value: "cuda" }, { key: "MACHINE_NAME", value: "runpod" }]'
            ;;
        *)
            env_block='env: [{ key: "AUTO_LOAD_MODEL", value: "true" }]'
            ;;
    esac
    # Avatar pod needs more container disk (SadTalker checkpoints + XTTS
    # weights baked in ~5 GB; leave headroom for HF cache).
    local container_disk="20"
    [[ "$PROFILE" == "avatar" ]] && container_disk="40"
    local mutation
    mutation=$(jq -n \
        --arg gpu "$gpu_id" \
        --arg img "$DOCKER_IMAGE" \
        --arg name "$pod_name" \
        --arg ports "$ports_spec" \
        --arg auth "$auth_clause" \
        --arg env "$env_block" \
        --arg disk "$container_disk" \
        '{
        query: "mutation { podFindAndDeployOnDemand(input: { cloudType: COMMUNITY, gpuCount: 1, volumeInGb: 50, volumeMountPath: \"/workspace\", containerDiskInGb: \($disk), gpuTypeId: \"\($gpu)\", name: \"\($name)\", imageName: \"\($img)\", ports: \"\($ports)\", \($env)\($auth) }) { id desiredStatus runtime { ports { ip publicPort privatePort isIpPublic } } } }"
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

    # ── Register URL with anamnesis-app ──────────────────────────────────
    # Avatar profile: POST to /api/avatar/runpod/pods (MongoDB), no .env touch.
    # Trainer/d2 profiles: legacy .env write + worker_registry POST.
    register_worker "$public_url" "runpod · $gpu_alias" "$pod_id"
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
    local pod_id="${3:-}"

    # Avatar profile uses the MongoDB-backed pod registry (POST /api/avatar/runpod/pods).
    # No .env writes — the URL is derived in-app from pod_id+port. This removes
    # the 5-step manual chain (AWS edit → pull_env.sh → docker restart → ...)
    # that plagued every prior pod cycle.
    if [[ "$PROFILE" == "avatar" ]]; then
        if [[ -z "$pod_id" ]]; then
            echo "register_worker: avatar profile requires pod_id (arg 3) — skipping" >&2
            return 0
        fi
        # Port 3013 is hard-coded in the avatar worker (Dockerfile ENV WORKER_PORT=3013).
        local port=3013
        local gpu_type="cuda"  # all runpod profiles ship CUDA
        echo "Registering pod with anamnesis-app: pod_id=$pod_id port=$port"
        curl -sf -X POST "http://192.168.10.20:3010/api/avatar/runpod/pods" \
            -H "Content-Type: application/json" \
            -d "$(jq -n \
                    --arg pod_id "$pod_id" \
                    --argjson port "$port" \
                    --arg label "$label" \
                    --arg gpu_type "$gpu_type" \
                    '{pod_id: $pod_id, port: $port, label: $label, gpu_type: $gpu_type}')" \
            >/dev/null 2>&1 \
            && echo "  → registered (next chat turn will see the pod in the worker pool)" \
            || echo "  → registry POST failed (anamnesis-app unreachable?) — add manually in UI"
        return 0
    fi

    # Non-avatar profiles keep the legacy .env behavior (trainer / d2 don't
    # have a Mongo registry yet — they use AWS Secrets via .env).
    case "$PROFILE" in
        d2)
            local env_key_url="D2_ENDPOINT_URL"
            local env_key_label=""
            ;;
        trainer|*)
            local env_key_url="NANOGPT_URLS_RUNPOD"
            local env_key_label=""
            ;;
    esac

    if [[ -f "$ENV_FILE" ]]; then
        local strip_args=( -e "^${env_key_url}=" )
        [[ -n "$env_key_label" ]] && strip_args+=( -e "^${env_key_label}=" )
        strip_args+=( -e '^RUNPOD_ENDPOINT_URL=' )
        grep -v "${strip_args[@]}" "$ENV_FILE" > "$ENV_FILE.tmp" || true
        echo "${env_key_url}=${url}" >> "$ENV_FILE.tmp"
        echo "RUNPOD_ENDPOINT_URL=${url}" >> "$ENV_FILE.tmp"
        mv "$ENV_FILE.tmp" "$ENV_FILE"
    fi

    # POST to the dashboard worker_registry (trainer/d2 visibility).
    curl -s -X POST "http://192.168.10.20:3010/api/workers/register" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg url "$url" --arg label "$label" --arg kind "runpod-${PROFILE}" \
                '{url: $url, label: $label, kind: $kind}')" \
        2>/dev/null || true
}

unregister_worker() {
    local pod_id="$1"

    # Avatar profile: delete from the MongoDB pod registry.
    if [[ "$PROFILE" == "avatar" ]]; then
        echo "Unregistering pod $pod_id from anamnesis-app..."
        curl -sf -X DELETE "http://192.168.10.20:3010/api/avatar/runpod/pods/${pod_id}" \
            >/dev/null 2>&1 \
            && echo "  → removed from worker pool" \
            || echo "  → DELETE returned non-OK (already gone, or anamnesis-app down)"
        return 0
    fi

    case "$PROFILE" in
        d2)
            local env_key_url="D2_ENDPOINT_URL"
            local env_key_label=""
            ;;
        trainer|*)
            local env_key_url="NANOGPT_URLS_RUNPOD"
            local env_key_label=""
            ;;
    esac

    if [[ -f "$ENV_FILE" ]]; then
        local strip_args=( -e "^${env_key_url}=" )
        [[ -n "$env_key_label" ]] && strip_args+=( -e "^${env_key_label}=" )
        strip_args+=( -e '^RUNPOD_ENDPOINT_URL=' )
        grep -v "${strip_args[@]}" "$ENV_FILE" > "$ENV_FILE.tmp" || true
        mv "$ENV_FILE.tmp" "$ENV_FILE"
    fi

    # DELETE from the registry (best effort — endpoint may not exist).
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
        echo ""
        echo "Profile selection (env var):"
        echo "  RUNPOD_PROFILE=trainer  → QLoRA trainer image, port 3011 (default)"
        echo "  RUNPOD_PROFILE=d2       → δ² engine image (ghcr.io/elfege/anamnesis-d2:cuda-runpod), port 3015"
        echo "  RUNPOD_PROFILE=avatar   → avatar worker (XTTS + SadTalker), port 3013 — additive to local workers"
        echo ""
        echo "Examples:"
        echo "  RUNPOD_PROFILE=d2 ./deploy_runpod.sh start --gpu rtx3090"
        echo "  RUNPOD_PROFILE=avatar ./deploy_runpod.sh start --gpu rtx3090"
        echo "  RUNPOD_PROFILE=avatar ./deploy_runpod.sh status"
        exit 1
        ;;
esac
