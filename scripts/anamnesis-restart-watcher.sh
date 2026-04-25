#!/usr/bin/env bash
#
# anamnesis-restart-watcher.sh
#
# WHAT THIS DOES, FOR THE DUMMIES:
# =================================
#
# A small daemon that watches a single file. When the Anamnesis app writes
# the word "reboot" to that file, this watcher notices, resets the file, and
# restarts the app's docker stack via ./start.sh.
#
# WHY?
# =====
#
# The Anamnesis app sometimes needs a full restart to pick up changes:
#   - new env vars (e.g. NANOGPT_URLS_RUNPOD just got added)
#   - new secrets that weren't loaded at startup
#   - δ² base-model swap (different model loaded at container start)
#   - authorized_machine_id change
#
# The container itself can't restart its own stack (it would kill itself
# mid-restart). It also shouldn't have access to the Docker socket
# (security: container that can talk to docker can escape). So instead:
#
#   1. The container writes "reboot" to a file on a shared tmpfs.
#   2. THIS WATCHER (running on the host) sees the write, runs ./start.sh.
#
# Container ↔ host interface = one file on /dev/shm. No SSH, no sudo,
# no Docker socket. That's the entire trust surface.
#
# This is the same pattern used by the NVR project:
#   ~/0_MOBIUS.NVR/scripts/nvr-restart-watcher.sh
#
# Forked from there with NVR → ANAMNESIS naming changes only.
#
# INSTALL:
#   sudo cp deployment/anamnesis-restart-watcher.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now anamnesis-restart-watcher
#
# DOCKER COMPOSE BIND MOUNT (must be added to anamnesis-app service):
#   volumes:
#     - /dev/shm/anamnesis-restart:/dev/shm/anamnesis-restart
#

TRIGGER_FILE="/dev/shm/anamnesis-restart/trigger"
PROJECT_DIR="${ANAMNESIS_PROJECT_DIR:-$HOME/0_GENESIS_PROJECT/0_ANAMNESIS}"
RESTART_LOG="$PROJECT_DIR/restart_from_app.log"
LAST_TRIGGER_TIME="never"

# Make sure the trigger directory exists. /dev/shm is tmpfs (RAM-backed)
# so it's wiped on reboot — we recreate the directory each time.
mkdir -p "$(dirname "$TRIGGER_FILE")"
chmod 777 "$(dirname "$TRIGGER_FILE")"  # container needs write access

# Initialize trigger file with the watcher's start time
SERVICE_START_TIME=$(date)
echo "Watcher started: $SERVICE_START_TIME" > "$TRIGGER_FILE"
chmod 666 "$TRIGGER_FILE"

echo "[anamnesis-restart-watcher] watching $TRIGGER_FILE for restart requests..."

start=$(date +%s)
while true; do
    # ── Read the trigger file ─────────────────────────────────────────────
    content="$(cat "$TRIGGER_FILE" 2>/dev/null || echo '')"

    if [[ "$content" == "reboot" ]]; then
        # ── Restart triggered: reset the file, run start.sh ──────────────
        LAST_TRIGGER_TIME=$(date)
        echo "[anamnesis-restart-watcher] Restart triggered at $LAST_TRIGGER_TIME"
        echo "[anamnesis-restart-watcher] Started: $SERVICE_START_TIME | triggered: $LAST_TRIGGER_TIME" > "$TRIGGER_FILE"

        # Background the restart so we don't block the watch loop on it.
        # The restart itself runs in a subshell with logging redirected.
        (
            cd "$PROJECT_DIR" || {
                echo "[anamnesis-restart-watcher] cannot cd to $PROJECT_DIR" >&2
                exit 1
            }
            echo "[$(date)] Restart triggered — running start.sh" >> "$RESTART_LOG"
            ./start.sh >> "$RESTART_LOG" 2>&1 || {
                echo "[$(date)] start.sh FAILED" >> "$RESTART_LOG"
            }
            echo "[$(date)] start.sh completed" >> "$RESTART_LOG"
        ) &
    fi

    # ── Periodic heartbeat write ─────────────────────────────────────────
    # Every ~3 seconds, refresh the trigger file with a status line so the
    # container can see the watcher is alive. This is also what reads
    # like "triggered" the first time, so we check before overwriting.
    if (( $(date +%s) - start > 3 )); then
        if [[ "$content" != *triggered* && "$content" != "reboot" ]]; then
            echo "[anamnesis-restart-watcher] Started: $SERVICE_START_TIME | last triggered: $LAST_TRIGGER_TIME" > "$TRIGGER_FILE"
        fi
        start=$(date +%s)
    fi

    sleep 2
done
