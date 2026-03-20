#!/usr/bin/env bash
# ============================================
# ANAMNESIS — Remote Source Sync
# ============================================
# Syncs project files from remote machines into local staging dirs so the
# Anamnesis crawler can ingest them without SSH inside the container.
#
# Staging layout: ~/0_ANAMNESIS_SOURCES/{host}/
#   {host}/{PROJECT_DIR}/CLAUDE.md, README.md, docker-compose.yml, *.sh, *.py
#   {host}/0_SCRIPTS/**/*.sh       (bash style authority)
#   {host}/0_CLAUDE_IC/intercom.md, user_profile_elfege.md
#
# Run hourly via cron. Gracefully skips unreachable hosts (ConnectTimeout=5).
# hvtmc (OHVD_APP_PROD) is always optional — may be unreachable.
# ============================================

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

. ~/.env.colors 2>/dev/null || true

STAGING="${HOME}/0_ANAMNESIS_SOURCES"
REMOTE_HOME="/home/elfege"
LOG_FILE="${HOME}/0_LOGS/anamnesis_sync.log"
mkdir -p "$(dirname "$LOG_FILE")"

_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

_is_reachable() {
	local host="$1"
	ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no \
		"$host" true 2>/dev/null
}

_sync_host() {
	local host="$1"
	local dst="${STAGING}/${host}"

	_log "Checking ${host}..."
	if ! _is_reachable "$host"; then
		_log "${host} unreachable — skipping"
		return 0
	fi

	mkdir -p "$dst"
	_log "Syncing from ${host}..."

	# ── Project files ────────────────────────────────────────────
	# Discover project dirs containing docker-compose.yml/yaml
	local project_dirs
	project_dirs=$(ssh "$host" \
		"find ${REMOTE_HOME} -maxdepth 2 \
		 \( -name 'docker-compose.yml' -o -name 'docker-compose.yaml' \) \
		 -exec dirname {} \; 2>/dev/null | sort -u") || true

	local project_count=0
	while IFS= read -r project_dir; do
		[[ -z "$project_dir" ]] && continue
		local rel="${project_dir#${REMOTE_HOME}/}"
		local local_dir="${dst}/${rel}"
		mkdir -p "$local_dir"

		rsync -a \
			--include="CLAUDE.md" \
			--include="README.md" \
			--include="docker-compose.yml" \
			--include="docker-compose.yaml" \
			--include="*.sh" \
			--include="*.py" \
			--exclude="*" \
			"${host}:${project_dir}/" "${local_dir}/" 2>/dev/null || true

		(( project_count++ )) || true
	done <<< "$project_dirs"
	_log "  ${host}: ${project_count} projects synced"

	# ── 0_SCRIPTS (bash authority — .sh only, skip junk dirs) ───
	local scripts_dst="${dst}/0_SCRIPTS"
	mkdir -p "$scripts_dst"
	rsync -a \
		--filter="- 0_DEPRECATED/" \
		--filter="- 0_TRASH/" \
		--filter="- 0_ARCHIVE/" \
		--filter="- 0_CONFLICTS/" \
		--include="*/" \
		--include="*.sh" \
		--exclude="*" \
		"${host}:${REMOTE_HOME}/0_SCRIPTS/" \
		"${scripts_dst}/" 2>/dev/null || true

	# ── 0_CLAUDE_IC (intercom + user profile) ───────────────────
	local ic_dst="${dst}/0_CLAUDE_IC"
	mkdir -p "$ic_dst"
	rsync -a \
		"${host}:${REMOTE_HOME}/0_CLAUDE_IC/intercom.md" \
		"${host}:${REMOTE_HOME}/0_CLAUDE_IC/user_profile_elfege.md" \
		"${ic_dst}/" 2>/dev/null || true

	# ── .claude/projects (JSONL conversation logs) ──────────────
	local claude_dst="${dst}/.claude/projects"
	mkdir -p "$claude_dst"
	rsync -a \
		--include="*/" \
		--include="*.jsonl" \
		--exclude="*" \
		"${host}:${REMOTE_HOME}/.claude/projects/" \
		"${claude_dst}/" 2>/dev/null || true
	_log "  ${host}: .claude/projects JSONL synced"

	_log "Done: ${host}"
}

# ── Main ─────────────────────────────────────────────────────────
mkdir -p "${STAGING}"

_sync_host "server"
_sync_host "officewsl"
_sync_host "hvtmc"    # OHVD_APP_PROD — not always reachable, always optional

_log "Sync complete."
