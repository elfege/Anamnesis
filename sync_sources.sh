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
# Some hosts are always optional — may be unreachable.
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
	local remote_home="${2:-${REMOTE_HOME}}"
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
		"find ${remote_home} -maxdepth 2 \
		 \( -name 'docker-compose.yml' -o -name 'docker-compose.yaml' \) \
		 -exec dirname {} \; 2>/dev/null | sort -u") || true

	local project_count=0
	while IFS= read -r project_dir; do
		[[ -z "$project_dir" ]] && continue
		local rel="${project_dir#${remote_home}/}"
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
		"${host}:${remote_home}/0_SCRIPTS/" \
		"${scripts_dst}/" 2>/dev/null || true

	# ── 0_CLAUDE_IC (intercom + user profile) ───────────────────
	local ic_dst="${dst}/0_CLAUDE_IC"
	mkdir -p "$ic_dst"
	rsync -a \
		"${host}:${remote_home}/0_CLAUDE_IC/intercom.md" \
		"${host}:${remote_home}/0_CLAUDE_IC/user_profile_elfege.md" \
		"${ic_dst}/" 2>/dev/null || true

	# ── Root CLAUDE.md ───────────────────────────────────────────
	rsync -a \
		"${host}:${remote_home}/CLAUDE.md" \
		"${dst}/" 2>/dev/null || true

	# ── .claude/projects (JSONL conversation logs) ──────────────
	local claude_dst="${dst}/.claude/projects"
	mkdir -p "$claude_dst"
	rsync -a \
		--include="*/" \
		--include="*.jsonl" \
		--exclude="*" \
		"${host}:${remote_home}/.claude/projects/" \
		"${claude_dst}/" 2>/dev/null || true
	_log "  ${host}: .claude/projects JSONL synced"

	_log "Done: ${host}"
}

# ── Main ─────────────────────────────────────────────────────────
mkdir -p "${STAGING}"

# Configure hosts via environment or edit this section.
# Hosts not reachable are silently skipped (ConnectTimeout=5).
_sync_host "server-1"
_sync_host "server-2"
_sync_host "server-3"
_sync_host "server-4"

_log "Sync complete."
