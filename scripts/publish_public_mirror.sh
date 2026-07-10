#!/usr/bin/env bash
# publish_public_mirror.sh — push private-dev main to public mirror verbatim.
#
# CRITICAL DESIGN CONSTRAINTS (operator directive 2026-07-09):
#   1. Public repo MUST preserve full commit history + hashes for IP provenance.
#      This script does NOT rewrite history. It performs a fast-forward push
#      of main → public/main. Historical commits (including any that predate
#      the scrub adoption) go through unchanged.
#   2. Public repo MUST show AGE + ACTIVITY (portfolio requirement).
#      - AGE: git preserves original author-dates on push; the mirror will
#        show commits going back to the project's earliest date on day one.
#      - ACTIVITY: every subsequent push carries real author-dates. Run this
#        script on a rhythm (post-wrap-up, weekly, or on a CI trigger) so the
#        contribution graph reflects ongoing development.
#
# Leak defense is FORWARD-ONLY:
#   - This script refuses to push if HEAD tree currently tracks any file that
#     matches a STRIP rule in scripts/scrub_rules.txt.
#   - .githooks/pre-push (installed via core.hooksPath) enforces the same
#     check on every push to `public/main`.
#   - The repo was public until 2026-07-09; historical commits already exposed
#     what was there. Preserving them costs no additional exposure.
#
# Usage:
#   scripts/publish_public_mirror.sh                # dry-run: reports what would push
#   scripts/publish_public_mirror.sh --push         # actually pushes
#
# Prereqs:
#   git remote add public git@github.com:elfege/Anamnesis-public.git
#   (script auto-adds if missing)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RULES="$SCRIPT_DIR/scrub_rules.txt"
PUBLIC_REMOTE="public"
PUBLIC_URL="https://github.com/elfege/Anamnesis.git"
PUBLIC_BRANCH="main"
LOCAL_BRANCH="main"

cd "$REPO_DIR"

# ─── Ensure the public remote exists ──────────────────────────
if ! git remote get-url "$PUBLIC_REMOTE" >/dev/null 2>&1; then
    echo "Adding remote '$PUBLIC_REMOTE' → $PUBLIC_URL"
    git remote add "$PUBLIC_REMOTE" "$PUBLIC_URL"
fi

# ─── Load scrub rules ─────────────────────────────────────────
if [[ ! -f "$RULES" ]]; then
    echo "ERROR: scrub rules missing at $RULES" >&2
    exit 1
fi

# Parse rules into two arrays; keep original order (first-match wins).
declare -a RULE_ACTIONS RULE_GLOBS
while IFS= read -r line; do
    # Trim + skip comments/blanks
    line="${line%%#*}"
    line="$(echo "$line" | xargs || true)"
    [[ -z "$line" ]] && continue
    action="$(echo "$line" | awk '{print $1}')"
    glob="$(echo "$line" | awk '{$1=""; print $0}' | xargs)"
    if [[ "$action" != "ALLOW" && "$action" != "STRIP" ]]; then
        echo "WARN: skipping invalid rule: $line" >&2
        continue
    fi
    RULE_ACTIONS+=("$action")
    RULE_GLOBS+=("$glob")
done < "$RULES"

# ─── Classify a path: returns "ALLOW" or "STRIP" ──────────────
classify() {
    local path="$1"
    local i
    for i in "${!RULE_GLOBS[@]}"; do
        local glob="${RULE_GLOBS[$i]}"
        # Bash extglob matching. '**' behavior: enable globstar.
        shopt -s extglob globstar 2>/dev/null || true
        # shellcheck disable=SC2053
        if [[ "$path" == $glob ]]; then
            echo "${RULE_ACTIONS[$i]}"
            return
        fi
    done
    # No rule matched → default STRIP (safety net)
    echo "STRIP"
}

# ─── Scan HEAD tree for STRIP-classified tracked files ────────
echo "Scanning HEAD tree against $RULES ..."
violations=0
while IFS= read -r path; do
    verdict="$(classify "$path")"
    if [[ "$verdict" == "STRIP" ]]; then
        echo "  VIOLATION: $path (STRIP-classified but tracked in HEAD)"
        violations=$((violations + 1))
    fi
done < <(git ls-files)

if (( violations > 0 )); then
    echo ""
    echo "REFUSING to push: $violations tracked file(s) match STRIP rules." >&2
    echo "Either 'git rm --cached <path>' those files, or add an explicit ALLOW rule." >&2
    exit 2
fi
echo "  scan clean — no violations"

# ─── Compare local vs public ──────────────────────────────────
echo ""
echo "Fetching $PUBLIC_REMOTE ..."
git fetch "$PUBLIC_REMOTE" 2>&1 | tail -3

# What would we push?
if git rev-parse --verify "$PUBLIC_REMOTE/$PUBLIC_BRANCH" >/dev/null 2>&1; then
    ahead="$(git rev-list --count "$PUBLIC_REMOTE/$PUBLIC_BRANCH..$LOCAL_BRANCH")"
    behind="$(git rev-list --count "$LOCAL_BRANCH..$PUBLIC_REMOTE/$PUBLIC_BRANCH")"
    echo "  local $LOCAL_BRANCH is $ahead commit(s) ahead of $PUBLIC_REMOTE/$PUBLIC_BRANCH"
    echo "  local $LOCAL_BRANCH is $behind commit(s) behind $PUBLIC_REMOTE/$PUBLIC_BRANCH"
    if (( behind > 0 )); then
        echo "REFUSING to push: public mirror has commits not in local main." >&2
        echo "This should never happen for a mirror — investigate before continuing." >&2
        exit 3
    fi
else
    total="$(git rev-list --count "$LOCAL_BRANCH")"
    echo "  $PUBLIC_REMOTE/$PUBLIC_BRANCH does not exist — first push will populate with $total commits (full history)"
fi

# ─── Push or dry-run ──────────────────────────────────────────
if [[ "${1:-}" == "--push" ]]; then
    echo ""
    echo "Pushing $LOCAL_BRANCH → $PUBLIC_REMOTE/$PUBLIC_BRANCH (full history, no rewrite) ..."
    git push "$PUBLIC_REMOTE" "$LOCAL_BRANCH:$PUBLIC_BRANCH"
    echo "Done. Public mirror updated."
else
    echo ""
    echo "DRY RUN — pass --push to actually publish."
fi
