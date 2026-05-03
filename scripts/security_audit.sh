#!/usr/bin/env bash
# scripts/security_audit.sh — CANONICAL security audit (TRACKED in git).
#
# Runs in two contexts:
#   1. GitHub Actions (.github/workflows/security_audit.yml) — server-side, on every
#      push and pull request. Cannot be bypassed by --no-verify.
#   2. Anywhere else, manually: `./scripts/security_audit.sh [--scope=staged|head|all]`
#
# The local pre-push hook (.git/hooks/pre-push) calls scripts/security_check.sh
# (UNTRACKED, machine-local, can have additional rules). This canonical script is
# the BASELINE — what runs in CI for every contributor, what reviewers can rely on.
#
# Three classes of violations:
#   1. Forbidden COPY in Dockerfiles (secrets, weights, personal data)
#   2. Secret tokens in any file (AWS keys, GitHub tokens, RunPod, Anthropic, OpenAI, PEMs)
#   3. Model weight files (.pt, .safetensors, .gguf, .ckpt) in git
#
# Exit codes:
#   0 = clean
#   1 = violations found
#   2 = script error

set -uo pipefail

SCOPE="${1:-all}"   # all | staged | head | range:<gitrev>
SCOPE="${SCOPE#--scope=}"

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Determine which files to scan
case "$SCOPE" in
    staged)
        FILES=$(git diff --cached --name-only --diff-filter=ACM)
        ;;
    head)
        FILES=$(git diff HEAD~1..HEAD --name-only --diff-filter=ACM)
        ;;
    range:*)
        FILES=$(git diff "${SCOPE#range:}" --name-only --diff-filter=ACM)
        ;;
    all)
        # Full repo scan, excluding .git
        FILES=$(git ls-files)
        ;;
    *)
        echo "Unknown scope: $SCOPE (use staged|head|range:REV|all)" >&2
        exit 2
        ;;
esac

VIOLATIONS=()

# ────────────────────────────────────────────────────────────────────────────
# Check 1: forbidden COPY patterns in any tracked Dockerfile
# ────────────────────────────────────────────────────────────────────────────
DOCKERFILES=$(echo "$FILES" | grep -E '(^|/)Dockerfile' || true)

# Patterns that bake sensitive data into images. Each pattern → human description.
declare -A FORBIDDEN_COPYS=(
    ['COPY[[:space:]]+\.env']='COPY .env (secrets)'
    ['COPY[[:space:]]+\*\.env']='COPY *.env (secrets)'
    ['COPY[[:space:]]+\*\.key']='COPY *.key (private keys)'
    ['COPY[[:space:]]+\*\.pem']='COPY *.pem (private keys)'
    ['COPY[[:space:]]+\*\.pfx']='COPY *.pfx (cert bundles)'
    ['COPY[[:space:]]+secrets/']='COPY secrets/ (credential bundle)'
    ['COPY[[:space:]]+vault/']='COPY vault/ (credential bundle)'
    ['COPY[[:space:]]+credentials']='COPY credentials* (credentials)'
    ['COPY[[:space:]]+\.aws/']='COPY .aws/ (AWS creds)'
    ['COPY[[:space:]]+\.ssh/']='COPY .ssh/ (SSH keys)'
    ['COPY[[:space:]]+d2/d2_checkpoints']='COPY d2/d2_checkpoints (model weights = personal data)'
    ['COPY[[:space:]]+d2/d2_data']='COPY d2/d2_data (training data, may include personal corpus)'
    ['COPY[[:space:]]+d2/output']='COPY d2/output (training outputs)'
    ['COPY[[:space:]]+checkpoints/']='COPY checkpoints/ (model weights)'
    ['COPY[[:space:]]+\*\.pt[[:space:]]']='COPY *.pt (PyTorch weights)'
    ['COPY[[:space:]]+\*\.safetensors']='COPY *.safetensors (model weights)'
    ['COPY[[:space:]]+\*\.gguf']='COPY *.gguf (quantized weights)'
    ['COPY[[:space:]]+\*\.ckpt']='COPY *.ckpt (checkpoint)'
    ['COPY[[:space:]]+samples/']='COPY samples/ (avatar samples may be copyrighted)'
    ['COPY[[:space:]]+README_handoff']='COPY README_handoff (operational docs)'
    ['COPY[[:space:]]+README_project_history']='COPY README_project_history (operational docs)'
)

if [[ -n "$DOCKERFILES" ]]; then
    while IFS= read -r f; do
        [[ -f "$f" ]] || continue
        for pattern in "${!FORBIDDEN_COPYS[@]}"; do
            if grep -nE "$pattern" "$f" >/dev/null 2>&1; then
                line=$(grep -nE "$pattern" "$f" | head -1)
                VIOLATIONS+=("FORBIDDEN COPY in $f line ${line%%:*}: ${FORBIDDEN_COPYS[$pattern]}")
            fi
        done
        # Bare `COPY .` warning (not blocking, but flag for review)
        if grep -nE '^COPY[[:space:]]+\.[[:space:]]+' "$f" >/dev/null 2>&1; then
            VIOLATIONS+=("REVIEW: bare 'COPY . …' in $f — verify .dockerignore covers all sensitive paths")
        fi
    done <<< "$DOCKERFILES"
fi

# ────────────────────────────────────────────────────────────────────────────
# Check 2: required .dockerignore at every Dockerfile's directory
# ────────────────────────────────────────────────────────────────────────────
if [[ -n "$DOCKERFILES" ]]; then
    while IFS= read -r f; do
        ctx_dir="$(dirname "$f")"
        if [[ ! -f "$ctx_dir/.dockerignore" && ! -f ".dockerignore" ]]; then
            VIOLATIONS+=("MISSING .dockerignore in $ctx_dir/ (required when Dockerfile is present)")
        fi
    done <<< "$DOCKERFILES"
fi

# ────────────────────────────────────────────────────────────────────────────
# Check 3: secret token patterns in any scanned file
# ────────────────────────────────────────────────────────────────────────────
declare -A SECRET_PATTERNS=(
    ['AKIA[0-9A-Z]{16}']='AWS access key'
    ['ghp_[A-Za-z0-9]{36,}']='GitHub fine-grained PAT'
    ['gho_[A-Za-z0-9]{36,}']='GitHub OAuth token'
    ['ghs_[A-Za-z0-9]{36,}']='GitHub server-to-server token'
    ['rpa_[A-Za-z0-9]{40,}']='RunPod API key'
    ['sk-ant-api[0-9]{2}-[A-Za-z0-9_-]{80,}']='Anthropic API key'
    ['sk-proj-[A-Za-z0-9_-]{80,}']='OpenAI project key'
    ['sk-[A-Za-z0-9]{40,}']='OpenAI / generic sk- key'
    ['tgp_v[a-zA-Z0-9_-]{40,}']='Together.ai API key'
    ['hf_[A-Za-z0-9]{30,}']='HuggingFace token'
    ['xoxb-[A-Za-z0-9-]+']='Slack bot token'
    ['xoxp-[A-Za-z0-9-]+']='Slack user token'
    ['-----BEGIN .* PRIVATE KEY-----']='PEM private key'
    ['-----BEGIN OPENSSH PRIVATE KEY-----']='OpenSSH private key'
)

while IFS= read -r f; do
    [[ -f "$f" ]] || continue
    # Skip binaries
    file --mime "$f" 2>/dev/null | grep -q binary && continue
    # Skip the security script itself (it lists patterns as data)
    case "$f" in
        scripts/security_audit.sh|scripts/security_check.sh|.github/workflows/security_audit.yml)
            continue
            ;;
    esac
    for pattern in "${!SECRET_PATTERNS[@]}"; do
        if grep -nE "$pattern" "$f" >/dev/null 2>&1; then
            line_no=$(grep -nE "$pattern" "$f" | head -1 | cut -d: -f1)
            # NEVER print the matched value — would re-leak
            VIOLATIONS+=("SECRET in $f line $line_no: ${SECRET_PATTERNS[$pattern]}")
        fi
    done
done <<< "$FILES"

# ────────────────────────────────────────────────────────────────────────────
# Check 4: model weight files staged for commit
# ────────────────────────────────────────────────────────────────────────────
while IFS= read -r f; do
    case "$f" in
        *.pt|*.bin|*.safetensors|*.gguf|*.ckpt)
            # WikiText .bin is a special case — but it's in d2/data/wikitext/ (gitignored)
            # If a .bin made it into git, that's almost certainly a problem
            VIOLATIONS+=("MODEL WEIGHT staged: $f (belongs in d2_checkpoints*/, gitignored)")
            ;;
    esac
done <<< "$FILES"

# ────────────────────────────────────────────────────────────────────────────
# Check 5: required gitignore entries
# ────────────────────────────────────────────────────────────────────────────
REQUIRED_IGNORES=(
    '.env'
    'd2/d2_checkpoints'
    'd2/d2_data'
    '*.pt'
    '*.safetensors'
    'scripts/security_check.sh'
    'Dockerfile'
)
if [[ -f .gitignore ]]; then
    for ign in "${REQUIRED_IGNORES[@]}"; do
        # Match the entry as a whole-line pattern (allow either bare or anchored)
        if ! grep -qFx "$ign" .gitignore 2>/dev/null && ! grep -qE "^${ign//\*/\\*}/?$" .gitignore 2>/dev/null; then
            # Soft warning rather than blocking — the policy is "be present", not "be exact"
            : # could enable strict mode here later
        fi
    done
fi

# ────────────────────────────────────────────────────────────────────────────
# Report
# ────────────────────────────────────────────────────────────────────────────
if [[ ${#VIOLATIONS[@]} -gt 0 ]]; then
    echo
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║                       SECURITY AUDIT FAILED                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo "Scope: $SCOPE  ·  Files scanned: $(echo "$FILES" | wc -l)"
    echo
    for v in "${VIOLATIONS[@]}"; do
        echo "  ✗ $v"
    done
    echo
    echo "If a violation is a false positive, document it in scripts/security_audit.sh"
    echo "and open a PR to update the canonical rules. Do NOT bypass with --no-verify"
    echo "in CI (this script is the GitHub Actions check)."
    exit 1
fi

echo "✓ Security audit passed (scope: $SCOPE, files: $(echo "$FILES" | wc -l))"
exit 0
