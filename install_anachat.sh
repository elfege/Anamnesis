#!/usr/bin/env bash
# install_anachat.sh — Install the anachat / chatana CLI tool
# Run as root or with sudo, or let the script escalate.
#
# Usage:
#   bash install_anachat.sh
#   bash install_anachat.sh --prefix /usr/local   # default
#   bash install_anachat.sh --prefix ~/bin         # user-local

set -euo pipefail

PREFIX="/usr/local"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        *) printf "Unknown option: %s\n" "$1" >&2; exit 1 ;;
    esac
done

DEST="$PREFIX/bin/anachat"
LINK="$PREFIX/bin/chatana"

# Use sudo only if needed
_install() {
    if [[ -w "$(dirname "$1")" ]]; then
        cp "$2" "$1"
        chmod +x "$1"
    else
        sudo cp "$2" "$1"
        sudo chmod +x "$1"
    fi
}

_symlink() {
    if [[ -w "$(dirname "$1")" ]]; then
        ln -sf "$2" "$1"
    else
        sudo ln -sf "$2" "$1"
    fi
}

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tools"
SRC="$TOOLS/anachat"
SRC_RENDER="$TOOLS/anachat_render.py"

for f in "$SRC" "$SRC_RENDER"; do
    if [[ ! -f "$f" ]]; then
        printf "Source not found: %s\n" "$f" >&2
        printf "Run this script from the 0_ANAMNESIS project root.\n" >&2
        exit 1
    fi
done

printf "Installing anachat        → %s\n" "$DEST"
_install "$DEST" "$SRC"

printf "Installing anachat_render → %s/bin/anachat_render.py\n" "$PREFIX"
_install "$PREFIX/bin/anachat_render.py" "$SRC_RENDER"

printf "Creating alias  chatana   → %s\n" "$DEST"
_symlink "$LINK" "$DEST"

printf "Done. Test with: anachat \"your query\"\n"
