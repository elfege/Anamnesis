#!/usr/bin/env bash
# ============================================
# ANAMNESIS - Install Ollama
# Called automatically by start.sh when Ollama is not found.
# ============================================

set -e

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

. ~/.env.colors 2>/dev/null || true

OLLAMA_DEFAULT_MODEL="${OLLAMA_DEFAULT_MODEL:-llama3.2}"

echo -e "${CYAN:-}Installing Ollama...${NC:-}"
curl -fsSL https://ollama.com/install.sh | sh

# Enable and start as a systemd service
echo -e "${CYAN:-}Enabling Ollama systemd service...${NC:-}"
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for Ollama API to become available
echo -n "Waiting for Ollama to be ready"
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/ >/dev/null 2>&1; then
        echo " ready."
        break
    fi
    sleep 1
    printf "."
done

if ! curl -sf http://localhost:11434/ >/dev/null 2>&1; then
    echo ""
    echo -e "${RED:-}Ollama did not start within 30s. Check: sudo journalctl -u ollama -n 50${NC:-}"
    exit 1
fi

echo -e "${CYAN:-}Pulling model: ${OLLAMA_DEFAULT_MODEL}${NC:-}"
ollama pull "$OLLAMA_DEFAULT_MODEL"

echo -e "${GREEN:-}Ollama installed. Model ready: ${OLLAMA_DEFAULT_MODEL}${NC:-}"
echo -e "  API:   http://localhost:11434"
echo -e "  Model: $OLLAMA_DEFAULT_MODEL"
