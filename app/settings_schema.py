"""
settings_schema.py — Canonical declaration of all UI-editable settings.

Each category has a description + a list of keys. Each key declares:
  - key            : env-var-style identifier (UPPER_SNAKE_CASE)
  - type           : "string" | "int" | "float" | "bool" | "path" | "select" | "secret"
  - default        : default value (None means "no default — must be set")
  - secret         : if True, value is redacted in API responses (unless localhost+reveal)
  - restart_required : if True, changing this key requires a container restart
                       to take effect (mount/env-baked-at-boot values)
  - description    : human-readable hint for the UI
  - options        : (only for type="select") allowed string values

Adding new categories: just extend SETTINGS_SCHEMA. The settings UI auto-renders
new cards. Backend resolution still falls back to env vars + module defaults.
"""

from typing import Any

# Sources that sync_sources.sh actually populates → these names map slot → host.
# server-1 (192.168.10.15), server-2 (192.168.10.110 — "office"), server-3 (dellserver).
# SOURCE_0 is reserved for $LOCAL_HOME (this machine, direct mount).
DEFAULT_SOURCE_NAMES = {
    0: "local",
    1: "server",       # SSH_HOST_SERVER  → 192.168.10.15
    2: "office",       # SSH_HOST_OFFICE  → 192.168.10.110
    3: "dellserver",   # SSH_HOST_DELLSERVER (optional)
    4: "host-4",
    5: "host-5",
}


def _source_keys() -> list[dict]:
    keys: list[dict] = []
    for i in range(6):
        keys.append({
            "key": f"SOURCE_{i}",
            "type": "path",
            "default": None,
            "secret": False,
            "restart_required": True,
            "description": (
                f"Host directory mounted at /sources/source-{i}. "
                + ("Slot 0 is typically your $HOME (LOCAL_HOME)." if i == 0
                   else f"Slot {i} is typically a sync_sources.sh staging dir.")
            ),
        })
        keys.append({
            "key": f"SOURCE_{i}_NAME",
            "type": "string",
            "default": DEFAULT_SOURCE_NAMES[i],
            "secret": False,
            "restart_required": True,
            "description": f"Display label / mount alias for slot {i} (e.g. 'office', 'server').",
        })
    keys.append({
        "key": "DOCUMENTS_DIR",
        "type": "path",
        "default": "/sources/documents",
        "secret": False,
        "restart_required": True,
        "description": "Container path for the documents source (mounted from host).",
    })
    keys.append({
        "key": "TEACHINGS_DIR",
        "type": "path",
        "default": "/sources/teachings",
        "secret": False,
        "restart_required": True,
        "description": "Container path for the teachings source (mounted from host).",
    })
    keys.append({
        "key": "CRAWLER_INTERVAL_SECONDS",
        "type": "int",
        "default": 300,
        "secret": False,
        "restart_required": False,
        "description": "How often (seconds) the crawler scans configured sources.",
    })
    return keys


SETTINGS_SCHEMA: dict[str, dict[str, Any]] = {
    "sources": {
        "description": (
            "Crawler source mounts. Each SOURCE_N is a host directory bind-mounted "
            "into /sources/source-N/. Slot 0 is the local machine; slots 1-5 are "
            "staging directories populated by sync_sources.sh from remote hosts."
        ),
        "keys": _source_keys(),
    },

    "ingestion": {
        "description": "How exchanges are summarized and embedded.",
        "keys": [
            {
                "key": "INGEST_BACKEND",
                "type": "select",
                "options": ["ollama", "claude-api", "claude-cli"],
                "default": "ollama",
                "secret": False,
                "restart_required": False,
                "description": "Which LLM backend summarizes incoming exchanges.",
            },
            {
                "key": "EMBED_MODEL",
                "type": "string",
                "default": "BAAI/bge-large-en-v1.5",
                "secret": False,
                "restart_required": True,
                "description": "Sentence-transformer model used to embed summaries (env: EMBEDDING_MODEL).",
            },
            {
                "key": "EMBEDDING_DIMENSIONS",
                "type": "int",
                "default": 1024,
                "secret": False,
                "restart_required": True,
                "description": "Dimensionality of the embedding vector (must match the model).",
            },
            {
                "key": "EMBEDDING_CPU_PCT",
                "type": "int",
                "default": 50,
                "secret": False,
                "restart_required": False,
                "description": "Percentage of logical cores assigned to embedding workers (1-100).",
            },
        ],
    },

    "training": {
        "description": "Defaults for training launches (LoRA fine-tunes, δ² runs, etc.).",
        "keys": [
            {"key": "DEFAULT_BASE_MODEL", "type": "string", "default": "gpt2-medium",
             "secret": False, "restart_required": False,
             "description": "Default HF model id when launching a new fine-tune."},
            {"key": "DEFAULT_BLOCK_SIZE", "type": "int", "default": 256,
             "secret": False, "restart_required": False,
             "description": "Default sequence length for training."},
            {"key": "DEFAULT_BATCH_SIZE", "type": "int", "default": 1,
             "secret": False, "restart_required": False,
             "description": "Default per-device batch size."},
            {"key": "DEFAULT_LR", "type": "float", "default": 1e-4,
             "secret": False, "restart_required": False,
             "description": "Default learning rate."},
        ],
    },

    "runpod": {
        "description": "RunPod (on-demand vLLM pods, OpenAI-compatible).",
        "keys": [
            {"key": "RUNPOD_API_KEY", "type": "secret", "default": "", "secret": True,
             "restart_required": True, "description": "Management API key for podFindAndDeployOnDemand."},
            {"key": "RUNPOD_POD_ID", "type": "string", "default": "", "secret": False,
             "restart_required": True, "description": "Currently-deployed pod identifier (informational)."},
            {"key": "RUNPOD_ENDPOINT_URL", "type": "string", "default": "", "secret": False,
             "restart_required": True, "description": "Full vLLM endpoint URL of running pod."},
            {"key": "RUNPOD_DEFAULT_MODEL", "type": "string",
             "default": "meta-llama/Meta-Llama-3.1-70B-Instruct",
             "secret": False, "restart_required": True,
             "description": "Which model to load on the pod by default."},
            {"key": "RUNPOD_REGISTRY_AUTH_ID", "type": "secret", "default": "", "secret": True,
             "restart_required": True, "description": "Opaque RunPod-side ID for private-registry creds."},
        ],
    },

    "together": {
        "description": "Together.ai hosted inference (pay-per-token).",
        "keys": [
            {"key": "TOGETHER_AI_KEY", "type": "secret", "default": "", "secret": True,
             "restart_required": True, "description": "Together.ai API key (canonical name)."},
            {"key": "TOGETHER_AI_ID", "type": "string", "default": "", "secret": False,
             "restart_required": True, "description": "Account identifier (informational)."},
            {"key": "TOGETHER_BASE_URL", "type": "string",
             "default": "https://api.together.xyz/v1",
             "secret": False, "restart_required": True,
             "description": "OpenAI-compatible API base URL."},
        ],
    },

    "anthropic": {
        "description": "Anthropic API + Claude CLI configuration.",
        "keys": [
            {"key": "ANTHROPIC_API_KEY", "type": "secret", "default": "", "secret": True,
             "restart_required": True, "description": "Anthropic API key (claude-api backend)."},
            {"key": "CLAUDE_MODEL", "type": "string", "default": "claude-sonnet-4-6",
             "secret": False, "restart_required": False,
             "description": "Default Claude model id (e.g. claude-sonnet-4-6)."},
            {"key": "CLAUDE_CLI_HOST", "type": "string", "default": "host.docker.internal",
             "secret": False, "restart_required": True,
             "description": "Host where the claude CLI binary lives (accessed via SSH)."},
            {"key": "CLAUDE_CLI_PATH", "type": "string", "default": "claude",
             "secret": False, "restart_required": True,
             "description": "Path to the claude binary on CLAUDE_CLI_HOST."},
        ],
    },

    "ollama": {
        "description": "Local Ollama endpoints (fallback chain).",
        "keys": [
            {"key": "OLLAMA_DEFAULT_MODEL", "type": "string", "default": "llama3.2",
             "secret": False, "restart_required": False,
             "description": "Default Ollama model name for chat/summarization."},
            {"key": "OLLAMA_URL_1", "type": "string", "default": "", "secret": False,
             "restart_required": True, "description": "First Ollama endpoint URL (highest priority)."},
            {"key": "OLLAMA_LABEL_1", "type": "string", "default": "", "secret": False,
             "restart_required": True, "description": "Display label for endpoint 1."},
            {"key": "OLLAMA_URL_2", "type": "string", "default": "", "secret": False,
             "restart_required": True, "description": "Second Ollama endpoint URL."},
            {"key": "OLLAMA_LABEL_2", "type": "string", "default": "", "secret": False,
             "restart_required": True, "description": "Display label for endpoint 2."},
            {"key": "OLLAMA_URL_3", "type": "string", "default": "", "secret": False,
             "restart_required": True, "description": "Third Ollama endpoint URL."},
            {"key": "OLLAMA_LABEL_3", "type": "string", "default": "", "secret": False,
             "restart_required": True, "description": "Display label for endpoint 3."},
        ],
    },

    "avatar": {
        "description": "Belle avatar pipeline (LLM → TTS → animation).",
        "keys": [
            {"key": "AVATAR_PERSONA_NAME", "type": "string", "default": "Belle",
             "secret": False, "restart_required": False,
             "description": "Persona display name."},
            {"key": "AVATAR_EDGE_VOICE", "type": "string", "default": "en-US-AvaNeural",
             "secret": False, "restart_required": False,
             "description": "Microsoft Edge TTS voice id (fallback voice)."},
            {"key": "AVATAR_ANIMATE_DEFAULT", "type": "select",
             "options": ["auto", "true", "false"],
             "default": "auto", "secret": False, "restart_required": True,
             "description": "Whether to animate by default (auto = on if worker + reference image exist)."},
            {"key": "AVATAR_WORKER_URL_1", "type": "string", "default": "", "secret": False,
             "restart_required": True, "description": "First avatar GPU worker URL."},
            {"key": "AVATAR_WORKER_LABEL_1", "type": "string", "default": "", "secret": False,
             "restart_required": True, "description": "Label for avatar worker 1."},
            {"key": "AVATAR_WORKER_URL_2", "type": "string", "default": "", "secret": False,
             "restart_required": True, "description": "Second avatar GPU worker URL (fallback)."},
            {"key": "AVATAR_WORKER_LABEL_2", "type": "string", "default": "", "secret": False,
             "restart_required": True, "description": "Label for avatar worker 2."},
        ],
    },

    "vault": {
        "description": (
            "Vault / AWS Secrets Manager integration. The pull_env.sh script "
            "regenerates .env from these — changing them here only affects the "
            "running container until the next pull."
        ),
        "keys": [
            {"key": "AUTHORIZED_MACHINE_ID", "type": "secret", "default": "",
             "secret": True, "restart_required": True,
             "description": "Machine-id allowed to mutate this Anamnesis instance."},
            {"key": "VAULT_PASSPHRASE", "type": "secret", "default": "",
             "secret": True, "restart_required": True,
             "description": "Passphrase for any local secrets vault (if used)."},
        ],
    },
}


def get_key_def(category: str, key: str) -> dict | None:
    """Look up the definition for a single key, or None if not declared."""
    cat = SETTINGS_SCHEMA.get(category)
    if not cat:
        return None
    for kd in cat.get("keys", []):
        if kd["key"] == key:
            return kd
    return None


def all_keys() -> list[tuple[str, str, dict]]:
    """Yield (category, key, key_def) for every declared key."""
    out = []
    for cat_name, cat in SETTINGS_SCHEMA.items():
        for kd in cat.get("keys", []):
            out.append((cat_name, kd["key"], kd))
    return out
