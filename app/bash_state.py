"""
Shared state for bash execution consent and terminal history.
Imported by both routes/chat.py (producer) and routes/bash.py (consumer).
"""
import asyncio

# consent_id -> {command, reason, host, event, approved, result}
_pending_consents: dict[str, dict] = {}
