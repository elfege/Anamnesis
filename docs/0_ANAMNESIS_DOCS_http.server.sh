#!/usr/bin/env bash
# Serve 0_ANAMNESIS docs locally via Python's built-in HTTP server.
# Port 8093 is registered in the central port registry.
# Usage: ./0_ANAMNESIS_DOCS_http.server.sh

PORT=8093
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Serving docs from: $DIR"
echo "URL: http://$(hostname -I | awk '{print $1}'):${PORT}/"
echo "Press Ctrl+C to stop."

cd "$DIR" && python3 -m http.server "$PORT"
