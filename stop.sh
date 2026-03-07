#!/usr/bin/env bash
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

docker compose down
echo "Anamnesis stopped."
