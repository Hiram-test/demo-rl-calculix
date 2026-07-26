#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
exec python scripts/mesh_need.py serve --host "${HOST:-127.0.0.1}" --port "${PORT:-8765}"
