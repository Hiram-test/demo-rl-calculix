#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_CMD="${PYTHON_CMD:-python3}"
ABAQUS_CMD="${ABAQUS_CMD:-abaqus}"
MODE="${1:-solve}"
shift || true

exec "${PYTHON_CMD}" rl_main_local.py \
  --backend abaqus \
  --mode "${MODE}" \
  --abaqus-cmd "${ABAQUS_CMD}" \
  --template-cae-file DEMO.cae \
  --goal-file examples/goal_local.json \
  "$@"
