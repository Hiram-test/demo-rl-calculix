#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_CMD="${PYTHON_CMD:-python3}"
GMSH_CMD="${GMSH_CMD:-gmsh}"
CCX_CMD="${CCX_CMD:-ccx}"
MODE="${1:-solve}"
shift || true

exec "${PYTHON_CMD}" rl_main_local.py \
  --backend calculix \
  --mode "${MODE}" \
  --gmsh-cmd "${GMSH_CMD}" \
  --ccx-cmd "${CCX_CMD}" \
  --plate-config examples/calculix_plate.json \
  --goal-file examples/goal_local.json \
  "$@"
