#!/bin/bash
# =============================================================================
# MCP3D: Run MCP on ModelNet-C with OpenShape backbone
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

# --- Configuration ---
BACKBONE="openshape"
DATASET="modelnet_c"
COR_TYPE="${1:-add_global_2}"
USE_RES="${2:-False}"

# --- Paths ---
MCP3D_CONFIG="mcp3d/configs"
DATA_ROOT="./Point-Cache/data"

echo "========================================"
echo "MCP3D: ${BACKBONE} × ${DATASET} (${COR_TYPE})"
echo "MCP++: ${USE_RES}"
echo "========================================"

python -u mcp3d/mcp3d_runner.py \
  --config "${MCP3D_CONFIG}" \
  --datasets "${DATASET}" \
  --data-root "${DATA_ROOT}" \
  --lm3d "${BACKBONE}" \
  --oshape-version vitg14 \
  --cache-type hierarchical \
  --cor-type "${COR_TYPE}" \
  --npoints 1024 \
  --res "${USE_RES}" \
  --tta-steps 1 \
  --seed 1
