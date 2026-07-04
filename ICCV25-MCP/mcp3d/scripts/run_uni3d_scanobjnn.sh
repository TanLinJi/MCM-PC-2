#!/bin/bash
# =============================================================================
# MCP3D: Run MCP on ScanObjNN with Uni3D backbone
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

# --- Configuration ---
BACKBONE="uni3d"
DATASET="scanobjnn"
SONN_VARIANT="${1:-hardest}"
USE_RES="${2:-False}"

# --- Paths ---
MCP3D_CONFIG="mcp3d/configs"
DATA_ROOT="./Point-Cache/data"

echo "========================================"
echo "MCP3D: ${BACKBONE} × ${DATASET} (${SONN_VARIANT})"
echo "MCP++: ${USE_RES}"
echo "========================================"

python -u mcp3d/mcp3d_runner.py \
  --config "${MCP3D_CONFIG}" \
  --datasets "${DATASET}" \
  --data-root "${DATA_ROOT}" \
  --lm3d "${BACKBONE}" \
  --cache-type hierarchical \
  --sonn-variant "${SONN_VARIANT}" \
  --npoints 1024 \
  --res "${USE_RES}" \
  --tta-steps 1 \
  --seed 1
