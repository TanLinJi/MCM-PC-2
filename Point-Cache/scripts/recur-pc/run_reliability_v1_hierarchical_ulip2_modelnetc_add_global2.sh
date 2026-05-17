#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Stage 2: MCM-PC Reliability Score v1
# Baseline: Point-Cache Hierarchical Cache
# Model: ULIP-2
# Dataset: ModelNet-C
# Corruption: add_global_2
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POINT_CACHE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${POINT_CACHE_ROOT}/.." && pwd)"

cd "${POINT_CACHE_ROOT}"

export PYTHONPATH="${POINT_CACHE_ROOT}:${PYTHONPATH:-}"

GPU_ID="${GPU_ID:-0}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="${POINT_CACHE_ROOT}/logs/recur-pc"
LOG_FILE="${LOG_DIR}/reliability_v1_hierarchical_ulip2_modelnetc_add_global2_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "============================================================" | tee "${LOG_FILE}"
echo "Stage 2: Reliability Score v1 + Hierarchical Cache" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Time: ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_FILE}"
echo "Point-Cache root: ${POINT_CACHE_ROOT}" | tee -a "${LOG_FILE}"
echo "GPU_ID: ${GPU_ID}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

echo "[Git info]" | tee -a "${LOG_FILE}"
cd "${REPO_ROOT}"
git branch --show-current | tee -a "${LOG_FILE}"
git rev-parse --short HEAD | tee -a "${LOG_FILE}"
git status --short | tee -a "${LOG_FILE}"

echo "============================================================" | tee -a "${LOG_FILE}"
echo "[Environment]" | tee -a "${LOG_FILE}"
cd "${POINT_CACHE_ROOT}"
which python | tee -a "${LOG_FILE}"
python --version | tee -a "${LOG_FILE}"
nvidia-smi | tee -a "${LOG_FILE}"

echo "============================================================" | tee -a "${LOG_FILE}"
echo "[Required files]" | tee -a "${LOG_FILE}"
ls weights/ulip2/point-encoder/pointbert_ulip2.pt | tee -a "${LOG_FILE}"
ls weights/ulip2/image-text-encoder/slip_base_100ep.pt | tee -a "${LOG_FILE}"

echo "============================================================" | tee -a "${LOG_FILE}"
echo "[Run] Reliability v1 hierarchical inference starts" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python runners/model_with_hierarchical_caches.py \
  --config configs \
  --lm3d ulip \
  --cache-type hierarchical \
  --ckpt_path weights/ulip2/point-encoder/pointbert_ulip2.pt \
  --slip-ckpt-path weights/ulip2/image-text-encoder/slip_base_100ep.pt \
  --dataset modelnet_c \
  --data-root data \
  --modelnet_c_root data/modelnet_c \
  --sonn_variant obj_only \
  --cor_type add_global_2 \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --ulip-version ulip2 \
  2>&1 | tee -a "${LOG_FILE}"

echo "============================================================" | tee -a "${LOG_FILE}"
echo "[Done] Reliability v1 hierarchical inference finished" | tee -a "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
