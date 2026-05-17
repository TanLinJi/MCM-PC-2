#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Baseline: Original Point-Cache Hierarchical Cache
# GPU: 0
# Dataset: ModelNet-C
# Corruptions: all 7 severity-2 corruptions
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POINT_CACHE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${POINT_CACHE_ROOT}/.." && pwd)"

cd "${POINT_CACHE_ROOT}"
export PYTHONPATH="${POINT_CACHE_ROOT}:${PYTHONPATH:-}"

GPU_ID="${GPU_ID:-0}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="${POINT_CACHE_ROOT}/logs/recur-pc/baseline_hierarchical_modelnetc_all_corruptions_${TIMESTAMP}"
SUMMARY_FILE="${RUN_DIR}/summary_baseline_hierarchical.csv"

RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches.py"

COR_TYPES=(
  add_global_2
  add_local_2
  dropout_global_2
  dropout_local_2
  rotate_2
  scale_2
  jitter_2
)

mkdir -p "${RUN_DIR}"

echo "method,corruption,accuracy,status,log_file" > "${SUMMARY_FILE}"

echo "============================================================"
echo "Baseline Hierarchical Cache: ModelNet-C all corruptions"
echo "============================================================"
echo "Time: ${TIMESTAMP}"
echo "GPU_ID: ${GPU_ID}"
echo "Runner: ${RUNNER}"
echo "Run dir: ${RUN_DIR}"
echo "Summary: ${SUMMARY_FILE}"
echo "============================================================"

cd "${REPO_ROOT}"
echo "[Git info]"
git branch --show-current
git rev-parse --short HEAD
git status --short

cd "${POINT_CACHE_ROOT}"

echo "============================================================"
echo "[Syntax check]"
echo "============================================================"
python -m py_compile "${RUNNER}"

for COR_TYPE in "${COR_TYPES[@]}"; do
  LOG_FILE="${RUN_DIR}/baseline_hierarchical_${COR_TYPE}.log"

  echo "============================================================" | tee "${LOG_FILE}"
  echo "[Baseline Hierarchical] Start: ${COR_TYPE}" | tee -a "${LOG_FILE}"
  echo "GPU_ID: ${GPU_ID}" | tee -a "${LOG_FILE}"
  echo "Runner: ${RUNNER}" | tee -a "${LOG_FILE}"
  echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
  echo "============================================================" | tee -a "${LOG_FILE}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${RUNNER}" \
    --config configs \
    --lm3d ulip \
    --cache-type hierarchical \
    --ckpt_path weights/ulip2/point-encoder/pointbert_ulip2.pt \
    --slip-ckpt-path weights/ulip2/image-text-encoder/slip_base_100ep.pt \
    --dataset modelnet_c \
    --data-root data \
    --modelnet_c_root data/modelnet_c \
    --sonn_variant obj_only \
    --cor_type "${COR_TYPE}" \
    --npoints 1024 \
    --sim2real_type so_obj_only_9 \
    --ulip-version ulip2 \
    2>&1 | tee -a "${LOG_FILE}"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  if [[ "${EXIT_CODE}" -eq 0 ]]; then
    ACC="$(grep -E '\*\*\*Final\*\*\*.*accuracy' "${LOG_FILE}" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1 || true)"
    if [[ -z "${ACC}" ]]; then
      ACC="NA"
      STATUS="NO_FINAL_ACC"
    else
      STATUS="OK"
    fi
  else
    ACC="NA"
    STATUS="FAILED_${EXIT_CODE}"
  fi

  echo "baseline_hierarchical,${COR_TYPE},${ACC},${STATUS},${LOG_FILE}" >> "${SUMMARY_FILE}"

  echo "============================================================" | tee -a "${LOG_FILE}"
  echo "[Baseline Hierarchical] Finished: ${COR_TYPE}" | tee -a "${LOG_FILE}"
  echo "Status: ${STATUS}" | tee -a "${LOG_FILE}"
  echo "Accuracy: ${ACC}" | tee -a "${LOG_FILE}"
  echo "============================================================" | tee -a "${LOG_FILE}"
done

echo "============================================================"
echo "Baseline Hierarchical finished."
echo "Summary:"
cat "${SUMMARY_FILE}"
echo "Logs saved to: ${RUN_DIR}"
echo "============================================================"
