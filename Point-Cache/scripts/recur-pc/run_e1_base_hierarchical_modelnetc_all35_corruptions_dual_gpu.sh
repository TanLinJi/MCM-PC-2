#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# E1-BASE: Original Point-Cache Hierarchical Cache
# ModelNet-C all-35-corruption evaluation
#
# Correct local suffix mapping:
#   paper severity 1 -> local suffix _0
#   paper severity 2 -> local suffix _1
#   paper severity 3 -> local suffix _2
#   paper severity 4 -> local suffix _3
#   paper severity 5 -> local suffix _4
#
# 35 corruptions = 7 corruption types × 5 local suffixes _0.._4.
#
# GPU0:
#   add_global_0-4, dropout_global_0-4, rotate_0-4, jitter_0-2
#
# GPU1:
#   add_local_0-4, dropout_local_0-4, scale_0-4, jitter_3-4
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POINT_CACHE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${POINT_CACHE_ROOT}/.." && pwd)"

cd "${POINT_CACHE_ROOT}"

export PYTHONPATH="${POINT_CACHE_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=disabled

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="${POINT_CACHE_ROOT}/logs/recur-pc/e1_base_dual_gpu_modelnetc_all35_corruptions_${TIMESTAMP}"

SUMMARY_GPU0="${RUN_DIR}/summary_gpu0.csv"
SUMMARY_GPU1="${RUN_DIR}/summary_gpu1.csv"
SUMMARY_ALL="${RUN_DIR}/summary_e1_base_dual_gpu_all35.csv"

RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches.py"

CSV_HEADER="method,gpu,corruption,accuracy,status,log_file"

if [[ "${QUICK_TEST:-0}" == "1" ]]; then
  # Keep add_global_2 for comparability with previous quick tests.
  GPU0_COR_TYPES=(add_global_2)
  GPU1_COR_TYPES=()
else
  # GPU0: 18 corruptions
  GPU0_COR_TYPES=(
    add_global_0
    add_global_1
    add_global_2
    add_global_3
    add_global_4

    dropout_global_0
    dropout_global_1
    dropout_global_2
    dropout_global_3
    dropout_global_4

    rotate_0
    rotate_1
    rotate_2
    rotate_3
    rotate_4

    jitter_0
    jitter_1
    jitter_2
  )

  # GPU1: 17 corruptions
  GPU1_COR_TYPES=(
    add_local_0
    add_local_1
    add_local_2
    add_local_3
    add_local_4

    dropout_local_0
    dropout_local_1
    dropout_local_2
    dropout_local_3
    dropout_local_4

    scale_0
    scale_1
    scale_2
    scale_3
    scale_4

    jitter_3
    jitter_4
  )
fi

mkdir -p "${RUN_DIR}"

echo "${CSV_HEADER}" > "${SUMMARY_GPU0}"
echo "${CSV_HEADER}" > "${SUMMARY_GPU1}"

echo "============================================================"
echo "E1-BASE Original Hierarchical Cache: ModelNet-C all35"
echo "============================================================"
echo "Time: ${TIMESTAMP}"
echo "Runner: ${RUNNER}"
echo "Run dir: ${RUN_DIR}"
echo "QUICK_TEST: ${QUICK_TEST:-0}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "============================================================"

cd "${REPO_ROOT}"
echo "[Git info]"
git branch --show-current
git rev-parse --short HEAD
git status --short

cd "${POINT_CACHE_ROOT}"

echo "============================================================"
echo "[File check]"
echo "============================================================"

if [[ ! -f "${RUNNER}" ]]; then
  echo "[ERROR] Runner not found:"
  echo "${RUNNER}"
  exit 1
fi

echo "============================================================"
echo "[Syntax check]"
echo "============================================================"
python -m py_compile "${RUNNER}"

append_summary_row() {
  local method="$1"
  local gpu_id="$2"
  local cor_type="$3"
  local status="$4"
  local log_file="$5"
  local summary_file="$6"

  local acc="NA"

  if [[ "${status}" == "OK" ]]; then
    acc="$(grep -E '\*\*\*Final\*\*\*.*accuracy' "${log_file}" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1 || true)"
    if [[ -z "${acc}" ]]; then
      acc="NA"
      status="NO_FINAL_ACC"
    fi
  fi

  echo "${method},${gpu_id},${cor_type},${acc},${status},${log_file}" >> "${summary_file}"
}

run_worker() {
  local gpu_id="$1"
  local summary_file="$2"
  shift 2
  local cor_types=("$@")

  if [[ "${#cor_types[@]}" -eq 0 ]]; then
    echo "[GPU${gpu_id}] No corruption assigned. Skip."
    return 0
  fi

  for COR_TYPE in "${cor_types[@]}"; do
    local LOG_FILE="${RUN_DIR}/e1_base_gpu${gpu_id}_${COR_TYPE}.log"

    echo "============================================================" | tee "${LOG_FILE}"
    echo "[E1-BASE] Start: ${COR_TYPE}" | tee -a "${LOG_FILE}"
    echo "GPU_ID: ${gpu_id}" | tee -a "${LOG_FILE}"
    echo "Runner: ${RUNNER}" | tee -a "${LOG_FILE}"
    echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"

    set +e
    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${RUNNER}" \
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
      STATUS="OK"
    else
      STATUS="FAILED_${EXIT_CODE}"
    fi

    append_summary_row \
      "e1_base_hierarchical" \
      "${gpu_id}" \
      "${COR_TYPE}" \
      "${STATUS}" \
      "${LOG_FILE}" \
      "${summary_file}"

    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "[E1-BASE] Finished: ${COR_TYPE}" | tee -a "${LOG_FILE}"
    echo "GPU_ID: ${gpu_id}" | tee -a "${LOG_FILE}"
    echo "Status: ${STATUS}" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
  done
}

echo "============================================================"
echo "[Launch workers]"
echo "============================================================"
echo "GPU0 corruptions: ${GPU0_COR_TYPES[*]:-None}"
echo "GPU1 corruptions: ${GPU1_COR_TYPES[*]:-None}"

run_worker 0 "${SUMMARY_GPU0}" "${GPU0_COR_TYPES[@]}" &
PID_GPU0=$!

run_worker 1 "${SUMMARY_GPU1}" "${GPU1_COR_TYPES[@]}" &
PID_GPU1=$!

echo "GPU0 PID: ${PID_GPU0}"
echo "GPU1 PID: ${PID_GPU1}"

set +e
wait "${PID_GPU0}"
STATUS_GPU0=$?

wait "${PID_GPU1}"
STATUS_GPU1=$?
set -e

echo "============================================================"
echo "[Merge summaries]"
echo "============================================================"

head -n 1 "${SUMMARY_GPU0}" > "${SUMMARY_ALL}"
tail -n +2 "${SUMMARY_GPU0}" >> "${SUMMARY_ALL}"
tail -n +2 "${SUMMARY_GPU1}" >> "${SUMMARY_ALL}"

echo "GPU0 exit code: ${STATUS_GPU0}"
echo "GPU1 exit code: ${STATUS_GPU1}"

echo
echo "Combined summary:"
cat "${SUMMARY_ALL}"

echo
echo "Logs saved to:"
echo "${RUN_DIR}"
echo "============================================================"

exit $((STATUS_GPU0 + STATUS_GPU1))
