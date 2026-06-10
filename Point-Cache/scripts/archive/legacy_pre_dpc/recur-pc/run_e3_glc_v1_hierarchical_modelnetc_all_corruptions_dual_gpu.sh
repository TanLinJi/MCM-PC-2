#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# E3-GLC-v1: Non-expansive Global-Local Consistency Gate
# ModelNet-C all-corruption evaluation
#
# GPU0:
#   add_global_2, dropout_global_2, rotate_2, jitter_2
#
# GPU1:
#   add_local_2, dropout_local_2, scale_2
#
# Each corruption is an independent online TTA stream.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POINT_CACHE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${POINT_CACHE_ROOT}/.." && pwd)"

cd "${POINT_CACHE_ROOT}"

export PYTHONPATH="${POINT_CACHE_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=disabled

# Server: 24 CPU cores, 2 GPU processes.
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="${POINT_CACHE_ROOT}/logs/recur-pc/e3_glc_v1_dual_gpu_modelnetc_all_corruptions_${TIMESTAMP}"

SUMMARY_GPU0="${RUN_DIR}/summary_gpu0.csv"
SUMMARY_GPU1="${RUN_DIR}/summary_gpu1.csv"
SUMMARY_ALL="${RUN_DIR}/summary_e3_glc_v1_dual_gpu_all.csv"

RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches_e3_glc_v1.py"

if [[ "${QUICK_TEST:-0}" == "1" ]]; then
  GPU0_COR_TYPES=(add_global_2)
  GPU1_COR_TYPES=()
else
  GPU0_COR_TYPES=(
    add_global_2
    dropout_global_2
    rotate_2
    jitter_2
  )

  GPU1_COR_TYPES=(
    add_local_2
    dropout_local_2
    scale_2
  )
fi

mkdir -p "${RUN_DIR}"

echo "method,gpu,corruption,accuracy,status,log_file" > "${SUMMARY_GPU0}"
echo "method,gpu,corruption,accuracy,status,log_file" > "${SUMMARY_GPU1}"

echo "============================================================"
echo "E3-GLC-v1 Dual-GPU ModelNet-C Evaluation"
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
    local LOG_FILE="${RUN_DIR}/e3_glc_v1_gpu${gpu_id}_${COR_TYPE}.log"

    echo "============================================================" | tee "${LOG_FILE}"
    echo "[E3-GLC-v1] Start: ${COR_TYPE}" | tee -a "${LOG_FILE}"
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

    echo "e3_glc_v1_hierarchical,${gpu_id},${COR_TYPE},${ACC},${STATUS},${LOG_FILE}" >> "${summary_file}"

    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "[E3-GLC-v1] Finished: ${COR_TYPE}" | tee -a "${LOG_FILE}"
    echo "GPU_ID: ${gpu_id}" | tee -a "${LOG_FILE}"
    echo "Status: ${STATUS}" | tee -a "${LOG_FILE}"
    echo "Accuracy: ${ACC}" | tee -a "${LOG_FILE}"
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

wait "${PID_GPU0}"
STATUS_GPU0=$?

wait "${PID_GPU1}"
STATUS_GPU1=$?

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
