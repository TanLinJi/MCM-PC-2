#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="e0_tpe_v2_trimmean10_zs_modelnetc_severity2_corruptions"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="logs/recur-pc/${EXP_NAME}_${TIMESTAMP}"
mkdir -p "${LOG_ROOT}"

SUMMARY_GPU0="${LOG_ROOT}/summary_gpu0.csv"
SUMMARY_GPU1="${LOG_ROOT}/summary_gpu1.csv"
SUMMARY_ALL="${LOG_ROOT}/summary_${EXP_NAME}.csv"

echo "gpu_id,status,corruption,accuracy,log_file" > "${SUMMARY_GPU0}"
echo "gpu_id,status,corruption,accuracy,log_file" > "${SUMMARY_GPU1}"

# severity 2 对应本地后缀 _1
GPU0_CORRUPTIONS=(
  add_global_1
  add_local_1
  dropout_global_1
  dropout_local_1
)

GPU1_CORRUPTIONS=(
  jitter_1
  rotate_1
  scale_1
)

run_one() {
  local gpu_id="$1"
  local corruption="$2"
  local summary_file="$3"
  local log_file="${LOG_ROOT}/${corruption}_gpu${gpu_id}.log"

  echo "[E0-TPE-v2-trimmean10-ZS] Start: ${corruption} on GPU ${gpu_id}"

  set +e
  TPE_AGG_MODE="trimmean10" \
  TPE_TRIM_RATIO="0.10" \
  CUDA_VISIBLE_DEVICES="${gpu_id}" python runners/zs_infer_e0_tpe_v2_template_score_agg.py \
    --config configs/modelnet_c.yaml \
    --dataset modelnet_c \
    --cor_type "${corruption}" \
    --cache-type global \
    --lm3d ulip \
    > "${log_file}" 2>&1
  exit_code=$?
  set -e

  if [[ "${exit_code}" -ne 0 ]]; then
    echo "${gpu_id},FAIL,${corruption},NA,${log_file}" >> "${summary_file}"
    echo "[E0-TPE-v2-trimmean10-ZS] Failed: ${corruption}. See ${log_file}"
    return 0
  fi

  acc="$(grep -E "Final|zero-shot test accuracy|accuracy" "${log_file}" \
    | tail -n 5 \
    | grep -oE '[0-9]+(\.[0-9]+)?' \
    | tail -n 1 || true)"

  if [[ -z "${acc}" ]]; then
    acc="NA"
    status="NO_ACC"
  else
    status="OK"
  fi

  echo "${gpu_id},${status},${corruption},${acc},${log_file}" >> "${summary_file}"
  echo "[E0-TPE-v2-trimmean10-ZS] Finished: ${corruption} | GPU ${gpu_id} | Status ${status} | Accuracy ${acc}"
}

run_worker() {
  local gpu_id="$1"
  local summary_file="$2"
  shift 2

  for corruption in "$@"; do
    run_one "${gpu_id}" "${corruption}" "${summary_file}"
  done
}

run_worker 0 "${SUMMARY_GPU0}" "${GPU0_CORRUPTIONS[@]}" &
PID0=$!

run_worker 1 "${SUMMARY_GPU1}" "${GPU1_CORRUPTIONS[@]}" &
PID1=$!

wait "${PID0}"
wait "${PID1}"

head -n 1 "${SUMMARY_GPU0}" > "${SUMMARY_ALL}"
tail -n +2 "${SUMMARY_GPU0}" >> "${SUMMARY_ALL}"
tail -n +2 "${SUMMARY_GPU1}" >> "${SUMMARY_ALL}"

echo
echo "[E0-TPE-v2-trimmean10-ZS] All done."
echo "Log root: ${LOG_ROOT}"
echo "Summary: ${SUMMARY_ALL}"
cat "${SUMMARY_ALL}"
