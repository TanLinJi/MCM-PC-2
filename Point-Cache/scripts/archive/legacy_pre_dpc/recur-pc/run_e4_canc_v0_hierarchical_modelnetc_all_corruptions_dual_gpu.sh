#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# E4-CANC-v0: Conservative Conflict-Aware Negative Cache
# ModelNet-C all-corruption evaluation
#
# GPU0:
#   add_global_2, dropout_global_2, rotate_2, jitter_2
#
# GPU1:
#   add_local_2, dropout_local_2, scale_2
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
RUN_DIR="${POINT_CACHE_ROOT}/logs/recur-pc/e4_canc_v0_dual_gpu_modelnetc_all_corruptions_${TIMESTAMP}"

SUMMARY_GPU0="${RUN_DIR}/summary_gpu0.csv"
SUMMARY_GPU1="${RUN_DIR}/summary_gpu1.csv"
SUMMARY_ALL="${RUN_DIR}/summary_e4_canc_v0_dual_gpu_all.csv"

RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches_e4_canc_v0.py"

CSV_HEADER="method,gpu,corruption,accuracy,base_negative_rate,e4_total_negative_rate,e4_extra_negative_rate,boundary_rate,conflict_rate,conflict_boundary_rate,status,log_file,json_file"

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

echo "${CSV_HEADER}" > "${SUMMARY_GPU0}"
echo "${CSV_HEADER}" > "${SUMMARY_GPU1}"

echo "============================================================"
echo "E4-CANC-v0 Dual-GPU ModelNet-C Evaluation"
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
  local json_file="$6"
  local summary_file="$7"

  python - "$method" "$gpu_id" "$cor_type" "$status" "$log_file" "$json_file" "$summary_file" <<'PY'
import csv
import json
import sys

method, gpu_id, cor_type, status, log_file, json_file, summary_file = sys.argv[1:]

fields = [
    "method",
    "gpu",
    "corruption",
    "accuracy",
    "base_negative_rate",
    "e4_total_negative_rate",
    "e4_extra_negative_rate",
    "boundary_rate",
    "conflict_rate",
    "conflict_boundary_rate",
    "status",
    "log_file",
    "json_file",
]

def fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)

data = {}
if status == "OK":
    with open(json_file, "r") as f:
        data = json.load(f)

row = {
    "method": method,
    "gpu": gpu_id,
    "corruption": cor_type,
    "status": status,
    "log_file": log_file,
    "json_file": json_file,
}

for key in fields:
    if key not in row:
        row[key] = fmt(data.get(key, "NA"))

with open(summary_file, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writerow(row)
PY
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
    local LOG_FILE="${RUN_DIR}/e4_canc_v0_gpu${gpu_id}_${COR_TYPE}.log"
    local JSON_FILE="${RUN_DIR}/e4_canc_v0_gpu${gpu_id}_${COR_TYPE}.json"

    echo "============================================================" | tee "${LOG_FILE}"
    echo "[E4-CANC-v0] Start: ${COR_TYPE}" | tee -a "${LOG_FILE}"
    echo "GPU_ID: ${gpu_id}" | tee -a "${LOG_FILE}"
    echo "Runner: ${RUNNER}" | tee -a "${LOG_FILE}"
    echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    echo "JSON: ${JSON_FILE}" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"

    set +e
    E4_CANC_OUT="${JSON_FILE}" CUDA_VISIBLE_DEVICES="${gpu_id}" python "${RUNNER}" \
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

    if [[ "${EXIT_CODE}" -eq 0 && -f "${JSON_FILE}" ]]; then
      STATUS="OK"
    else
      STATUS="FAILED_${EXIT_CODE}"
    fi

    append_summary_row \
      "e4_canc_v0" \
      "${gpu_id}" \
      "${COR_TYPE}" \
      "${STATUS}" \
      "${LOG_FILE}" \
      "${JSON_FILE}" \
      "${summary_file}"

    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "[E4-CANC-v0] Finished: ${COR_TYPE}" | tee -a "${LOG_FILE}"
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
