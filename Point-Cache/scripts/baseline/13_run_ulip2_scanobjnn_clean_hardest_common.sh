# 本公共脚本用于执行 ULIP-2 在 ScanObjNN clean hardest 上的单方法 baseline 复现实验。
#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "Usage: bash 13_run_ulip2_scanobjnn_clean_hardest_common.sh EXP_ID METHOD METHOD_FULL RUNNER CACHE_TYPE [GPU]"
  exit 1
fi

EXP_ID="$1"
METHOD="$2"
METHOD_FULL="$3"
RUNNER="$4"
CACHE_TYPE="$5"
PHYSICAL_GPU="${6:-0}"

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

RESULT_ROOT="${PC_ROOT}/results/baseline"
RUN_DIR="${RESULT_ROOT}/${EXP_ID}"
LOG_DIR="${RUN_DIR}/logs"
WANDB_DIR_LOCAL="${RUN_DIR}/wandb"
SUMMARY_FILE="${RUN_DIR}/summary.csv"

mkdir -p "${LOG_DIR}" "${WANDB_DIR_LOCAL}"

cd "${PC_ROOT}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR_LOCAL}"
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1

DATASET="sonn_c"
DATA_ROOT="data/sonn_c"
SONN_VARIANT="hardest"
CORRUPTION="clean"
SEVERITY="-"
COR_TYPE="clean"
DATA_FILE="${DATA_ROOT}/${SONN_VARIANT}/${COR_TYPE}.h5"
NPOINTS=1024

BACKBONE="ULIP-2"
LM3D="ulip"
ULIP_VERSION="ulip2"
CKPT_PATH="weights/ulip/pointbert_ulip2.pt"
SLIP_CKPT_PATH="weights/ulip/slip_base_100ep.pt"

OSHAPE_VERSION="vitg14"
S2R_TYPE="so_obj_only_9"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${EXP_ID}_${COR_TYPE}_${TIMESTAMP}.log"

echo "exp_id,dataset,data_root,corruption,severity,cor_type,file,sonn_variant,backbone,method,method_full,acc,status,gpu,log_path" > "${SUMMARY_FILE}"

extract_acc() {
  local log_file="$1"
  grep -Ei '\*\*\*Final\*\*\*.*accuracy:' "${log_file}" | tail -n 1 | sed -E 's/.*accuracy: *([0-9]+(\.[0-9]+)?).*/\1/'
}

echo "============================================================"
echo "EXP_ID: ${EXP_ID}"
echo "Method: ${METHOD_FULL}"
echo "Physical GPU: ${PHYSICAL_GPU}"
echo "Internal device: 0"
echo "Dataset: ${DATASET}"
echo "Data root: ${DATA_ROOT}"
echo "SONN variant: ${SONN_VARIANT}"
echo "cor_type: ${COR_TYPE}"
echo "Data file: ${DATA_FILE}"
echo "Backbone: ${BACKBONE}"
echo "Runner: ${RUNNER}"
echo "Cache type: ${CACHE_TYPE}"
echo "Result dir: ${RUN_DIR}"
echo "Log file: ${LOG_FILE}"
echo "============================================================"

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "${EXP_ID},${DATASET},${DATA_ROOT},${CORRUPTION},${SEVERITY},${COR_TYPE},${DATA_FILE},${SONN_VARIANT},${BACKBONE},${METHOD},${METHOD_FULL},,missing_file,${PHYSICAL_GPU},${LOG_FILE}" >> "${SUMMARY_FILE}"
  echo "ERROR: missing file ${DATA_FILE}"
  exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "${EXP_ID},${DATASET},${DATA_ROOT},${CORRUPTION},${SEVERITY},${COR_TYPE},${DATA_FILE},${SONN_VARIANT},${BACKBONE},${METHOD},${METHOD_FULL},,missing_ckpt,${PHYSICAL_GPU},${LOG_FILE}" >> "${SUMMARY_FILE}"
  echo "ERROR: missing checkpoint ${CKPT_PATH}"
  exit 1
fi

if [[ ! -f "${SLIP_CKPT_PATH}" ]]; then
  echo "${EXP_ID},${DATASET},${DATA_ROOT},${CORRUPTION},${SEVERITY},${COR_TYPE},${DATA_FILE},${SONN_VARIANT},${BACKBONE},${METHOD},${METHOD_FULL},,missing_slip_ckpt,${PHYSICAL_GPU},${LOG_FILE}" >> "${SUMMARY_FILE}"
  echo "ERROR: missing SLIP checkpoint ${SLIP_CKPT_PATH}"
  exit 1
fi

set +e
python "${RUNNER}" \
  --config configs \
  --wandb-log \
  --lm3d "${LM3D}" \
  --cache-type "${CACHE_TYPE}" \
  --ckpt_path "${CKPT_PATH}" \
  --slip-ckpt-path "${SLIP_CKPT_PATH}" \
  --dataset "${DATASET}" \
  --sonn_variant "${SONN_VARIANT}" \
  --cor_type "${COR_TYPE}" \
  --npoints "${NPOINTS}" \
  --sim2real_type "${S2R_TYPE}" \
  --oshape-version "${OSHAPE_VERSION}" \
  --ulip-version "${ULIP_VERSION}" \
  --device 0 \
  --print-freq 500 \
  2>&1 | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ "${EXIT_CODE}" -ne 0 ]]; then
  echo "${EXP_ID},${DATASET},${DATA_ROOT},${CORRUPTION},${SEVERITY},${COR_TYPE},${DATA_FILE},${SONN_VARIANT},${BACKBONE},${METHOD},${METHOD_FULL},,failed,${PHYSICAL_GPU},${LOG_FILE}" >> "${SUMMARY_FILE}"
  echo "ERROR: run failed, exit_code=${EXIT_CODE}"
  exit "${EXIT_CODE}"
fi

ACC="$(extract_acc "${LOG_FILE}")"

if [[ -z "${ACC}" ]]; then
  echo "${EXP_ID},${DATASET},${DATA_ROOT},${CORRUPTION},${SEVERITY},${COR_TYPE},${DATA_FILE},${SONN_VARIANT},${BACKBONE},${METHOD},${METHOD_FULL},,failed_parse_acc,${PHYSICAL_GPU},${LOG_FILE}" >> "${SUMMARY_FILE}"
  echo "ERROR: failed to parse acc"
  exit 1
fi

echo "${EXP_ID},${DATASET},${DATA_ROOT},${CORRUPTION},${SEVERITY},${COR_TYPE},${DATA_FILE},${SONN_VARIANT},${BACKBONE},${METHOD},${METHOD_FULL},${ACC},done,${PHYSICAL_GPU},${LOG_FILE}" >> "${SUMMARY_FILE}"

echo
echo "============================================================"
echo "Finished: ${EXP_ID}"
echo "acc=${ACC}"
echo "summary: ${SUMMARY_FILE}"
echo "============================================================"
cat "${SUMMARY_FILE}"
