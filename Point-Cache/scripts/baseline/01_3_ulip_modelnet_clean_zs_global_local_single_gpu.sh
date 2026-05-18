# 本脚本用于复现 Point-Cache baseline 实验 01_3：ULIP 在 ModelNet clean 上的 Zero-shot + Global Cache + Local Cache 推理。
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"
EXP_ID="01_3_ulip_modelnet_clean_zs_global_local"
METHOD="zs_global_local"
METHOD_FULL="Zero-shot + Global Cache + Local Cache"
RUNNER="runners/model_with_hierarchical_caches.py"
CACHE_TYPE="hierarchical"

RESULT_ROOT="${PC_ROOT}/results/baseline"
RUN_DIR="${RESULT_ROOT}/${EXP_ID}"
LOG_DIR="${RUN_DIR}/logs"
WANDB_DIR_LOCAL="${RUN_DIR}/wandb"
SUMMARY_FILE="${RUN_DIR}/summary.csv"

PHYSICAL_GPU="${1:-0}"

mkdir -p "${LOG_DIR}" "${WANDB_DIR_LOCAL}"

cd "${PC_ROOT}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR_LOCAL}"
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1

DATASET="modelnet_c"
DATA_FILE="data/modelnet_c/clean.h5"
COR_TYPE="clean"
SONN_VARIANT="-"
NPOINTS=1024

BACKBONE="ULIP"
LM3D="ulip"
ULIP_VERSION="ulip1"
CKPT_PATH="weights/ulip/pointbert_ulip1.pt"
SLIP_CKPT_PATH="weights/ulip/slip_base_100ep.pt"

OSHAPE_VERSION="vitg14"
S2R_TYPE="so_obj_only_9"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${EXP_ID}_${TIMESTAMP}.log"

echo "exp_id,dataset,file,cor_type,sonn_variant,backbone,method,method_full,acc,status,gpu,log_path" > "${SUMMARY_FILE}"

echo "============================================================"
echo "EXP_ID: ${EXP_ID}"
echo "Method: ${METHOD_FULL}"
echo "Physical GPU: ${PHYSICAL_GPU}"
echo "Internal device: 0"
echo "Dataset: ${DATASET}"
echo "Data file: ${DATA_FILE}"
echo "Runner: ${RUNNER}"
echo "Cache type: ${CACHE_TYPE}"
echo "Result dir: ${RUN_DIR}"
echo "Log file: ${LOG_FILE}"
echo "============================================================"

set +e
python "${RUNNER}" \
  --config configs \
  --wandb-log \
  --lm3d "${LM3D}" \
  --cache-type "${CACHE_TYPE}" \
  --ckpt_path "${CKPT_PATH}" \
  --slip-ckpt-path "${SLIP_CKPT_PATH}" \
  --dataset "${DATASET}" \
  --sonn_variant "hardest" \
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
  echo "${EXP_ID},${DATASET},${DATA_FILE},${COR_TYPE},${SONN_VARIANT},${BACKBONE},${METHOD},${METHOD_FULL},,failed,${PHYSICAL_GPU},${LOG_FILE}" >> "${SUMMARY_FILE}"
  exit "${EXIT_CODE}"
fi

ACC="$(grep -Ei '\*\*\*Final\*\*\*.*accuracy:' "${LOG_FILE}" | tail -n 1 | sed -E 's/.*accuracy: *([0-9]+(\.[0-9]+)?).*/\1/')"

if [[ -z "${ACC}" ]]; then
  echo "${EXP_ID},${DATASET},${DATA_FILE},${COR_TYPE},${SONN_VARIANT},${BACKBONE},${METHOD},${METHOD_FULL},,failed_parse_acc,${PHYSICAL_GPU},${LOG_FILE}" >> "${SUMMARY_FILE}"
  echo "ERROR: 无法从日志中解析准确率"
  exit 1
fi

echo "${EXP_ID},${DATASET},${DATA_FILE},${COR_TYPE},${SONN_VARIANT},${BACKBONE},${METHOD},${METHOD_FULL},${ACC},done,${PHYSICAL_GPU},${LOG_FILE}" >> "${SUMMARY_FILE}"

echo
echo "实验完成：${EXP_ID}"
cat "${SUMMARY_FILE}"
