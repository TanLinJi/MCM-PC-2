# 本公共脚本用于执行 Uni3D 在 ModelNet clean 上的单方法 baseline 复现实验。
#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "Usage: bash 31_run_uni3d_modelnet_clean_common.sh EXP_ID METHOD METHOD_FULL RUNNER CACHE_TYPE [GPU]"
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

DATASET="modelnet_c"
DATA_ROOT="data/modelnet_c"
SONN_VARIANT="-"
CORRUPTION="clean"
SEVERITY="-"
COR_TYPE="clean"
DATA_FILE="${DATA_ROOT}/${COR_TYPE}.h5"
NPOINTS=1024

BACKBONE="Uni3D"
LM3D="uni3d"
CKPT_PATH="weights/uni3d/modelnet40/model.pt"
PRETRAINED_PATH="weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin"

# Uni3D-g 参数，与 Point-Cache 原始 eval 脚本中的 uni3d 分支保持一致。
PC_MODEL="eva_giant_patch14_560"
CLIP_MODEL="EVA02-E-14-plus"
PC_FEAT_DIM=1408
NUM_GROUP=512
GROUP_SIZE=64
PC_ENCODER_DIM=512
EMBED_DIM=1024

ULIP_VERSION="-"
OSHAPE_VERSION="-"
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
echo "cor_type: ${COR_TYPE}"
echo "Data file: ${DATA_FILE}"
echo "Backbone: ${BACKBONE}"
echo "LM3D: ${LM3D}"
echo "Runner: ${RUNNER}"
echo "Cache type arg: ${CACHE_TYPE}"
echo "Uni3D ckpt: ${CKPT_PATH}"
echo "Uni3D text pretrained: ${PRETRAINED_PATH}"
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
  echo "ERROR: missing Uni3D checkpoint ${CKPT_PATH}"
  exit 1
fi

if [[ ! -f "${PRETRAINED_PATH}" ]]; then
  echo "${EXP_ID},${DATASET},${DATA_ROOT},${CORRUPTION},${SEVERITY},${COR_TYPE},${DATA_FILE},${SONN_VARIANT},${BACKBONE},${METHOD},${METHOD_FULL},,missing_pretrained,${PHYSICAL_GPU},${LOG_FILE}" >> "${SUMMARY_FILE}"
  echo "ERROR: missing Uni3D text pretrained checkpoint ${PRETRAINED_PATH}"
  exit 1
fi

set +e
python "${RUNNER}" \
  --config configs \
  --wandb-log \
  --lm3d "${LM3D}" \
  --cache-type "${CACHE_TYPE}" \
  --pc-model "${PC_MODEL}" \
  --clip-model "${CLIP_MODEL}" \
  --pretrained "${PRETRAINED_PATH}" \
  --pc-feat-dim "${PC_FEAT_DIM}" \
  --num-group "${NUM_GROUP}" \
  --group-size "${GROUP_SIZE}" \
  --pc-encoder-dim "${PC_ENCODER_DIM}" \
  --embed-dim "${EMBED_DIM}" \
  --ckpt_path "${CKPT_PATH}" \
  --dataset "${DATASET}" \
  --sonn_variant obj_only \
  --cor_type "${COR_TYPE}" \
  --npoints "${NPOINTS}" \
  --sim2real_type "${S2R_TYPE}" \
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
