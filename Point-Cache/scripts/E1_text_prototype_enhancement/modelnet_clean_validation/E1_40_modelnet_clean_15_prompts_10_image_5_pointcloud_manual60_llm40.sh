#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"
PC_ROOT="/root/autodl-tmp/MCM-PC-2/Point-Cache"

PROMPT_COUNT="15"
LLM_PROMPT_MODE_VALUE="image10_pointcloud5"
DEFAULT_PROMPT_FILE="modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json"
DEFAULT_EXP_ID="E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40"

STATIC_WEIGHT="0.60"
DYNAMIC_WEIGHT="0.40"
USE_DEFAULT_PROMPTS="${E1_USE_DEFAULT_PROMPTS:-1}"
PHYSICAL_GPU="${1:-0}"

if [[ "${USE_DEFAULT_PROMPTS}" == "1" ]]; then
  EXP_ID="${E1_EXP_ID:-${DEFAULT_EXP_ID}}"
  PROMPT_CACHE_DIR_VALUE="${E1_PROMPT_CACHE_DIR:-llm}"
  PROMPT_CACHE_FILE_VALUE="${E1_PROMPT_CACHE_FILE:-${DEFAULT_PROMPT_FILE}}"
elif [[ "${USE_DEFAULT_PROMPTS}" == "0" ]]; then
  GENERATED_TAG="${E1_GENERATED_PROMPT_TAG:-$(date +%Y%m%d_%H%M%S)}"
  EXP_ID="${E1_EXP_ID:-${DEFAULT_EXP_ID}_regenerated_${GENERATED_TAG}}"
  PROMPT_CACHE_DIR_VALUE="${E1_GENERATED_PROMPT_DIR:-llm/generated/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_${GENERATED_TAG}}"
  PROMPT_CACHE_FILE_VALUE="${E1_GENERATED_PROMPT_FILE:-${DEFAULT_PROMPT_FILE}}"

  echo "============================================================"
  echo "E1_40 clean ModelNet prompt generation mode"
  echo "Generated prompt dir: ${PROMPT_CACHE_DIR_VALUE}"
  echo "Generated prompt file: ${PROMPT_CACHE_FILE_VALUE}"
  echo "============================================================"

  if [[ "${PROMPT_CACHE_DIR_VALUE}" == /* ]]; then
    mkdir -p "${PROMPT_CACHE_DIR_VALUE}"
  else
    mkdir -p "${PC_ROOT}/${PROMPT_CACHE_DIR_VALUE}"
  fi
  PROMPT_CACHE_DIR="${PROMPT_CACHE_DIR_VALUE}" \
  FORCE_REGENERATE=1 \
  bash "${SCRIPT_DIR}/common/generate_modelnet_c_prompts.sh" \
    "${PROMPT_COUNT}" \
    "${LLM_PROMPT_MODE_VALUE}" \
    "${PROMPT_CACHE_FILE_VALUE}"
else
  echo "ERROR: E1_USE_DEFAULT_PROMPTS must be 1 or 0."
  echo "  1: use the existing default prompt JSON"
  echo "  0: call the LLM API and save regenerated prompts under Point-Cache/llm/"
  exit 1
fi

cd "${PC_ROOT}"

PROMPT_CACHE_PATH="${PROMPT_CACHE_DIR_VALUE}/${PROMPT_CACHE_FILE_VALUE}"
if [[ ! -f "${PROMPT_CACHE_PATH}" ]]; then
  echo "ERROR: E1 LLM prompt JSON not found:"
  echo "  ${PROMPT_CACHE_PATH}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export WANDB_DIR="${PC_ROOT}/results/E1_text_prototype_enhancement/${EXP_ID}/wandb"
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1

read -r -a E1_PYTHON <<< "${E1_PYTHON_CMD:-python}"

mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "E1 clean ModelNet text prototype fusion"
echo "EXP_ID: ${EXP_ID}"
echo "METHOD: manual_full + LLM"
echo "DATASET_LABEL: modelnet_clean"
echo "LOADER_DATASET: modelnet_c"
echo "COR_TYPE: clean"
echo "DATA_FILE: data/modelnet_c/clean.h5"
echo "STATIC_WEIGHT(manual_full): ${STATIC_WEIGHT}"
echo "DYNAMIC_WEIGHT(LLM): ${DYNAMIC_WEIGHT}"
echo "PROMPT_COUNT: ${PROMPT_COUNT}"
echo "LLM_PROMPT_MODE: ${LLM_PROMPT_MODE_VALUE}"
echo "PROMPT_CACHE_PATH: ${PROMPT_CACHE_PATH}"
echo "PURPOSE: clean ModelNet validation of E1_36 selected setting, 15 prompts = 10 image + 5 pointcloud, manual60_llm40"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "Runner: runners/E1_text_prototype_enhancement/run_modelnet_clean_text_prototype_fusion.py"
echo "Result root: results/E1_text_prototype_enhancement"
echo "Python command: ${E1_PYTHON[*]}"
echo "============================================================"

"${E1_PYTHON[@]}" runners/E1_text_prototype_enhancement/run_modelnet_clean_text_prototype_fusion.py \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "zs" \
  --baseline-method-full "E1 manual_full + LLM fusion: manual_full=${STATIC_WEIGHT}, LLM=${DYNAMIC_WEIGHT}; clean ModelNet validation of E1_36 selected setting, 15 prompts = 10 image + 5 pointcloud" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/E1_text_prototype_enhancement" \
  --modelnet-clean-data-root "data/modelnet_c" \
  --modelnet-clean-cor-type "clean" \
  --config configs \
  --wandb-log \
  --lm3d ulip \
  --cache-type "global" \
  --prompt-source "manualfull_llm_dynamic_init" \
  --prompt-cache-dir "${PROMPT_CACHE_DIR_VALUE}" \
  --prompt-cache-file "${PROMPT_CACHE_FILE_VALUE}" \
  --llm-provider "deepseek" \
  --llm-model "deepseek-v4-pro" \
  --llm-api-key-file "llm/secrets/llm_api_key.txt" \
  --llm-api-base-url "https://api.deepseek.com/chat/completions" \
  --llm-temperature "0.3" \
  --llm-prompt-mode "${LLM_PROMPT_MODE_VALUE}" \
  --dynamic-prompt-count "${PROMPT_COUNT}" \
  --prompt-static-weight "${STATIC_WEIGHT}" \
  --prompt-dynamic-weight "${DYNAMIC_WEIGHT}" \
  --ckpt_path weights/ulip/pointbert_ulip1.pt \
  --slip-ckpt-path weights/ulip/slip_base_100ep.pt \
  --dataset modelnet_c \
  --sonn_variant hardest \
  --cor_type clean \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --oshape-version vitg14 \
  --ulip-version ulip1 \
  --device 0 \
  --print-freq 500
