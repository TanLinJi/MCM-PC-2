#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"
PC_ROOT="/root/autodl-tmp/MCM-PC-2/Point-Cache"

PROMPT_COUNT="15"
LLM_PROMPT_MODE_VALUE="image10_pointcloud5"
DEFAULT_PROMPT_FILE="modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json"
DEFAULT_EXP_ID="E1_33_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual60_llm40"

USE_DEFAULT_PROMPTS="${E1_USE_DEFAULT_PROMPTS:-1}"
PHYSICAL_GPU="${1:-0}"

if [[ "${USE_DEFAULT_PROMPTS}" == "1" ]]; then
  EXP_ID="${E1_EXP_ID:-${DEFAULT_EXP_ID}}"
  PROMPT_CACHE_DIR_VALUE="${E1_PROMPT_CACHE_DIR:-llm}"
  PROMPT_CACHE_FILE_VALUE="${E1_PROMPT_CACHE_FILE:-${DEFAULT_PROMPT_FILE}}"
elif [[ "${USE_DEFAULT_PROMPTS}" == "0" ]]; then
  GENERATED_TAG="${E1_GENERATED_PROMPT_TAG:-$(date +%Y%m%d_%H%M%S)}"
  EXP_ID="${E1_EXP_ID:-${DEFAULT_EXP_ID}_regenerated_${GENERATED_TAG}}"
  PROMPT_CACHE_DIR_VALUE="${E1_GENERATED_PROMPT_DIR:-llm/generated/E1_33_15_prompts_10_image_5_pointcloud_${GENERATED_TAG}}"
  PROMPT_CACHE_FILE_VALUE="${E1_GENERATED_PROMPT_FILE:-${DEFAULT_PROMPT_FILE}}"

  echo "============================================================"
  echo "E1_33 severity2 prompt generation mode"
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
  exit 1
fi

MODELNET_C_SEVERITIES=2 \
PROMPT_CACHE_DIR="${PROMPT_CACHE_DIR_VALUE}" \
bash "${SCRIPT_DIR}/common/run_modelnet_c_fusion.sh" \
  "${EXP_ID}" \
  "0.60" \
  "0.40" \
  "${PROMPT_COUNT}" \
  "${LLM_PROMPT_MODE_VALUE}" \
  "${PROMPT_CACHE_FILE_VALUE}" \
  "ModelNet-C severity2 prompt composition diagnostic, 15 prompts = 10 image + 5 pointcloud, manual60_llm40" \
  "${PHYSICAL_GPU}"
