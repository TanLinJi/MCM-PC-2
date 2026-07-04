#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 7 ]]; then
  echo "Usage: bash run_modelnet_c_fusion.sh EXP_ID STATIC_WEIGHT DYNAMIC_WEIGHT PROMPT_COUNT LLM_PROMPT_MODE PROMPT_CACHE_FILE PURPOSE [GPU]"
  exit 1
fi

EXP_ID="$1"
STATIC_WEIGHT="$2"
DYNAMIC_WEIGHT="$3"
PROMPT_COUNT="$4"
LLM_PROMPT_MODE_VALUE="$5"
PROMPT_CACHE_FILE="$6"
PURPOSE="$7"
PHYSICAL_GPU="${8:-0}"

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

cd "${PC_ROOT}"

PROMPT_CACHE_DIR="${PROMPT_CACHE_DIR:-llm}"
PROMPT_CACHE_PATH="${PROMPT_CACHE_DIR}/${PROMPT_CACHE_FILE}"
MODELNET_C_SEVERITIES_VALUE="${MODELNET_C_SEVERITIES:-all}"

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
echo "E1 ModelNet-C text prototype fusion"
echo "EXP_ID: ${EXP_ID}"
echo "METHOD: manual_full + LLM"
echo "MODELNET_C_SEVERITIES: ${MODELNET_C_SEVERITIES_VALUE}"
echo "STATIC_WEIGHT(manual_full): ${STATIC_WEIGHT}"
echo "DYNAMIC_WEIGHT(LLM): ${DYNAMIC_WEIGHT}"
echo "PROMPT_COUNT: ${PROMPT_COUNT}"
echo "LLM_PROMPT_MODE: ${LLM_PROMPT_MODE_VALUE}"
echo "PROMPT_CACHE_PATH: ${PROMPT_CACHE_PATH}"
echo "PURPOSE: ${PURPOSE}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "Runner: runners/E1_text_prototype_enhancement/run_modelnet_c_full_text_prototype_fusion.py"
echo "Result root: results/E1_text_prototype_enhancement"
echo "Python command: ${E1_PYTHON[*]}"
echo "============================================================"

"${E1_PYTHON[@]}" runners/E1_text_prototype_enhancement/run_modelnet_c_full_text_prototype_fusion.py \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "zs" \
  --baseline-method-full "E1 manual_full + LLM fusion: manual_full=${STATIC_WEIGHT}, LLM=${DYNAMIC_WEIGHT}; ${PURPOSE}" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/E1_text_prototype_enhancement" \
  --modelnet-c-severities "${MODELNET_C_SEVERITIES_VALUE}" \
  --config configs \
  --wandb-log \
  --lm3d ulip \
  --cache-type "global" \
  --prompt-source "manualfull_llm_dynamic_init" \
  --prompt-cache-dir "${PROMPT_CACHE_DIR}" \
  --prompt-cache-file "${PROMPT_CACHE_FILE}" \
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
  --cor_type add_global_0 \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --oshape-version vitg14 \
  --ulip-version ulip1 \
  --device 0 \
  --print-freq 500
