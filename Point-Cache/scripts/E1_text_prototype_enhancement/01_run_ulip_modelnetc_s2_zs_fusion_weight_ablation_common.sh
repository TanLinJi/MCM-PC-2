#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "Usage: bash 01_run_ulip_modelnetc_s2_zs_fusion_weight_ablation_common.sh EXP_ID STATIC_WEIGHT DYNAMIC_WEIGHT PURPOSE [GPU]"
  exit 1
fi

EXP_ID="$1"
STATIC_WEIGHT="$2"
DYNAMIC_WEIGHT="$3"
PURPOSE="$4"
PHYSICAL_GPU="${5:-0}"

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

cd "${PC_ROOT}"

LLM_PROVIDER_VALUE="${LLM_PROVIDER:-deepseek}"
LLM_MODEL_VALUE="${LLM_MODEL:-deepseek-v4-pro}"
LLM_PROMPT_MODE_VALUE="${LLM_PROMPT_MODE:-multiview_2d3d}"
DYNAMIC_PROMPT_COUNT_VALUE="${DYNAMIC_PROMPT_COUNT:-10}"

SHARED_PROMPT_DIR="${SHARED_PROMPT_DIR:-results/E1_text_prototype_enhancement/shared_prompts}"
SHARED_PROMPT_FILE="${SHARED_PROMPT_DIR}/modelnet_c_${LLM_PROVIDER_VALUE}_${LLM_MODEL_VALUE}_${LLM_PROMPT_MODE_VALUE}_${DYNAMIC_PROMPT_COUNT_VALUE}_prompts.json"

if [[ ! -f "${SHARED_PROMPT_FILE}" ]]; then
  echo "ERROR: shared LLM prompt cache not found:"
  echo "  ${SHARED_PROMPT_FILE}"
  echo
  echo "This weight ablation should reuse existing LLM prompts and should not regenerate prompts."
  echo "Please create or copy the shared prompt cache before running this script."
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export WANDB_DIR="${PC_ROOT}/results/E1_text_prototype_enhancement/${EXP_ID}/wandb"
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1

mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "E1-S1 Fusion Weight Ablation"
echo "Setting: ULIP x ModelNet-C severity=2 x zero-shot"
echo "EXP_ID: ${EXP_ID}"
echo "METHOD: manual_full_llm_fusion"
echo "STATIC_WEIGHT(manual_full): ${STATIC_WEIGHT}"
echo "DYNAMIC_WEIGHT(LLM): ${DYNAMIC_WEIGHT}"
echo "PURPOSE: ${PURPOSE}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "Shared prompt cache: ${SHARED_PROMPT_FILE}"
echo "Runner: runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py"
echo "Result root: results/E1_text_prototype_enhancement"
echo "============================================================"

python runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "zs" \
  --baseline-method-full "Fusion weight ablation: manual_full=${STATIC_WEIGHT}, LLM=${DYNAMIC_WEIGHT}" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/E1_text_prototype_enhancement" \
  --config configs \
  --wandb-log \
  --lm3d ulip \
  --cache-type "global" \
  --prompt-source "manualfull_llm_dynamic_init" \
  --prompt-cache-dir "${SHARED_PROMPT_DIR}" \
  --llm-provider "${LLM_PROVIDER_VALUE}" \
  --llm-model "${LLM_MODEL_VALUE}" \
  --llm-api-key-file "${LLM_API_KEY_FILE:-llm/secrets/llm_api_key.txt}" \
  --llm-api-base-url "${LLM_API_BASE_URL:-https://api.deepseek.com/chat/completions}" \
  --llm-temperature "${LLM_TEMPERATURE:-0.3}" \
  --llm-prompt-mode "${LLM_PROMPT_MODE_VALUE}" \
  --dynamic-prompt-count "${DYNAMIC_PROMPT_COUNT_VALUE}" \
  --prompt-static-weight "${STATIC_WEIGHT}" \
  --prompt-dynamic-weight "${DYNAMIC_WEIGHT}" \
  --ckpt_path weights/ulip/pointbert_ulip1.pt \
  --slip-ckpt-path weights/ulip/slip_base_100ep.pt \
  --dataset modelnet_c \
  --sonn_variant hardest \
  --cor_type add_global_2 \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --oshape-version vitg14 \
  --ulip-version ulip1 \
  --device 0 \
  --print-freq 500
