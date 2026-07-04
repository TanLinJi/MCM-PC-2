#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "Usage: bash 00_run_ulip_modelnetc_s2_zs_text_method_smoke_common.sh EXP_ID PROMPT_SOURCE METHOD_FULL PURPOSE [GPU]"
  exit 1
fi

EXP_ID="$1"
PROMPT_SOURCE="$2"
METHOD_FULL="$3"
PURPOSE="$4"
PHYSICAL_GPU="${5:-0}"

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

cd "${PC_ROOT}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export WANDB_DIR="${PC_ROOT}/results/E1_text_prototype_enhancement/${EXP_ID}/wandb"
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1

read -r -a E1_PYTHON <<< "${E1_PYTHON_CMD:-python}"

mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "E1 Text Prototype Enhancement Smoke Test"
echo "Setting: ULIP x ModelNet-C severity=2 x zero-shot"
echo "EXP_ID: ${EXP_ID}"
echo "PROMPT_SOURCE: ${PROMPT_SOURCE}"
echo "METHOD_FULL: ${METHOD_FULL}"
echo "PURPOSE: ${PURPOSE}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "Runner: runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py"
echo "Result root: results/E1_text_prototype_enhancement"
echo "Python command: ${E1_PYTHON[*]}"
echo "============================================================"

"${E1_PYTHON[@]}" runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "zs" \
  --baseline-method-full "${METHOD_FULL}" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/E1_text_prototype_enhancement" \
  --config configs \
  --wandb-log \
  --lm3d ulip \
  --cache-type "global" \
  --prompt-source "${PROMPT_SOURCE}" \
  --prompt-cache-dir "${PROMPT_CACHE_DIR:-llm/e1_prompt_bank}" \
  --llm-provider "${LLM_PROVIDER:-deepseek}" \
  --llm-model "${LLM_MODEL:-deepseek-v4-pro}" \
  --llm-api-key-file "${LLM_API_KEY_FILE:-llm/secrets/llm_api_key.txt}" \
  --llm-api-base-url "${LLM_API_BASE_URL:-https://api.deepseek.com/chat/completions}" \
  --llm-temperature "${LLM_TEMPERATURE:-0.3}" \
  --llm-prompt-mode "${LLM_PROMPT_MODE:-multiview_2d3d}" \
  --dynamic-prompt-count "${DYNAMIC_PROMPT_COUNT:-10}" \
  --prompt-static-weight "${PROMPT_STATIC_WEIGHT:-0.75}" \
  --prompt-dynamic-weight "${PROMPT_DYNAMIC_WEIGHT:-0.25}" \
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
