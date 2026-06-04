#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "Usage: bash 00_run_ulip_modelnetc_s2_cache_transfer_common.sh EXP_ID CACHE_METHOD PROMPT_SOURCE METHOD_FULL PURPOSE [GPU]"
  echo
  echo "CACHE_METHOD choices:"
  echo "  zs_global"
  echo "  zs_global_local"
  echo
  echo "PROMPT_SOURCE choices for E2:"
  echo "  manual_full"
  echo "  manualfull_llm_dynamic_init"
  exit 1
fi

EXP_ID="$1"
CACHE_METHOD="$2"
PROMPT_SOURCE="$3"
METHOD_FULL="$4"
PURPOSE="$5"
PHYSICAL_GPU="${6:-0}"

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

cd "${PC_ROOT}"

SHARED_PROMPT_DIR="results/E1_text_prototype_enhancement/shared_prompts"
SHARED_PROMPT_FILE="${SHARED_PROMPT_DIR}/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"

if [[ "${CACHE_METHOD}" == "zs_global" ]]; then
  CACHE_TYPE="global"
elif [[ "${CACHE_METHOD}" == "zs_global_local" ]]; then
  CACHE_TYPE="hierarchical"
else
  echo "ERROR: unsupported CACHE_METHOD: ${CACHE_METHOD}"
  echo "Supported choices: zs_global, zs_global_local"
  exit 1
fi

if [[ "${PROMPT_SOURCE}" == "manualfull_llm_dynamic_init" ]]; then
  if [[ ! -f "${SHARED_PROMPT_FILE}" ]]; then
    echo "ERROR: shared E1 LLM prompt cache not found:"
    echo "  ${SHARED_PROMPT_FILE}"
    echo
    echo "E2 should reuse E1 shared prompts and should not regenerate LLM prompts."
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export WANDB_DIR="${PC_ROOT}/results/E2_text_prototype_transfer_to_pointcache/${EXP_ID}/wandb"
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1

mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "E2 Text Prototype Transfer to Point-Cache"
echo "Setting: ULIP x ModelNet-C severity=2"
echo "EXP_ID: ${EXP_ID}"
echo "CACHE_METHOD: ${CACHE_METHOD}"
echo "CACHE_TYPE: ${CACHE_TYPE}"
echo "PROMPT_SOURCE: ${PROMPT_SOURCE}"
echo "METHOD_FULL: ${METHOD_FULL}"
echo "PURPOSE: ${PURPOSE}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "Runner: runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py"
echo "Result root: results/E2_text_prototype_transfer_to_pointcache"
echo "Shared prompt dir: ${SHARED_PROMPT_DIR}"
echo "============================================================"

python runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "${CACHE_METHOD}" \
  --baseline-method-full "${METHOD_FULL}" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/E2_text_prototype_transfer_to_pointcache" \
  --config configs \
  --wandb-log \
  --lm3d ulip \
  --cache-type "${CACHE_TYPE}" \
  --prompt-source "${PROMPT_SOURCE}" \
  --prompt-cache-dir "${SHARED_PROMPT_DIR}" \
  --llm-provider "deepseek" \
  --llm-model "deepseek-v4-pro" \
  --llm-api-key-file "llm/secrets/llm_api_key.txt" \
  --llm-api-base-url "https://api.deepseek.com/chat/completions" \
  --llm-temperature "0.3" \
  --llm-prompt-mode "multiview_2d3d" \
  --dynamic-prompt-count "10" \
  --prompt-static-weight "0.75" \
  --prompt-dynamic-weight "0.25" \
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
