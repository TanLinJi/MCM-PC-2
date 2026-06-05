#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "Usage: bash 02_run_ulip_modelnetc_s2_parallel_gpa_center_source_ablation_common.sh EXP_ID RUNNER PROMPT_SOURCE METHOD_FULL PURPOSE [GPU]"
  echo
  echo "This common script is for E3-V2 parallel GPA center-source ablation:"
  echo "  02_1: Parallel GPA + GPA-only center"
  echo "  02_2: Parallel GPA + Entropy-only center"
  echo "  02_3: Parallel GPA + Entropy+GPA union center"
  exit 1
fi

EXP_ID="$1"
RUNNER="$2"
PROMPT_SOURCE="$3"
METHOD_FULL="$4"
PURPOSE="$5"
PHYSICAL_GPU="${6:-0}"

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

cd "${PC_ROOT}"

SHARED_PROMPT_DIR="results/E1_text_prototype_enhancement/shared_prompts"
SHARED_PROMPT_FILE="${SHARED_PROMPT_DIR}/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"

if [[ "${PROMPT_SOURCE}" == "manualfull_llm_dynamic_init" ]]; then
  if [[ ! -f "${SHARED_PROMPT_FILE}" ]]; then
    echo "ERROR: shared E1 LLM prompt cache not found:"
    echo "  ${SHARED_PROMPT_FILE}"
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export WANDB_DIR="${PC_ROOT}/results/E3_global_prototype_alignment_cache/${EXP_ID}/wandb"
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1
export GPA_SAVE_STATS="${GPA_SAVE_STATS:-1}"

mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "E3-V2 Parallel Global Prototype-Alignment Cache"
echo "Setting: ULIP x ModelNet-C severity=2 x zs_global_local"
echo "EXP_ID: ${EXP_ID}"
echo "RUNNER: ${RUNNER}"
echo "PROMPT_SOURCE: ${PROMPT_SOURCE}"
echo "METHOD_FULL: ${METHOD_FULL}"
echo "PURPOSE: ${PURPOSE}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "CACHE_METHOD: zs_global_local"
echo "CACHE_TYPE: hierarchical"
echo "Result root: results/E3_global_prototype_alignment_cache"
echo "Shared prompt dir: ${SHARED_PROMPT_DIR}"
echo "============================================================"

python "${RUNNER}" \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "zs_global_local" \
  --baseline-method-full "${METHOD_FULL}" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/E3_global_prototype_alignment_cache" \
  --config configs \
  --wandb-log \
  --lm3d ulip \
  --cache-type "hierarchical" \
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
