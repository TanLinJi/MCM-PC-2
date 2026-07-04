#!/usr/bin/env bash
set -euo pipefail

EXP_ID="$1"
RUNNER="$2"
PROMPT_SOURCE="$3"
METHOD_FULL="$4"
PURPOSE="$5"
PHYSICAL_GPU="${6:-0}"

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

cd "${PC_ROOT}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export GPA_SAVE_STATS="${GPA_SAVE_STATS:-1}"
export E4_DIST_EPS="${E4_DIST_EPS:-1e-4}"
export E4_DIST_MIN_VAR="${E4_DIST_MIN_VAR:-1e-4}"
export E4_TEXT_DIST_EPS="${E4_TEXT_DIST_EPS:-${E4_DIST_EPS}}"
export E4_TEXT_DIST_MIN_VAR="${E4_TEXT_DIST_MIN_VAR:-${E4_DIST_MIN_VAR}}"
export E4_TEXT_SCORE_WEIGHT="${E4_TEXT_SCORE_WEIGHT:-0.1}"
export E4_SCORE_NORM_MODE="${E4_SCORE_NORM_MODE:-none}"
export E4_SCORE_NORM_MIN_COUNT="${E4_SCORE_NORM_MIN_COUNT:-8}"
export E4_SCORE_NORM_EPS="${E4_SCORE_NORM_EPS:-1e-6}"
export E4_SCORE_NORM_CLIP="${E4_SCORE_NORM_CLIP:-0}"

echo "============================================================"
echo "E4-C Accepted-History Text-Visual Distribution-Guided GPA Cache"
echo "EXP_ID: ${EXP_ID}"
echo "RUNNER: ${RUNNER}"
echo "PROMPT_SOURCE: ${PROMPT_SOURCE}"
echo "METHOD_FULL: ${METHOD_FULL}"
echo "PURPOSE: ${PURPOSE}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "E4_DIST_EPS: ${E4_DIST_EPS}"
echo "E4_DIST_MIN_VAR: ${E4_DIST_MIN_VAR}"
echo "E4_TEXT_DIST_EPS: ${E4_TEXT_DIST_EPS}"
echo "E4_TEXT_DIST_MIN_VAR: ${E4_TEXT_DIST_MIN_VAR}"
echo "E4_TEXT_GATE_MODE: ${E4_TEXT_GATE_MODE:-distribution}"
echo "E4_TEXT_PROTO_SCORE_SCALE: ${E4_TEXT_PROTO_SCORE_SCALE:-1.0}"
echo "E4_TEXT_SCORE_WEIGHT: ${E4_TEXT_SCORE_WEIGHT}"
echo "E4_SCORE_NORM_MODE: ${E4_SCORE_NORM_MODE}"
echo "E4_SCORE_NORM_MIN_COUNT: ${E4_SCORE_NORM_MIN_COUNT}"
echo "E4_SCORE_NORM_EPS: ${E4_SCORE_NORM_EPS}"
echo "E4_SCORE_NORM_CLIP: ${E4_SCORE_NORM_CLIP}"
echo "E4_TEXT_DIST_PROMPT_SOURCE: ${E4_TEXT_DIST_PROMPT_SOURCE:-${PROMPT_SOURCE}}"
echo "E4_PROMPT_CACHE_DIR: ${E4_PROMPT_CACHE_DIR:-results/E1_text_prototype_enhancement/shared_prompts}"
echo "E4_PROMPT_CACHE_FILE: ${E4_PROMPT_CACHE_FILE:-<auto>}"
echo "E4_LLM_PROMPT_MODE: ${E4_LLM_PROMPT_MODE:-multiview_2d3d}"
echo "E4_DYNAMIC_PROMPT_COUNT: ${E4_DYNAMIC_PROMPT_COUNT:-10}"
echo "E4_PROMPT_STATIC_WEIGHT: ${E4_PROMPT_STATIC_WEIGHT:-0.75}"
echo "E4_PROMPT_DYNAMIC_WEIGHT: ${E4_PROMPT_DYNAMIC_WEIGHT:-0.25}"
echo "============================================================"

python "${RUNNER}" \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "zs_global_local" \
  --baseline-method-full "${METHOD_FULL}" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/E4_distribution_guided_cache" \
  --config configs \
  --lm3d ulip \
  --cache-type "hierarchical" \
  --prompt-source "${PROMPT_SOURCE}" \
  --prompt-cache-dir "${E4_PROMPT_CACHE_DIR:-results/E1_text_prototype_enhancement/shared_prompts}" \
  --prompt-cache-file "${E4_PROMPT_CACHE_FILE:-}" \
  --llm-provider "deepseek" \
  --llm-model "deepseek-v4-pro" \
  --llm-api-key-file "llm/secrets/llm_api_key.txt" \
  --llm-api-base-url "https://api.deepseek.com/chat/completions" \
  --llm-temperature "0.3" \
  --llm-prompt-mode "${E4_LLM_PROMPT_MODE:-multiview_2d3d}" \
  --dynamic-prompt-count "${E4_DYNAMIC_PROMPT_COUNT:-10}" \
  --prompt-static-weight "${E4_PROMPT_STATIC_WEIGHT:-0.75}" \
  --prompt-dynamic-weight "${E4_PROMPT_DYNAMIC_WEIGHT:-0.25}" \
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
