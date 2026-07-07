#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 6 ]]; then
  echo "Usage: bash 09_run_e4_c_a0_e1_explicit_final_score_scanobjnnc_all35_common.sh SETTING ENTROPY_CAP GPA_CAP LOCAL_CAP NEG_CAP GPU"
  exit 1
fi

SETTING="$1"
ENTROPY_CAP="$2"
GPA_CAP="$3"
LOCAL_CAP="$4"
NEG_CAP="$5"
PHYSICAL_GPU="$6"

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

SHARED_PROMPT_FILE="${PC_ROOT}/results/E1_text_prototype_enhancement/shared_prompts/sonn_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"
SCANOBJNNC_CLASS_FILE="${PC_ROOT}/data/sonn_c/shape_names.txt"

if [[ ! -f "${SHARED_PROMPT_FILE}" ]]; then
  echo "ERROR: shared ScanObjNN-C E1 LLM prompt cache not found:"
  echo "  ${SHARED_PROMPT_FILE}"
  echo
  echo "09_2 all35 must reuse the existing prompt cache and must not call the LLM API."
  exit 1
fi

if [[ ! -f "${SCANOBJNNC_CLASS_FILE}" ]]; then
  echo "ERROR: ScanObjNN-C class file not found:"
  echo "  ${SCANOBJNNC_CLASS_FILE}"
  exit 1
fi

python - "${SHARED_PROMPT_FILE}" "${SCANOBJNNC_CLASS_FILE}" <<'PY'
import json
import sys
from pathlib import Path

cache_path = Path(sys.argv[1])
class_file = Path(sys.argv[2])
required_prompt_count = 10

with cache_path.open("r", encoding="utf-8") as f:
    saved = json.load(f)

prompts = saved.get("prompts", saved)
if not isinstance(prompts, dict):
    raise SystemExit(f"ERROR: invalid prompt cache format: {cache_path}")

with class_file.open("r", encoding="utf-8") as f:
    classnames = [line.strip() for line in f if line.strip()]

missing = []
short = []
for classname in classnames:
    clean_name = classname.replace("_", " ")
    class_prompts = prompts.get(clean_name)
    if class_prompts is None:
        missing.append(clean_name)
    elif len(class_prompts) < required_prompt_count:
        short.append((clean_name, len(class_prompts)))

if missing or short:
    print("ERROR: shared E1 prompt cache is incomplete.")
    if missing:
        print("Missing classes:")
        for name in missing:
            print(f"  {name}")
    if short:
        print("Classes with too few prompts:")
        for name, count in short:
            print(f"  {name}: {count}/{required_prompt_count}")
    print()
    print("This script intentionally stops here to avoid regenerating prompts through the LLM API.")
    raise SystemExit(1)

print(
    "Verified shared ScanObjNN-C E1 prompt cache: "
    f"{len(classnames)} classes, at least {required_prompt_count} prompts per class."
)
PY

export SONN_VARIANT="${SONN_VARIANT:-hardest}"

FINAL_SCORE_SWEEP_TAG="${E4_FINAL_SCORE_SWEEP_TAG:-ag4p4_al3p9_an0p19}"
EXP_ID="09_2_${SETTING}_${FINAL_SCORE_SWEEP_TAG}_ulip_scanobjnnc_${SONN_VARIANT}_all35_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist"
LOCAL_CENTERS="${E4_LOCAL_CENTERS:-3}"
METHOD_FULL="09_2 ScanObjNN-C ${SONN_VARIANT} all35 ${SETTING}: explicit final-score weight sweep; entropy_cap=${ENTROPY_CAP}; gpa_cap=${GPA_CAP}; local_cap=${LOCAL_CAP}; neg_cap=${NEG_CAP}; local_centers=${LOCAL_CENTERS}; formula=y_zs+alpha_g*y_g+alpha_l*y_l-alpha_n*y_n; text_weight=0.15; score_norm=running_zscore"
PURPOSE="基于当前 ModelNet-C all35 最优配置迁移到 ScanObjNN-C all35；缓存更新和容量逻辑保持一致，只把 final score 写成显式公式 y=y_zs+alpha_g*y_g+alpha_l*y_l-alpha_n*y_n，并支持一次完整 ScanObjNN-C 推理统计多组 alpha。"

export E4_SCORE_NORM_MODE="${E4_SCORE_NORM_MODE:-running_zscore}"
export E4_SCORE_NORM_MIN_COUNT="${E4_SCORE_NORM_MIN_COUNT:-8}"
export E4_SCORE_NORM_EPS="${E4_SCORE_NORM_EPS:-1e-6}"
export E4_SCORE_NORM_CLIP="${E4_SCORE_NORM_CLIP:-0}"

export E4_TEXT_DIST_PROMPT_SOURCE="${E4_TEXT_DIST_PROMPT_SOURCE:-manualfull_llm_dynamic_init}"
export E4_TEXT_SCORE_WEIGHT="${E4_TEXT_SCORE_WEIGHT:-0.15}"

export E4_ENTROPY_CAP="${ENTROPY_CAP}"
export E4_GPA_CAP="${GPA_CAP}"
export E4_LOCAL_CAP="${LOCAL_CAP}"
export E4_NEG_CAP="${NEG_CAP}"
export E4_LOCAL_CENTERS="${LOCAL_CENTERS}"

echo "============================================================"
echo "09_2 explicit final-score all35 setting: ${SETTING}"
echo "EXP_ID: ${EXP_ID}"
echo "SONN_VARIANT: ${SONN_VARIANT}"
echo "Data root: data/sonn_c/${SONN_VARIANT}"
echo "E4_ENTROPY_CAP: ${E4_ENTROPY_CAP}"
echo "E4_GPA_CAP: ${E4_GPA_CAP}"
echo "E4_LOCAL_CAP: ${E4_LOCAL_CAP}"
echo "E4_NEG_CAP: ${E4_NEG_CAP}"
echo "E4_LOCAL_CENTERS: ${E4_LOCAL_CENTERS}"
echo "E4_FINAL_SCORE_SWEEP_TAG: ${FINAL_SCORE_SWEEP_TAG}"
echo "E4_FINAL_SCORE_WEIGHTS: ${E4_FINAL_SCORE_WEIGHTS:-<config-default>}"
echo "Result dir:"
echo "${PC_ROOT}/results/E4_distribution_guided_cache/${EXP_ID}"
echo "============================================================"

cd "${PC_ROOT}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export GPA_SAVE_STATS="${GPA_SAVE_STATS:-1}"
export E4_DIST_EPS="${E4_DIST_EPS:-1e-4}"
export E4_DIST_MIN_VAR="${E4_DIST_MIN_VAR:-1e-4}"
export E4_TEXT_DIST_EPS="${E4_TEXT_DIST_EPS:-${E4_DIST_EPS}}"
export E4_TEXT_DIST_MIN_VAR="${E4_TEXT_DIST_MIN_VAR:-${E4_DIST_MIN_VAR}}"
export E4_PROMPT_CACHE_DIR="${E4_PROMPT_CACHE_DIR:-results/E1_text_prototype_enhancement/shared_prompts}"
export E4_PROMPT_CACHE_FILE="${E4_PROMPT_CACHE_FILE:-}"
export E4_LLM_PROMPT_MODE="${E4_LLM_PROMPT_MODE:-multiview_2d3d}"
export E4_DYNAMIC_PROMPT_COUNT="${E4_DYNAMIC_PROMPT_COUNT:-10}"
export E4_PROMPT_STATIC_WEIGHT="${E4_PROMPT_STATIC_WEIGHT:-0.75}"
export E4_PROMPT_DYNAMIC_WEIGHT="${E4_PROMPT_DYNAMIC_WEIGHT:-0.25}"

python "runners/E4_distribution_guided_cache/run_e4_c_a0_e1_explicit_final_score_ulip_scanobjnnc_hardest_all35.py" \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "zs_global_local" \
  --baseline-method-full "${METHOD_FULL}" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/E4_distribution_guided_cache" \
  --config configs \
  --lm3d ulip \
  --cache-type "hierarchical" \
  --prompt-source "manual_full" \
  --prompt-cache-dir "${E4_PROMPT_CACHE_DIR}" \
  --prompt-cache-file "${E4_PROMPT_CACHE_FILE}" \
  --llm-provider "deepseek" \
  --llm-model "deepseek-v4-pro" \
  --llm-api-key-file "llm/secrets/llm_api_key.txt" \
  --llm-api-base-url "https://api.deepseek.com/chat/completions" \
  --llm-temperature "0.3" \
  --llm-prompt-mode "${E4_LLM_PROMPT_MODE}" \
  --dynamic-prompt-count "${E4_DYNAMIC_PROMPT_COUNT}" \
  --prompt-static-weight "${E4_PROMPT_STATIC_WEIGHT}" \
  --prompt-dynamic-weight "${E4_PROMPT_DYNAMIC_WEIGHT}" \
  --ckpt_path weights/ulip/pointbert_ulip1.pt \
  --slip-ckpt-path weights/ulip/slip_base_100ep.pt \
  --dataset sonn_c \
  --sonn_c_root data/sonn_c \
  --sonn_variant "${SONN_VARIANT}" \
  --cor_type add_global_0 \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --oshape-version vitg14 \
  --ulip-version ulip1 \
  --device 0 \
  --print-freq 500
