#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 6 ]]; then
  echo "Usage: bash 09_run_e4_c_a0_e1_explicit_final_score_common.sh SETTING ENTROPY_CAP GPA_CAP LOCAL_CAP NEG_CAP GPU"
  exit 1
fi

SETTING="$1"
ENTROPY_CAP="$2"
GPA_CAP="$3"
LOCAL_CAP="$4"
NEG_CAP="$5"
PHYSICAL_GPU="$6"

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"
PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

SHARED_PROMPT_FILE="${PC_ROOT}/results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"
MODELNETC_CLASS_FILE="${PC_ROOT}/data/modelnet_c/shape_names.txt"

if [[ ! -f "${SHARED_PROMPT_FILE}" ]]; then
  echo "ERROR: shared E1 LLM prompt cache not found:"
  echo "  ${SHARED_PROMPT_FILE}"
  echo
    echo "09_2 must reuse the existing prompt cache and must not call the LLM API."
  exit 1
fi

python - "${SHARED_PROMPT_FILE}" "${MODELNETC_CLASS_FILE}" <<'PY'
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
    "Verified shared E1 prompt cache: "
    f"{len(classnames)} classes, at least {required_prompt_count} prompts per class."
)
PY

FINAL_SCORE_SWEEP_TAG="${E4_FINAL_SCORE_SWEEP_TAG:-score_sweep}"
EXP_ID="09_2_${SETTING}_${FINAL_SCORE_SWEEP_TAG}_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist"
LOCAL_CENTERS="${E4_LOCAL_CENTERS:-3}"
METHOD_FULL="09_2 ${SETTING}: explicit final-score weight sweep; entropy_cap=${ENTROPY_CAP}; gpa_cap=${GPA_CAP}; local_cap=${LOCAL_CAP}; neg_cap=${NEG_CAP}; local_centers=${LOCAL_CENTERS}; formula=y_zs+alpha_g*y_g+alpha_l*y_l-alpha_n*y_n; text_weight=0.15; score_norm=running_zscore"
PURPOSE="基于 02_9_2/09_1 的新增代码；缓存更新和容量逻辑保持一致，只把 final score 写成显式公式 y=y_zs+alpha_g*y_g+alpha_l*y_l-alpha_n*y_n，并支持一次推理统计多组 alpha。"

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
echo "09_2 explicit final-score setting: ${SETTING}"
echo "EXP_ID: ${EXP_ID}"
echo "E4_ENTROPY_CAP: ${E4_ENTROPY_CAP}"
echo "E4_GPA_CAP: ${E4_GPA_CAP}"
echo "E4_LOCAL_CAP: ${E4_LOCAL_CAP}"
echo "E4_NEG_CAP: ${E4_NEG_CAP}"
echo "E4_LOCAL_CENTERS: ${E4_LOCAL_CENTERS}"
echo "E4_FINAL_SCORE_SWEEP_TAG: ${FINAL_SCORE_SWEEP_TAG}"
echo "E4_FINAL_SCORE_WEIGHTS: ${E4_FINAL_SCORE_WEIGHTS:-<config-default>}"
echo "============================================================"

bash "${SCRIPT_DIR}/02_run_e4_c_ulip_modelnetc_s2_common.sh" \
  "${EXP_ID}" \
  "runners/E4_distribution_guided_cache/run_e4_c_a0_e1_explicit_final_score_ulip_modelnetc_s2.py" \
  "manual_full" \
  "${METHOD_FULL}" \
  "${PURPOSE}" \
  "${PHYSICAL_GPU}"
