#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"
PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

EXP_ID="02_10_1_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_distfinal_tw0p15_text1_visual1_score_norm_manualfull_llm_dynamic_init_textdist"
RUNNER="runners/E4_distribution_guided_cache/run_e4_c_a0_e1_distribution_final_score_ulip_modelnetc_s2.py"
PHYSICAL_GPU="${1:-0}"

SHARED_PROMPT_FILE="${PC_ROOT}/results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"
MODELNETC_CLASS_FILE="${PC_ROOT}/data/modelnet_c/shape_names.txt"

if [[ ! -f "${SHARED_PROMPT_FILE}" ]]; then
  echo "ERROR: shared E1 LLM prompt cache not found:"
  echo "  ${SHARED_PROMPT_FILE}"
  echo
  echo "This E4-C-A0+E1 distribution-final experiment must reuse the existing prompt cache and must not call the LLM API."
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

export E4_SCORE_NORM_MODE="${E4_SCORE_NORM_MODE:-running_zscore}"
export E4_SCORE_NORM_MIN_COUNT="${E4_SCORE_NORM_MIN_COUNT:-8}"
export E4_SCORE_NORM_EPS="${E4_SCORE_NORM_EPS:-1e-6}"
export E4_SCORE_NORM_CLIP="${E4_SCORE_NORM_CLIP:-0}"

export E4_TEXT_DIST_PROMPT_SOURCE="${E4_TEXT_DIST_PROMPT_SOURCE:-manualfull_llm_dynamic_init}"
export E4_TEXT_SCORE_WEIGHT="${E4_TEXT_SCORE_WEIGHT:-0.15}"

export E4_FINAL_TEXT_DIST_WEIGHT="${E4_FINAL_TEXT_DIST_WEIGHT:-1.0}"
export E4_FINAL_VISUAL_DIST_WEIGHT="${E4_FINAL_VISUAL_DIST_WEIGHT:-1.0}"
export E4_FINAL_DIST_NORM_MODE="${E4_FINAL_DIST_NORM_MODE:-per_sample_zscore}"
export E4_FINAL_DIST_NORM_EPS="${E4_FINAL_DIST_NORM_EPS:-1e-6}"
export E4_FINAL_MISSING_SCORE_MARGIN="${E4_FINAL_MISSING_SCORE_MARGIN:-10.0}"

echo "============================================================"
echo "02_10_1 E4-C-A0+E1 distribution-final score"
echo "EXP_ID: ${EXP_ID}"
echo "RUNNER: ${RUNNER}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "E4_TEXT_SCORE_WEIGHT: ${E4_TEXT_SCORE_WEIGHT}"
echo "E4_TEXT_DIST_PROMPT_SOURCE: ${E4_TEXT_DIST_PROMPT_SOURCE}"
echo "E4_SCORE_NORM_MODE: ${E4_SCORE_NORM_MODE}"
echo "E4_FINAL_TEXT_DIST_WEIGHT: ${E4_FINAL_TEXT_DIST_WEIGHT}"
echo "E4_FINAL_VISUAL_DIST_WEIGHT: ${E4_FINAL_VISUAL_DIST_WEIGHT}"
echo "E4_FINAL_DIST_NORM_MODE: ${E4_FINAL_DIST_NORM_MODE}"
echo "============================================================"

bash "${SCRIPT_DIR}/02_run_e4_c_ulip_modelnetc_s2_common.sh" \
  "${EXP_ID}" \
  "${RUNNER}" \
  "manual_full" \
  "02_10_1: E4-C-A0+E1 distribution-final score; final logits use normalized text distribution + visual accepted-history distribution, no clip logits/cache voting" \
  "Based on 02_9_2, but final scoring is entirely distribution-based; E1 textdist replacement weight=0.15, final text/visual weights=1.0/1.0" \
  "${PHYSICAL_GPU}"
