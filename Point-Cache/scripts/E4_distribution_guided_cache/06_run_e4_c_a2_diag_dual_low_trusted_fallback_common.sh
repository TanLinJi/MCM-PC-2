#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 7 ]]; then
  echo "Usage: bash 06_run_e4_c_a2_diag_dual_low_trusted_fallback_common.sh EXP_ID TEXT_WEIGHT ENTROPY_THRESHOLD ENERGY_Z_THRESHOLD ENERGY_MIN_COUNT PURPOSE GPU"
  exit 1
fi

EXP_ID="$1"
TEXT_WEIGHT="$2"
ENTROPY_THRESHOLD="$3"
ENERGY_Z_THRESHOLD="$4"
ENERGY_MIN_COUNT="$5"
PURPOSE="$6"
PHYSICAL_GPU="$7"

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"
PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

SHARED_PROMPT_FILE="${PC_ROOT}/results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"
MODELNETC_CLASS_FILE="${PC_ROOT}/data/modelnet_c/shape_names.txt"

if [[ ! -f "${SHARED_PROMPT_FILE}" ]]; then
  echo "ERROR: shared E1 LLM prompt cache not found:"
  echo "  ${SHARED_PROMPT_FILE}"
  echo
  echo "This A2 diagnostic run must reuse the existing prompt cache and must not call the LLM API."
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
export E4_TEXT_SCORE_WEIGHT="${E4_TEXT_SCORE_WEIGHT:-${TEXT_WEIGHT}}"

export E4_TRUSTED_FALLBACK="${E4_TRUSTED_FALLBACK:-dual_low_entropy_energy}"
export E4_TRUSTED_ENTROPY_THRESHOLD="${E4_TRUSTED_ENTROPY_THRESHOLD:-${ENTROPY_THRESHOLD}}"
export E4_TRUSTED_ENERGY_Z_THRESHOLD="${E4_TRUSTED_ENERGY_Z_THRESHOLD:-${ENERGY_Z_THRESHOLD}}"
export E4_TRUSTED_ENERGY_MIN_COUNT="${E4_TRUSTED_ENERGY_MIN_COUNT:-${ENERGY_MIN_COUNT}}"
export E4_TRUSTED_ENERGY_EPS="${E4_TRUSTED_ENERGY_EPS:-1e-6}"

export E4_DIAG_SAVE_SAMPLE_ENERGY="${E4_DIAG_SAVE_SAMPLE_ENERGY:-1}"
export E4_DIAG_SAVE_REPLACEMENT_SNAPSHOT="${E4_DIAG_SAVE_REPLACEMENT_SNAPSHOT:-1}"

echo "============================================================"
echo "E4-C-A2 diagnostic dual-low trusted fallback common runner"
echo "EXP_ID: ${EXP_ID}"
echo "TEXT_WEIGHT: ${E4_TEXT_SCORE_WEIGHT}"
echo "E4_TRUSTED_FALLBACK: ${E4_TRUSTED_FALLBACK}"
echo "E4_TRUSTED_ENTROPY_THRESHOLD: ${E4_TRUSTED_ENTROPY_THRESHOLD}"
echo "E4_TRUSTED_ENERGY_Z_THRESHOLD: ${E4_TRUSTED_ENERGY_Z_THRESHOLD}"
echo "E4_TRUSTED_ENERGY_MIN_COUNT: ${E4_TRUSTED_ENERGY_MIN_COUNT}"
echo "E4_DIAG_SAVE_SAMPLE_ENERGY: ${E4_DIAG_SAVE_SAMPLE_ENERGY}"
echo "E4_DIAG_SAVE_REPLACEMENT_SNAPSHOT: ${E4_DIAG_SAVE_REPLACEMENT_SNAPSHOT}"
echo "PURPOSE: ${PURPOSE}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "============================================================"

bash "${SCRIPT_DIR}/02_run_e4_c_ulip_modelnetc_s2_common.sh" \
  "${EXP_ID}" \
  "runners/E4_distribution_guided_cache/run_e4_c_a2_diag_dual_low_trusted_fallback_ulip_modelnetc_s2_accepted_history_text_visual_distribution_guided_gpa.py" \
  "manual_full" \
  "06_1: E4-C-A2 diagnostic dual-low trusted fallback, text_weight=${E4_TEXT_SCORE_WEIGHT}, entropy<=${E4_TRUSTED_ENTROPY_THRESHOLD}, energy_z<=${E4_TRUSTED_ENERGY_Z_THRESHOLD}, energy_min_count=${E4_TRUSTED_ENERGY_MIN_COUNT}, diag=sample_energy+replacement_snapshot" \
  "${PURPOSE}" \
  "${PHYSICAL_GPU}"
