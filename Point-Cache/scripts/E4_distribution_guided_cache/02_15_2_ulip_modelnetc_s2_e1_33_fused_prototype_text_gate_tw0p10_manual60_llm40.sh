#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"
SCRIPT_DIR="${PC_ROOT}/scripts/E4_distribution_guided_cache"

EXP_ID="02_15_2_ulip_modelnetc_s2_e4_c_a0_e1_33_fused_prototype_text_gate_tw0p10_manual60_llm40"
RUNNER="runners/E4_distribution_guided_cache/run_e4_c_a0_e1_textdist_only_ulip_modelnetc_s2_accepted_history_text_visual_distribution_guided_gpa.py"
if [[ "${1:-}" == "--dry-run" ]]; then
  PHYSICAL_GPU="0"
  E4_DRY_RUN="1"
else
  PHYSICAL_GPU="${1:-0}"
  E4_DRY_RUN="${E4_DRY_RUN:-0}"
fi
if [[ "${2:-}" == "--dry-run" ]]; then
  E4_DRY_RUN="1"
fi
PROMPT_CACHE_FILE="${PC_ROOT}/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json"
MODELNETC_CLASS_FILE="${PC_ROOT}/data/modelnet_c/shape_names.txt"

if [[ ! -f "${PROMPT_CACHE_FILE}" ]]; then
  echo "ERROR: fixed E1-33 prompt JSON not found:"
  echo "  ${PROMPT_CACHE_FILE}"
  exit 1
fi

python - "${PROMPT_CACHE_FILE}" "${MODELNETC_CLASS_FILE}" <<'PY'
import json
import sys
from pathlib import Path

cache_path = Path(sys.argv[1])
class_file = Path(sys.argv[2])
required_prompt_count = 15

with cache_path.open("r", encoding="utf-8") as f:
    saved = json.load(f)

expected = {
    "dataset_name": "modelnet_c",
    "dynamic_prompt_count": required_prompt_count,
    "llm_prompt_mode": "image10_pointcloud5",
}
for key, expected_value in expected.items():
    actual_value = saved.get(key)
    if actual_value != expected_value:
        raise SystemExit(
            f"ERROR: prompt cache metadata mismatch for {key}: "
            f"expected {expected_value!r}, got {actual_value!r}"
        )

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
    elif len(class_prompts) != required_prompt_count:
        short.append((clean_name, len(class_prompts)))

if missing or short:
    print("ERROR: E1-33 prompt cache is incomplete.")
    if missing:
        print("Missing classes:")
        for name in missing:
            print(f"  {name}")
    if short:
        print("Classes with wrong prompt count:")
        for name, count in short:
            print(f"  {name}: {count}/{required_prompt_count}")
    raise SystemExit(1)

print(
    "Verified E1-33 prompt cache: "
    f"{len(classnames)} classes, exactly {required_prompt_count} prompts per class."
)
PY

export E4_TEXT_GATE_MODE="fused_prototype"
export E4_TEXT_PROTO_SCORE_SCALE="${E4_TEXT_PROTO_SCORE_SCALE:-1.0}"
export E4_TEXT_DIST_PROMPT_SOURCE="manualfull_llm_dynamic_init"
export E4_TEXT_SCORE_WEIGHT="0.10"
export E4_PROMPT_CACHE_DIR="llm"
export E4_PROMPT_CACHE_FILE="$(basename "${PROMPT_CACHE_FILE}")"
export E4_LLM_PROMPT_MODE="image10_pointcloud5"
export E4_DYNAMIC_PROMPT_COUNT="15"
export E4_PROMPT_STATIC_WEIGHT="0.60"
export E4_PROMPT_DYNAMIC_WEIGHT="0.40"
export E4_SCORE_NORM_MODE="running_zscore"
export E4_SCORE_NORM_MIN_COUNT="${E4_SCORE_NORM_MIN_COUNT:-8}"
export E4_SCORE_NORM_EPS="${E4_SCORE_NORM_EPS:-1e-6}"
export E4_SCORE_NORM_CLIP="${E4_SCORE_NORM_CLIP:-0}"

echo "============================================================"
echo "02_15_2 ModelNet-C severity=2 fused-prototype text gate"
echo "EXP_ID: ${EXP_ID}"
echo "RUNNER: ${RUNNER}"
echo "DATASET: ModelNet-C severity=2, 7 corruptions"
echo "CARRIER: 02_9_2 E4-C-A0+E1-textdist-only"
echo "FINAL CLASSIFIER/LOGITS: manual_full Point-Cache voting, unchanged"
echo "TEXT GATE MODE: ${E4_TEXT_GATE_MODE}"
echo "TEXT PROTOTYPE SCORE SCALE: ${E4_TEXT_PROTO_SCORE_SCALE}"
echo "E4_TEXT_SCORE_WEIGHT: ${E4_TEXT_SCORE_WEIGHT}"
echo "E4_SCORE_NORM_MODE: ${E4_SCORE_NORM_MODE}"
echo "E1 PROMPT CONFIG: 15 prompts/class = 10 image + 5 pointcloud"
echo "E1 PROMPT FUSION: manual_full=0.60, LLM=0.40"
echo "PROMPT_CACHE_FILE: llm/$(basename "${PROMPT_CACHE_FILE}")"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "RESULT_DIR: results/E4_distribution_guided_cache/${EXP_ID}"
echo "DRY_RUN: ${E4_DRY_RUN}"
echo "============================================================"

cd "${PC_ROOT}"

if [[ "${E4_DRY_RUN}" == "1" ]]; then
  echo "Dry run only. The experiment command would be:"
  printf 'bash %q %q %q %q %q %q %q\n' \
    "${SCRIPT_DIR}/02_run_e4_c_ulip_modelnetc_s2_common.sh" \
    "${EXP_ID}" \
    "${RUNNER}" \
    "manual_full" \
    "02-15-2: 02-9-2 carrier with E1-33 fused-prototype text gate; final Point-Cache voting unchanged; text_weight=0.10; manual_full=0.60, LLM=0.40" \
    "S2 diagnostic for reducing fused-prototype text gate strength from 0.15 to 0.10 after 02_15_1 improved over 02_14_1 but stayed below 02_9_2" \
    "${PHYSICAL_GPU}"
  exit 0
fi

bash "${SCRIPT_DIR}/02_run_e4_c_ulip_modelnetc_s2_common.sh" \
  "${EXP_ID}" \
  "${RUNNER}" \
  "manual_full" \
  "02-15-2: 02-9-2 carrier with E1-33 fused-prototype text gate; final Point-Cache voting unchanged; text_weight=0.10; manual_full=0.60, LLM=0.40" \
  "S2 diagnostic for reducing fused-prototype text gate strength from 0.15 to 0.10 after 02_15_1 improved over 02_14_1 but stayed below 02_9_2" \
  "${PHYSICAL_GPU}"
