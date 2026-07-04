#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"
RUNNER="runners/E4_distribution_guided_cache/run_e4_c_a0_e1_decoupled_cache_capacity_ulip_modelnetc_clean.py"

# Edit this list to choose clean ModelNet capacity settings.
# Format: entropy_cap,gpa_cap,local_cap,neg_cap
COMBINATIONS=(
  "2,3,3,5"
  "3,3,3,6"
)

# Local centers means KMeans centers saved for each local-cache sample, not cache capacity.
LOCAL_CENTERS="${E4_LOCAL_CENTERS:-3}"

# Single physical GPU id. The default matches a one-card 4090 machine.
PHYSICAL_GPU="${1:-0}"

SHARED_PROMPT_FILE="${PC_ROOT}/results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"
MODELNETC_CLASS_FILE="${PC_ROOT}/data/modelnet_c/shape_names.txt"
CLEAN_FILE="${PC_ROOT}/data/modelnet_c/clean.h5"

if [[ ! -f "${CLEAN_FILE}" ]]; then
  echo "ERROR: clean ModelNet-C file not found:"
  echo "  ${CLEAN_FILE}"
  exit 1
fi

if [[ ! -f "${SHARED_PROMPT_FILE}" ]]; then
  echo "ERROR: shared E1 LLM prompt cache not found:"
  echo "  ${SHARED_PROMPT_FILE}"
  echo
  echo "This clean 09_1 run must reuse the existing prompt cache and must not call the LLM API."
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

cd "${PC_ROOT}"

read -r -a E4_PYTHON <<< "${E4_PYTHON_CMD:-python}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export GPA_SAVE_STATS="${GPA_SAVE_STATS:-1}"

export E4_DIST_EPS="${E4_DIST_EPS:-1e-4}"
export E4_DIST_MIN_VAR="${E4_DIST_MIN_VAR:-1e-4}"
export E4_TEXT_DIST_EPS="${E4_TEXT_DIST_EPS:-${E4_DIST_EPS}}"
export E4_TEXT_DIST_MIN_VAR="${E4_TEXT_DIST_MIN_VAR:-${E4_DIST_MIN_VAR}}"
export E4_SCORE_NORM_MODE="${E4_SCORE_NORM_MODE:-running_zscore}"
export E4_SCORE_NORM_MIN_COUNT="${E4_SCORE_NORM_MIN_COUNT:-8}"
export E4_SCORE_NORM_EPS="${E4_SCORE_NORM_EPS:-1e-6}"
export E4_SCORE_NORM_CLIP="${E4_SCORE_NORM_CLIP:-0}"
export E4_TEXT_DIST_PROMPT_SOURCE="${E4_TEXT_DIST_PROMPT_SOURCE:-manualfull_llm_dynamic_init}"
export E4_TEXT_SCORE_WEIGHT="${E4_TEXT_SCORE_WEIGHT:-0.15}"

parse_combo() {
  local raw="$1"
  local cleaned

  cleaned="${raw//[()]/}"
  cleaned="${cleaned// /}"
  cleaned="${cleaned//，/,}"

  IFS=',' read -r ENTROPY_CAP GPA_CAP LOCAL_CAP NEG_CAP extra <<< "${cleaned}"

  if [[ -n "${extra:-}" || -z "${ENTROPY_CAP:-}" || -z "${GPA_CAP:-}" || -z "${LOCAL_CAP:-}" || -z "${NEG_CAP:-}" ]]; then
    echo "ERROR: invalid capacity combo: ${raw}"
    echo "Expected format: 3,3,3,8 or '(3,3,3,8)'"
    exit 1
  fi

  for value in "${ENTROPY_CAP}" "${GPA_CAP}" "${LOCAL_CAP}" "${NEG_CAP}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ || "${value}" -le 0 ]]; then
      echo "ERROR: capacity values must be positive integers: ${raw}"
      exit 1
    fi
  done
}

run_combo() {
  local combo="$1"
  local setting
  local exp_id
  local result_dir

  parse_combo "${combo}"
  setting="e${ENTROPY_CAP}_g${GPA_CAP}_l${LOCAL_CAP}_n${NEG_CAP}"
  exp_id="09_1_${setting}_clean_ulip_modelnetc_clean_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist"
  result_dir="${PC_ROOT}/results/E4_distribution_guided_cache/${exp_id}"

  export E4_ENTROPY_CAP="${ENTROPY_CAP}"
  export E4_GPA_CAP="${GPA_CAP}"
  export E4_LOCAL_CAP="${LOCAL_CAP}"
  export E4_NEG_CAP="${NEG_CAP}"
  export E4_LOCAL_CENTERS="${LOCAL_CENTERS}"

  echo "============================================================"
  echo "09_1 clean ModelNet-C decoupled cache-capacity run"
  echo "Setting: ${setting}"
  echo "EXP_ID: ${exp_id}"
  echo "Physical GPU: ${PHYSICAL_GPU}"
  echo "DATA_FILE: data/modelnet_c/clean.h5"
  echo "ENTROPY_CAP: ${ENTROPY_CAP}"
  echo "GPA_CAP: ${GPA_CAP}"
  echo "LOCAL_CAP: ${LOCAL_CAP}"
  echo "NEG_CAP: ${NEG_CAP}"
  echo "LOCAL_CENTERS: ${LOCAL_CENTERS}"
  echo "E4_TEXT_SCORE_WEIGHT: ${E4_TEXT_SCORE_WEIGHT}"
  echo "E4_TEXT_DIST_PROMPT_SOURCE: ${E4_TEXT_DIST_PROMPT_SOURCE}"
  echo "E4_SCORE_NORM_MODE: ${E4_SCORE_NORM_MODE}"
  echo "Result dir: ${result_dir}"
  echo "Python command: ${E4_PYTHON[*]}"
  echo "============================================================"

  "${E4_PYTHON[@]}" "${RUNNER}" \
    --baseline-exp-id "${exp_id}" \
    --baseline-method "zs_global_local" \
    --baseline-method-full "09_1 clean ${setting}: decoupled cache-capacity; entropy_cap=${ENTROPY_CAP}; gpa_cap=${GPA_CAP}; local_cap=${LOCAL_CAP}; neg_cap=${NEG_CAP}; local_centers=${LOCAL_CENTERS}; text_weight=0.15; score_norm=running_zscore" \
    --baseline-gpu "${PHYSICAL_GPU}" \
    --baseline-result-root "results/E4_distribution_guided_cache" \
    --config configs \
    --lm3d ulip \
    --cache-type "hierarchical" \
    --prompt-source "manual_full" \
    --prompt-cache-dir "results/E1_text_prototype_enhancement/shared_prompts" \
    --prompt-cache-file "" \
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
    --cor_type clean \
    --npoints 1024 \
    --sim2real_type so_obj_only_9 \
    --oshape-version vitg14 \
    --ulip-version ulip1 \
    --device 0 \
    --print-freq 500
}

echo "============================================================"
echo "09_1 queued clean ModelNet-C capacity validation"
echo "Combinations: ${COMBINATIONS[*]}"
echo "Physical GPU: ${PHYSICAL_GPU}"
echo "============================================================"

for combo in "${COMBINATIONS[@]}"; do
  run_combo "${combo}"
done

echo "============================================================"
echo "Finished 09_1 clean ModelNet-C capacity validation queue"
echo "Combinations: ${COMBINATIONS[*]}"
echo "============================================================"
