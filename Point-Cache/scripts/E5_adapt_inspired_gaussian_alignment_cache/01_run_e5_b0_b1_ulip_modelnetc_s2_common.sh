#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "Usage: bash 01_run_e5_b0_b1_ulip_modelnetc_s2_common.sh EXP_ID PURPOSE GPU RUNNER"
  exit 1
fi

EXP_ID="$1"
PURPOSE="$2"
PHYSICAL_GPU="$3"
RUNNER="$4"

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

SHARED_PROMPT_FILE="${PC_ROOT}/results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"
MODELNETC_CLASS_FILE="${PC_ROOT}/data/modelnet_c/shape_names.txt"

if [[ ! -f "${SHARED_PROMPT_FILE}" ]]; then
  echo "ERROR: shared E1 LLM prompt cache not found:"
  echo "  ${SHARED_PROMPT_FILE}"
  echo
  echo "E5-B0/B1 reuses the existing prompt cache and must not call the LLM API."
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

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export WANDB_DIR="${PC_ROOT}/results/E5_adapt_inspired_gaussian_alignment_cache/${EXP_ID}/wandb"
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1
export GPA_SAVE_STATS="${GPA_SAVE_STATS:-1}"

# Keep the E4-C-A0+E1-textdist-only baseline identical to the current best 02_9_2.
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

# E5-B0/B1 posterior prototype residual.
export E5_STATSBANK_CAPACITY="${E5_STATSBANK_CAPACITY:-16}"
export E5_POSTERIOR_KAPPA="${E5_POSTERIOR_KAPPA:-16}"
export E5_POSTERIOR_MIN_TOTAL="${E5_POSTERIOR_MIN_TOTAL:-8}"
export E5_POSTERIOR_MIN_CLASSES="${E5_POSTERIOR_MIN_CLASSES:-2}"
export E5_POSTERIOR_NORM_EPS="${E5_POSTERIOR_NORM_EPS:-1e-6}"
export E5_POSTERIOR_NORM_CLIP="${E5_POSTERIOR_NORM_CLIP:-3.0}"
export E5_POSTERIOR_GAMMAS="${E5_POSTERIOR_GAMMAS:-0.05,0.10,0.20}"
export E5_SAVE_SAMPLE_DIAGNOSTICS="${E5_SAVE_SAMPLE_DIAGNOSTICS:-1}"
export E5_CORRUPTIONS="${E5_CORRUPTIONS:-}"

mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "E5-B0/B1 text-prior posterior prototype residual"
echo "EXP_ID: ${EXP_ID}"
echo "RUNNER: ${RUNNER}"
echo "PURPOSE: ${PURPOSE}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "E4_TEXT_SCORE_WEIGHT: ${E4_TEXT_SCORE_WEIGHT}"
echo "E4_TEXT_DIST_PROMPT_SOURCE: ${E4_TEXT_DIST_PROMPT_SOURCE}"
echo "E4_SCORE_NORM_MODE: ${E4_SCORE_NORM_MODE}"
echo "E5_STATSBANK_CAPACITY: ${E5_STATSBANK_CAPACITY}"
echo "E5_POSTERIOR_KAPPA: ${E5_POSTERIOR_KAPPA}"
echo "E5_POSTERIOR_MIN_TOTAL: ${E5_POSTERIOR_MIN_TOTAL}"
echo "E5_POSTERIOR_MIN_CLASSES: ${E5_POSTERIOR_MIN_CLASSES}"
echo "E5_POSTERIOR_NORM_EPS: ${E5_POSTERIOR_NORM_EPS}"
echo "E5_POSTERIOR_NORM_CLIP: ${E5_POSTERIOR_NORM_CLIP}"
echo "E5_POSTERIOR_GAMMAS: ${E5_POSTERIOR_GAMMAS}"
echo "E5_SAVE_SAMPLE_DIAGNOSTICS: ${E5_SAVE_SAMPLE_DIAGNOSTICS}"
echo "E5_CORRUPTIONS: ${E5_CORRUPTIONS:-all}"
echo "============================================================"

python "${RUNNER}" \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "zs_global_local" \
  --baseline-method-full "E5-B0/B1: E4-C-A0+E1-textdist-only baseline with delayed StatsBank text-prior posterior prototype residual" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/E5_adapt_inspired_gaussian_alignment_cache" \
  --config configs \
  --wandb-log \
  --lm3d ulip \
  --cache-type "hierarchical" \
  --prompt-source "manual_full" \
  --prompt-cache-dir "results/E1_text_prototype_enhancement/shared_prompts" \
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
