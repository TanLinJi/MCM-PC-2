#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "Usage: bash 04_run_e4_c_a0_e1_textdist_only_ulip_scanobjnnc_hardest_s4_common.sh EXP_ID TEXT_WEIGHT PURPOSE GPU"
  exit 1
fi

EXP_ID="$1"
TEXT_WEIGHT="$2"
PURPOSE="$3"
PHYSICAL_GPU="$4"

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"
RUNNER="runners/E4_distribution_guided_cache/run_e4_c_a0_e1_textdist_only_ulip_scanobjnnc_hardest_s4_accepted_history_text_visual_distribution_guided_gpa.py"

SHARED_PROMPT_FILE="${PC_ROOT}/results/E1_text_prototype_enhancement/shared_prompts/sonn_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"
SCANOBJNNC_CLASS_FILE="${PC_ROOT}/data/sonn_c/shape_names.txt"

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

with class_file.open("r", encoding="utf-8") as f:
    classnames = [line.strip() for line in f if line.strip()]

if not cache_path.exists():
    print("ScanObjNN-C E1 prompt cache not found; the runner will call the configured LLM API and save it:")
    print(f"  {cache_path}")
    raise SystemExit(0)

with cache_path.open("r", encoding="utf-8") as f:
    saved = json.load(f)

prompts = saved.get("prompts", saved)
if not isinstance(prompts, dict):
    raise SystemExit(f"ERROR: invalid prompt cache format: {cache_path}")

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
    print("ScanObjNN-C E1 prompt cache is incomplete; the runner will call the configured LLM API for missing/short classes.")
    if missing:
        print("Missing classes:")
        for name in missing:
            print(f"  {name}")
    if short:
        print("Classes with too few prompts:")
        for name, count in short:
            print(f"  {name}: {count}/{required_prompt_count}")
    print(f"Cache path: {cache_path}")
    raise SystemExit(0)

print(
    "Verified shared ScanObjNN-C E1 prompt cache: "
    f"{len(classnames)} classes, at least {required_prompt_count} prompts per class."
)
PY

cd "${PC_ROOT}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export WANDB_DIR="${PC_ROOT}/results/E4_distribution_guided_cache/${EXP_ID}/wandb"
export WANDB_SILENT=true
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
export E4_TEXT_SCORE_WEIGHT="${TEXT_WEIGHT}"
export SONN_VARIANT="${SONN_VARIANT:-hardest}"

mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "E4-C-A0+E1-textdist-only ScanObjNN-C hardest severity=4"
echo "EXP_ID: ${EXP_ID}"
echo "RUNNER: ${RUNNER}"
echo "TEXT_WEIGHT: ${TEXT_WEIGHT}"
echo "PURPOSE: ${PURPOSE}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "Dataset loader: sonn_c"
echo "SONN_VARIANT: ${SONN_VARIANT}"
echo "Data root: data/sonn_c/${SONN_VARIANT}"
echo "Prompt source: manual_full"
echo "Text distribution prompt source: ${E4_TEXT_DIST_PROMPT_SOURCE}"
echo "Shared prompt cache: ${SHARED_PROMPT_FILE}"
echo "E4_SCORE_NORM_MODE: ${E4_SCORE_NORM_MODE}"
echo "============================================================"

python "${RUNNER}" \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "zs_global_local" \
  --baseline-method-full "E4-C-A0+E1-textdist-only text-weight ablation on ScanObjNN-C hardest: text_weight=${TEXT_WEIGHT}" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/E4_distribution_guided_cache" \
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
  --dataset sonn_c \
  --sonn_c_root data/sonn_c \
  --sonn_variant "${SONN_VARIANT}" \
  --cor_type add_global_4 \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --oshape-version vitg14 \
  --ulip-version ulip1 \
  --device 0 \
  --print-freq 500
