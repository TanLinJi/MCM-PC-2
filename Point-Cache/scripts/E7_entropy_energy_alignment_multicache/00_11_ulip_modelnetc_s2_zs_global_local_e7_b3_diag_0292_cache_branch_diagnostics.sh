#!/usr/bin/env bash
set -euo pipefail

# E7-B3-Diag-02_9_2 corrupted entry:
# ULIP + ModelNet-C severity=2, cache-branch diagnostics on the current best 02_9_2 carrier.
# This run does not change prediction logic; it only records diagnostics.

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

EXP_ID="00_11_ulip_modelnetc_s2_zs_global_local_e7_b3_diag_0292_cache_branch_diagnostics"
RUNNER="runners/E7_entropy_energy_alignment_multicache/run_e7_b3_diag_0292_ulip_modelnetc_s2_cache_branch_diagnostics.py"
PHYSICAL_GPU="${1:-0}"

SHARED_PROMPT_FILE="${PC_ROOT}/results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"
MODELNETC_CLASS_FILE="${PC_ROOT}/data/modelnet_c/shape_names.txt"

if [[ ! -f "${SHARED_PROMPT_FILE}" ]]; then
  echo "ERROR: shared E1 LLM prompt cache not found:"
  echo "  ${SHARED_PROMPT_FILE}"
  echo
  echo "E7-B3-Diag-02_9_2 reuses the existing prompt cache and must not call the LLM API."
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
export WANDB_DIR="${PC_ROOT}/results/E7_entropy_energy_alignment_multicache/${EXP_ID}/wandb"
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1
export GPA_SAVE_STATS="${GPA_SAVE_STATS:-1}"

# Exact 02_9_2 method settings.
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

# B3-Diag output controls.
export E7_B3_DIAG_SAVE_SAMPLES="${E7_B3_DIAG_SAVE_SAMPLES:-1}"
export E7_B3_DIAG_SAVE_RAW_LOGITS="${E7_B3_DIAG_SAVE_RAW_LOGITS:-0}"
export E7_B3_DIAG_EPS="${E7_B3_DIAG_EPS:-1e-6}"

mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "E7-B3-Diag-02_9_2 Cache-Branch Diagnostics on ModelNet-C severity=2"
echo "EXP_ID: ${EXP_ID}"
echo "RUNNER: ${RUNNER}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "E4_TEXT_SCORE_WEIGHT: ${E4_TEXT_SCORE_WEIGHT}"
echo "E4_TEXT_DIST_PROMPT_SOURCE: ${E4_TEXT_DIST_PROMPT_SOURCE}"
echo "E4_SCORE_NORM_MODE: ${E4_SCORE_NORM_MODE}"
echo "E7_B3_DIAG_SAVE_SAMPLES: ${E7_B3_DIAG_SAVE_SAMPLES}"
echo "E7_B3_DIAG_SAVE_RAW_LOGITS: ${E7_B3_DIAG_SAVE_RAW_LOGITS}"
echo "============================================================"

python "${RUNNER}" \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "zs_global_local" \
  --baseline-method-full "E7-B3-Diag-02_9_2: cache-branch diagnostics only, no prediction change" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/E7_entropy_energy_alignment_multicache" \
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
