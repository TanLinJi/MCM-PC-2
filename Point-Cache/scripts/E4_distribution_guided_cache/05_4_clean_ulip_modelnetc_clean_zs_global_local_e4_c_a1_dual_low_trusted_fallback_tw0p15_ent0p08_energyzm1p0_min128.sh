#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

EXP_ID="05_4_clean_ulip_modelnetc_clean_zs_global_local_e4_c_a1_dual_low_trusted_fallback_tw0p15_ent0p08_energyzm1p0_min128"
RUNNER="runners/E4_distribution_guided_cache/run_e4_c_a1_dual_low_trusted_fallback_ulip_modelnetc_clean.py"
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
  echo "This A1 clean run must reuse the existing prompt cache and must not call the LLM API."
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
export E4_TEXT_SCORE_WEIGHT="${E4_TEXT_SCORE_WEIGHT:-0.15}"

export E4_TRUSTED_FALLBACK="${E4_TRUSTED_FALLBACK:-dual_low_entropy_energy}"
export E4_TRUSTED_ENTROPY_THRESHOLD="${E4_TRUSTED_ENTROPY_THRESHOLD:-0.08}"
if [[ -z "${E4_TRUSTED_ENERGY_Z_THRESHOLD:-}" ]]; then
  export E4_TRUSTED_ENERGY_Z_THRESHOLD="-1.0"
fi
export E4_TRUSTED_ENERGY_MIN_COUNT="${E4_TRUSTED_ENERGY_MIN_COUNT:-128}"
export E4_TRUSTED_ENERGY_EPS="${E4_TRUSTED_ENERGY_EPS:-1e-6}"

mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "05_4 clean ModelNet-C A1 strict dual-low trusted fallback run"
echo "EXP_ID: ${EXP_ID}"
echo "RUNNER: ${RUNNER}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "DATA_FILE: data/modelnet_c/clean.h5"
echo "E4_TEXT_SCORE_WEIGHT: ${E4_TEXT_SCORE_WEIGHT}"
echo "E4_TEXT_DIST_PROMPT_SOURCE: ${E4_TEXT_DIST_PROMPT_SOURCE}"
echo "E4_SCORE_NORM_MODE: ${E4_SCORE_NORM_MODE}"
echo "E4_TRUSTED_FALLBACK: ${E4_TRUSTED_FALLBACK}"
echo "E4_TRUSTED_ENTROPY_THRESHOLD: ${E4_TRUSTED_ENTROPY_THRESHOLD}"
echo "E4_TRUSTED_ENERGY_Z_THRESHOLD: ${E4_TRUSTED_ENERGY_Z_THRESHOLD}"
echo "E4_TRUSTED_ENERGY_MIN_COUNT: ${E4_TRUSTED_ENERGY_MIN_COUNT}"
echo "============================================================"

python "${RUNNER}" \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "zs_global_local" \
  --baseline-method-full "05_4 clean: E4-C-A1 strict dual-low trusted fallback tw0.15 entropy<=0.08 energy_z<=-1.0 min_count=128 on modelnet_c clean.h5" \
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
  --dataset modelnet_c \
  --sonn_variant hardest \
  --cor_type clean \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --oshape-version vitg14 \
  --ulip-version ulip1 \
  --device 0 \
  --print-freq 500
