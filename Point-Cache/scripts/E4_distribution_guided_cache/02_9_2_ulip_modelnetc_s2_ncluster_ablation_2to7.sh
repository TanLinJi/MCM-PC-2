#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"
RESULT_ROOT="${PC_ROOT}/results/E4_distribution_guided_cache"
RUNNER="runners/E4_distribution_guided_cache/run_e4_c_a0_e1_textdist_only_ulip_modelnetc_s2_accepted_history_text_visual_distribution_guided_gpa.py"

PHYSICAL_GPU="${1:-0}"
DRY_RUN="${2:-0}"
if [[ "${DRY_RUN}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
PROMPT_CACHE_DIR="results/E1_text_prototype_enhancement/shared_prompts"
PROMPT_CACHE_FILE="modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"
PROMPT_CACHE_PATH="${PC_ROOT}/${PROMPT_CACHE_DIR}/${PROMPT_CACHE_FILE}"
MODELNETC_CLASS_FILE="${PC_ROOT}/data/modelnet_c/shape_names.txt"

mkdir -p "${RESULT_ROOT}"
cd "${PC_ROOT}"

python - "${PROMPT_CACHE_PATH}" "${MODELNETC_CLASS_FILE}" <<'PY'
import json
import sys
from pathlib import Path

cache_path = Path(sys.argv[1])
class_file = Path(sys.argv[2])
required_prompt_count = 10

if not cache_path.exists():
    raise SystemExit(f"ERROR: prompt cache not found: {cache_path}")
if not class_file.exists():
    raise SystemExit(f"ERROR: class file not found: {class_file}")

with cache_path.open("r", encoding="utf-8") as f:
    saved = json.load(f)

expected = {
    "dataset_name": "modelnet_c",
    "dynamic_prompt_count": required_prompt_count,
    "llm_prompt_mode": "multiview_2d3d",
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
    elif len(class_prompts) < required_prompt_count:
        short.append((clean_name, len(class_prompts)))

if missing or short:
    print("ERROR: prompt cache is incomplete.")
    if missing:
        print("Missing classes:")
        for name in missing:
            print(f"  {name}")
    if short:
        print("Classes with too few prompts:")
        for name, count in short:
            print(f"  {name}: {count}/{required_prompt_count}")
    raise SystemExit(1)

print(
    "Verified prompt cache: "
    f"{len(classnames)} classes, exactly {required_prompt_count} prompts per class."
)
PY

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1
export GPA_SAVE_STATS="${GPA_SAVE_STATS:-1}"

export E4_TEXT_GATE_MODE="${E4_TEXT_GATE_MODE:-distribution}"
export E4_TEXT_PROTO_SCORE_SCALE="${E4_TEXT_PROTO_SCORE_SCALE:-1.0}"
export E4_DIST_EPS="${E4_DIST_EPS:-1e-4}"
export E4_DIST_MIN_VAR="${E4_DIST_MIN_VAR:-1e-4}"
export E4_TEXT_DIST_EPS="${E4_TEXT_DIST_EPS:-${E4_DIST_EPS}}"
export E4_TEXT_DIST_MIN_VAR="${E4_TEXT_DIST_MIN_VAR:-${E4_DIST_MIN_VAR}}"
export E4_TEXT_DIST_PROMPT_SOURCE="manualfull_llm_dynamic_init"
export E4_TEXT_SCORE_WEIGHT="0.15"
export E4_SCORE_NORM_MODE="running_zscore"
export E4_SCORE_NORM_MIN_COUNT="${E4_SCORE_NORM_MIN_COUNT:-8}"
export E4_SCORE_NORM_EPS="${E4_SCORE_NORM_EPS:-1e-6}"
export E4_SCORE_NORM_CLIP="${E4_SCORE_NORM_CLIP:-0}"
export E4_PROMPT_CACHE_DIR="${PROMPT_CACHE_DIR}"
export E4_PROMPT_CACHE_FILE="${PROMPT_CACHE_FILE}"
export E4_LLM_PROMPT_MODE="multiview_2d3d"
export E4_DYNAMIC_PROMPT_COUNT="10"
export E4_PROMPT_STATIC_WEIGHT="0.75"
export E4_PROMPT_DYNAMIC_WEIGHT="0.25"

run_one_cluster() {
  local n_cluster="$1"
  local exp_id="02_9_2_ncluster${n_cluster}_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_textdist_only_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist"
  local method_full="02-9-2 n_cluster ablation on ModelNet-C severity=2; same carrier as 02-9-2; manual_full final classifier/logits; E1 descriptions only for text distribution; text_weight=0.15; running_zscore score normalization; k_shot=3; alpha=4.0; beta=3.0; n_cluster=${n_cluster}"
  local wandb_dir="${RESULT_ROOT}/${exp_id}/wandb"

  mkdir -p "${wandb_dir}"
  export WANDB_DIR="${wandb_dir}"

  echo "============================================================"
  echo "ModelNet-C severity=2 n_cluster ablation"
  echo "n_cluster: ${n_cluster}"
  echo "EXP_ID: ${exp_id}"
  echo "RESULT_DIR: ${RESULT_ROOT}/${exp_id}"
  echo "WANDB_DIR: ${WANDB_DIR}"
  echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
  echo "E4_TEXT_GATE_MODE: ${E4_TEXT_GATE_MODE}"
  echo "E4_TEXT_SCORE_WEIGHT: ${E4_TEXT_SCORE_WEIGHT}"
  echo "E4_SCORE_NORM_MODE: ${E4_SCORE_NORM_MODE}"
  echo "============================================================"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${PYTHON_BIN}" "${RUNNER}" \
      --baseline-exp-id "${exp_id}" \
      --baseline-method "zs_global_local" \
      --baseline-method-full "${method_full}" \
      --baseline-gpu "${PHYSICAL_GPU}" \
      --baseline-result-root "results/E4_distribution_guided_cache" \
      --config configs \
      --wandb-log \
      --lm3d ulip \
      --cache-type hierarchical \
      --prompt-source manual_full \
      --prompt-cache-dir "${PROMPT_CACHE_DIR}" \
      --prompt-cache-file "${PROMPT_CACHE_FILE}" \
      --llm-provider deepseek \
      --llm-model deepseek-v4-pro \
      --llm-api-key-file llm/secrets/llm_api_key.txt \
      --llm-api-base-url https://api.deepseek.com/chat/completions \
      --llm-temperature 0.3 \
      --llm-prompt-mode multiview_2d3d \
      --dynamic-prompt-count 10 \
      --prompt-static-weight 0.75 \
      --prompt-dynamic-weight 0.25 \
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
      --print-freq 500 \
      --seed 1 \
      --k_shot 3 \
      --alpha 4.0 \
      --beta 3.0 \
      --n_cluster "${n_cluster}"
    echo
    return
  fi

  "${PYTHON_BIN}" "${RUNNER}" \
    --baseline-exp-id "${exp_id}" \
    --baseline-method "zs_global_local" \
    --baseline-method-full "${method_full}" \
    --baseline-gpu "${PHYSICAL_GPU}" \
    --baseline-result-root "results/E4_distribution_guided_cache" \
    --config configs \
    --wandb-log \
    --lm3d ulip \
    --cache-type hierarchical \
    --prompt-source manual_full \
    --prompt-cache-dir "${PROMPT_CACHE_DIR}" \
    --prompt-cache-file "${PROMPT_CACHE_FILE}" \
    --llm-provider deepseek \
    --llm-model deepseek-v4-pro \
    --llm-api-key-file llm/secrets/llm_api_key.txt \
    --llm-api-base-url https://api.deepseek.com/chat/completions \
    --llm-temperature 0.3 \
    --llm-prompt-mode multiview_2d3d \
    --dynamic-prompt-count 10 \
    --prompt-static-weight 0.75 \
    --prompt-dynamic-weight 0.25 \
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
    --print-freq 500 \
    --seed 1 \
    --k_shot 3 \
    --alpha 4.0 \
    --beta 3.0 \
    --n_cluster "${n_cluster}"
}

read -r -a CLUSTERS <<< "${CLUSTERS_OVERRIDE:-2 3 4 5 6 7}"
for n_cluster in "${CLUSTERS[@]}"; do
  run_one_cluster "${n_cluster}"
done

echo "============================================================"
echo "Finished n_cluster sweep: ${CLUSTERS[*]}"
echo "Results are under: ${RESULT_ROOT}"
echo "============================================================"
