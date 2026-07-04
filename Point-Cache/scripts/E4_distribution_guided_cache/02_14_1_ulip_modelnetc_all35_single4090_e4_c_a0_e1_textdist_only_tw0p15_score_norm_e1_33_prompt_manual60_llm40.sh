#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

EXP_ID="02_14_1_all35_ulip_modelnetc_zs_global_local_e4_c_a0_e1_textdist_only_tw0p15_score_norm_e1_33_prompt_single4090"
RUNNER="runners/E4_distribution_guided_cache/modelnetc_all35_02_9_2/launch_02_9_2_modelnetc_all35_e1_33_prompt_single4090.py"
PHYSICAL_GPU="${1:-0}"
PROMPT_CACHE_FILE="llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json"

cd "${PC_ROOT}"

if [[ ! -f "${PROMPT_CACHE_FILE}" ]]; then
  echo "ERROR: fixed E1 prompt JSON not found:"
  echo "  ${PROMPT_CACHE_FILE}"
  exit 1
fi

read -r -a E4_PYTHON <<< "${E4_PYTHON_CMD:-python}"

echo "============================================================"
echo "02_14_1 ModelNet-C all35 single-4090 run"
echo "EXP_ID: ${EXP_ID}"
echo "RUNNER: ${RUNNER}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "DATASET: ModelNet-C full, 35 evaluations = 7 corruptions x 5 severities"
echo "CARRIER: 02_9_2 E4-C-A0+E1-textdist-only"
echo "FINAL CLASSIFIER/LOGITS: manual_full, unchanged from 02_9_2"
echo "E4_TEXT_SCORE_WEIGHT: 0.15"
echo "E4_SCORE_NORM_MODE: running_zscore"
echo "E1 PROMPT CONFIG: 15 prompts/class = 10 image + 5 pointcloud"
echo "E1 PROMPT FUSION: manual_full=0.60, LLM=0.40"
echo "PROMPT_CACHE_FILE: ${PROMPT_CACHE_FILE}"
echo "RESULT_DIR: results/E4_distribution_guided_cache/${EXP_ID}"
echo "Python command: ${E4_PYTHON[*]}"
echo "============================================================"

"${E4_PYTHON[@]}" "${RUNNER}" \
  --exp-id "${EXP_ID}" \
  --gpus "${PHYSICAL_GPU}" \
  --result-root "results/E4_distribution_guided_cache" \
  --text-weight "0.15" \
  --print-freq "500" \
  --python "${E4_WORKER_PYTHON:-python}"
