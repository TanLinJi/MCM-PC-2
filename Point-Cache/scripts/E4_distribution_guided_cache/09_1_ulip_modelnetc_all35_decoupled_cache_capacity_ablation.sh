#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"
RUNNER="runners/E4_distribution_guided_cache/modelnetc_all35_09_1_decoupled_cache_capacity/launch_09_1_modelnetc_all35_decoupled_cache_capacity.py"

# Edit this list to choose all35 capacity settings.
# Format: entropy_cap,gpa_cap,local_cap,neg_cap
COMBINATIONS=(
  "3,3,3,6"
  "3,3,3,7"
  "3,3,3,8"
)

# Local centers means KMeans centers saved for each local-cache sample, not cache capacity.
LOCAL_CENTERS="${E4_LOCAL_CENTERS:-3}"

# Single physical GPU id. The default matches a one-card 4090 machine.
PHYSICAL_GPU="${1:-0}"

cd "${PC_ROOT}"

read -r -a E4_PYTHON <<< "${E4_PYTHON_CMD:-python}"

parse_combo() {
  local raw="$1"
  local cleaned

  cleaned="${raw//[()]/}"
  cleaned="${cleaned// /}"
  cleaned="${cleaned//，/,}"

  IFS=',' read -r ENTROPY_CAP GPA_CAP LOCAL_CAP NEG_CAP extra <<< "${cleaned}"

  if [[ -n "${extra:-}" || -z "${ENTROPY_CAP:-}" || -z "${GPA_CAP:-}" || -z "${LOCAL_CAP:-}" || -z "${NEG_CAP:-}" ]]; then
    echo "ERROR: invalid capacity combo: ${raw}"
    echo "Expected format: 3,3,3,5 or '(3,3,3,5)'"
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
  exp_id="09_1_${setting}_all35_ulip_modelnetc_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist"
  result_dir="${PC_ROOT}/results/E4_distribution_guided_cache/${exp_id}"

  echo "============================================================"
  echo "09_1 ModelNet-C all35 decoupled cache-capacity run"
  echo "Setting: ${setting}"
  echo "EXP_ID: ${exp_id}"
  echo "Physical GPU: ${PHYSICAL_GPU}"
  echo "ENTROPY_CAP: ${ENTROPY_CAP}"
  echo "GPA_CAP: ${GPA_CAP}"
  echo "LOCAL_CAP: ${LOCAL_CAP}"
  echo "NEG_CAP: ${NEG_CAP}"
  echo "LOCAL_CENTERS: ${LOCAL_CENTERS}"
  echo "Result dir: ${result_dir}"
  echo "Python command: ${E4_PYTHON[*]}"
  echo "============================================================"

  "${E4_PYTHON[@]}" "${RUNNER}" \
    --setting "${setting}" \
    --entropy-cap "${ENTROPY_CAP}" \
    --gpa-cap "${GPA_CAP}" \
    --local-cap "${LOCAL_CAP}" \
    --neg-cap "${NEG_CAP}" \
    --local-centers "${LOCAL_CENTERS}" \
    --exp-id "${exp_id}" \
    --gpus "${PHYSICAL_GPU}" \
    --result-root "results/E4_distribution_guided_cache" \
    --text-weight "0.15" \
    --print-freq "500" \
    --python "${E4_WORKER_PYTHON:-python}"
}

echo "============================================================"
echo "09_1 queued ModelNet-C all35 capacity validation"
echo "Combinations: ${COMBINATIONS[*]}"
echo "Physical GPU: ${PHYSICAL_GPU}"
echo "============================================================"

for combo in "${COMBINATIONS[@]}"; do
  run_combo "${combo}"
done

echo "============================================================"
echo "Finished 09_1 ModelNet-C all35 capacity validation queue"
echo "Combinations: ${COMBINATIONS[*]}"
echo "============================================================"
