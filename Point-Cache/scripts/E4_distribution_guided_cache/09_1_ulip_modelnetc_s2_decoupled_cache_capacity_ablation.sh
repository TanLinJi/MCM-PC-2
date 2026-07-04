#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"
COMMON_SCRIPT="${SCRIPT_DIR}/09_run_e4_c_a0_e1_decoupled_cache_capacity_common.sh"

# 手动修改这里即可切换容量设置。容量单位：每类最多缓存多少个样本。
# 格式：entropy_cap,gpa_cap,local_cap,neg_cap
COMBINATIONS=(
  "3,3,3,9"
  "3,3,3,10"
)

# local centers 是每个进入局部缓存的样本保存多少个 KMeans 中心，不是缓存容量。
export E4_LOCAL_CENTERS="${E4_LOCAL_CENTERS:-3}"

PHYSICAL_GPU="${1:-0}"

parse_combo() {
  local raw="$1"
  local cleaned

  cleaned="${raw//[()]/}"
  cleaned="${cleaned// /}"
  cleaned="${cleaned//，/,}"

  IFS=',' read -r ENTROPY_CAP GPA_CAP LOCAL_CAP NEG_CAP extra <<< "${cleaned}"

  if [[ -n "${extra:-}" || -z "${ENTROPY_CAP:-}" || -z "${GPA_CAP:-}" || -z "${LOCAL_CAP:-}" || -z "${NEG_CAP:-}" ]]; then
    echo "ERROR: invalid capacity combo: ${raw}"
    echo "Expected format: 2,4,4,3 or '(2,4,4,3)'"
    exit 1
  fi

  for value in "${ENTROPY_CAP}" "${GPA_CAP}" "${LOCAL_CAP}" "${NEG_CAP}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
      echo "ERROR: capacity values must be non-negative integers: ${raw}"
      exit 1
    fi
  done
}

run_combo() {
  local combo="$1"
  local setting

  parse_combo "${combo}"
  setting="e${ENTROPY_CAP}_g${GPA_CAP}_l${LOCAL_CAP}_n${NEG_CAP}"

  echo "================================================------------"
  echo "09_1 ModelNet-C severity=2 decoupled cache-capacity ablation"
  echo "GPU: ${PHYSICAL_GPU}"
  echo "Setting: ${setting}"
  echo "ENTROPY_CAP: ${ENTROPY_CAP}"
  echo "GPA_CAP: ${GPA_CAP}"
  echo "LOCAL_CAP: ${LOCAL_CAP}"
  echo "NEG_CAP: ${NEG_CAP}"
  echo "E4_LOCAL_CENTERS: ${E4_LOCAL_CENTERS}"
  echo "Result dir:"
  echo "/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_1_${setting}_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist"
  echo "================================================------------"

  bash "${COMMON_SCRIPT}" \
    "${setting}" \
    "${ENTROPY_CAP}" \
    "${GPA_CAP}" \
    "${LOCAL_CAP}" \
    "${NEG_CAP}" \
    "${PHYSICAL_GPU}"
}

echo "============================================================"
echo "09_1 queued decoupled cache-capacity ablation"
echo "GPU: ${PHYSICAL_GPU}"
echo "Combinations: ${COMBINATIONS[*]}"
echo "============================================================"

for combo in "${COMBINATIONS[@]}"; do
  run_combo "${combo}"
done

echo "============================================================"
echo "Finished 09_1 decoupled cache-capacity ablation queue"
echo "Combinations: ${COMBINATIONS[*]}"
echo "============================================================"
