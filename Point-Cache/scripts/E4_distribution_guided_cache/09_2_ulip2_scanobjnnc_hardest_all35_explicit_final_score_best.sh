#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"
COMMON_SCRIPT="${SCRIPT_DIR}/09_run_e4_c_a0_e1_explicit_final_score_scanobjnnc_all35_backbone_common.sh"
BACKBONE_KEY="ulip2"

# 容量单位：每类最多缓存多少个样本。
# 格式：entropy_cap,gpa_cap,local_cap,neg_cap
COMBINATIONS=(
  "3,3,3,6"
)

# 对应公式：y = y_zs + alpha_g * y_g + alpha_l * y_l - alpha_n * y_n
# 格式：alpha_g,alpha_l,alpha_n，也可以写成 name:alpha_g,alpha_l,alpha_n
FINAL_SCORE_WEIGHTS=(
  "modelnetc_best:4.4,3.9,0.19"
)

export E4_FINAL_SCORE_SWEEP_TAG="${E4_FINAL_SCORE_SWEEP_TAG:-modelnetc_best_w4p4_3p9_0p19}"
export SONN_VARIANT="${SONN_VARIANT:-hardest}"
export E4_LOCAL_CENTERS="${E4_LOCAL_CENTERS:-3}"

PHYSICAL_GPU="${1:-0}"

join_by_semicolon() {
  local IFS=';'
  echo "$*"
}

export E4_FINAL_SCORE_WEIGHTS="$(join_by_semicolon "${FINAL_SCORE_WEIGHTS[@]}")"

parse_combo() {
  local raw="$1"
  local cleaned

  cleaned="${raw//[()]/}"
  cleaned="${cleaned// /}"
  cleaned="${cleaned//，/,}"

  IFS=',' read -r ENTROPY_CAP GPA_CAP LOCAL_CAP NEG_CAP extra <<< "${cleaned}"

  if [[ -n "${extra:-}" || -z "${ENTROPY_CAP:-}" || -z "${GPA_CAP:-}" || -z "${LOCAL_CAP:-}" || -z "${NEG_CAP:-}" ]]; then
    echo "ERROR: invalid capacity combo: ${raw}"
    echo "Expected format: 3,3,3,6 or '(3,3,3,6)'"
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
  echo "09_2 ULIP-2 ScanObjNN-C all35 explicit final-score evaluation"
  echo "GPU: ${PHYSICAL_GPU}"
  echo "SONN_VARIANT: ${SONN_VARIANT}"
  echo "Setting: ${setting}"
  echo "E4_FINAL_SCORE_WEIGHTS: ${E4_FINAL_SCORE_WEIGHTS}"
  echo "Result dir:"
  echo "/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_${setting}_${E4_FINAL_SCORE_SWEEP_TAG}_ulip2_scanobjnnc_${SONN_VARIANT}_all35_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist"
  echo "================================================------------"

  bash "${COMMON_SCRIPT}" \
    "${BACKBONE_KEY}" \
    "${setting}" \
    "${ENTROPY_CAP}" \
    "${GPA_CAP}" \
    "${LOCAL_CAP}" \
    "${NEG_CAP}" \
    "${PHYSICAL_GPU}"
}

for combo in "${COMBINATIONS[@]}"; do
  run_combo "${combo}"
done
