#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"

export E4_SCORE_NORM_MODE="${E4_SCORE_NORM_MODE:-running_zscore}"
export E4_SCORE_NORM_MIN_COUNT="${E4_SCORE_NORM_MIN_COUNT:-8}"
export E4_SCORE_NORM_EPS="${E4_SCORE_NORM_EPS:-1e-6}"
export E4_SCORE_NORM_CLIP="${E4_SCORE_NORM_CLIP:-0}"
export E4_TEXT_DIST_PROMPT_SOURCE="${E4_TEXT_DIST_PROMPT_SOURCE:-manualfull_llm_dynamic_init}"
export E4_TEXT_SCORE_WEIGHT="${E4_TEXT_SCORE_WEIGHT:-0.15}"

bash "${SCRIPT_DIR}/02_run_e4_c_ulip_modelnetc_s2_common.sh" \
  "02_11_1_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_protective_soft_rejection_gpa_tw0p15_score_norm_manualfull_llm_dynamic_init_textdist" \
  "runners/E4_distribution_guided_cache/run_e4_c_a0_e1_protective_soft_rejection_gpa_ulip_modelnetc_s2.py" \
  "manual_full" \
  "E4-02_11 protective soft-rejection GPA replacement" \
  "基于 02_9_2 的最小改动实验；最终 clip_weights 与 final logits 保持 manual_full/E4-C-A0；E1 cached descriptions 只进入 text distribution；E4_TEXT_SCORE_WEIGHT=0.15；GPA 满缓存替换从 joint_score_new >= joint_score_old 放宽为保护性软拒绝：若低熵且 joint_score_new 不低于当前类别 GPA cache 最小 joint_score，则允许替换最高熵样本" \
  "${1:-0}"
