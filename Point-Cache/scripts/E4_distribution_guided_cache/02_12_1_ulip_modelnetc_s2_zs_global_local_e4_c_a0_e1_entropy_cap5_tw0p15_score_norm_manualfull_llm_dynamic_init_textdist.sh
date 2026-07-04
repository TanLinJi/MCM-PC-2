#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"

export E4_SCORE_NORM_MODE="${E4_SCORE_NORM_MODE:-running_zscore}"
export E4_SCORE_NORM_MIN_COUNT="${E4_SCORE_NORM_MIN_COUNT:-8}"
export E4_SCORE_NORM_EPS="${E4_SCORE_NORM_EPS:-1e-6}"
export E4_SCORE_NORM_CLIP="${E4_SCORE_NORM_CLIP:-0}"
export E4_TEXT_DIST_PROMPT_SOURCE="${E4_TEXT_DIST_PROMPT_SOURCE:-manualfull_llm_dynamic_init}"
export E4_TEXT_SCORE_WEIGHT="${E4_TEXT_SCORE_WEIGHT:-0.15}"
export E4_ENTROPY_CACHE_CAPACITY="${E4_ENTROPY_CACHE_CAPACITY:-5}"

bash "${SCRIPT_DIR}/02_run_e4_c_ulip_modelnetc_s2_common.sh" \
  "02_12_1_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_entropy_cap5_tw0p15_score_norm_manualfull_llm_dynamic_init_textdist" \
  "runners/E4_distribution_guided_cache/run_e4_c_a0_e1_entropy_cap5_ulip_modelnetc_s2.py" \
  "manual_full" \
  "E4-02_12 entropy cache capacity 5" \
  "基于 02_11 的最小改动实验；全局熵缓存容量从 3 扩大到 5（E4_ENTROPY_CACHE_CAPACITY=5）；GPA 缓存容量保持 3；GPA 替换仍用保护性软拒绝；最终 clip_weights 与 final logits 保持 manual_full/E4-C-A0；E1 cached descriptions 只进入 text distribution；E4_TEXT_SCORE_WEIGHT=0.15" \
  "${1:-0}"
