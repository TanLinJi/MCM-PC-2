#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"

export E4_SCORE_NORM_MODE="${E4_SCORE_NORM_MODE:-running_zscore}"
export E4_SCORE_NORM_MIN_COUNT="${E4_SCORE_NORM_MIN_COUNT:-8}"
export E4_SCORE_NORM_EPS="${E4_SCORE_NORM_EPS:-1e-6}"
export E4_SCORE_NORM_CLIP="${E4_SCORE_NORM_CLIP:-0}"

bash "${SCRIPT_DIR}/02_5_run_e4_c_a0_c2_ulip_modelnetc_s2_common.sh" \
  "02_5_ulip_modelnetc_s2_zs_global_local_e4_c_a0_c2_prior_corrected_gpa_global_logits_manual_full" \
  "runners/E4_distribution_guided_cache/run_e4_c_a0_c2_ulip_modelnetc_s2_prior_corrected_gpa_global_logits.py" \
  "manual_full" \
  "E4-C-A0-c2: z-score text-visual distribution-guided GPA-Cache with prior-corrected GPA global logits" \
  "文本-视觉类别概率分布引导的 GPA-Cache 净化；在融合 text_score 与 visual_score 前使用 running z-score 做分数尺度归一化；视觉分布累计曾被 EntropyCache 或 GPACache 接受过的历史可信样本；未被正缓存接受的样本不参与分布；同一次运行输出原公式、raw GPA global 和 prior-corrected GPA global gamma sweep" \
  "${1:-0}"
