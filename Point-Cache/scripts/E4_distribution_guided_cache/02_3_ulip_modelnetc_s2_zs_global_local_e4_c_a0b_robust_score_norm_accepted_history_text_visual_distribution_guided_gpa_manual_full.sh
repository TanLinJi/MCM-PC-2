#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"

export E4_SCORE_NORM_MODE="${E4_SCORE_NORM_MODE:-running_robust_iqr}"
export E4_SCORE_NORM_MIN_COUNT="${E4_SCORE_NORM_MIN_COUNT:-8}"
export E4_SCORE_NORM_EPS="${E4_SCORE_NORM_EPS:-1e-6}"
export E4_SCORE_NORM_CLIP="${E4_SCORE_NORM_CLIP:-0}"

bash "${SCRIPT_DIR}/02_3_run_e4_c_a0b_ulip_modelnetc_s2_common.sh" \
  "02_3_ulip_modelnetc_s2_zs_global_local_e4_c_a0b_robust_score_norm_accepted_history_text_visual_distribution_guided_gpa_manual_full" \
  "runners/E4_distribution_guided_cache/run_e4_c_a0b_ulip_modelnetc_s2_robust_score_norm_accepted_history_text_visual_distribution_guided_gpa.py" \
  "manual_full" \
  "E4-C-A0b: robust score-normalized accepted-history text-visual distribution-guided GPA-Cache" \
  "文本-视觉类别概率分布引导的 GPA-Cache 净化；在融合 text_score 与 visual_score 前使用 running robust median/IQR 做分数尺度归一化；视觉分布累计曾被 EntropyCache 或 GPACache 接受过的历史可信样本；未被正缓存接受的样本不参与分布；文本分布来自 prompt-level embeddings；沿用未满直接加入、满后替换最高熵样本、低熵门控和最终预测公式" \
  "${1:-0}"
