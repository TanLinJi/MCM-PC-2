#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"

bash "${SCRIPT_DIR}/01_run_e4_b_ulip_modelnetc_s2_common.sh" \
  "01_1_ulip_modelnetc_s2_zs_global_local_e4_b_text_visual_distribution_guided_gpa_manual_full" \
  "runners/E4_distribution_guided_cache/run_e4_b_ulip_modelnetc_s2_text_visual_distribution_guided_gpa.py" \
  "manual_full" \
  "E4-B: text-visual distribution-guided GPA-Cache with current EntropyCache-GPACache union visual distribution" \
  "文本-视觉类别概率分布引导的 GPA-Cache 净化；视觉分布来自当前 EntropyCache 与 GPACache 并集，不累加历史淘汰样本；文本分布来自 prompt-level embeddings；沿用未满直接加入、满后替换最高熵样本、低熵门控和最终预测公式" \
  "${1:-0}"
