#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"

bash "${SCRIPT_DIR}/02_9_run_e4_c_a0_e1_textdist_only_text_weight_common.sh" \
  "02_9_1_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_textdist_only_tw0p05_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist" \
  "0.05" \
  "基于 02_8 的 E4-C-A0+E1-textdist-only 文本权重消融；最终 clip_weights 与 final logits 保持 manual_full/E4-C-A0；E1 cached descriptions 只进入 text distribution；本次设置 E4_TEXT_SCORE_WEIGHT=0.05，测试较弱文本分布项是否优于 02_8 的 0.10" \
  "${1:-0}"
