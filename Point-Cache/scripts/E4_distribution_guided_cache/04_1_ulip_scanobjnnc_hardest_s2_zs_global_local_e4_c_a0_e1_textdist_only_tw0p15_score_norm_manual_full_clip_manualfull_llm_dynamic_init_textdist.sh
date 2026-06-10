#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"

bash "${SCRIPT_DIR}/04_run_e4_c_a0_e1_textdist_only_ulip_scanobjnnc_hardest_s2_common.sh" \
  "04_1_ulip_scanobjnnc_hardest_s2_zs_global_local_e4_c_a0_e1_textdist_only_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist" \
  "0.15" \
  "ScanObjNN-C hardest severity=2 复用 02_9_2 的 E4-C-A0+E1-textdist-only 设置；final clip_weights 与 final logits 保持 manual_full/E4-C-A0；E1 cached descriptions 只进入 text distribution；E4_TEXT_SCORE_WEIGHT=0.15；使用 sonn_c loader 读取 data/sonn_c/hardest" \
  "${1:-0}"
