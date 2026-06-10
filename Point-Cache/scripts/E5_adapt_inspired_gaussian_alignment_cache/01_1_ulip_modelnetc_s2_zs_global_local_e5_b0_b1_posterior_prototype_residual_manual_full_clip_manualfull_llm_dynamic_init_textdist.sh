#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache"

bash "${SCRIPT_DIR}/01_run_e5_b0_b1_ulip_modelnetc_s2_common.sh" \
  "01_1_ulip_modelnetc_s2_zs_global_local_e5_b0_b1_posterior_prototype_residual_manual_full_clip_manualfull_llm_dynamic_init_textdist" \
  "E5-B0/B1: 在当前最优 02_9_2 设置上新增独立 StatsBank、delayed-update、text-prior posterior prototype residual；保持原始 Point-Cache/E4 公式输出，并在同一次实验中输出多个 gamma 的 residual-enhanced 准确率" \
  "${1:-0}" \
  "runners/E5_adapt_inspired_gaussian_alignment_cache/run_e5_b0_b1_ulip_modelnetc_s2_posterior_prototype_residual.py"
