#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache"

bash "${SCRIPT_DIR}/00_run_e5_a0_a1_ulip_modelnetc_s2_common.sh" \
  "00_1_ulip_modelnetc_s2_zs_global_local_e5_a0_a1_adapt_gda_diagnostics_manual_full_clip_manualfull_llm_dynamic_init_textdist" \
  "E5-A0/A1: 在当前最优 02_9_2 设置上新增独立 StatsBank、delayed-update、shared-covariance GDA standalone diagnostics；不改变 final logits，不覆盖 E4 结果" \
  "${1:-0}" \
  "runners/E5_adapt_inspired_gaussian_alignment_cache/run_e5_a0_a1_ulip_modelnetc_s2_adapt_gda_diagnostics.py"
