#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/01_run_ulip_modelnetc_s2_zs_fusion_weight_ablation_common.sh" \
  "01_5_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020" \
  "0.80" \
  "0.20" \
  "补充 manual_full:LLM = 0.80:0.20 的中间权重消融" \
  "${1:-0}"
