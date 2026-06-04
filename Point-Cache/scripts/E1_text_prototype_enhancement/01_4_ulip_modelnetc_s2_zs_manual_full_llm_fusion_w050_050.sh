#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/01_run_ulip_modelnetc_s2_zs_fusion_weight_ablation_common.sh" \
  "01_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w050_050" \
  "0.50" \
  "0.50" \
  "验证较高 LLM 权重是否会导致文本原型偏移" \
  "${1:-0}"
