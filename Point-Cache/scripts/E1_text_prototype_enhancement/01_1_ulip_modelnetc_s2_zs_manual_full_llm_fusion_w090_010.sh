#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/01_run_ulip_modelnetc_s2_zs_fusion_weight_ablation_common.sh" \
  "01_1_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w090_010" \
  "0.90" \
  "0.10" \
  "验证更保守地引入 LLM 描述是否比 0.75:0.25 更稳定" \
  "${1:-0}"
