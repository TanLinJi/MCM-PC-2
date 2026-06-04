#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/01_run_ulip_modelnetc_s2_zs_fusion_weight_ablation_common.sh" \
  "01_2_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w085_015" \
  "0.85" \
  "0.15" \
  "验证中等保守的 LLM 融合权重是否优于当前默认权重" \
  "${1:-0}"
