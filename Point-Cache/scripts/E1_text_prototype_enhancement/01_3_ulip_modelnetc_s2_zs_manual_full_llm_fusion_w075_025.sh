#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/01_run_ulip_modelnetc_s2_zs_fusion_weight_ablation_common.sh" \
  "01_3_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w075_025" \
  "0.75" \
  "0.25" \
  "复现实验 smoke test 中取得正结果的默认融合权重" \
  "${1:-0}"
