#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

DYNAMIC_PROMPT_COUNT=15 \
LLM_PROMPT_MODE=multiview_2d3d_1to2 \
bash "${SCRIPT_DIR}/01_run_ulip_modelnetc_s2_zs_fusion_weight_ablation_common.sh" \
  "02_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020_p15_2d3d_1to2" \
  "0.80" \
  "0.20" \
  "15 条 LLM 描述，2D:3D = 1:2，manual_full:LLM = 0.80:0.20" \
  "${1:-0}"
