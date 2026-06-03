#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/01_run_ulip_modelnetc_s2_zs_prompt_ablation_common.sh" \
  "01_4_ulip_modelnetc_s2_zs_fuse_manualfull_multiview_llm" \
  "manualfull_llm_dynamic_init" \
  "Zero-shot with fused manual_full and multi-view LLM descriptions" \
  "验证保留 manual_full 视觉语义锚点并融合 LLM 多视角描述是否有帮助" \
  "${1:-0}"
