#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/01_run_ulip_modelnetc_s2_zs_prompt_ablation_common.sh" \
  "01_2_ulip_modelnetc_s2_zs_dynamic_multiview_llm_descriptions" \
  "llm_dynamic_init" \
  "Zero-shot with dynamic multi-view LLM descriptions" \
  "验证 LLM 生成的 2D 视觉语义 + 3D 点云几何多视角类别描述是否有帮助" \
  "${1:-0}"
