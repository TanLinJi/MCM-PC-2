#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/00_run_ulip_modelnetc_s2_zs_text_method_smoke_common.sh" \
  "00_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_smoke" \
  "manualfull_llm_dynamic_init" \
  "Smoke test with manual_full and LLM description fusion" \
  "验证原始完整手工模板文本原型与 LLM 多视角描述文本原型加权融合是否带来提升" \
  "${1:-0}"
