#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/00_run_ulip_modelnetc_s2_zs_text_method_smoke_common.sh" \
  "00_3_ulip_modelnetc_s2_zs_llm_only_smoke" \
  "llm_dynamic_init" \
  "Smoke test with LLM-only category descriptions" \
  "验证只使用 LLM 生成的类别级多视角描述是否可以替代原始手工模板" \
  "${1:-0}"
