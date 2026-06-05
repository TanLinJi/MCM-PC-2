#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E3_global_prototype_alignment_cache"

bash "${SCRIPT_DIR}/00_run_ulip_modelnetc_s2_gpa_common.sh" \
  "00_2_ulip_modelnetc_s2_zs_global_local_gpa_manual_full_llm_fusion_smoke" \
  "manualfull_llm_dynamic_init" \
  "E3 smoke test: manual_full_llm_fusion with sequential Global Prototype-Alignment Cache" \
  "验证使用 E2 文本融合时，顺序式 GPA Cache 是否继续提升完整 Point-Cache" \
  "${1:-0}"
