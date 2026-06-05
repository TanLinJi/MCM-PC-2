#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E3_global_prototype_alignment_cache"

bash "${SCRIPT_DIR}/00_run_ulip_modelnetc_s2_gpa_common.sh" \
  "00_1_ulip_modelnetc_s2_zs_global_local_gpa_manual_full_smoke" \
  "manual_full" \
  "E3 smoke test: manual_full with sequential Global Prototype-Alignment Cache" \
  "验证不使用 E2 文本融合时，顺序式 GPA Cache 是否提升完整 Point-Cache" \
  "${1:-0}"
