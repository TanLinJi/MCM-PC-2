#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/00_run_ulip_modelnetc_s2_zs_text_method_smoke_common.sh" \
  "00_2_ulip_modelnetc_s2_zs_manual_3d_smoke" \
  "manual_3d" \
  "Smoke test with 3D-only manual templates" \
  "验证删除 2D 图像风格模板、只保留 3D 几何相关模板是否会导致性能下降" \
  "${1:-0}"
