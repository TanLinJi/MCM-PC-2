#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/00_run_ulip_modelnetc_s2_zs_text_method_smoke_common.sh" \
  "00_1_ulip_modelnetc_s2_zs_manual_full_smoke" \
  "manual_full" \
  "Smoke test with original full manual templates" \
  "验证 E1 新接口不破坏 Point-Cache 原始完整手工模板 baseline" \
  "${1:-0}"
