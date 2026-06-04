#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache"

bash "${SCRIPT_DIR}/00_run_ulip_modelnetc_s2_cache_transfer_common.sh" \
  "00_1_ulip_modelnetc_s2_zs_global_manual_full_smoke" \
  "zs_global" \
  "manual_full" \
  "E2 smoke test: manual_full text prototype with global cache" \
  "验证 Point-Cache 原始完整手工模板在 global cache 设置下的 severity=2 结果" \
  "${1:-0}"
