#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache"

bash "${SCRIPT_DIR}/00_run_ulip_modelnetc_s2_cache_transfer_common.sh" \
  "00_2_ulip_modelnetc_s2_zs_global_manual_full_llm_fusion_smoke" \
  "zs_global" \
  "manualfull_llm_dynamic_init" \
  "E2 smoke test: manual_full_llm_fusion text prototype with global cache" \
  "验证 E1 文本原型融合收益能否传递到 Point-Cache global cache 设置" \
  "${1:-0}"
