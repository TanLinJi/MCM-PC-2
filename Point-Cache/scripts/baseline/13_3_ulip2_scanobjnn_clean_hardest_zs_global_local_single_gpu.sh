# 本脚本用于复现 Point-Cache baseline 实验 13_3：ULIP-2 在 ScanObjNN clean hardest 上的 Zero-shot + Global Cache + Local Cache 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/13_run_ulip2_scanobjnn_clean_hardest_common.sh" \
  "13_3_ulip2_scanobjnn_clean_hardest_zs_global_local" \
  "zs_global_local" \
  "Zero-shot + Global Cache + Local Cache" \
  "runners/model_with_hierarchical_caches.py" \
  "hierarchical" \
  "${1:-0}"
