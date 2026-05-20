# 本脚本用于复现 Point-Cache baseline 实验 13_2：ULIP-2 在 ScanObjNN clean hardest 上的 Zero-shot + Global Cache 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/13_run_ulip2_scanobjnn_clean_hardest_common.sh" \
  "13_2_ulip2_scanobjnn_clean_hardest_zs_global" \
  "zs_global" \
  "Zero-shot + Global Cache" \
  "runners/model_with_global_cache.py" \
  "global" \
  "${1:-0}"
