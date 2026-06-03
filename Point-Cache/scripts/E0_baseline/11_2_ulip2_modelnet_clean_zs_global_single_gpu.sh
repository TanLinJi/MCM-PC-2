# 本脚本用于复现 Point-Cache baseline 实验 11_2：ULIP-2 在 ModelNet clean 上的 Zero-shot + Global Cache 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/11_run_ulip2_modelnet_clean_common.sh" \
  "11_2_ulip2_modelnet_clean_zs_global" \
  "zs_global" \
  "Zero-shot + Global Cache" \
  "runners/model_with_global_cache.py" \
  "global" \
  "${1:-0}"
