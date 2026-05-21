# 本脚本用于复现 Point-Cache baseline 实验 14_3：ULIP-2 在 ScanObjNN-C hardest 全部 35 个损坏设置上的 Zero-shot + Global Cache + Local Cache 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/14_run_ulip2_scanobjnnc_hardest_corruptions_all35_common.sh" \
  "14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local" \
  "zs_global_local" \
  "Zero-shot + Global Cache + Local Cache" \
  "runners/model_with_hierarchical_caches.py" \
  "hierarchical" \
  "${1:-0}"
