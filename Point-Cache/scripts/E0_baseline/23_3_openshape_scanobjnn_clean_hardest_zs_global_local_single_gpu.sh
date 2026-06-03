# 本脚本用于复现 Point-Cache baseline 实验 23_3：OpenShape 在 ScanObjNN clean hardest 上的 Zero-shot + Global Cache + Local Cache 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/23_run_openshape_scanobjnn_clean_hardest_common.sh" \
  "23_3_openshape_scanobjnn_clean_hardest_zs_global_local" \
  "zs_global_local" \
  "Zero-shot + Global Cache + Local Cache" \
  "runners/model_with_hierarchical_caches.py" \
  "hierarchical" \
  "${1:-0}"
