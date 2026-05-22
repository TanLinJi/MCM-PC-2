# 本脚本用于复现 Point-Cache baseline 实验 22_3：OpenShape 在 ModelNet-C 全部 35 个损坏设置上的 Zero-shot + Global Cache + Local Cache 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/22_run_openshape_modelnetc_corruptions_all35_common.sh" \
  "22_3_openshape_modelnetc_corruptions_all35_zs_global_local" \
  "zs_global_local" \
  "Zero-shot + Global Cache + Local Cache" \
  "runners/model_with_hierarchical_caches.py" \
  "hierarchical" \
  "${1:-0}"
