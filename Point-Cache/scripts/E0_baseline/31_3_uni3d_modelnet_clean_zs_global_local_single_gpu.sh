# 本脚本用于复现 Point-Cache baseline 实验 31_3：Uni3D 在 ModelNet clean 上的 Zero-shot + Global Cache + Local Cache 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/31_run_uni3d_modelnet_clean_common.sh" \
  "31_3_uni3d_modelnet_clean_zs_global_local" \
  "zs_global_local" \
  "Zero-shot + Global Cache + Local Cache" \
  "runners/model_with_hierarchical_caches.py" \
  "hierarchical" \
  "${1:-0}"
