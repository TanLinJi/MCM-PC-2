# 本脚本用于复现 Point-Cache baseline 实验 32_3：Uni3D 在 ModelNet-C 全部 35 个损坏设置上的 Zero-shot + Global Cache + Local Cache 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/32_run_uni3d_modelnetc_corruptions_all35_common.sh" \
  "32_3_uni3d_modelnetc_corruptions_all35_zs_global_local" \
  "zs_global_local" \
  "Zero-shot + Global Cache + Local Cache" \
  "runners/model_with_hierarchical_caches.py" \
  "hierarchical" \
  "${1:-0}"
