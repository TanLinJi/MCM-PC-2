# 本脚本用于复现 Point-Cache baseline 实验 22_2：OpenShape 在 ModelNet-C 全部 35 个损坏设置上的 Zero-shot + Global Cache 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/22_run_openshape_modelnetc_corruptions_all35_common.sh" \
  "22_2_openshape_modelnetc_corruptions_all35_zs_global" \
  "zs_global" \
  "Zero-shot + Global Cache" \
  "runners/model_with_global_cache.py" \
  "global" \
  "${1:-0}"
