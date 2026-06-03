# 本脚本用于复现 Point-Cache baseline 实验 21_1：OpenShape 在 ModelNet clean 上的 Zero-shot 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/21_run_openshape_modelnet_clean_common.sh" \
  "21_1_openshape_modelnet_clean_zs" \
  "zs" \
  "Zero-shot" \
  "runners/zs_infer.py" \
  "global" \
  "${1:-0}"
