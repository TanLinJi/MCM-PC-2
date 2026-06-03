# 本脚本用于复现 Point-Cache baseline 实验 23_1：OpenShape 在 ScanObjNN clean hardest 上的 Zero-shot 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/23_run_openshape_scanobjnn_clean_hardest_common.sh" \
  "23_1_openshape_scanobjnn_clean_hardest_zs" \
  "zs" \
  "Zero-shot" \
  "runners/zs_infer.py" \
  "global" \
  "${1:-0}"
