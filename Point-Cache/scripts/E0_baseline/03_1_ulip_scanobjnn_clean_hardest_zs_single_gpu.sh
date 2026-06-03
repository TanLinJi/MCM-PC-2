# 本脚本用于复现 Point-Cache baseline 实验 03_1：ULIP 在 ScanObjNN clean hardest 上的 Zero-shot 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/03_run_ulip_scanobjnn_clean_hardest_common.sh" \
  "03_1_ulip_scanobjnn_clean_hardest_zs" \
  "zs" \
  "Zero-shot" \
  "runners/zs_infer.py" \
  "global" \
  "${1:-0}"
