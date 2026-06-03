# 本脚本用于复现 Point-Cache baseline 实验 14_1：ULIP-2 在 ScanObjNN-C hardest 全部 35 个损坏设置上的 Zero-shot 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/14_run_ulip2_scanobjnnc_hardest_corruptions_all35_common.sh" \
  "14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs" \
  "zs" \
  "Zero-shot" \
  "runners/zs_infer.py" \
  "global" \
  "${1:-0}"
