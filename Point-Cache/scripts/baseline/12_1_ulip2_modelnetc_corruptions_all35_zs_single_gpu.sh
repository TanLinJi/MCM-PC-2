# 本脚本用于复现 Point-Cache baseline 实验 12_1：ULIP-2 在 ModelNet-C 全部 35 个损坏设置上的 Zero-shot 推理。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/baseline"

bash "${SCRIPT_DIR}/12_run_ulip2_modelnetc_corruptions_all35_common.sh" \
  "12_1_ulip2_modelnetc_corruptions_all35_zs" \
  "zs" \
  "Zero-shot" \
  "runners/zs_infer.py" \
  "global" \
  "${1:-0}"
