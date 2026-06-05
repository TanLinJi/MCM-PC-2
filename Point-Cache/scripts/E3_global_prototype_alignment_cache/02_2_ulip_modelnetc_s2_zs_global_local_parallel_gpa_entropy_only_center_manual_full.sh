#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E3_global_prototype_alignment_cache"

bash "${SCRIPT_DIR}/02_run_ulip_modelnetc_s2_parallel_gpa_center_source_ablation_common.sh" \
  "02_2_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_only_center_manual_full" \
  "runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_parallel_gpa_entropy_only_center.py" \
  "manual_full" \
  "E3-V2: parallel GPA Cache with Entropy-only center and manual_full" \
  "验证并列式 GPA Cache 下，仅使用 Global Entropy Cache 构造原型中心是否优于 E3-V1 和 E2 baseline" \
  "${1:-0}"
