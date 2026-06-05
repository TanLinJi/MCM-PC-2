#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E3_global_prototype_alignment_cache"

bash "${SCRIPT_DIR}/01_run_ulip_modelnetc_s2_gpa_center_source_ablation_common.sh" \
  "01_1_ulip_modelnetc_s2_zs_global_local_gpa_entropy_only_center_manual_full" \
  "runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_gpa_entropy_only_center.py" \
  "manual_full" \
  "E3 center-source ablation: sequential GPA Cache with Entropy-only center and manual_full" \
  "固定顺序式 GPA Cache，只将原型中心来源改为 Global Entropy Cache，验证 Entropy-only center 是否优于 GPA-only center" \
  "${1:-0}"
