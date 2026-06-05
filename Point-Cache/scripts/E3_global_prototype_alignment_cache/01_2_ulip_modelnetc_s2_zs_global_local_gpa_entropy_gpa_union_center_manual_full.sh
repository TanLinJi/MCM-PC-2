#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E3_global_prototype_alignment_cache"

bash "${SCRIPT_DIR}/01_run_ulip_modelnetc_s2_gpa_center_source_ablation_common.sh" \
  "01_2_ulip_modelnetc_s2_zs_global_local_gpa_entropy_gpa_union_center_manual_full" \
  "runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_gpa_entropy_gpa_union_center.py" \
  "manual_full" \
  "E3 center-source ablation: sequential GPA Cache with Entropy+GPA union center and manual_full" \
  "固定顺序式 GPA Cache，只将原型中心来源改为 Global Entropy Cache 与 GPA Cache 的并集，验证 union center 是否优于 GPA-only center" \
  "${1:-0}"
