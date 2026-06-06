#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E3_global_prototype_alignment_cache"

bash "${SCRIPT_DIR}/02_run_ulip_modelnetc_s2_parallel_gpa_center_source_ablation_common.sh" \
  "03_2_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_bootstrap_init_manual_full" \
  "runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_parallel_gpa_entropy_bootstrap_init.py" \
  "manual_full" \
  "E3-V3-B: parallel GPA Cache with Entropy-bootstrap initialization and manual_full" \
  "初始化阶段用 Global Entropy Cache 启动 GPA Cache；初始化完成后使用 Entropy+GPA union center 进行并列式 GPA 更新" \
  "${1:-0}"
