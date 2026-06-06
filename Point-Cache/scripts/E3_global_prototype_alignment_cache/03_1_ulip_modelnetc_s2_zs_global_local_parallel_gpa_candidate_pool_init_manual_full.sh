#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E3_global_prototype_alignment_cache"

export GPA_CANDIDATE_MULTIPLIER="${GPA_CANDIDATE_MULTIPLIER:-2}"

bash "${SCRIPT_DIR}/02_run_ulip_modelnetc_s2_parallel_gpa_center_source_ablation_common.sh" \
  "03_1_ulip_modelnetc_s2_zs_global_local_parallel_gpa_candidate_pool_init_manual_full" \
  "runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_parallel_gpa_candidate_pool_init.py" \
  "manual_full" \
  "E3-V3-A: parallel GPA Cache with candidate-pool initialization and manual_full" \
  "验证候选池初始化是否能缓解 GPA Cache 前 K 个样本无筛选进入的问题；候选池大小默认为 2K，中心为 Entropy Cache 与 GPA candidate pool 的联合中心" \
  "${1:-0}"
