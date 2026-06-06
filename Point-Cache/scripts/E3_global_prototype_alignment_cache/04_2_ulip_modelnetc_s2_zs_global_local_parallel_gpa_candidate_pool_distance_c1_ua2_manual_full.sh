#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E3_global_prototype_alignment_cache"

export GPA_CANDIDATE_MULTIPLIER="${GPA_CANDIDATE_MULTIPLIER:-2}"

bash "${SCRIPT_DIR}/02_run_ulip_modelnetc_s2_parallel_gpa_center_source_ablation_common.sh" \
  "04_2_ulip_modelnetc_s2_zs_global_local_parallel_gpa_candidate_pool_distance_c1_ua2_manual_full" \
  "runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_parallel_gpa_candidate_pool_distance_c1_ua2.py" \
  "manual_full" \
  "E3-V3-C1-Ua2: candidate-pool distance initialization with low-entropy gate and farthest replacement" \
  "候选池距离初始化 GPA-Cache；中心只来自每类 2K candidate pool；初始化选距离中心最近的 K 个；GPA-Cache 满后使用低熵门控，且新样本距离小于当前最远样本距离时替换最远样本；每次替换后同步 local cache 并立即重算 GPA-Center；最终预测权重暂时不改" \
  "${1:-0}"
