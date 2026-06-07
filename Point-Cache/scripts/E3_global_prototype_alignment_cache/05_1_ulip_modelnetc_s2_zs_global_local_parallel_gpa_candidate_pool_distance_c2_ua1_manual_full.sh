#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E3_global_prototype_alignment_cache"

export GPA_CANDIDATE_MULTIPLIER="${GPA_CANDIDATE_MULTIPLIER:-2}"

bash "${SCRIPT_DIR}/02_run_ulip_modelnetc_s2_parallel_gpa_center_source_ablation_common.sh" \
  "05_1_ulip_modelnetc_s2_zs_global_local_parallel_gpa_candidate_pool_distance_c2_ua1_manual_full" \
  "runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_parallel_gpa_candidate_pool_distance_c2_ua1.py" \
  "manual_full" \
  "E3-V3-C2-Ua1: candidate-pool distance initialization with Candidate+Entropy center and low-entropy highest-entropy replacement" \
  "候选池距离初始化 GPA-Cache；临时中心来自每类 2K candidate pool + Entropy Cache；初始化时仍然只从 candidate pool 中选择距离中心最近的 K 个进入 GPA-Cache；GPA-Cache 满后使用低熵门控，且新样本距离小于当前最高熵样本距离时替换最高熵样本；每次替换后同步 local cache 并立即重算 GPA-Center；最终预测权重暂时不改" \
  "${1:-0}"
