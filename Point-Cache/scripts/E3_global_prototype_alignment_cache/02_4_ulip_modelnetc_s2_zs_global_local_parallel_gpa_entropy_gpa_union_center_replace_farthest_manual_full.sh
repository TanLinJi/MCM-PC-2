#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E3_global_prototype_alignment_cache"

bash "${SCRIPT_DIR}/02_run_ulip_modelnetc_s2_parallel_gpa_center_source_ablation_common.sh" \
  "02_4_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_gpa_union_center_replace_farthest_manual_full" \
  "runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_parallel_gpa_entropy_gpa_union_center_replace_farthest.py" \
  "manual_full" \
  "E3-V2-Cb: parallel GPA Cache with Entropy+GPA union center and low-entropy farthest replacement" \
  "补做 2+C+b：并列式 GPA Cache；Global Entropy Cache 与 GPA Cache 并集构造原型中心；GPA-Cache 未满直接初始化；GPA-Cache 满后使用低熵门控，同时要求新样本距离小于当前最远样本距离，并替换当前离中心最远样本；最终 logits 公式保持不变；保存 GPA 事件日志用于后续统计熵与距离关系" \
  "${1:-0}"
