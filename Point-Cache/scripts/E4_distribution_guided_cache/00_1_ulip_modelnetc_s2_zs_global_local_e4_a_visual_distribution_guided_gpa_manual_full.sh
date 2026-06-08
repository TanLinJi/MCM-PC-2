#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"

bash "${SCRIPT_DIR}/00_run_e4_a_ulip_modelnetc_s2_common.sh" \
  "00_1_ulip_modelnetc_s2_zs_global_local_e4_a_visual_distribution_guided_gpa_manual_full" \
  "runners/E4_distribution_guided_cache/run_e4_a_ulip_modelnetc_s2_visual_distribution_guided_gpa.py" \
  "manual_full" \
  "E4-A: distribution-guided GPA-Cache based on E3-V2-C initialization and highest-entropy replacement" \
  "类别概率分布引导的 GPA-Cache 净化；沿用 E3-V2-C 的未满直接初始化与替换最高熵样本规则；保留低熵门控；将距离单中心更近替换为更符合类别分布；最终预测公式暂时不改" \
  "${1:-0}"
