#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"

bash "${SCRIPT_DIR}/06_run_e4_c_a2_diag_dual_low_trusted_fallback_common.sh" \
  "06_3_ulip_modelnetc_s2_zs_global_local_e4_c_a2_diag_dual_low_trusted_fallback_tw0p15_ent0p08_energyzm3p0_min128_diag" \
  "0.15" \
  "0.08" \
  "-3.0" \
  "128" \
  "E4-C-A2 diagnostic: 低熵+极严格低能量可信样本回退；energy_z 阈值从 -1.5 收紧到 -3.0；记录每个样本的熵/能量，并记录 GPA 替换时当前样本、被替换样本和替换前后缓存快照；TTA 过程不使用 clean/corruption 标签；验证 ModelNet-C severity=2 七类损坏" \
  "${1:-0}"
