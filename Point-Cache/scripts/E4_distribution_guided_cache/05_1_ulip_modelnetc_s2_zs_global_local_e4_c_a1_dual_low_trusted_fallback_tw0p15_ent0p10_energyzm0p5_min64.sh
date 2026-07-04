#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"

bash "${SCRIPT_DIR}/05_run_e4_c_a1_dual_low_trusted_fallback_common.sh" \
  "05_1_ulip_modelnetc_s2_zs_global_local_e4_c_a1_dual_low_trusted_fallback_tw0p15_ent0p10_energyzm0p5_min64" \
  "0.15" \
  "0.10" \
  "-0.5" \
  "64" \
  "E4-C-A1: 低熵+低能量可信样本回退；TTA 过程不使用 clean/corruption 标签；在 02_9_2 设置基础上验证 ModelNet-C severity=2 七类损坏" \
  "${1:-0}"
