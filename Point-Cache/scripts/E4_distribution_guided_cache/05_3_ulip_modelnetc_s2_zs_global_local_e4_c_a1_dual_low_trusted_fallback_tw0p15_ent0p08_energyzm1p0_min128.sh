#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"

bash "${SCRIPT_DIR}/05_run_e4_c_a1_dual_low_trusted_fallback_common.sh" \
  "05_3_ulip_modelnetc_s2_zs_global_local_e4_c_a1_dual_low_trusted_fallback_tw0p15_ent0p08_energyzm1p0_min128" \
  "0.15" \
  "0.08" \
  "-1.0" \
  "128" \
  "E4-C-A1 strict: 低熵+低能量可信样本回退；相对 05_1 收紧 entropy 0.10->0.08, energy_z -0.5->-1.0, min_count 64->128；TTA 过程不使用 clean/corruption 标签；验证 ModelNet-C severity=2 七类损坏" \
  "${1:-0}"
