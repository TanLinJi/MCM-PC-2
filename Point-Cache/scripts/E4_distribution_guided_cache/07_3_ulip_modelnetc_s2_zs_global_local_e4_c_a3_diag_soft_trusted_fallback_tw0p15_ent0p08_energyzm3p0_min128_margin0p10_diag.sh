#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"

bash "${SCRIPT_DIR}/07_run_e4_c_a3_diag_soft_trusted_fallback_common.sh" \
  "07_3_ulip_modelnetc_s2_zs_global_local_e4_c_a3_diag_soft_trusted_fallback_tw0p15_ent0p08_energyzm3p0_min128_margin0p10_diag" \
  "0.15" \
  "0.08" \
  "-3.0" \
  "128" \
  "0.10" \
  "E4-C-A3 diagnostic: 低熵+极严格低能量可信样本不再直接替换，只用 margin=0.10 放宽 text-visual joint score 门槛；记录每个样本的熵/能量，并记录 GPA 替换时当前样本、被替换样本和替换前后缓存快照；TTA 过程不使用 clean/corruption 标签；验证 ModelNet-C severity=2 七类损坏" \
  "${1:-0}"
