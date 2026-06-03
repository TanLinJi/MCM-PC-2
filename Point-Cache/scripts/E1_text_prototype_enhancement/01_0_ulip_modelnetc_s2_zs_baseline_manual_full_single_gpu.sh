#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/01_run_ulip_modelnetc_s2_zs_prompt_ablation_common.sh" \
  "01_0_ulip_modelnetc_s2_zs_baseline_manual_full" \
  "manual_full" \
  "Zero-shot with original full manual prompt set" \
  "验证 E1 新接口不破坏 Point-Cache 原始完整手工模板 baseline" \
  "${1:-0}"
