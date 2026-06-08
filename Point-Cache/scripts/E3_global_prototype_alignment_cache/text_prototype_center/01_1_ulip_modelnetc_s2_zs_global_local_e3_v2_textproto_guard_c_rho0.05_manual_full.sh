#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E3_global_prototype_alignment_cache/text_prototype_center"

export TEXT_PROTO_GUARD_RHO="0.05"

bash "${SCRIPT_DIR}/01_run_ulip_modelnetc_s2_textproto_guard_common.sh" \
  "01_1_ulip_modelnetc_s2_zs_global_local_e3_v2_textproto_guard_c_rho0.05_manual_full" \
  "runners/E3_global_prototype_alignment_cache/text_prototype_center/run_e3_ulip_modelnetc_s2_textproto_guard_center.py" \
  "manual_full" \
  "E3-V2-TextProto-Guard-C-rho0.05: visual denoising branch with Text Prototype semantic guard" \
  "在 E3-V2-C 基础上引入 TextProto Guard；visual_center 每次由当前 Entropy Cache + 当前 GPA-Cache 计算；Text Prototype 固定来自 clip_weights；不使用历史累计；替换时保留视觉距离更近分支，并增加视觉距离最多退化 5% 且文本距离更近的语义保护分支；未满直接初始化、替换最高熵样本、local cache 同步和最终预测公式均沿用 E3-V2-C" \
  "${1:-0}"
