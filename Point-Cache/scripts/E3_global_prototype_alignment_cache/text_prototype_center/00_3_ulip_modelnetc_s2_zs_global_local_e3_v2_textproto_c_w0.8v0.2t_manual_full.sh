#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E3_global_prototype_alignment_cache/text_prototype_center"

export TEXT_PROTO_VISUAL_WEIGHT="0.8"
export TEXT_PROTO_TEXT_WEIGHT="0.2"

bash "${SCRIPT_DIR}/00_run_ulip_modelnetc_s2_textproto_common.sh" \
  "00_3_ulip_modelnetc_s2_zs_global_local_e3_v2_textproto_c_w0.8v0.2t_manual_full" \
  "runners/E3_global_prototype_alignment_cache/text_prototype_center/run_e3_ulip_modelnetc_s2_textproto_union_center.py" \
  "manual_full" \
  "E3-V2-TextProto-C-w0.8v0.2t: Text Prototype + Entropy/GPA visual union center" \
  "在 E3-V2-C 基础上引入 Text Prototype；视觉中心由 Entropy Cache + GPA-Cache 构造；最终中心为 normalize(0.8 visual + 0.2 text)；未满直接初始化、低熵门控、替换最高熵样本、local cache 同步和最终预测公式均沿用 E3-V2-C" \
  "${1:-0}"
