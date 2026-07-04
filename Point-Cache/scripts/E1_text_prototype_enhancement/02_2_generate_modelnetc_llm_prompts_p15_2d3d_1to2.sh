#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/02_generate_modelnetc_llm_prompts_common.sh" \
  "15" \
  "multiview_2d3d_1to2"
