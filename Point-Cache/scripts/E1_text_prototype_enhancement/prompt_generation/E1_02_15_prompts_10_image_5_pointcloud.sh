#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/common/generate_modelnet_c_prompts.sh" \
  "15" \
  "image10_pointcloud5" \
  "modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json"

