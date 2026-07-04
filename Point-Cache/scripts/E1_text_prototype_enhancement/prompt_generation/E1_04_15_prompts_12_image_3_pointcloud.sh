#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/common/generate_modelnet_c_prompts.sh" \
  "15" \
  "image12_pointcloud3" \
  "modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_12_image_3_pointcloud.json"
