#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E1_text_prototype_enhancement"

bash "${SCRIPT_DIR}/common/run_modelnet_c_fusion.sh" \
  "E1_13_modelnet_c_full_manual75_llm25" \
  "0.75" \
  "0.25" \
  "10" \
  "image4_pointcloud4_bridge2" \
  "modelnet_c_llm_descriptions_deepseek_v4pro_10_prompts_4_image_4_pointcloud_2_bridge.json" \
  "ModelNet-C full fusion weight ablation, 10 prompts = 4 image + 4 pointcloud + 2 bridge" \
  "${1:-0}"
