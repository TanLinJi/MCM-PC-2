#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"

bash "${SCRIPT_DIR}/08_run_e4_c_a0_e1_textdist_only_neg_highent_common.sh" \
  "clean" \
  "${1:-0}"
