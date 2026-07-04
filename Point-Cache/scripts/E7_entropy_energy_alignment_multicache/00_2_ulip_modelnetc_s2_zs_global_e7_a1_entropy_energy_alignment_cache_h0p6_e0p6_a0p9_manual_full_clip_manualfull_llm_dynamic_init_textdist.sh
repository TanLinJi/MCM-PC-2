#!/usr/bin/env bash
set -euo pipefail

# E7-A1 entry: same E7-A multi-cache structure as A0, but with weaker
# positive cache weights and logits-norm diagnostics enabled.
#   alpha_H = 0.6
#   alpha_E = 0.6
#   alpha_A = 0.9

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E7_entropy_energy_alignment_multicache"

export E7_ALPHA_ENTROPY="0.6"
export E7_ALPHA_ENERGY="0.6"
export E7_ALPHA_ALIGNMENT="0.9"
export GPA_SAVE_STATS="${GPA_SAVE_STATS:-1}"

bash "${SCRIPT_DIR}/00_run_e7_a_ulip_modelnetc_s2_common.sh" \
  "00_2_ulip_modelnetc_s2_zs_global_e7_a1_entropy_energy_alignment_cache_h0p6_e0p6_a0p9_manual_full_clip_manualfull_llm_dynamic_init_textdist" \
  "E7-A1: entropy+energy+alignment multi-cache, global feature only (no local cache), reduced manual weights alpha_H=0.6 alpha_E=0.6 alpha_A=0.9, beta=3.0, text_weight=0.15, score_norm=running_zscore, logits norm diagnostics enabled; final classifier manual_full, E1 LLM only for text distribution; ModelNet-C severity=2" \
  "${1:-0}"
