#!/usr/bin/env bash
set -euo pipefail

# E7-A2 entry: E7-A1 reduced cache weights plus sample-wise gated fusion.
# Final fusion:
#   S_final = S_zs + g(x) * S_cache
# Gate rule:
#   cache_pred == zs_pred -> g = 1.0
#   low zs_margin + high cache_margin + high cache_similarity -> g = 1.2
#   otherwise -> g = 0.2

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E7_entropy_energy_alignment_multicache"

export E7_ALPHA_ENTROPY="0.6"
export E7_ALPHA_ENERGY="0.6"
export E7_ALPHA_ALIGNMENT="0.9"
export E7_GATED_FUSION="1"
export E7_GATE_AGREE="1.0"
export E7_GATE_CORRECT="1.2"
export E7_GATE_FALLBACK="0.2"
export E7_GATE_ZS_MARGIN_MAX="5.0"
export E7_GATE_CACHE_MARGIN_MIN="1.0"
export E7_GATE_SIM_MIN="0.60"
export GPA_SAVE_STATS="${GPA_SAVE_STATS:-1}"

bash "${SCRIPT_DIR}/00_run_e7_a_ulip_modelnetc_s2_common.sh" \
  "00_3_ulip_modelnetc_s2_zs_global_e7_a2_gated_cache_h0p6_e0p6_a0p9_gagree1p0_gcorrect1p2_gfallback0p2_zsm5p0_cm1p0_sim0p60_manual_full_clip_manualfull_llm_dynamic_init_textdist" \
  "E7-A2: E7-A1 reduced weights plus sample-wise gated fusion S_final=S_zs+g*S_cache; g=1.0 if cache_pred==zs_pred, g=1.2 if low zs_margin<=5.0 and cache_margin>=1.0 and cache_similarity>=0.60, otherwise g=0.2; alpha_H=0.6 alpha_E=0.6 alpha_A=0.9; logits norm and gate diagnostics enabled; ModelNet-C severity=2" \
  "${1:-0}"
