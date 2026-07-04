#!/usr/bin/env bash
set -euo pipefail

# E7-A3 entry: A0-style multi-cache carrier with alignment-entry zero-shot
# correctness diagnostics.
# Core purpose:
#   Measure cumulative correctness of zero-shot pseudo-labels for samples that
#   actually enter/replace the alignment cache.
# Carrier settings:
#   entropy capacity = 4
#   energy capacity = 4
#   alignment capacity = 3
#   alpha_H = 1.8
#   alpha_E = 1.8
#   alpha_A = 2.5

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E7_entropy_energy_alignment_multicache"

export E7_ENTROPY_CAPACITY="4"
export E7_ENERGY_CAPACITY="4"
export E7_ALIGNMENT_CAPACITY="3"
export E7_ALPHA_ENTROPY="1.8"
export E7_ALPHA_ENERGY="1.8"
export E7_ALPHA_ALIGNMENT="2.5"
export E7_GATED_FUSION="0"
export GPA_SAVE_STATS="${GPA_SAVE_STATS:-1}"

bash "${SCRIPT_DIR}/00_run_e7_a_ulip_modelnetc_s2_common.sh" \
  "00_4_ulip_modelnetc_s2_zs_global_e7_a3_alignment_zs_diag_c4_h1p8_e1p8_a2p5_manual_full_clip_manualfull_llm_dynamic_init_textdist" \
  "E7-A3: A0-style entropy+energy+alignment multi-cache carrier, no gated fusion; diagnostic target is cumulative zero-shot pseudo-label correctness for samples that are alignment-eligible and samples that actually enter/replace alignment cache; capacities H=4 E=4 A=3; weights alpha_H=1.8 alpha_E=1.8 alpha_A=2.5; final classifier manual_full, E1 LLM only for text distribution; ModelNet-C severity=2" \
  "${1:-0}"
