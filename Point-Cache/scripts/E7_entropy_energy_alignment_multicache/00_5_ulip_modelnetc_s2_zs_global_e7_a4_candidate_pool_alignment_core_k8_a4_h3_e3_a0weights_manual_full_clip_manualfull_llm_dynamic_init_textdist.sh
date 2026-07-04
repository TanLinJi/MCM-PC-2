#!/usr/bin/env bash
set -euo pipefail

# E7-A4 entry: ULIP + ModelNet-C severity=2, candidate-pool alignment-core cache.
# Final classifier = manual_full; E1 LLM descriptions only feed text distribution.

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E7_entropy_energy_alignment_multicache"

bash "${SCRIPT_DIR}/00_run_e7_a4_ulip_modelnetc_s2_common.sh" \
  "00_5_ulip_modelnetc_s2_zs_global_e7_a4_candidate_pool_alignment_core_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist" \
  "E7-A4: candidate pool K=8, alignment core K=4, entropy/energy cache K=3, A0 weights alpha_ZS=1.0 alpha_A=alpha_H=alpha_E=2.0, score old/new average for samples entering true caches; final classifier manual_full, E1 LLM only for text distribution" \
  "${1:-0}"
