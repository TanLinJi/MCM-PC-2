#!/usr/bin/env bash
set -euo pipefail

# E7-A4-B1 entry: ULIP + ModelNet-C severity=2, cache-norm-clipped candidate-pool alignment-core cache.
# Final classifier = manual_full; E1 LLM descriptions only feed text distribution.

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E7_entropy_energy_alignment_multicache"

bash "${SCRIPT_DIR}/00_run_e7_a4_b1_ulip_modelnetc_s2_common.sh" \
  "00_6_ulip_modelnetc_s2_zs_global_e7_a4_b1_cache_norm_clip_cap20_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist" \
  "E7-A4-B1: A4 fixed + cache total logits norm clipping cap=20, candidate pool K=8, alignment core K=4, entropy/energy cache K=3, A0 weights alpha_ZS=1.0 alpha_A=alpha_H=alpha_E=2.0, old/new score average for samples entering alignment/entropy/energy caches; final classifier manual_full, E1 LLM only for text distribution" \
  "${1:-0}"
