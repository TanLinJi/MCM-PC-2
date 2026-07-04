#!/usr/bin/env bash
set -euo pipefail

# E7-A0 entry: ULIP + ModelNet-C severity=2, entropy/energy/alignment multi-cache,
# manual weights, no negative cache. Final classifier = manual_full;
# E1 LLM descriptions only feed the cache-replacement text distribution.

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E7_entropy_energy_alignment_multicache"

bash "${SCRIPT_DIR}/00_run_e7_a_ulip_modelnetc_s2_common.sh" \
  "00_1_ulip_modelnetc_s2_zs_global_e7_a_entropy_energy_alignment_cache_manual_full_clip_manualfull_llm_dynamic_init_textdist" \
  "E7-A0: entropy+energy+alignment multi-cache, global feature only (no local cache), manual weights alpha_H=alpha_E=alpha_A=2.0, beta=3.0, text_weight=0.15, score_norm=running_zscore; final classifier manual_full, E1 LLM only for text distribution; first run on ModelNet-C severity=2 to compare against 02_9_2 (54.71)" \
  "${1:-0}"
