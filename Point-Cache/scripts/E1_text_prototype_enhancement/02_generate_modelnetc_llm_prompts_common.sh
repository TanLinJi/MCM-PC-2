#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: bash 02_generate_modelnetc_llm_prompts_common.sh PROMPT_COUNT LLM_PROMPT_MODE"
  exit 1
fi

PROMPT_COUNT="$1"
LLM_PROMPT_MODE_VALUE="$2"

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
LLM_PROVIDER_VALUE="${LLM_PROVIDER:-deepseek}"
LLM_MODEL_VALUE="${LLM_MODEL:-deepseek-v4-pro}"
PROMPT_BANK_DIR="${PROMPT_BANK_DIR:-llm/e1_prompt_bank}"
SEED_CACHE="${PROMPT_BANK_DIR}/modelnet_c_${LLM_PROVIDER_VALUE}_${LLM_MODEL_VALUE}_multiview_2d3d_10_prompts.json"

if [[ ! -f "${SEED_CACHE}" ]]; then
  echo "ERROR: seed ModelNet-C class-name cache not found:"
  echo "  ${SEED_CACHE}"
  echo "The generator uses its class_names metadata to avoid duplicating dataset-loader logic."
  exit 1
fi

read -r -a E1_PYTHON <<< "${E1_PYTHON_CMD:-python}"

# `conda run` drops stdin unless capture is disabled; this script feeds Python by stdin.
if [[ "${E1_PYTHON[0]:-}" == "conda" && "${E1_PYTHON[1]:-}" == "run" ]]; then
  has_no_capture=0
  for token in "${E1_PYTHON[@]}"; do
    if [[ "${token}" == "--no-capture-output" ]]; then
      has_no_capture=1
      break
    fi
  done
  if [[ "${has_no_capture}" == "0" ]]; then
    E1_PYTHON=(conda run --no-capture-output "${E1_PYTHON[@]:2}")
  fi
fi

"${E1_PYTHON[@]}" - <<PY
import json
from argparse import Namespace
from pathlib import Path

from llm.e1_dynamic_prompt_generator import generate_llm_prompts

seed_cache = Path("${SEED_CACHE}")
with seed_cache.open("r", encoding="utf-8") as f:
    seed_data = json.load(f)

classnames = seed_data["class_names"]
force_regenerate = "${FORCE_REGENERATE}" == "1"

args = Namespace(
    prompt_source="llm_dynamic_init",
    llm_provider="${LLM_PROVIDER_VALUE}",
    llm_model="${LLM_MODEL_VALUE}",
    llm_api_key_file="${LLM_API_KEY_FILE:-llm/secrets/llm_api_key.txt}",
    llm_api_base_url="${LLM_API_BASE_URL:-https://api.deepseek.com/chat/completions}",
    llm_temperature=float("${LLM_TEMPERATURE:-0.3}"),
    dynamic_prompt_count=int("${PROMPT_COUNT}"),
    llm_prompt_mode="${LLM_PROMPT_MODE_VALUE}",
    prompt_cache_dir="${PROMPT_BANK_DIR}",
    force_regenerate_prompts=force_regenerate,
    dataset="modelnet_c",
    llm_max_retries=int("${LLM_MAX_RETRIES:-3}"),
)

prompts = generate_llm_prompts(
    classnames=classnames,
    args=args,
    dataset_name="modelnet_c",
)

print("[OK] Prompt bank:", "${PROMPT_BANK_DIR}")
print("[OK] Classes:", len(prompts))
print("[OK] Prompts per class:", len(next(iter(prompts.values()))))
PY
