#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
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
from argparse import Namespace
from llm.e1_dynamic_prompt_generator import generate_llm_prompts

force_regenerate = "${FORCE_REGENERATE}" == "1"

args = Namespace(
    prompt_source="llm_dynamic_init",
    llm_provider="deepseek",
    llm_model="deepseek-v4-pro",
    llm_api_key_file="llm/secrets/llm_api_key.txt",
    llm_api_base_url="https://api.deepseek.com/chat/completions",
    llm_temperature=0.7,
    dynamic_prompt_count=2,
    prompt_cache_dir="llm/e1_prompt_bank",
    force_regenerate_prompts=force_regenerate,
    dataset="api_test",
)

prompts = generate_llm_prompts(
    classnames=["airplane"],
    args=args,
    dataset_name="api_test",
)

print("[OK] Generated or loaded classes:", list(prompts.keys()))
print("[OK] Number of prompts for airplane:", len(prompts["airplane"]))
print("[OK] First prompt:", prompts["airplane"][0])
PY
