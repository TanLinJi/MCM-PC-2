#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

python - <<PY
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
    prompt_cache_dir="results/E1_text_prototype_enhancement/prompts",
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
