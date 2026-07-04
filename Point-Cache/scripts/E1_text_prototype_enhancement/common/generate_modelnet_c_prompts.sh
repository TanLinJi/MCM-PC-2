#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: bash generate_modelnet_c_prompts.sh PROMPT_COUNT LLM_PROMPT_MODE PROMPT_CACHE_FILE"
  exit 1
fi

PROMPT_COUNT="$1"
LLM_PROMPT_MODE_VALUE="$2"
PROMPT_CACHE_FILE="$3"

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
LLM_PROVIDER_VALUE="${LLM_PROVIDER:-deepseek}"
LLM_MODEL_VALUE="${LLM_MODEL:-deepseek-v4-pro}"
PROMPT_CACHE_DIR="${PROMPT_CACHE_DIR:-llm}"
CLASS_NAMES_FILE="${CLASS_NAMES_FILE:-data/modelnet_c/shape_names.txt}"

if [[ ! -f "${CLASS_NAMES_FILE}" ]]; then
  echo "ERROR: class names file not found:"
  echo "  ${CLASS_NAMES_FILE}"
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
from argparse import Namespace
from pathlib import Path

from llm.e1_dynamic_prompt_generator import generate_llm_prompts, get_prompt_cache_path

class_names_file = Path("${CLASS_NAMES_FILE}")
classnames = [
    line.strip()
    for line in class_names_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

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
    prompt_cache_dir="${PROMPT_CACHE_DIR}",
    prompt_cache_file="${PROMPT_CACHE_FILE}",
    force_regenerate_prompts=force_regenerate,
    dataset="modelnet_c",
    llm_max_retries=int("${LLM_MAX_RETRIES:-5}"),
)

cache_path = get_prompt_cache_path(args, "modelnet_c")
print("[E1] Dataset: modelnet_c")
print("[E1] Classes:", len(classnames))
print("[E1] Prompt count:", args.dynamic_prompt_count)
print("[E1] Prompt mode:", args.llm_prompt_mode)
print("[E1] Output:", cache_path)

prompts = generate_llm_prompts(
    classnames=classnames,
    args=args,
    dataset_name="modelnet_c",
)

per_class_counts = sorted({len(items) for items in prompts.values()})
print("[OK] Classes:", len(prompts))
print("[OK] Prompts per class:", per_class_counts)
PY

