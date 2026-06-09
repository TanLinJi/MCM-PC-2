#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache"
PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

SHARED_PROMPT_FILE="${PC_ROOT}/results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"
MODELNETC_CLASS_FILE="${PC_ROOT}/data/modelnet_c/shape_names.txt"

if [[ ! -f "${SHARED_PROMPT_FILE}" ]]; then
  echo "ERROR: shared E1 LLM prompt cache not found:"
  echo "  ${SHARED_PROMPT_FILE}"
  echo
  echo "This E4-C-A0+E1-textdist experiment must reuse the existing prompt cache and must not call the LLM API."
  exit 1
fi

python - "${SHARED_PROMPT_FILE}" "${MODELNETC_CLASS_FILE}" <<'PY'
import json
import sys
from pathlib import Path

cache_path = Path(sys.argv[1])
class_file = Path(sys.argv[2])
required_prompt_count = 10

with cache_path.open("r", encoding="utf-8") as f:
    saved = json.load(f)

prompts = saved.get("prompts", saved)
if not isinstance(prompts, dict):
    raise SystemExit(f"ERROR: invalid prompt cache format: {cache_path}")

with class_file.open("r", encoding="utf-8") as f:
    classnames = [line.strip() for line in f if line.strip()]

missing = []
short = []
for classname in classnames:
    clean_name = classname.replace("_", " ")
    class_prompts = prompts.get(clean_name)
    if class_prompts is None:
        missing.append(clean_name)
    elif len(class_prompts) < required_prompt_count:
        short.append((clean_name, len(class_prompts)))

if missing or short:
    print("ERROR: shared E1 prompt cache is incomplete.")
    if missing:
        print("Missing classes:")
        for name in missing:
            print(f"  {name}")
    if short:
        print("Classes with too few prompts:")
        for name, count in short:
            print(f"  {name}: {count}/{required_prompt_count}")
    print()
    print("This script intentionally stops here to avoid regenerating prompts through the LLM API.")
    raise SystemExit(1)

print(
    "Verified shared E1 prompt cache: "
    f"{len(classnames)} classes, at least {required_prompt_count} prompts per class."
)
PY

export E4_SCORE_NORM_MODE="${E4_SCORE_NORM_MODE:-running_zscore}"
export E4_SCORE_NORM_MIN_COUNT="${E4_SCORE_NORM_MIN_COUNT:-8}"
export E4_SCORE_NORM_EPS="${E4_SCORE_NORM_EPS:-1e-6}"
export E4_SCORE_NORM_CLIP="${E4_SCORE_NORM_CLIP:-0}"

# Keep the base classifier and final logits identical to E4-C-A0/manual_full.
# Use E1 cached descriptions only inside the E4 text-distribution replacement score.
export E4_TEXT_DIST_PROMPT_SOURCE="${E4_TEXT_DIST_PROMPT_SOURCE:-manualfull_llm_dynamic_init}"

bash "${SCRIPT_DIR}/02_run_e4_c_ulip_modelnetc_s2_common.sh" \
  "02_8_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_textdist_only_score_norm_accepted_history_text_visual_distribution_guided_gpa_manual_full_clip_manualfull_llm_dynamic_init_textdist" \
  "runners/E4_distribution_guided_cache/run_e4_c_a0_e1_textdist_only_ulip_modelnetc_s2_accepted_history_text_visual_distribution_guided_gpa.py" \
  "manual_full" \
  "E4-C-A0+E1-textdist-only: manual_full clip logits with cached E1 weighted text distribution for GPA replacement" \
  "修正 02_7 的耦合问题：最终 clip_weights、cache 初始化预测、global entropy cache、negative cache 和最终 logits 均保持 E4-C-A0 的 manual_full；只在 E4-C 的 text distribution 替换准则中引入 E1 已缓存的 manual_full/LLM 描述融合；E1 文本分布按分支权重 manual_full=0.75、LLM=0.25 做加权均值和方差估计；直接复用 E1 共享缓存，不重新调用 LLM API" \
  "${1:-0}"
