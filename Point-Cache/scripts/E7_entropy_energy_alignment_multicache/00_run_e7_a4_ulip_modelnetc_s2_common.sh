#!/usr/bin/env bash
set -euo pipefail

# E7-A4 common launcher: ULIP + ModelNet-C severity=2 candidate-pool alignment-core cache.
#
# 用法:
#   bash 00_run_e7_a4_ulip_modelnetc_s2_common.sh EXP_ID PURPOSE GPU
#
# 该脚本:
#   1. 校验 E1 共享 prompt 缓存完整，不调用 LLM API；
#   2. 导出 E7-A4 超参数（可被外部环境变量覆盖）；
#   3. 路由到 E7-A4 runner，cache_type 在 runner 内部强制为 global。

if [[ "$#" -lt 3 ]]; then
  echo "Usage: bash 00_run_e7_a4_ulip_modelnetc_s2_common.sh EXP_ID PURPOSE GPU"
  exit 1
fi

EXP_ID="$1"
PURPOSE="$2"
PHYSICAL_GPU="$3"

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

SHARED_PROMPT_FILE="${PC_ROOT}/results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"
MODELNETC_CLASS_FILE="${PC_ROOT}/data/modelnet_c/shape_names.txt"

if [[ ! -f "${SHARED_PROMPT_FILE}" ]]; then
  echo "ERROR: shared E1 LLM prompt cache not found:"
  echo "  ${SHARED_PROMPT_FILE}"
  echo
    echo "E7-A4 reuses the existing prompt cache and must not call the LLM API."
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

cd "${PC_ROOT}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export WANDB_DIR="${PC_ROOT}/results/E7_entropy_energy_alignment_multicache/${EXP_ID}/wandb"
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1
export GPA_SAVE_STATS="${GPA_SAVE_STATS:-1}"

# 缓存容量（A4 文档第 5 节）。
export E7_CANDIDATE_CAPACITY="${E7_CANDIDATE_CAPACITY:-8}"
export E7_ALIGNMENT_CAPACITY="${E7_ALIGNMENT_CAPACITY:-4}"
export E7_ENTROPY_CAPACITY="${E7_ENTROPY_CAPACITY:-3}"
export E7_ENERGY_CAPACITY="${E7_ENERGY_CAPACITY:-3}"

# 手动权重：沿用 E7-A0，暂不引入自适应权重。
export E7_ALPHA_ZS="${E7_ALPHA_ZS:-1.0}"
export E7_ALPHA_ENTROPY="${E7_ALPHA_ENTROPY:-2.0}"
export E7_ALPHA_ENERGY="${E7_ALPHA_ENERGY:-2.0}"
export E7_ALPHA_ALIGNMENT="${E7_ALPHA_ALIGNMENT:-2.0}"

# 相似度温度。
export E7_BETA_ENTROPY="${E7_BETA_ENTROPY:-3.0}"
export E7_BETA_ENERGY="${E7_BETA_ENERGY:-3.0}"
export E7_BETA_ALIGNMENT="${E7_BETA_ALIGNMENT:-3.0}"

# 分布打分参数（沿用 E4/02_9_2 经验）。
export E7_DIST_EPS="${E7_DIST_EPS:-1e-4}"
export E7_DIST_MIN_VAR="${E7_DIST_MIN_VAR:-1e-4}"
export E7_TEXT_DIST_EPS="${E7_TEXT_DIST_EPS:-${E7_DIST_EPS}}"
export E7_TEXT_DIST_MIN_VAR="${E7_TEXT_DIST_MIN_VAR:-${E7_DIST_MIN_VAR}}"
export E7_TEXT_SCORE_WEIGHT="${E7_TEXT_SCORE_WEIGHT:-0.15}"
export E7_SCORE_NORM_MODE="${E7_SCORE_NORM_MODE:-running_zscore}"
export E7_SCORE_NORM_MIN_COUNT="${E7_SCORE_NORM_MIN_COUNT:-8}"
export E7_SCORE_NORM_EPS="${E7_SCORE_NORM_EPS:-1e-6}"
export E7_SCORE_NORM_CLIP="${E7_SCORE_NORM_CLIP:-0}"
export E7_ALIGNMENT_MIN_TOTAL="${E7_ALIGNMENT_MIN_TOTAL:-0}"
export E7_A4_RELIABILITY_EPS="${E7_A4_RELIABILITY_EPS:-1e-6}"
export E7_A4_SCORE_OLD_WEIGHT="${E7_A4_SCORE_OLD_WEIGHT:-0.5}"
export E7_A4_SCORE_NEW_WEIGHT="${E7_A4_SCORE_NEW_WEIGHT:-0.5}"

# 文本分布来源：E1 LLM 描述仅用于缓存替换打分，不替换最终 manual_full 分类器。
export E7_TEXT_DIST_PROMPT_SOURCE="${E7_TEXT_DIST_PROMPT_SOURCE:-manualfull_llm_dynamic_init}"

RUNNER="runners/E7_entropy_energy_alignment_multicache/run_e7_a4_ulip_modelnetc_s2_candidate_pool_alignment_core.py"

mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "E7-A4 Candidate-Pool Alignment-Core Cache (global feature only)"
echo "EXP_ID: ${EXP_ID}"
echo "RUNNER: ${RUNNER}"
echo "PURPOSE: ${PURPOSE}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "E7_CANDIDATE_CAPACITY: ${E7_CANDIDATE_CAPACITY}"
echo "E7_ALIGNMENT_CAPACITY: ${E7_ALIGNMENT_CAPACITY}"
echo "E7_ENTROPY_CAPACITY: ${E7_ENTROPY_CAPACITY}"
echo "E7_ENERGY_CAPACITY: ${E7_ENERGY_CAPACITY}"
echo "E7_ALPHA_ZS: ${E7_ALPHA_ZS}"
echo "E7_ALPHA_ENTROPY: ${E7_ALPHA_ENTROPY}"
echo "E7_ALPHA_ENERGY: ${E7_ALPHA_ENERGY}"
echo "E7_ALPHA_ALIGNMENT: ${E7_ALPHA_ALIGNMENT}"
echo "E7_BETA_ENTROPY: ${E7_BETA_ENTROPY}"
echo "E7_BETA_ENERGY: ${E7_BETA_ENERGY}"
echo "E7_BETA_ALIGNMENT: ${E7_BETA_ALIGNMENT}"
echo "E7_TEXT_SCORE_WEIGHT: ${E7_TEXT_SCORE_WEIGHT}"
echo "E7_SCORE_NORM_MODE: ${E7_SCORE_NORM_MODE}"
echo "E7_TEXT_DIST_PROMPT_SOURCE: ${E7_TEXT_DIST_PROMPT_SOURCE}"
echo "E7_A4_SCORE_OLD_WEIGHT: ${E7_A4_SCORE_OLD_WEIGHT}"
echo "E7_A4_SCORE_NEW_WEIGHT: ${E7_A4_SCORE_NEW_WEIGHT}"
echo "============================================================"

python "${RUNNER}" \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "zs_global" \
  --baseline-method-full "E7-A4 candidate-pool alignment-core cache (manual_full classifier, E1 textdist replacement only)" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/E7_entropy_energy_alignment_multicache" \
  --config configs \
  --wandb-log \
  --lm3d ulip \
  --cache-type "global" \
  --prompt-source "manual_full" \
  --prompt-cache-dir "results/E1_text_prototype_enhancement/shared_prompts" \
  --llm-provider "deepseek" \
  --llm-model "deepseek-v4-pro" \
  --llm-api-key-file "llm/secrets/llm_api_key.txt" \
  --llm-api-base-url "https://api.deepseek.com/chat/completions" \
  --llm-temperature "0.3" \
  --llm-prompt-mode "multiview_2d3d" \
  --dynamic-prompt-count "10" \
  --prompt-static-weight "0.75" \
  --prompt-dynamic-weight "0.25" \
  --ckpt_path weights/ulip/pointbert_ulip1.pt \
  --slip-ckpt-path weights/ulip/slip_base_100ep.pt \
  --dataset modelnet_c \
  --sonn_variant hardest \
  --cor_type add_global_2 \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --oshape-version vitg14 \
  --ulip-version ulip1 \
  --device 0 \
  --print-freq 500
