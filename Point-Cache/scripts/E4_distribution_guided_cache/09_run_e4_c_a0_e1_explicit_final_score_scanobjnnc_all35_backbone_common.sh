#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 7 ]]; then
  echo "Usage: bash 09_run_e4_c_a0_e1_explicit_final_score_scanobjnnc_all35_backbone_common.sh BACKBONE SETTING ENTROPY_CAP GPA_CAP LOCAL_CAP NEG_CAP GPU"
  echo "BACKBONE choices: ulip2, openshape, uni3d"
  exit 1
fi

BACKBONE_KEY="$1"
SETTING="$2"
ENTROPY_CAP="$3"
GPA_CAP="$4"
LOCAL_CAP="$5"
NEG_CAP="$6"
PHYSICAL_GPU="$7"

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

SHARED_PROMPT_FILE="${PC_ROOT}/results/E1_text_prototype_enhancement/shared_prompts/sonn_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"
SCANOBJNNC_CLASS_FILE="${PC_ROOT}/data/sonn_c/shape_names.txt"

if [[ ! -f "${SHARED_PROMPT_FILE}" ]]; then
  echo "ERROR: shared ScanObjNN-C E1 LLM prompt cache not found:"
  echo "  ${SHARED_PROMPT_FILE}"
  echo
  echo "This all35 script must reuse the existing prompt cache and must not call the LLM API."
  exit 1
fi

if [[ ! -f "${SCANOBJNNC_CLASS_FILE}" ]]; then
  echo "ERROR: ScanObjNN-C class file not found:"
  echo "  ${SCANOBJNNC_CLASS_FILE}"
  exit 1
fi

python - "${SHARED_PROMPT_FILE}" "${SCANOBJNNC_CLASS_FILE}" <<'PY'
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
    print("ERROR: shared ScanObjNN-C E1 prompt cache is incomplete.")
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
    "Verified shared ScanObjNN-C E1 prompt cache: "
    f"{len(classnames)} classes, at least {required_prompt_count} prompts per class."
)
PY

BACKBONE_ARGS=()
case "${BACKBONE_KEY}" in
  ulip2)
    BACKBONE_NAME="ULIP-2"
    BACKBONE_TOKEN="ulip2"
    LM3D="ulip"
    CKPT_PATH="weights/ulip/pointbert_ulip2.pt"
    SLIP_CKPT_PATH="weights/ulip/slip_base_100ep.pt"
    if [[ ! -f "${PC_ROOT}/${CKPT_PATH}" ]]; then
      echo "ERROR: missing ULIP-2 checkpoint: ${PC_ROOT}/${CKPT_PATH}"
      exit 1
    fi
    if [[ ! -f "${PC_ROOT}/${SLIP_CKPT_PATH}" ]]; then
      echo "ERROR: missing SLIP text checkpoint: ${PC_ROOT}/${SLIP_CKPT_PATH}"
      exit 1
    fi
    BACKBONE_ARGS=(
      --lm3d "${LM3D}"
      --ckpt_path "${CKPT_PATH}"
      --slip-ckpt-path "${SLIP_CKPT_PATH}"
      --oshape-version vitg14
      --ulip-version ulip2
    )
    ;;
  openshape)
    BACKBONE_NAME="OpenShape"
    BACKBONE_TOKEN="openshape"
    LM3D="openshape"
    CKPT_PATH="weights/openshape/openshape-pointbert-vitg14-rgb/model.pt"
    OPENCLIP_PATH="weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin"
    if [[ ! -f "${PC_ROOT}/${CKPT_PATH}" ]]; then
      echo "ERROR: missing OpenShape checkpoint: ${PC_ROOT}/${CKPT_PATH}"
      exit 1
    fi
    if [[ ! -f "${PC_ROOT}/${OPENCLIP_PATH}" ]]; then
      echo "ERROR: missing OpenShape OpenCLIP checkpoint: ${PC_ROOT}/${OPENCLIP_PATH}"
      exit 1
    fi
    BACKBONE_ARGS=(
      --lm3d "${LM3D}"
      --ckpt_path "${CKPT_PATH}"
      --oshape-version vitg14
      --ulip-version ulip2
    )
    ;;
  uni3d)
    BACKBONE_NAME="Uni3D"
    BACKBONE_TOKEN="uni3d"
    LM3D="uni3d"
    PC_FEAT_DIM=1408
    NUM_GROUP=512
    GROUP_SIZE=64
    PC_ENCODER_DIM=512
    EMBED_DIM=1024
    CKPT_PATH="weights/uni3d/scanobjnn/model.pt"
    TEXT_CKPT_PATH=""
    for p in \
      "weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin" \
      "weights/uni3d/open_clip_pytorch_model/eva02_enormous_patch14_plus_clip_224/laion2b_s9b_b144k.bin" \
      "weights/uni3d/laion2b_s9b_b144k.bin"
    do
      if [[ -f "${PC_ROOT}/${p}" ]]; then
        TEXT_CKPT_PATH="${p}"
        break
      fi
    done
    if [[ -z "${TEXT_CKPT_PATH}" ]]; then
      echo "ERROR: missing Uni3D OpenCLIP text checkpoint under ${PC_ROOT}/weights/uni3d"
      find "${PC_ROOT}/weights/uni3d" -maxdepth 5 -type f 2>/dev/null | sort || true
      exit 1
    fi
    if [[ ! -f "${PC_ROOT}/${CKPT_PATH}" ]]; then
      echo "ERROR: missing Uni3D checkpoint: ${PC_ROOT}/${CKPT_PATH}"
      find "${PC_ROOT}/weights/uni3d" -maxdepth 4 -type f 2>/dev/null | sort || true
      exit 1
    fi
    BACKBONE_ARGS=(
      --lm3d "${LM3D}"
      --pc-feat-dim "${PC_FEAT_DIM}"
      --num-group "${NUM_GROUP}"
      --group-size "${GROUP_SIZE}"
      --pc-encoder-dim "${PC_ENCODER_DIM}"
      --embed-dim "${EMBED_DIM}"
      --ckpt_path "${CKPT_PATH}"
      --pretrained "${TEXT_CKPT_PATH}"
      --oshape-version vitg14
      --ulip-version ulip2
    )
    ;;
  *)
    echo "ERROR: unsupported BACKBONE=${BACKBONE_KEY}. Expected: ulip2, openshape, uni3d"
    exit 1
    ;;
esac

export SONN_VARIANT="${SONN_VARIANT:-hardest}"

FINAL_SCORE_SWEEP_TAG="${E4_FINAL_SCORE_SWEEP_TAG:-modelnetc_best_w4p4_3p9_0p19}"
EXP_ID="09_2_${SETTING}_${FINAL_SCORE_SWEEP_TAG}_${BACKBONE_TOKEN}_scanobjnnc_${SONN_VARIANT}_all35_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist"
LOCAL_CENTERS="${E4_LOCAL_CENTERS:-3}"
METHOD_FULL="09_2 ScanObjNN-C ${SONN_VARIANT} all35 ${BACKBONE_NAME} ${SETTING}: explicit final-score; entropy_cap=${ENTROPY_CAP}; gpa_cap=${GPA_CAP}; local_cap=${LOCAL_CAP}; neg_cap=${NEG_CAP}; local_centers=${LOCAL_CENTERS}; formula=y_zs+alpha_g*y_g+alpha_l*y_l-alpha_n*y_n; text_weight=0.15; score_norm=running_zscore"

export E4_SCORE_NORM_MODE="${E4_SCORE_NORM_MODE:-running_zscore}"
export E4_SCORE_NORM_MIN_COUNT="${E4_SCORE_NORM_MIN_COUNT:-8}"
export E4_SCORE_NORM_EPS="${E4_SCORE_NORM_EPS:-1e-6}"
export E4_SCORE_NORM_CLIP="${E4_SCORE_NORM_CLIP:-0}"

export E4_TEXT_DIST_PROMPT_SOURCE="${E4_TEXT_DIST_PROMPT_SOURCE:-manualfull_llm_dynamic_init}"
export E4_TEXT_SCORE_WEIGHT="${E4_TEXT_SCORE_WEIGHT:-0.15}"

export E4_ENTROPY_CAP="${ENTROPY_CAP}"
export E4_GPA_CAP="${GPA_CAP}"
export E4_LOCAL_CAP="${LOCAL_CAP}"
export E4_NEG_CAP="${NEG_CAP}"
export E4_LOCAL_CENTERS="${LOCAL_CENTERS}"
export E4_FINAL_SCORE_WEIGHTS="${E4_FINAL_SCORE_WEIGHTS:-modelnetc_best:4.4,3.9,0.19}"

echo "============================================================"
echo "09_2 explicit final-score ScanObjNN-C all35"
echo "EXP_ID: ${EXP_ID}"
echo "BACKBONE: ${BACKBONE_NAME}"
echo "SONN_VARIANT: ${SONN_VARIANT}"
echo "Data root: data/sonn_c/${SONN_VARIANT}"
echo "E4_ENTROPY_CAP: ${E4_ENTROPY_CAP}"
echo "E4_GPA_CAP: ${E4_GPA_CAP}"
echo "E4_LOCAL_CAP: ${E4_LOCAL_CAP}"
echo "E4_NEG_CAP: ${E4_NEG_CAP}"
echo "E4_LOCAL_CENTERS: ${E4_LOCAL_CENTERS}"
echo "E4_FINAL_SCORE_SWEEP_TAG: ${FINAL_SCORE_SWEEP_TAG}"
echo "E4_FINAL_SCORE_WEIGHTS: ${E4_FINAL_SCORE_WEIGHTS}"
echo "Result dir:"
echo "${PC_ROOT}/results/E4_distribution_guided_cache/${EXP_ID}"
echo "============================================================"

cd "${PC_ROOT}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export GPA_SAVE_STATS="${GPA_SAVE_STATS:-1}"
export E4_DIST_EPS="${E4_DIST_EPS:-1e-4}"
export E4_DIST_MIN_VAR="${E4_DIST_MIN_VAR:-1e-4}"
export E4_TEXT_DIST_EPS="${E4_TEXT_DIST_EPS:-${E4_DIST_EPS}}"
export E4_TEXT_DIST_MIN_VAR="${E4_TEXT_DIST_MIN_VAR:-${E4_DIST_MIN_VAR}}"
export E4_PROMPT_CACHE_DIR="${E4_PROMPT_CACHE_DIR:-results/E1_text_prototype_enhancement/shared_prompts}"
export E4_PROMPT_CACHE_FILE="${E4_PROMPT_CACHE_FILE:-}"
export E4_LLM_PROMPT_MODE="${E4_LLM_PROMPT_MODE:-multiview_2d3d}"
export E4_DYNAMIC_PROMPT_COUNT="${E4_DYNAMIC_PROMPT_COUNT:-10}"
export E4_PROMPT_STATIC_WEIGHT="${E4_PROMPT_STATIC_WEIGHT:-0.75}"
export E4_PROMPT_DYNAMIC_WEIGHT="${E4_PROMPT_DYNAMIC_WEIGHT:-0.25}"

python "runners/E4_distribution_guided_cache/run_e4_c_a0_e1_explicit_final_score_scanobjnnc_hardest_all35.py" \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "zs_global_local" \
  --baseline-method-full "${METHOD_FULL}" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/E4_distribution_guided_cache" \
  --config configs \
  --cache-type "hierarchical" \
  --prompt-source "manual_full" \
  --prompt-cache-dir "${E4_PROMPT_CACHE_DIR}" \
  --prompt-cache-file "${E4_PROMPT_CACHE_FILE}" \
  --llm-provider "deepseek" \
  --llm-model "deepseek-v4-pro" \
  --llm-api-key-file "llm/secrets/llm_api_key.txt" \
  --llm-api-base-url "https://api.deepseek.com/chat/completions" \
  --llm-temperature "0.3" \
  --llm-prompt-mode "${E4_LLM_PROMPT_MODE}" \
  --dynamic-prompt-count "${E4_DYNAMIC_PROMPT_COUNT}" \
  --prompt-static-weight "${E4_PROMPT_STATIC_WEIGHT}" \
  --prompt-dynamic-weight "${E4_PROMPT_DYNAMIC_WEIGHT}" \
  "${BACKBONE_ARGS[@]}" \
  --dataset sonn_c \
  --sonn_c_root data/sonn_c \
  --sonn_variant "${SONN_VARIANT}" \
  --cor_type add_global_0 \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --device 0 \
  --print-freq 500
