# 本公共脚本用于执行 Uni3D 在 ModelNet-C 全部 35 个损坏设置上的单方法复现实验；当前版本只启动一次 Python，并在 Python 内部循环 35 个 cor_type。
#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "Usage: bash 32_run_uni3d_modelnetc_corruptions_all35_common.sh EXP_ID METHOD METHOD_FULL RUNNER CACHE_TYPE [GPU]"
  exit 1
fi

EXP_ID="$1"
METHOD="$2"
METHOD_FULL="$3"
RUNNER="$4"
CACHE_TYPE="$5"
PHYSICAL_GPU="${6:-0}"

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

cd "${PC_ROOT}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export WANDB_DIR="${PC_ROOT}/results/baseline/${EXP_ID}/wandb"
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1

mkdir -p "${WANDB_DIR}"

PC_FEAT_DIM=1408
NUM_GROUP=512
GROUP_SIZE=64
PC_ENCODER_DIM=512
EMBED_DIM=1024

CKPT_PATH="weights/uni3d/modelnet40/model.pt"

TEXT_CKPT_PATH=""
for p in \
  "weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin" \
  "weights/uni3d/open_clip_pytorch_model/eva02_enormous_patch14_plus_clip_224/laion2b_s9b_b144k.bin" \
  "weights/uni3d/laion2b_s9b_b144k.bin"
do
  if [[ -f "${p}" ]]; then
    TEXT_CKPT_PATH="${p}"
    break
  fi
done

if [[ -z "${TEXT_CKPT_PATH}" ]]; then
  echo "ERROR: missing Uni3D OpenCLIP text checkpoint under weights/uni3d"
  find weights/uni3d -maxdepth 5 -type f 2>/dev/null | sort || true
  exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "ERROR: missing Uni3D checkpoint ${CKPT_PATH}"
  find weights/uni3d -maxdepth 4 -type f 2>/dev/null | sort || true
  exit 1
fi

echo "============================================================"
echo "Optimized common script"
echo "EXP_ID: ${EXP_ID}"
echo "METHOD: ${METHOD}"
echo "METHOD_FULL: ${METHOD_FULL}"
echo "OLD_RUNNER_ARG: ${RUNNER}"
echo "CACHE_TYPE: ${CACHE_TYPE}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "Dataset: ModelNet-C all35"
echo "Backbone: Uni3D"
echo "Uni3D checkpoint: ${CKPT_PATH}"
echo "Uni3D text checkpoint: ${TEXT_CKPT_PATH}"
echo "pc_feat_dim: ${PC_FEAT_DIM}"
echo "num_group: ${NUM_GROUP}"
echo "group_size: ${GROUP_SIZE}"
echo "pc_encoder_dim: ${PC_ENCODER_DIM}"
echo "embed_dim: ${EMBED_DIM}"
echo "Python runner: runners/baseline/run_uni3d_modelnetc_corruptions_all35.py"
echo "============================================================"

python runners/baseline/run_uni3d_modelnetc_corruptions_all35.py \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "${METHOD}" \
  --baseline-method-full "${METHOD_FULL}" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/baseline" \
  --config configs \
  --wandb-log \
  --lm3d uni3d \
  --cache-type "${CACHE_TYPE}" \
  --pc-feat-dim "${PC_FEAT_DIM}" \
  --num-group "${NUM_GROUP}" \
  --group-size "${GROUP_SIZE}" \
  --pc-encoder-dim "${PC_ENCODER_DIM}" \
  --embed-dim "${EMBED_DIM}" \
  --ckpt_path "${CKPT_PATH}" \
  --pretrained "${TEXT_CKPT_PATH}" \
  --dataset modelnet_c \
  --cor_type add_global_0 \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --device 0 \
  --print-freq 500
