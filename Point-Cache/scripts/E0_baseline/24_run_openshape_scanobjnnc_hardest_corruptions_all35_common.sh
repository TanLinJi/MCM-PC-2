# 本公共脚本用于执行 OpenShape 在 ScanObjNN-C hardest 全部 35 个损坏设置上的单方法复现实验；当前版本只启动一次 Python，并在 Python 内部循环 35 个 cor_type。
#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "Usage: bash 24_run_openshape_scanobjnnc_hardest_corruptions_all35_common.sh EXP_ID METHOD METHOD_FULL RUNNER CACHE_TYPE [GPU]"
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

echo "============================================================"
echo "Optimized common script"
echo "EXP_ID: ${EXP_ID}"
echo "METHOD: ${METHOD}"
echo "METHOD_FULL: ${METHOD_FULL}"
echo "OLD_RUNNER_ARG: ${RUNNER}"
echo "CACHE_TYPE: ${CACHE_TYPE}"
echo "PHYSICAL_GPU: ${PHYSICAL_GPU}"
echo "Dataset: ScanObjNN-C hardest all35"
echo "Backbone: OpenShape"
echo "OpenShape version: vitg14"
echo "Python runner: runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py"
echo "============================================================"

python runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py \
  --baseline-exp-id "${EXP_ID}" \
  --baseline-method "${METHOD}" \
  --baseline-method-full "${METHOD_FULL}" \
  --baseline-gpu "${PHYSICAL_GPU}" \
  --baseline-result-root "results/baseline" \
  --config configs \
  --wandb-log \
  --lm3d openshape \
  --cache-type "${CACHE_TYPE}" \
  --ckpt_path weights/openshape/openshape-pointbert-vitg14-rgb/model.pt \
  --dataset sonn_c \
  --sonn_variant hardest \
  --cor_type add_global_0 \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --oshape-version vitg14 \
  --device 0 \
  --print-freq 500
