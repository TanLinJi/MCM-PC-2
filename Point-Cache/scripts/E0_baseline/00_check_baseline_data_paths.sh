# 本脚本用于检查 Point-Cache baseline 第一轮复现实验所需的数据文件、脚本目录和结果目录是否正确。
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

SCRIPT_DIR="${PC_ROOT}/scripts/baseline"
RESULT_ROOT="${PC_ROOT}/results/baseline"
RUN_NAME="00_check_baseline_data_paths"
RUN_DIR="${RESULT_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/check_baseline_data_paths_${TIMESTAMP}.log"

{
  echo "============================================================"
  echo "Point-Cache baseline 数据路径检查"
  echo "时间: ${TIMESTAMP}"
  echo "项目根目录: ${PROJECT_ROOT}"
  echo "Point-Cache目录: ${PC_ROOT}"
  echo "脚本目录: ${SCRIPT_DIR}"
  echo "结果目录: ${RESULT_ROOT}"
  echo "当前检查输出目录: ${RUN_DIR}"
  echo "日志文件: ${LOG_FILE}"
  echo "============================================================"
  echo

  cd "${PC_ROOT}"

  echo "[1/5] 检查基础目录"
  test -d "${SCRIPT_DIR}" && echo "OK: ${SCRIPT_DIR}"
  test -d "${RESULT_ROOT}" && echo "OK: ${RESULT_ROOT}"
  echo

  echo "[2/5] 检查 clean 文件"
  CLEAN_FILES=(
    "data/modelnet_c/clean.h5"
    "data/sonn_c/hardest/clean.h5"
  )

  missing_count=0

  for f in "${CLEAN_FILES[@]}"; do
    if [[ -f "${f}" ]]; then
      echo "OK: ${f}"
    else
      echo "MISSING: ${f}"
      missing_count=$((missing_count + 1))
    fi
  done
  echo

  echo "[3/5] 检查 ModelNet-C 35 个 corrupted 文件"
  CORRUPTIONS=(
    "add_global"
    "add_local"
    "dropout_global"
    "dropout_local"
    "rotate"
    "scale"
    "jitter"
  )

  SEVERITIES=(0 1 2 3 4)

  modelnet_total=0
  modelnet_missing=0

  for corruption in "${CORRUPTIONS[@]}"; do
    for severity in "${SEVERITIES[@]}"; do
      f="data/modelnet_c/${corruption}_${severity}.h5"
      modelnet_total=$((modelnet_total + 1))
      if [[ -f "${f}" ]]; then
        echo "OK: ${f}"
      else
        echo "MISSING: ${f}"
        modelnet_missing=$((modelnet_missing + 1))
        missing_count=$((missing_count + 1))
      fi
    done
  done

  echo "ModelNet-C corrupted 文件检查: $((modelnet_total - modelnet_missing))/${modelnet_total} 存在"
  echo

  echo "[4/5] 检查 ScanObjNN-C hardest 35 个 corrupted 文件"
  sonn_total=0
  sonn_missing=0

  for corruption in "${CORRUPTIONS[@]}"; do
    for severity in "${SEVERITIES[@]}"; do
      f="data/sonn_c/hardest/${corruption}_${severity}.h5"
      sonn_total=$((sonn_total + 1))
      if [[ -f "${f}" ]]; then
        echo "OK: ${f}"
      else
        echo "MISSING: ${f}"
        sonn_missing=$((sonn_missing + 1))
        missing_count=$((missing_count + 1))
      fi
    done
  done

  echo "ScanObjNN-C hardest corrupted 文件检查: $((sonn_total - sonn_missing))/${sonn_total} 存在"
  echo

  echo "[5/5] 检查 .gitignore 是否忽略 results"
  cd "${PROJECT_ROOT}"

  if grep -qxF "Point-Cache/results/" .gitignore; then
    echo "OK: .gitignore 已包含 Point-Cache/results/"
  else
    echo "MISSING: .gitignore 未包含 Point-Cache/results/"
    missing_count=$((missing_count + 1))
  fi

  echo
  echo "============================================================"
  if [[ "${missing_count}" -eq 0 ]]; then
    echo "检查结果: PASS，第一轮 baseline 数据文件齐全。"
  else
    echo "检查结果: FAIL，共缺失 ${missing_count} 项。"
    exit 1
  fi
  echo "============================================================"

} 2>&1 | tee "${LOG_FILE}"
