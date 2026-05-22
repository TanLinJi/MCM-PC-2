# 本脚本用于下载 Point-Cache baseline 复现实验中 Uni3D 所需的官方 checkpoints。
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 说明：
# 1. D 组 Uni3D baseline 需要使用与数据集对应的 Uni3D-g checkpoint。
# 2. 31 / 32 组对应 ModelNet / ModelNet-C，使用：
#      weights/uni3d/modelnet40/model.pt
# 3. 33 / 34 组对应 ScanObjNN / ScanObjNN-C hardest，使用：
#      weights/uni3d/scanobjnn/model.pt
# 4. 原先服务器中只有：
#      weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt
#    使用该 checkpoint 跑 31 组时，ModelNet clean 结果整体偏低。
#    下载并切换到 modelnet40/model.pt 后，31 组结果与 Point-Cache 原文高度对齐。
# 5. 本脚本默认使用 Hugging Face 镜像 https://hf-mirror.com。
#    如果服务器可以直接访问 Hugging Face，可将 HF_ENDPOINT 改为 https://huggingface.co
# 6. 这些 checkpoint 文件较大，不应提交到 Git。
# ============================================================

PROJECT_ROOT="/root/autodl-tmp/MCM-PC-2"
PC_ROOT="${PROJECT_ROOT}/Point-Cache"

cd "${PC_ROOT}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "============================================================"
echo "Download Uni3D checkpoints"
echo "PC_ROOT: ${PC_ROOT}"
echo "HF_ENDPOINT: ${HF_ENDPOINT}"
echo "============================================================"

mkdir -p \
  weights/uni3d/modelnet40 \
  weights/uni3d/scanobjnn \
  weights/uni3d/lvis \
  weights/uni3d/_hf_download

echo
echo ">>> Download Uni3D-g ModelNet40 checkpoint"
python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="BAAI/Uni3D",
    filename="modelzoo/uni3d-g/mnet40/model.pt",
    local_dir="weights/uni3d/_hf_download",
)
print(path)
PY

cp weights/uni3d/_hf_download/modelzoo/uni3d-g/mnet40/model.pt \
   weights/uni3d/modelnet40/model.pt

echo
echo ">>> Download Uni3D-g ScanObjNN checkpoint"
python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="BAAI/Uni3D",
    filename="modelzoo/uni3d-g/scanobjnn/model.pt",
    local_dir="weights/uni3d/_hf_download",
)
print(path)
PY

cp weights/uni3d/_hf_download/modelzoo/uni3d-g/scanobjnn/model.pt \
   weights/uni3d/scanobjnn/model.pt

echo
echo ">>> Download optional Uni3D-g general checkpoint"
python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="BAAI/Uni3D",
    filename="modelzoo/uni3d-g/model.pt",
    local_dir="weights/uni3d/_hf_download",
)
print(path)
PY

cp weights/uni3d/_hf_download/modelzoo/uni3d-g/model.pt \
   weights/uni3d/model.pt

echo
echo ">>> Download optional Uni3D-g LVIS checkpoint"
python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="BAAI/Uni3D",
    filename="modelzoo/uni3d-g/lvis/model.pt",
    local_dir="weights/uni3d/_hf_download",
)
print(path)
PY

cp weights/uni3d/_hf_download/modelzoo/uni3d-g/lvis/model.pt \
   weights/uni3d/lvis/model.pt

echo
echo ">>> Check downloaded checkpoints"
find weights/uni3d -maxdepth 5 -type f \( -name "*.pt" -o -name "*.bin" \) \
  -printf "%p\t%s bytes\n" | sort

echo
echo ">>> SHA256"
sha256sum \
  weights/uni3d/modelnet40/model.pt \
  weights/uni3d/scanobjnn/model.pt \
  weights/uni3d/model.pt \
  weights/uni3d/lvis/model.pt \
  weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin \
  weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt 2>/dev/null || true

echo
echo ">>> Remove duplicated Hugging Face local download cache"
du -sh weights/uni3d/_hf_download || true
rm -rf weights/uni3d/_hf_download

echo
echo ">>> Final Uni3D checkpoint list"
find weights/uni3d -maxdepth 4 -type f \( -name "*.pt" -o -name "*.bin" \) \
  -printf "%p\t%s bytes\n" | sort

echo
echo "Done."
