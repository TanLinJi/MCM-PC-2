# 33_1_uni3d_scanobjnn_clean_hardest_zs

## 1. 实验目的

本实验用于复现 Uni3D 在 ScanObjNN clean hardest split 上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 33_1_uni3d_scanobjnn_clean_hardest_zs |
| Backbone | Uni3D |
| Dataset | ScanObjNN clean hardest |
| Dataset 参数 | sonn_c |
| SONN variant | hardest |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 Uni3D 在 ScanObjNN clean hardest 上的无缓存基础性能。该结果后续会作为 33_2 和 33_3 的对照基线，但本文件只记录 33_1 本身，不展开完整 33 组的方法间对比。

需要特别注意：33 组属于 Uni3D × ScanObjNN 系列实验，必须使用 ScanObjNN 对应 checkpoint：

weights/uni3d/scanobjnn/model.pt

不能使用 ModelNet 系列的：

weights/uni3d/modelnet40/model.pt

也不能使用此前服务器原有的通用 pc_encoder checkpoint：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | Uni3D |
| 数据集参数 | sonn_c |
| 数据目录 | data/sonn_c |
| 数据集变体 | hardest |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 方法脚本 | Point-Cache/scripts/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/33_run_uni3d_scanobjnn_clean_hardest_common.sh |
| 输入点数 | 1024 |
| Uni3D point encoder checkpoint | weights/uni3d/scanobjnn/model.pt |
| Uni3D text encoder checkpoint | weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin |
| pc_model | eva_giant_patch14_560 |
| clip_model | EVA02-E-14-plus |
| pc_feat_dim | 1408 |
| num_group | 512 |
| group_size | 64 |
| pc_encoder_dim | 512 |
| embed_dim | 1024 |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 0 |

本实验使用 `sonn_c` 作为 dataset 参数，并指定：

| 参数 | 值 |
|---|---|
| sonn_variant | hardest |
| cor_type | clean |
| npoints | 1024 |
| sim2real_type | so_obj_only_9 |

实际读取文件为：

data/sonn_c/hardest/clean.h5

---

## 3. 当前实现方式

33 组是 clean 单文件实验，因此不需要 all35 优化 runner。

| 项目 | 说明 |
|---|---|
| 是否需要 35 个 cor_type 循环 | 否 |
| 是否需要 Python 内部循环 | 否 |
| 是否需要 all35 runner | 否 |
| 是否每个子实验只生成 1 行 summary | 是 |
| 是否每个子实验只生成 1 个 log | 是 |

33_1 的执行路径为：

| 层级 | 文件 |
|---|---|
| 单方法脚本 | Point-Cache/scripts/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/33_run_uni3d_scanobjnn_clean_hardest_common.sh |
| 原始推理 runner | Point-Cache/runners/zs_infer.py |

该结构与 03、13、23、31 等 clean 单文件实验保持一致。

---

## 4. 方法说明

33_1 是纯 Zero-shot 推理，不使用任何 cache。

| 组成部分 | 是否使用 |
|---|---:|
| Text prototype / 文本原型 | 是 |
| Point cloud global feature | 是 |
| Zero-shot logits | 是 |
| Global Cache logits | 否 |
| Local Cache logits | 否 |
| Hierarchical Cache | 否 |

需要注意：公共脚本中仍然会传入 `cache_type` 参数，这是为了统一脚本接口；但是 33_1 实际调用的是 `runners/zs_infer.py`，该 runner 不会构建 Global Cache 或 Local Cache。因此，33_1 应明确记录为无缓存 Zero-shot baseline。

---

## 5. Uni3D checkpoint 说明

本实验使用的 Uni3D point encoder checkpoint 为：

weights/uni3d/scanobjnn/model.pt

这是本次 33 组复现中最关键的设置。

此前曾使用服务器原有 checkpoint：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

旧 checkpoint 下 33_1 的结果为 41.33，明显低于原文 46.04。检查 33 组公共脚本后发现，当时脚本的 checkpoint 候选列表没有包含 `weights/uni3d/scanobjnn/model.pt`，因此自动优先选择了已存在的 `weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt`。

修正公共脚本后，将 ScanObjNN checkpoint 加入候选列表第一位：

weights/uni3d/scanobjnn/model.pt

重新运行后，33_1 的结果提升到 45.63，与原文 46.04 高度接近。因此，33 组后续正式记录均以 `weights/uni3d/scanobjnn/model.pt` 为准。

该 checkpoint 的下载脚本已记录在：

Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh

完整的 checkpoint 特殊情况会在 33 组 summary 文档中统一说明。

---

## 6. 输出结构

输出目录：

Point-Cache/results/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

33_1_uni3d_scanobjnn_clean_hardest_zs_clean_YYYYMMDD_HHMMSS.log

因为本实验只测试 `clean.h5`，所以期望只有 1 行 summary 记录和 1 个对应 log。

---

## 7. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 1 | 1 | clean 实验只包含 1 个测试文件 |
| summary 中唯一 cor_type 数 | 1 | 1 | 只测试 clean |
| summary 中唯一 log_path 数 | 1 | 1 | 每次运行对应 1 个 log |
| logs 目录 .log 文件数 | 1 | 1 | 没有旧日志或重复日志残留 |
| status=done 数 | 1 | 1 | 脚本执行成功 |
| 当前复现 accuracy | 45.63 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |
| 实际 checkpoint | weights/uni3d/scanobjnn/model.pt | scanobjnn checkpoint | 已修正 |

从执行完整性看，33_1 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 8. 当前结果表

| 实验编号 | Dataset | File | Method | Checkpoint | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 33_1_uni3d_scanobjnn_clean_hardest_zs | sonn_c hardest | data/sonn_c/hardest/clean.h5 | Zero-shot | weights/uni3d/scanobjnn/model.pt | 45.63 | done |

该结果表示：Uni3D 在 ScanObjNN clean hardest 上的 Zero-shot 准确率为 45.63。

---

## 9. 与原文结果对比

原文 Point-Cache Table 7 中，Uni3D 在 ScanObjNN hardest clean/original data 上的 Zero-shot 结果为 46.04。

当前复现结果为 45.63。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Uni3D / ScanObjNN clean hardest / Zero-shot | 46.04 | 45.63 | -0.41 | 0.41 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与原文差异 | -0.41 |
| 绝对差异 | 0.41 |

分析：

当前复现结果 45.63 比原文 46.04 低 -0.41，差异较小。因此，33_1 不只是脚本执行成功，而且数值与原文较好对齐。

该结果说明，使用 `weights/uni3d/scanobjnn/model.pt` 后，Uni3D × ScanObjNN clean hardest 的 Zero-shot baseline 复现有效。

---

## 10. 与旧 checkpoint 结果对比

本实验曾使用服务器原有 checkpoint 运行过一次：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

旧 checkpoint 下 33_1 的结果为 41.33。切换到 ScanObjNN checkpoint 后，33_1 的结果为 45.63。

| Checkpoint | 33_1 Zero-shot |
|---|---:|
| weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt | 41.33 |
| weights/uni3d/scanobjnn/model.pt | 45.63 |
| 原文 Point-Cache Table 7 | 46.04 |

变化：

| 比较 | 变化 |
|---|---:|
| scanobjnn checkpoint - old pc_encoder checkpoint | +4.30 |
| scanobjnn checkpoint - 原文 | -0.41 |
| old pc_encoder checkpoint - 原文 | -4.71 |

分析：

旧 checkpoint 下的 41.33 明显低于原文 46.04，差异为 -4.71。切换到 `weights/uni3d/scanobjnn/model.pt` 后，结果提升到 45.63，与原文高度接近。

因此，33 组最初偏低的主要原因可以归结为：Uni3D point encoder checkpoint 与 ScanObjNN 数据设置不对应。后续 33 / 34 组应统一使用 `weights/uni3d/scanobjnn/model.pt`。

---

## 11. 与 03 / 13 / 23 组 ScanObjNN clean hardest 的关系

33_1 可以与前面三个 backbone 的 ScanObjNN clean hardest Zero-shot 结果进行横向比较。

| Backbone | 实验编号 | ScanObjNN clean hardest Zero-shot |
|---|---|---:|
| ULIP | 03_1_ulip_scanobjnn_clean_hardest_zs | 29.08 |
| ULIP-2 | 13_1_ulip2_scanobjnn_clean_hardest_zs | 33.31 |
| OpenShape | 23_1_openshape_scanobjnn_clean_hardest_zs | 41.88 |
| Uni3D | 33_1_uni3d_scanobjnn_clean_hardest_zs | 45.63 |

分析：

Uni3D 在 ScanObjNN clean hardest 上明显强于 ULIP、ULIP-2 和 OpenShape。与前面 ModelNet clean 上 OpenShape 略强于 Uni3D 不同，在 ScanObjNN clean hardest 上，Uni3D 的 Zero-shot 表现最高。

这说明 Uni3D 的 ScanObjNN 对应 checkpoint 对真实扫描 hardest split 有明显优势，也进一步证明 checkpoint 与数据集匹配的重要性。

---

## 12. 与后续子实验的关系

33_1 是 33 组第一个子实验，因此没有前序 33 组子实验可供对比。

它后续将作为以下实验的基线：

| 后续实验 | 对比方式 |
|---|---|
| 33_2_uni3d_scanobjnn_clean_hardest_zs_global | 与 33_1 比较，评估 Global Cache 在 Uni3D × ScanObjNN clean hardest 上的影响 |
| 33_3_uni3d_scanobjnn_clean_hardest_zs_global_local | 与 33_1 和 33_2 比较，评估完整 Point-Cache 及 Local Cache 额外影响 |

原文中 Uni3D 在 ScanObjNN clean hardest 上的趋势为：

| 方法 | 原文值 |
|---|---:|
| Zero-shot | 46.04 |
| + Global Cache | 50.28 |
| + Hierarchical Cache | 51.13 |

因此，后续 33_2 和 33_3 的重点是观察：

1. Global Cache 是否提升；
2. Local Cache 是否在 Global Cache 基础上继续提升；
3. 最终完整 Point-Cache 是否接近原文 51.13；
4. 当前方法趋势是否保持 Zero-shot < Global < Global + Local。

---

## 13. 结果含义分析

33_1 的意义不只是给出一个 clean hardest accuracy，而是确认 Uni3D 的 ScanObjNN checkpoint 设置已经正确。

| 观察 | 含义 |
|---|---|
| 旧 checkpoint 得到 41.33 | 与原文差异较大，不能直接归档 |
| 新 checkpoint 得到 45.63 | 与原文 46.04 高度接近 |
| 差异为 -0.41 | ScanObjNN checkpoint 是当前 33 组正确选择 |
| Uni3D 高于 OpenShape | 与 ScanObjNN hardest 上的 backbone 表现关系合理 |
| Uni3D 高于 ULIP / ULIP-2 | 符合强 backbone 预期 |

因此，33_1 是 Uni3D ScanObjNN 组能否继续复现的关键校准实验。只有 33_1 使用正确 checkpoint 对齐原文后，33_2 和 33_3 的 cache 结果才具有可信对照意义。

---

## 14. 对后续 MCM-PC 的启发

当前 33_1 对后续 MCM-PC 方法设计和实验管理有以下启发：

| 观察 | 启发 |
|---|---|
| Uni3D 对 checkpoint 选择非常敏感 | 后续 33–34 组必须明确记录 checkpoint 路径 |
| ScanObjNN clean hardest 应使用 scanobjnn checkpoint | 33 / 34 组统一使用 weights/uni3d/scanobjnn/model.pt |
| ModelNet checkpoint 不应混用到 ScanObjNN | 否则会影响复现判断 |
| 旧 pc_encoder checkpoint 会导致明显偏低 | 不应再作为正式 baseline checkpoint |
| 下载脚本需要归档 | 便于复现实验环境和避免后续混淆 |

这次 checkpoint 修正说明：对于 Uni3D，复现实验前不能只检查脚本是否跑通，还必须核对 checkpoint 是否与数据设置匹配。

---

## 15. 阶段性结论

本实验完成了 Uni3D × ScanObjNN clean hardest 的 Zero-shot baseline 复现。

主要结论如下：

1. 33_1 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前使用 checkpoint 为 weights/uni3d/scanobjnn/model.pt。
3. 当前复现准确率为 45.63。
4. 原文 Point-Cache Table 7 中 Uni3D / ScanObjNN clean hardest / Zero-shot 结果为 46.04。
5. 当前复现结果比原文低 -0.41，差异较小，可以认为结果有效。
6. 旧 checkpoint weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt 下结果为 41.33，比原文低 -4.71。
7. 切换到 ScanObjNN checkpoint 后，结果从 41.33 提升到 45.63，说明此前偏低主要来自 checkpoint 不匹配。
8. 33_1 结果有效，不需要重跑。
9. 本实验是 33_2 Global Cache 和 33_3 Global + Local Cache 的基础对照。
10. 33 / 34 组后续应统一使用 weights/uni3d/scanobjnn/model.pt。

---

## 16. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs_single_gpu.sh 1

---

## 17. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs/summary.csv | wc -l

tail -n +2 results/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs/summary.csv
