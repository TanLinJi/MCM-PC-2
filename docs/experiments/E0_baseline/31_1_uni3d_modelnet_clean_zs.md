# 31_1_uni3d_modelnet_clean_zs

## 1. 实验目的

本实验用于复现 Uni3D 在 ModelNet clean 上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 31_1_uni3d_modelnet_clean_zs |
| Backbone | Uni3D |
| Dataset | ModelNet clean |
| Dataset 参数 | modelnet_c |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 Uni3D 在 ModelNet clean 上的无缓存基础性能。该结果后续会作为 31_2 和 31_3 的对照基线，但本文件只记录 31_1 本身，不展开完整 31 组的方法间对比。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | Uni3D |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 实际数据文件 | data/modelnet_c/clean.h5 |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 方法脚本 | Point-Cache/scripts/baseline/31_1_uni3d_modelnet_clean_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/31_run_uni3d_modelnet_clean_common.sh |
| 输入点数 | 1024 |
| Uni3D point encoder checkpoint | weights/uni3d/modelnet40/model.pt |
| Uni3D text encoder checkpoint | weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin |
| pc_model | eva_giant_patch14_560 |
| clip_model | EVA02-E-14-plus |
| pc_feat_dim | 1408 |
| num_group | 512 |
| group_size | 64 |
| pc_encoder_dim | 512 |
| embed_dim | 1024 |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 0 |

本实验使用 `modelnet_c` 作为 dataset 参数，并指定：

| 参数 | 值 |
|---|---|
| cor_type | clean |
| npoints | 1024 |
| sonn_variant | obj_only |

实际读取文件为：

data/modelnet_c/clean.h5

---

## 3. 当前实现方式

31 组是 clean 单文件实验，因此不需要 all35 优化 runner。

| 项目 | 说明 |
|---|---|
| 是否需要 35 个 cor_type 循环 | 否 |
| 是否需要 Python 内部循环 | 否 |
| 是否需要 all35 runner | 否 |
| 是否每个子实验只生成 1 行 summary | 是 |
| 是否每个子实验只生成 1 个 log | 是 |

31_1 的执行路径为：

| 层级 | 文件 |
|---|---|
| 单方法脚本 | Point-Cache/scripts/baseline/31_1_uni3d_modelnet_clean_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/31_run_uni3d_modelnet_clean_common.sh |
| 原始推理 runner | Point-Cache/runners/zs_infer.py |

该结构与 01、11、21 等 ModelNet clean 单文件实验保持一致。

---

## 4. 方法说明

31_1 是纯 Zero-shot 推理，不使用任何 cache。

| 组成部分 | 是否使用 |
|---|---:|
| Text prototype / 文本原型 | 是 |
| Point cloud global feature | 是 |
| Zero-shot logits | 是 |
| Global Cache logits | 否 |
| Local Cache logits | 否 |
| Hierarchical Cache | 否 |

需要注意：公共脚本中仍然会传入 `cache_type` 参数，这是为了统一脚本接口；但是 31_1 实际调用的是 `runners/zs_infer.py`，该 runner 不会构建 Global Cache 或 Local Cache。因此，31_1 应明确记录为无缓存 Zero-shot baseline。

---

## 5. Uni3D checkpoint 说明

本实验使用的 Uni3D point encoder checkpoint 为：

weights/uni3d/modelnet40/model.pt

这是本次 31 组复现中非常关键的设置。

此前曾使用服务器原有 checkpoint：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

旧 checkpoint 下 31 组 ModelNet clean 结果整体偏低，31_1 Zero-shot 为 80.11，明显低于原文 81.81。后续下载并切换到 Uni3D-g 的 ModelNet40 checkpoint：

weights/uni3d/modelnet40/model.pt

重新运行后，31_1 Zero-shot 提升到 81.85，与原文 81.81 高度对齐。因此，31 组后续正式记录均以 `weights/uni3d/modelnet40/model.pt` 为准。

该 checkpoint 的下载脚本已记录在：

Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh

完整的 checkpoint 特殊情况会在 31 组 summary 文档中统一说明。

---

## 6. 输出结构

输出目录：

Point-Cache/results/baseline/31_1_uni3d_modelnet_clean_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

31_1_uni3d_modelnet_clean_zs_clean_YYYYMMDD_HHMMSS.log

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
| 当前复现 accuracy | 81.85 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，31_1 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 8. 当前结果表

| 实验编号 | Dataset | File | Method | Checkpoint | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 31_1_uni3d_modelnet_clean_zs | modelnet_c | data/modelnet_c/clean.h5 | Zero-shot | weights/uni3d/modelnet40/model.pt | 81.85 | done |

该结果表示：Uni3D 在 ModelNet clean 上的 Zero-shot 准确率为 81.85。

---

## 9. 与原文结果对比

原文 Point-Cache Table 1 中，Uni3D 在 ModelNet clean 上的 Zero-shot 结果为 81.81。

当前复现结果为 81.85。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Uni3D / ModelNet clean / Zero-shot | 81.81 | 81.85 | +0.04 | 0.04 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与原文差异 | +0.04 |
| 绝对差异 | 0.04 |

分析：

当前复现结果 81.85 比原文 81.81 高 +0.04，差异极小。因此，31_1 不只是脚本执行成功，而且数值与原文高度对齐。

该结果说明，使用 `weights/uni3d/modelnet40/model.pt` 后，Uni3D × ModelNet clean 的 Zero-shot baseline 复现成功。

---

## 10. 与旧 checkpoint 结果对比

本实验曾使用服务器原有 checkpoint 运行过一次：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

旧 checkpoint 下 31_1 的结果为 80.11。切换到 ModelNet40 checkpoint 后，31_1 的结果为 81.85。

| Checkpoint | 31_1 Zero-shot |
|---|---:|
| weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt | 80.11 |
| weights/uni3d/modelnet40/model.pt | 81.85 |
| 原文 Point-Cache Table 1 | 81.81 |

变化：

| 比较 | 变化 |
|---|---:|
| modelnet40 checkpoint - old pc_encoder checkpoint | +1.74 |
| modelnet40 checkpoint - 原文 | +0.04 |
| old pc_encoder checkpoint - 原文 | -1.70 |

分析：

旧 checkpoint 下的 80.11 明显低于原文 81.81，差异为 -1.70。切换到 `weights/uni3d/modelnet40/model.pt` 后，结果提升到 81.85，与原文几乎一致。

因此，31 组最初偏低的主要原因可以归结为：Uni3D point encoder checkpoint 与原文 ModelNet40 设置不对应。后续 31 / 32 组应统一使用 `weights/uni3d/modelnet40/model.pt`。

---

## 11. 与 01 / 11 / 21 组 ModelNet clean 的关系

31_1 可以与前面三个 backbone 的 ModelNet clean Zero-shot 结果进行横向比较。

| Backbone | 实验编号 | ModelNet clean Zero-shot |
|---|---|---:|
| ULIP | 01_1_ulip_modelnet_clean_zs | 约 56 |
| ULIP-2 | 11_1_ulip2_modelnet_clean_zs | 72.20 |
| OpenShape | 21_1_openshape_modelnet_clean_zs | 84.72 |
| Uni3D | 31_1_uni3d_modelnet_clean_zs | 81.85 |

分析：

Uni3D 在 ModelNet clean 上明显强于 ULIP 和 ULIP-2，低于 OpenShape。这个相对排序与原文大方向一致：OpenShape clean performance 最高，Uni3D 次之，ULIP-2 和 ULIP 更低。

这说明当前 Uni3D checkpoint 切换后，31_1 已经回到合理的 backbone 排序中。

---

## 12. 与后续子实验的关系

31_1 是 31 组第一个子实验，因此没有前序 31 组子实验可供对比。

它后续将作为以下实验的基线：

| 后续实验 | 对比方式 |
|---|---|
| 31_2_uni3d_modelnet_clean_zs_global | 与 31_1 比较，评估 Global Cache 在 Uni3D × ModelNet clean 上的影响 |
| 31_3_uni3d_modelnet_clean_zs_global_local | 与 31_1 和 31_2 比较，评估完整 Point-Cache 及 Local Cache 额外影响 |

原文中 Uni3D 在 ModelNet clean 上的趋势为：

| 方法 | 原文值 |
|---|---:|
| Zero-shot | 81.81 |
| + Global Cache | 83.14 |
| + Hierarchical Cache | 83.87 |

因此，后续 31_2 和 31_3 的重点是观察：

1. Global Cache 是否提升；
2. Local Cache 是否在 Global Cache 基础上继续提升；
3. 最终完整 Point-Cache 是否接近原文 83.87；
4. 当前方法趋势是否保持 Zero-shot < Global < Global + Local。

---

## 13. 结果含义分析

31_1 的意义不只是给出一个 clean accuracy，而是确认 Uni3D 的 ModelNet40 checkpoint 设置已经正确。

| 观察 | 含义 |
|---|---|
| 旧 checkpoint 得到 80.11 | 与原文差异较大，不能直接归档 |
| 新 checkpoint 得到 81.85 | 与原文 81.81 高度对齐 |
| 差异仅 +0.04 | ModelNet40 checkpoint 是当前 31 组正确选择 |
| Uni3D 低于 OpenShape | 与前面 backbone 结果关系合理 |
| Uni3D 高于 ULIP / ULIP-2 | 符合强 backbone 预期 |

因此，31_1 是 Uni3D 组能否继续复现的关键校准实验。只有 31_1 使用正确 checkpoint 对齐原文后，31_2 和 31_3 的 cache 结果才具有可信对照意义。

---

## 14. 对后续 MCM-PC 的启发

当前 31_1 对后续 MCM-PC 方法设计和实验管理有以下启发：

| 观察 | 启发 |
|---|---|
| Uni3D 对 checkpoint 选择非常敏感 | 后续 31–34 组必须明确记录 checkpoint 路径 |
| ModelNet clean 应使用 modelnet40 checkpoint | 31 / 32 组统一使用 weights/uni3d/modelnet40/model.pt |
| ScanObjNN 后续应使用 scanobjnn checkpoint | 33 / 34 组应使用 weights/uni3d/scanobjnn/model.pt |
| 旧 pc_encoder checkpoint 会导致偏低 | 不应再作为正式 baseline checkpoint |
| 下载脚本需要归档 | 便于复现实验环境和避免后续混淆 |

这次 checkpoint 修正说明：对于 Uni3D 这种大模型，复现实验前不能只检查脚本是否跑通，还必须核对 checkpoint 是否与数据设置匹配。

---

## 15. 阶段性结论

本实验完成了 Uni3D × ModelNet clean 的 Zero-shot baseline 复现。

主要结论如下：

1. 31_1 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前使用 checkpoint 为 weights/uni3d/modelnet40/model.pt。
3. 当前复现准确率为 81.85。
4. 原文 Point-Cache Table 1 中 Uni3D / ModelNet clean / Zero-shot 结果为 81.81。
5. 当前复现结果比原文高 +0.04，差异极小，可以认为高度对齐。
6. 旧 checkpoint weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt 下结果为 80.11，比原文低 -1.70。
7. 切换到 ModelNet40 checkpoint 后，结果从 80.11 提升到 81.85，说明此前偏低主要来自 checkpoint 不匹配。
8. 31_1 结果有效，不需要重跑。
9. 本实验是 31_2 Global Cache 和 31_3 Global + Local Cache 的基础对照。
10. 31 / 32 组后续应统一使用 weights/uni3d/modelnet40/model.pt。

---

## 16. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/31_1_uni3d_modelnet_clean_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/31_1_uni3d_modelnet_clean_zs_single_gpu.sh 1

---

## 17. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/31_1_uni3d_modelnet_clean_zs/summary.csv | wc -l

tail -n +2 results/baseline/31_1_uni3d_modelnet_clean_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/31_1_uni3d_modelnet_clean_zs/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/31_1_uni3d_modelnet_clean_zs/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/31_1_uni3d_modelnet_clean_zs/summary.csv
