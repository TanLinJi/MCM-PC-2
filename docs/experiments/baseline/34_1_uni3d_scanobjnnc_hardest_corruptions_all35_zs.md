# 34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs

## 1. 实验目的

本实验用于复现 Uni3D 在 ScanObjNN-C hardest 全部 35 个损坏设置上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs |
| Backbone | Uni3D |
| Dataset | ScanObjNN-C hardest |
| Dataset 参数 | sonn_c |
| SONN variant | hardest |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 Uni3D 在 ScanObjNN-C hardest corrupted setting 上的无缓存基础鲁棒性。该结果后续会作为 34_2 和 34_3 的对照基线，但本文件只记录 34_1 本身，不展开完整 34 组的方法间对比。

需要特别注意：原文 Point-Cache Table 7 报告的是 ScanObjNN-C hardest 在 severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原文 Point-Cache Table 7 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | Uni3D |
| 数据集参数 | sonn_c |
| 数据目录 | data/sonn_c |
| 实际 h5 文件目录 | data/sonn_c/hardest |
| SONN variant | hardest |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 优化 runner | Point-Cache/runners/baseline/run_uni3d_scanobjnnc_hardest_corruptions_all35.py |
| 方法脚本 | Point-Cache/scripts/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/34_run_uni3d_scanobjnnc_hardest_corruptions_all35_common.sh |
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
| cor_type | 由 runner 内部循环 35 个 corruption setting |
| npoints | 1024 |
| sim2real_type | so_obj_only_9 |

实际 h5 文件形式为：

data/sonn_c/hardest/{corruption}_{severity}.h5

---

## 3. 当前实现方式

34 组属于 ScanObjNN-C hardest all35 实验，因此使用优化版 Python runner：

Point-Cache/runners/baseline/run_uni3d_scanobjnnc_hardest_corruptions_all35.py

该 runner 的设计是：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 | 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 | 模型只加载一次，每个 cor_type 重新创建 DataLoader |
| bash 通过 tee 生成单个 cor_type 的 log | Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv | summary.csv 的列结构保持不变，每个 cor_type 一行 |

该优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、预测逻辑和输出结构。

---

## 4. 34 runner 修正记录

34 组最初从 32 组 ModelNet-C all35 runner 复制而来，因此曾出现过 runner 内部 metadata 残留问题。旧 runner 中存在类似字段：

| 错误字段 | 问题 |
|---|---|
| args.modelnet_c_root = "data/modelnet_c" | 继承自 32 组 ModelNet-C |
| data_root = args.modelnet_c_root | summary 中写成 ModelNet-C 路径 |
| sonn_variant = "-" | 没有记录 hardest split |
| file = data/modelnet_c/{cor_type}.h5 | summary 文件路径错误 |

修正后，当前 runner 已确认使用以下设置：

| 字段 | 当前正确值 |
|---|---|
| args.dataset | sonn_c |
| args.sonn_variant | hardest |
| args.sonn_c_root | data/sonn_c |
| args.data_root | data/sonn_c |
| summary data_root | data/sonn_c/hardest |
| summary file | data/sonn_c/hardest/{cor_type}.h5 |
| summary sonn_variant | hardest |
| expected checkpoint | weights/uni3d/scanobjnn/model.pt |

需要注意：`SONN_C` 数据集类需要在 `data/sonn_c` 下读取 `shape_names.txt`，因此 loader root 应为 `data/sonn_c`；但实际 h5 文件在 `data/sonn_c/hardest`，因此 summary 中记录的实际文件路径应为 `data/sonn_c/hardest/{cor_type}.h5`。

当前 34_1 summary 已确认 metadata 正确。

---

## 5. 方法说明

34_1 是纯 Zero-shot 推理，不使用任何 cache。

| 组成部分 | 是否使用 |
|---|---:|
| Text prototype / 文本原型 | 是 |
| Point cloud global feature | 是 |
| Zero-shot logits | 是 |
| Global Cache logits | 否 |
| Local Cache logits | 否 |
| Hierarchical Cache | 否 |

需要注意：公共脚本中仍会传入 `cache_type` 参数，这是为了统一脚本接口；但是 34_1 的方法为 `zs`，实际只使用 Zero-shot 推理逻辑，不构建 Global Cache 或 Local Cache。

---

## 6. Uni3D checkpoint 说明

本实验使用的 Uni3D point encoder checkpoint 为：

weights/uni3d/scanobjnn/model.pt

这是 33 / 34 组 Uni3D × ScanObjNN 系列实验的正式 checkpoint。

后续规则固定如下：

| 实验组 | 数据设置 | 应使用 checkpoint |
|---|---|---|
| 31 | Uni3D × ModelNet clean | weights/uni3d/modelnet40/model.pt |
| 32 | Uni3D × ModelNet-C all35 | weights/uni3d/modelnet40/model.pt |
| 33 | Uni3D × ScanObjNN clean hardest | weights/uni3d/scanobjnn/model.pt |
| 34 | Uni3D × ScanObjNN-C hardest all35 | weights/uni3d/scanobjnn/model.pt |

不能使用：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

也不能使用：

weights/uni3d/modelnet40/model.pt

该 checkpoint 的下载脚本已记录在：

Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh

---

## 7. 损坏类型

本实验包含 7 种 corruption，每种 corruption 包含 5 个 severity level：

| corruption | severity 编号 |
|---|---|
| add_global | 0, 1, 2, 3, 4 |
| add_local | 0, 1, 2, 3, 4 |
| dropout_global | 0, 1, 2, 3, 4 |
| dropout_local | 0, 1, 2, 3, 4 |
| rotate | 0, 1, 2, 3, 4 |
| scale | 0, 1, 2, 3, 4 |
| jitter | 0, 1, 2, 3, 4 |

注意：severity 文件编号从 0 开始，到 4 结束。不要写成 1 到 5。

文件命名形式为：

| 示例 | 含义 |
|---|---|
| add_global_0.h5 | add_global corruption，severity=0 |
| add_global_2.h5 | add_global corruption，severity=2 |
| jitter_4.h5 | jitter corruption，severity=4 |

---

## 8. 输出结构

输出目录：

Point-Cache/results/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs_{cor_type}_YYYYMMDD_HHMMSS.log

---

## 9. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 35 | 35 | 说明 35 个 cor_type 全部完成 |
| summary 中唯一 cor_type 数 | 35 | 35 | 说明没有漏跑或重复写入 |
| summary 中唯一 log_path 数 | 35 | 35 | 说明每个 cor_type 都有独立日志路径 |
| logs 目录当前 .log 文件数 | 35 | 35 | 说明没有旧日志或重复日志残留 |
| status=done 数 | 35 | 35 | 说明没有失败项 |
| dataset | sonn_c | sonn_c | 正确 |
| data_root | data/sonn_c/hardest | data/sonn_c/hardest | 正确 |
| file | data/sonn_c/hardest/{cor_type}.h5 | data/sonn_c/hardest/{cor_type}.h5 | 正确 |
| sonn_variant | hardest | hardest | 正确 |
| severity=2 Average | 37.75 | 用于论文对齐 | 与原文 Point-Cache Table 7 对比 |
| all35 Average | 37.66 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性和 metadata 看，34_1 现在是干净结果。

---

## 10. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 46.70 | 47.99 | 47.99 | 47.40 | 46.88 | 47.39 |
| add_local | 41.60 | 39.87 | 37.68 | 37.58 | 36.71 | 38.69 |
| dropout_global | 42.54 | 39.83 | 36.92 | 30.88 | 25.19 | 35.07 |
| dropout_local | 42.09 | 37.89 | 31.47 | 27.97 | 23.91 | 32.67 |
| rotate | 45.35 | 45.07 | 43.79 | 41.08 | 38.90 | 42.84 |
| scale | 40.35 | 39.24 | 37.72 | 36.88 | 36.50 | 38.14 |
| jitter | 40.04 | 33.34 | 28.66 | 23.66 | 18.53 | 28.85 |
| **Average** | **42.67** | **40.46** | **37.75** | **35.06** | **32.37** | **37.66** |

整体观察：

1. all35 Average 为 37.66，表示 Uni3D 在 ScanObjNN-C hardest 全 35 个 corrupted setting 上的 Zero-shot 鲁棒性水平。
2. severity=2 Average 为 37.75，用于和原文 Point-Cache Table 7 对齐。
3. add_global 的平均准确率最高，为 47.39。
4. jitter 的平均准确率最低，为 28.85。
5. jitter_4 为 18.53，是全部 35 个 setting 中最低结果。
6. high-severity jitter、dropout_local 和 dropout_global 是主要困难区域。

---

## 11. 与原文结果对比

原文 Point-Cache Table 7 报告的是 ScanObjNN-C hardest 在 severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

原文 Uni3D 在 ScanObjNN-C hardest severity=2 下的 Zero-shot Average 为 37.38。

当前复现 severity=2 Average 为 37.75。

| 对比对象 | 原文 S2 Avg | 当前复现 S2 Avg | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Uni3D / ScanObjNN-C hardest / Zero-shot | 37.38 | 37.75 | +0.37 | 0.37 |

分析：

当前复现结果 37.75 比原文 37.38 高 +0.37，差异很小。因此，34_1 不只是脚本执行成功，而且数值与原文高度接近。

该结果可以作为 34_2 和 34_3 的有效 Zero-shot baseline。

---

## 12. Severity 维度分析

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 42.67 | — | 0.00 |
| S1 | 40.46 | -2.21 | -2.21 |
| S2 | 37.75 | -2.71 | -4.92 |
| S3 | 35.06 | -2.69 | -7.61 |
| S4 | 32.37 | -2.69 | -10.30 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 42.67 下降到 32.37，总下降 10.30 个百分点。整体上，severity 越高，Uni3D Zero-shot 准确率越低。

---

## 13. Corruption 难度分析

### 13.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 28.85 | 平均最低，高 severity 下下降最明显 |
| 2 | dropout_local | 32.67 | 高 severity 下明显困难 |
| 3 | dropout_global | 35.07 | 高 severity 下明显困难 |
| 4 | scale | 38.14 | 中等偏难 |
| 5 | add_local | 38.69 | 中等偏难 |
| 6 | rotate | 42.84 | 相对较高 |
| 7 | add_global | 47.39 | 当前最高 |

分析：

jitter 是最困难 corruption，平均准确率只有 28.85，尤其 jitter_4 只有 18.53。dropout_local 和 dropout_global 也明显困难，说明坐标扰动和点云缺失是 Uni3D 在 ScanObjNN-C hardest Zero-shot 下的主要失败模式。

### 13.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|
| add_global | 46.70 | 46.88 | +0.18 | 47.39 |
| add_local | 41.60 | 36.71 | -4.89 | 38.69 |
| scale | 40.35 | 36.50 | -3.85 | 38.14 |
| rotate | 45.35 | 38.90 | -6.45 | 42.84 |
| dropout_global | 42.54 | 25.19 | -17.35 | 35.07 |
| dropout_local | 42.09 | 23.91 | -18.18 | 32.67 |
| jitter | 40.04 | 18.53 | -21.51 | 28.85 |

分析：

jitter 的退化最强，从 S0 的 40.04 下降到 S4 的 18.53，下降 -21.51。dropout_local 和 dropout_global 的退化也很强，分别下降 -18.18 和 -17.35。

这说明最高 severity 的坐标扰动和点云缺失对 ScanObjNN-C hardest 的 Zero-shot 结果破坏最大。

---

## 14. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | jitter | 40.04 | add_global | 46.70 | 6.66 |
| S1 | jitter | 33.34 | add_global | 47.99 | 14.65 |
| S2 | jitter | 28.66 | add_global | 47.99 | 19.33 |
| S3 | jitter | 23.66 | add_global | 47.40 | 23.74 |
| S4 | jitter | 18.53 | add_global | 46.88 | 28.35 |

分析：

jitter 在所有 severity 下都是最低项。随着 severity 增大，best-worst gap 从 S0 的 6.66 扩大到 S4 的 28.35，说明高 severity 下不同 corruption 的难度差异明显扩大。

---

## 15. 低准确率区域分析

| 条件 | 数量 | 占比 | 主要涉及 corruption |
|---|---:|---:|---|
| Acc < 40 | 22 / 35 | 62.86% | jitter, dropout_local, dropout_global, scale, add_local |
| Acc < 35 | 11 / 35 | 31.43% | jitter, dropout_local, dropout_global |
| Acc < 30 | 6 / 35 | 17.14% | jitter, dropout_local, dropout_global |
| Acc < 25 | 3 / 35 | 8.57% | jitter_3, jitter_4, dropout_local_4 |

分析：

34_1 的低准确率区域非常明显。Acc < 40 的 setting 有 22 个，占 62.86%。最低区域主要集中在 high-severity jitter、dropout_local 和 dropout_global。

---

## 16. 关键困难 setting

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 18.53 | 最高 severity 坐标扰动，当前最低 setting |
| jitter_3 | 23.66 | 中高 severity 坐标扰动 |
| dropout_local_4 | 23.91 | 高 severity 局部缺失 |
| dropout_global_4 | 25.19 | 高 severity 全局缺失 |
| jitter_2 | 28.66 | severity=2 jitter |
| dropout_local_3 | 27.97 | 中高 severity 局部缺失 |
| dropout_global_3 | 30.88 | 中高 severity 全局缺失 |
| dropout_local_2 | 31.47 | severity=2 局部缺失 |

分析：

34_1 中最困难 setting 是 jitter_4，准确率只有 18.53。高 severity dropout_local 和 dropout_global 也显著困难。

这些困难 setting 是后续 34_2 和 34_3 判断 cache 是否有效的重要观察对象。

---

## 17. 与 33_1 ScanObjNN clean hardest 的关系

33_1 是 Uni3D 在 ScanObjNN clean hardest 上的 Zero-shot 结果；34_1 是 Uni3D 在 ScanObjNN-C hardest all35 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | Accuracy / Avg |
|---|---|---|---:|
| 33_1_uni3d_scanobjnn_clean_hardest_zs | ScanObjNN clean hardest | Zero-shot | 45.63 |
| 34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs | ScanObjNN-C hardest S2 Avg | Zero-shot | 37.75 |
| 34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs | ScanObjNN-C hardest all35 Avg | Zero-shot | 37.66 |

当前结果说明：从 ScanObjNN clean hardest 到 ScanObjNN-C hardest corruption，Uni3D Zero-shot 明显下降。

| 比较 | 变化 |
|---|---:|
| 34_1 S2 Avg - 33_1 clean | 37.75 - 45.63 = -7.88 |
| 34_1 all35 Avg - 33_1 clean | 37.66 - 45.63 = -7.97 |

分析：

ScanObjNN-C hardest corruption 在 clean hardest 的基础上进一步降低 Uni3D Zero-shot 性能。34_1 是后续 34_2 Global Cache 和 34_3 Hierarchical Cache 的必要基础对照。

---

## 18. 与其他 backbone 的 ScanObjNN-C hardest Zero-shot 关系

34_1 可以与前面 ULIP、OpenShape 等 ScanObjNN-C hardest Zero-shot 结果进行横向比较。

| Backbone | 实验编号 | ScanObjNN-C hardest S2 Avg | ScanObjNN-C hardest all35 Avg |
|---|---|---:|---:|
| ULIP | 04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs | 23.91 | 23.65 |
| OpenShape | 24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs | 32.75 | 32.72 |
| Uni3D | 34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs | 37.75 | 37.66 |

分析：

Uni3D 在 ScanObjNN-C hardest Zero-shot 上明显强于 ULIP 和 OpenShape。当前 Uni3D all35 Avg 为 37.66，比 OpenShape 的 32.72 高 +4.94。

这说明在真实扫描 hardest corruption 场景中，Uni3D 是当前最强 backbone，但其 Zero-shot 绝对准确率仍然不高，说明该数据设置本身非常困难。

---

## 19. 与后续子实验的关系

34_1 是 34 组第一个子实验，因此没有前序 34 组子实验可供对比。

它后续将作为以下实验的基线：

| 后续实验 | 对比方式 |
|---|---|
| 34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global | 与 34_1 比较，评估 Global Cache 在 Uni3D × ScanObjNN-C hardest 上的鲁棒性增益 |
| 34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local | 与 34_1 和 34_2 比较，评估完整 Point-Cache 及 Local Cache 额外影响 |

原文中 Uni3D 在 ScanObjNN-C hardest severity=2 上 cache 是有正增益的：Zero-shot Avg 为 37.38，+Global Cache Avg 为 42.03，+Hierarchical Cache Avg 为 43.10。因此，后续 34_2 和 34_3 的重点是观察：

1. Global Cache 是否提升；
2. Local Cache 是否在 Global Cache 基础上继续提升；
3. 完整 Point-Cache 是否接近原文 43.10；
4. 当前方法趋势是否保持 Zero-shot < Global < Global + Local。

---

## 20. 结果含义分析

34_1 的意义不只是给出一个 corrupted accuracy，而是说明 Uni3D 在 ScanObjNN-C hardest 上的基础鲁棒性仍然有限。

| 观察 | 含义 |
|---|---|
| 34_1 all35 Avg = 37.66 | Uni3D 在 ScanObjNN-C hardest 上的 Zero-shot 总体鲁棒性 |
| 34_1 S2 Avg = 37.75 | 与原文 Table 7 对齐的复现结果 |
| 比原文 S2 高 +0.37 | 数值高度接近 |
| 比 33_1 clean 低 -7.97 | corruption 明显削弱 Uni3D zero-shot |
| jitter 最难 | 坐标扰动是最强失败模式 |
| dropout_local / dropout_global 也很难 | 点云缺失是重要失败模式 |
| metadata 已修正 | 当前结果可以归档 |

因此，34_1 是后续 34_2 和 34_3 判断 cache 是否能改善 ScanObjNN-C hardest corruption robustness 的必要基础。

---

## 21. 对后续 MCM-PC 的启发

当前 34_1 对后续 MCM-PC 方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| ScanObjNN-C hardest Zero-shot all35 只有 37.66 | 真实扫描 corrupted setting 非常困难 |
| jitter 是最困难 corruption | 坐标扰动需要专门鲁棒机制 |
| dropout_local / dropout_global 也很困难 | 点云缺失需要额外处理 |
| Uni3D 强于 OpenShape，但绝对准确率仍低 | 强 backbone 仍需要测试时适应 |
| 34 组必须使用 scanobjnn checkpoint | checkpoint 是 Uni3D ScanObjNN 系列关键设置 |
| loader root 与 summary root 不完全相同 | 复制 runner 时必须特别检查数据加载逻辑和 summary metadata |

这对 MCM-PC 很重要：后续方法若希望在最困难的真实扫描 corruption 场景中取得改进，必须重点处理 high-severity jitter 和 dropout 造成的错误预测。

---

## 22. 阶段性结论

本实验完成了 Uni3D × ScanObjNN-C hardest all35 的 Zero-shot baseline 复现。

主要结论如下：

1. 34_1 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前使用 checkpoint 为 weights/uni3d/scanobjnn/model.pt。
3. 当前 metadata 已修正并确认正确：dataset=sonn_c，data_root=data/sonn_c/hardest，sonn_variant=hardest。
4. 当前 severity=2 Average 为 37.75，原文 Point-Cache Table 7 中 Uni3D Zero-shot Avg 为 37.38，差异 +0.37。
5. 当前 all35 Average 为 37.66，是本实验额外统计的 35 个 corrupted setting 总平均。
6. 当前结果与原文高度接近，可以认为 34_1 复现有效。
7. 相比 33_1 ScanObjNN clean hardest 的 45.63，34_1 all35 Average 下降到 37.66，下降 -7.97。
8. jitter 是最困难 corruption，平均准确率只有 28.85。
9. jitter_4 是全部 35 个 setting 中最低结果，只有 18.53。
10. dropout_local 和 dropout_global 也是主要困难 corruption。
11. 本实验是 34_2 Global Cache 和 34_3 Hierarchical Cache 的基础对照，不在本文件中展开完整 34 组方法间对比。
12. 33 / 34 组后续应统一使用 weights/uni3d/scanobjnn/model.pt。

---

## 23. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh 1

---

## 24. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f6 | sort -u | wc -l

tail -n +2 results/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f13 | sort | uniq -c

head -2 results/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs/summary.csv

cat results/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs/summary.csv
