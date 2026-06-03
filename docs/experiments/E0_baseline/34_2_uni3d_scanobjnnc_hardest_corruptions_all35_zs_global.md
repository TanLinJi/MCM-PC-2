# 34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global

## 1. 实验目的

本实验用于复现 Uni3D 在 ScanObjNN-C hardest 全部 35 个损坏设置上使用 Zero-shot + Global Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global |
| Backbone | Uni3D |
| Dataset | ScanObjNN-C hardest |
| Dataset 参数 | sonn_c |
| SONN variant | hardest |
| Method | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |

本实验是在 34_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存是否能够提升 Uni3D 在 ScanObjNN-C hardest corrupted setting 上的鲁棒性。

需要特别注意：原文 Point-Cache Table 7 报告的是 ScanObjNN-C hardest 在 severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原文 Point-Cache Table 7 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

本文件只记录 34_2 本身，并与前序子实验 34_1 进行对比。完整 34 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 34 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | Uni3D |
| 数据集参数 | sonn_c |
| 数据目录 | data/sonn_c |
| 实际 h5 文件目录 | data/sonn_c/hardest |
| SONN variant | hardest |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 优化 runner | Point-Cache/runners/baseline/run_uni3d_scanobjnnc_hardest_corruptions_all35.py |
| 方法脚本 | Point-Cache/scripts/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/34_run_uni3d_scanobjnnc_hardest_corruptions_all35_common.sh |
| cache_type | global |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |
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
| 每个 cor_type 都重新启动 Python、重新加载模型 | 模型只加载一次，每个 cor_type 重新创建 DataLoader、重新初始化 Global Cache |
| bash 通过 tee 生成单个 cor_type 的 log | Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv | summary.csv 的列结构保持不变，每个 cor_type 一行 |

该优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、Global Cache 初始化、预测逻辑和输出结构。

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

当前 34_2 summary 已确认 metadata 正确。

---

## 5. 方法说明

34_2 在 Zero-shot logits 的基础上加入 Global Cache logits。

| 组成部分 | 是否使用 |
|---|---:|
| Text prototype / 文本原型 | 是 |
| Point cloud global feature | 是 |
| Zero-shot logits | 是 |
| Global Cache logits | 是 |
| Local Cache logits | 否 |
| Hierarchical Cache | 否 |

Global Cache 的基本作用是：在测试过程中动态缓存高置信度样本的全局点云特征和伪标签，然后对后续样本进行全局特征检索，生成 cache logits，并与 zero-shot logits 融合。

34_2 与 34_1 的主要区别如下：

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 34_1 | 是 | 否 | 否 |
| 34_2 | 是 | 是 | 否 |

因此，34_2 可以用于单独评估 Global Cache 在 Uni3D × ScanObjNN-C hardest 上的影响。

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

Point-Cache/results/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_{cor_type}_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 42.31 | 用于论文对齐 | 与原文 Point-Cache Table 7 对比 |
| all35 Average | 41.98 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性和 metadata 看，34_2 现在是干净结果。

---

## 10. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 51.35 | 53.23 | 53.71 | 51.77 | 51.80 | 52.37 |
| add_local | 45.42 | 41.67 | 41.81 | 40.63 | 38.93 | 41.69 |
| dropout_global | 46.53 | 44.27 | 42.44 | 37.89 | 32.65 | 40.76 |
| dropout_local | 47.33 | 41.74 | 36.29 | 32.55 | 28.21 | 37.22 |
| rotate | 49.69 | 48.89 | 46.98 | 45.42 | 42.61 | 46.72 |
| scale | 44.86 | 42.96 | 40.53 | 40.28 | 39.66 | 41.66 |
| jitter | 44.55 | 36.75 | 34.42 | 28.11 | 23.32 | 33.43 |
| **Average** | **47.10** | **44.22** | **42.31** | **39.52** | **36.74** | **41.98** |

整体观察：

1. all35 Average 为 41.98，表示 Uni3D 在 ScanObjNN-C hardest 全 35 个 corrupted setting 上加入 Global Cache 后的总体鲁棒性水平。
2. severity=2 Average 为 42.31，用于和原文 Point-Cache Table 7 对齐。
3. add_global 的平均准确率最高，为 52.37。
4. jitter 的平均准确率最低，为 33.43。
5. jitter_4 为 23.32，是全部 35 个 setting 中最低结果。
6. 相比 34_1 Zero-shot，34_2 在所有 corruption 上均有明显提升。

---

## 11. 与原文结果对比

原文 Point-Cache Table 7 报告的是 ScanObjNN-C hardest 在 severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

原文 Uni3D 在 ScanObjNN-C hardest severity=2 下的 +Global Cache Average 为 42.03。

当前复现 severity=2 Average 为 42.31。

| 对比对象 | 原文 S2 Avg | 当前复现 S2 Avg | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Uni3D / ScanObjNN-C hardest / +Global Cache | 42.03 | 42.31 | +0.28 | 0.28 |

分析：

当前复现结果 42.31 比原文 42.03 高 +0.28，差异很小。因此，34_2 不只是脚本执行成功，而且数值与原文高度接近。

该结果可以作为 34_3 分析 Local Cache 额外贡献的直接对照。

---

## 12. 与前序实验 34_1 的对比

34_1 是本实验的直接前序子实验，方法为 Zero-shot，不使用缓存。

| 实验编号 | 方法 | S2 Avg | all35 Avg |
|---|---|---:|---:|
| 34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs | Zero-shot | 37.75 | 37.66 |
| 34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global | Zero-shot + Global Cache | 42.31 | 41.98 |

Global Cache 带来的当前复现变化为：

| 比较 | S2 Gain | all35 Gain |
|---|---:|---:|
| 34_2 - 34_1 | +4.56 | +4.32 |

原文中对应变化为：

| 比较 | 原文变化 |
|---|---:|
| Global Cache - Zero-shot | 42.03 - 37.38 = +4.65 |

对比：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 相对 Zero-shot 的 S2 变化 | +4.65 | +4.56 | -0.09 |

分析：

当前复现中，Global Cache 将 severity=2 Average 从 34_1 的 37.75 提升到 34_2 的 42.31，提升 +4.56。原文中 Global Cache 提升为 +4.65。二者差异仅 -0.09。

因此，34_2 不仅绝对准确率与原文对齐，Global Cache 的相对增益也与原文高度一致。这说明当前 Uni3D ScanObjNN-C hardest 的 Global Cache 复现是可靠的。

---

## 13. Severity 维度分析

### 13.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 47.10 | — | 0.00 |
| S1 | 44.22 | -2.88 | -2.88 |
| S2 | 42.31 | -1.91 | -4.79 |
| S3 | 39.52 | -2.79 | -7.58 |
| S4 | 36.74 | -2.78 | -10.36 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 47.10 下降到 36.74，总下降 10.36 个百分点。整体上，severity 越高，Uni3D + Global Cache 的准确率越低。

### 13.2 与 34_1 Zero-shot 的 severity 维度对比

| Severity | 34_1 Zero-shot Avg | 34_2 ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 42.67 | 47.10 | +4.43 |
| S1 | 40.46 | 44.22 | +3.76 |
| S2 | 37.75 | 42.31 | +4.56 |
| S3 | 35.06 | 39.52 | +4.46 |
| S4 | 32.37 | 36.74 | +4.37 |
| **all35** | **37.66** | **41.98** | **+4.32** |

分析：

Global Cache 在所有 severity 上都带来正向提升。提升幅度比较稳定，S0 到 S4 的增益均在约 +3.76 到 +4.56 之间。说明 Global Cache 并不是只改善某个特定 severity，而是在整个 severity 范围内都有效。

---

## 14. Corruption 维度分析

### 14.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 33.43 | 平均最低，高 severity 下仍然困难 |
| 2 | dropout_local | 37.22 | 点云局部缺失仍然困难 |
| 3 | dropout_global | 40.76 | 点云全局缺失仍然困难 |
| 4 | scale | 41.66 | 中等偏难 |
| 5 | add_local | 41.69 | 中等偏难 |
| 6 | rotate | 46.72 | 相对较高 |
| 7 | add_global | 52.37 | 当前最高 |

分析：

加入 Global Cache 后，jitter 仍然是最困难 corruption，但平均准确率从 34_1 的 28.85 提升到 33.43。dropout_local 和 dropout_global 仍然较难，说明点云缺失仍是 ScanObjNN-C hardest 的主要困难类型。

### 14.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|
| add_global | 51.35 | 51.80 | +0.45 | 52.37 |
| add_local | 45.42 | 38.93 | -6.49 | 41.69 |
| scale | 44.86 | 39.66 | -5.20 | 41.66 |
| rotate | 49.69 | 42.61 | -7.08 | 46.72 |
| dropout_global | 46.53 | 32.65 | -13.88 | 40.76 |
| dropout_local | 47.33 | 28.21 | -19.12 | 37.22 |
| jitter | 44.55 | 23.32 | -21.23 | 33.43 |

分析：

jitter 的退化最强，从 S0 的 44.55 下降到 S4 的 23.32，下降 -21.23。dropout_local 的退化也很强，从 47.33 下降到 28.21。说明 Global Cache 虽然提高了整体准确率，但不能完全解决 high-severity jitter 和 dropout_local 的困难。

---

## 15. Global Cache 相比 34_1 的逐项提升

| Corruption | ZS Avg | ZS + Global Avg | Gain |
|---|---:|---:|---:|
| add_global | 47.39 | 52.37 | +4.98 |
| add_local | 38.69 | 41.69 | +3.00 |
| dropout_global | 35.07 | 40.76 | +5.69 |
| dropout_local | 32.67 | 37.22 | +4.55 |
| rotate | 42.84 | 46.72 | +3.88 |
| scale | 38.14 | 41.66 | +3.52 |
| jitter | 28.85 | 33.43 | +4.58 |
| **Average** | **37.66** | **41.98** | **+4.32** |

分析：

Global Cache 对全部 7 种 corruption 均为正增益，没有负增益。提升最大的 corruption 是 dropout_global，平均提升约 +5.69；其次是 add_global、jitter 和 dropout_local。

这说明 Global Cache 在 ScanObjNN-C hardest 上是稳定有效的主模块，尤其能改善点云缺失和坐标扰动带来的性能下降。

---

## 16. 低准确率区域分析

| 条件 | 34_1 Zero-shot 数量 | 34_2 ZS + Global 数量 | 减少数量 |
|---|---:|---:|---:|
| Acc < 40 | 22 / 35 | 11 / 35 | -11 |
| Acc < 35 | 11 / 35 | 7 / 35 | -4 |
| Acc < 30 | 6 / 35 | 3 / 35 | -3 |
| Acc < 25 | 3 / 35 | 1 / 35 | -2 |

分析：

Global Cache 明显减少了低准确率区域。Acc < 40 的 setting 从 22 个减少到 11 个，减少一半。Acc < 30 的 setting 从 6 个减少到 3 个。

这说明 Global Cache 不只是提高平均准确率，也减少了极低准确率 setting 的数量。

---

## 17. 当前仍然困难的低准确率 setting

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 23.32 | 最高 severity 坐标扰动，当前最低 setting |
| dropout_local_4 | 28.21 | 高 severity 局部缺失 |
| jitter_3 | 28.11 | 中高 severity 坐标扰动 |
| dropout_global_4 | 32.65 | 高 severity 全局缺失 |
| dropout_local_3 | 32.55 | 中高 severity 局部缺失 |
| jitter_2 | 34.42 | severity=2 坐标扰动 |
| jitter_1 | 36.75 | severity=1 坐标扰动 |
| dropout_global_3 | 37.89 | 中高 severity 全局缺失 |

分析：

34_2 中最困难区域仍然集中在 high-severity jitter、dropout_local 和 dropout_global。说明 Global Cache 对这些 setting 有明显提升，但不能完全解决强坐标扰动和点云缺失问题。

---

## 18. 与 33_2 ScanObjNN clean hardest 的关系

33_2 是 Uni3D 在 ScanObjNN clean hardest 上的 Zero-shot + Global Cache 结果；34_2 是 Uni3D 在 ScanObjNN-C hardest all35 上的 Zero-shot + Global Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy / Avg |
|---|---|---|---:|
| 33_2_uni3d_scanobjnn_clean_hardest_zs_global | ScanObjNN clean hardest | ZS + Global Cache | 50.03 |
| 34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global | ScanObjNN-C hardest S2 Avg | ZS + Global Cache | 42.31 |
| 34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global | ScanObjNN-C hardest all35 Avg | ZS + Global Cache | 41.98 |

当前结果说明：从 ScanObjNN clean hardest 到 ScanObjNN-C hardest corruption，Uni3D + Global Cache 明显下降。

| 比较 | 变化 |
|---|---:|
| 34_2 S2 Avg - 33_2 clean | 42.31 - 50.03 = -7.72 |
| 34_2 all35 Avg - 33_2 clean | 41.98 - 50.03 = -8.05 |

分析：

ScanObjNN-C hardest corruption 在 clean hardest 的基础上进一步降低 Uni3D + Global Cache 性能。Global Cache 提高了 corrupted accuracy，但 corruption gap 仍然存在。

---

## 19. 与其他 backbone 的 ScanObjNN-C hardest +Global Cache 关系

34_2 可以与前面 ULIP、OpenShape 等 ScanObjNN-C hardest +Global Cache 结果进行横向比较。

| Backbone | 实验编号 | ScanObjNN-C hardest S2 Avg | ScanObjNN-C hardest all35 Avg |
|---|---|---:|---:|
| ULIP | 04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global | 26.84 | 26.60 |
| OpenShape | 24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global | 37.30 | 36.71 |
| Uni3D | 34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global | 42.31 | 41.98 |

分析：

加入 Global Cache 后，Uni3D 在 ScanObjNN-C hardest 上明显强于 ULIP 和 OpenShape。当前 Uni3D all35 Avg 为 41.98，比 OpenShape 的 36.71 高 +5.27。

这说明在真实扫描 hardest corruption 场景中，Uni3D 是当前最强 backbone，但其 corrupted accuracy 仍然存在明显提升空间。

---

## 20. 与后续子实验的关系

34_2 是 34_3 的直接前序实验。

| 后续实验 | 对比方式 |
|---|---|
| 34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local | 与 34_2 比较，评估 Local Cache 在 Global Cache 基础上的额外影响 |

本文件不展开 34_3 的实际结果。34_3 的数值及 Local Cache 额外影响应记录在 34_3 子实验文档和 34 组 summary 文档中。

需要注意的是，当前 34_2 已经证明 Global Cache 在 Uni3D × ScanObjNN-C hardest 上有效，因此 34_3 的关键问题是：

| 问题 | 说明 |
|---|---|
| Local Cache 是否能在 Global Cache 基础上继续提升？ | 比较 34_3 - 34_2 |
| 完整 Point-Cache 是否接近原文 43.10？ | 比较 34_3 与原文 |
| 当前趋势是否仍是 ZS < Global < Global + Local？ | 判断整体方法趋势 |
| Local Cache 对 high-severity jitter / dropout 是否继续有效？ | 观察 34_3 的 corruption 维度变化 |

---

## 21. 结果含义分析

34_2 的结果说明：Global Cache 在 Uni3D × ScanObjNN-C hardest all35 上非常有效，并且当前复现与原文高度一致。

| 观察 | 含义 |
|---|---|
| 34_2 all35 Avg = 41.98 | Uni3D + Global Cache 在 ScanObjNN-C hardest 上的总体结果 |
| 34_2 S2 Avg = 42.31 | 与原文 Table 7 对齐的复现结果 |
| 原文 +Global Avg = 42.03 | 当前只高 +0.28，复现高度接近 |
| 相比 34_1 S2 提升 +4.56 | Global Cache 有明确正增益 |
| 原文 Global 增益为 +4.65 | 当前增益与原文高度一致 |
| 7 种 corruption 均有正增益 | Global Cache 提升稳定 |
| 低准确率 setting 数量减少 | Global Cache 不只提高平均值，也改善低性能区域 |

因此，34_2 是 34 组中确认 Global Cache 复现可靠的关键实验。

---

## 22. 对后续 MCM-PC 的启发

当前 34_2 对后续 MCM-PC 方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| Global Cache 在 S2 上提升 +4.56 | 全局缓存是稳定主模块 |
| Global Cache 在 all35 上提升 +4.32 | 全局缓存对整体 corruption 范围有效 |
| 7 种 corruption 均有正增益 | Global Cache 稳定性强 |
| jitter 仍然最低 | 坐标扰动需要专门鲁棒机制 |
| dropout_local / dropout_global 仍然困难 | 点云缺失需要额外处理 |
| 低准确率 setting 明显减少 | cache 有助于降低极端失败区域 |
| 34 组必须使用 scanobjnn checkpoint | checkpoint 是 Uni3D ScanObjNN 系列关键设置 |

这对 MCM-PC 很重要：后续方法不应削弱 Global Cache，而应在保留其稳定增益的基础上，引入更强的局部证据筛选、错误伪标签抑制或 corruption-aware 调节机制。

---

## 23. 阶段性结论

本实验完成了 Uni3D × ScanObjNN-C hardest all35 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 34_2 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前使用 checkpoint 为 weights/uni3d/scanobjnn/model.pt。
3. 当前 metadata 已修正并确认正确：dataset=sonn_c，data_root=data/sonn_c/hardest，sonn_variant=hardest。
4. 当前 severity=2 Average 为 42.31，原文 Point-Cache Table 7 中 Uni3D +Global Cache Avg 为 42.03，差异 +0.28。
5. 当前 all35 Average 为 41.98，是本实验额外统计的 35 个 corrupted setting 总平均。
6. 当前结果与原文高度接近，可以认为 34_2 复现有效。
7. 相比 34_1 Zero-shot，34_2 的 severity=2 Average 提升 +4.56，all35 Average 提升 +4.32。
8. 当前 Global Cache 的 severity=2 增益 +4.56，与原文 +4.65 几乎完全一致。
9. Global Cache 对 7 种 corruption 均为正增益。
10. Global Cache 明显减少了低准确率 setting 数量，Acc < 40 的 setting 从 22 个减少到 11 个。
11. jitter 仍然是最困难 corruption，Global Cache 后平均仍只有 33.43。
12. jitter_4 是全部 35 个 setting 中最低结果，只有 23.32。
13. 本实验是 34_3 分析 Local Cache 额外贡献的直接对照，不在本文件中展开完整 34 组方法间对比。
14. 33 / 34 组后续应统一使用 weights/uni3d/scanobjnn/model.pt。

---

## 24. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh 1

---

## 25. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f6 | sort -u | wc -l

tail -n +2 results/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f13 | sort | uniq -c

head -2 results/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv

cat results/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv
