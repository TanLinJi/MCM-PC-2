# 24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs

## 1. 实验目的

本实验用于复现 OpenShape 在 ScanObjNN-C hardest 全部 35 个损坏设置上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs |
| Backbone | OpenShape |
| Dataset | ScanObjNN-C hardest |
| Dataset 参数 | sonn_c |
| SONN variant | hardest |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 OpenShape 在 ScanObjNN-C hardest corrupted setting 上的无缓存基础鲁棒性。该结果后续会作为 24_2 和 24_3 的对照基线，但本文件只记录 24_1 本身，不展开完整 24 组的方法间对比。

需要特别注意：原文 Point-Cache Table 7 只报告 corruption severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原文 Point-Cache Table 7 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

---

## 2. 当前实现方式

本实验的外部命名规则如下：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs |
| 方法脚本 | Point-Cache/scripts/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/24_run_openshape_scanobjnnc_hardest_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs/ |

本实验是 all35 实验，因此使用优化 runner：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 |
| 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 |
| 模型只加载一次，每个 cor_type 重新创建 DataLoader |
| bash 通过 tee 生成单个 cor_type 的 log |
| Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv |
| summary.csv 的列结构保持不变 |

该优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、预测逻辑和输出结构。

---

## 3. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | OpenShape |
| 数据集参数 | sonn_c |
| 数据目录 | data/sonn_c |
| 数据集变体 | hardest |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 优化 runner | runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |
| Severity 编号 | 0, 1, 2, 3, 4 |
| 损坏组合数 | 7 × 5 = 35 |
| 输入点数 | 1024 |
| OpenShape version | vitg14 |
| OpenShape 权重 | weights/openshape/openshape-pointbert-vitg14-rgb/model.pt |
| OpenCLIP 权重 | weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 1 |

本实验使用 `sonn_c` 作为 dataset 参数，并指定：

| 参数 | 值 |
|---|---|
| sonn_variant | hardest |
| cor_type | 由 runner 内部循环 35 个 corruption setting |

实际读取文件形式为：

data/sonn_c/hardest/{corruption}_{severity}.h5

---

## 4. 损坏类型

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

## 5. 输出结构

输出目录：

Point-Cache/results/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs_add_global_0_YYYYMMDD_HHMMSS.log

也就是说，优化 runner 虽然只启动一次 Python，但仍然会为 35 个 cor_type 生成 35 个独立 log。

---

## 6. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 35 | 35 | 说明 35 个 cor_type 全部完成 |
| summary 中唯一 cor_type 数 | 35 | 35 | 说明没有漏跑或重复写入 |
| summary 中唯一 log_path 数 | 35 | 35 | 说明每个 cor_type 都有独立日志路径 |
| logs 目录当前 .log 文件数 | 35 | 35 | 说明日志目录状态正常 |
| status=done 数 | 35 | 35 | 说明没有失败项 |
| severity=2 Average | 32.75 | 用于论文对齐 | 与原文 Point-Cache Table 7 对比 |
| all35 Average | 32.72 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，24_1 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ScanObjNN-C hardest severity=2 参考值进行对比。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 39.59 | 36.19 | 35.32 | 33.41 | 32.93 | 35.49 |
| add_local | 38.41 | 34.21 | 32.96 | 31.99 | 30.71 | 33.66 |
| dropout_global | 41.12 | 40.94 | 40.63 | 37.65 | 24.08 | 36.88 |
| dropout_local | 36.40 | 31.40 | 27.17 | 23.21 | 19.47 | 27.53 |
| rotate | 41.19 | 40.11 | 37.51 | 34.66 | 31.09 | 36.91 |
| scale | 37.54 | 37.75 | 35.91 | 34.66 | 33.31 | 35.83 |
| jitter | 37.47 | 27.86 | 19.78 | 15.27 | 13.43 | 22.76 |
| **Average** | **38.82** | **35.49** | **32.75** | **30.12** | **26.43** | **32.72** |

整体观察：

1. all35 Average 为 32.72，表示 OpenShape 在 ScanObjNN-C hardest 全 35 个 corrupted setting 上的 Zero-shot 鲁棒性水平。
2. severity=2 Average 为 32.75，用于和原文 Point-Cache Table 7 对齐。
3. rotate 的平均准确率最高，为 36.91；dropout_global 平均为 36.88，与 rotate 非常接近。
4. jitter 的平均准确率最低，为 22.76。
5. jitter_4 为 13.43，是全部 35 个 setting 中最低的结果。
6. dropout_local 的平均准确率也很低，为 27.53，是除 jitter 之外最困难的 corruption。

---

## 8. 与原文结果对比

原文 Point-Cache Table 7 报告的是 ScanObjNN-C hardest / S-PB T50-RS-C 在 severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 35.32 | 32.41 | +2.91 | 2.91 |
| add_local | 32.96 | 35.60 | -2.64 | 2.64 |
| dropout_global | 40.63 | 37.80 | +2.83 | 2.83 |
| dropout_local | 27.17 | 27.34 | -0.17 | 0.17 |
| rotate | 37.51 | 36.61 | +0.90 | 0.90 |
| scale | 35.91 | 35.22 | +0.69 | 0.69 |
| jitter | 19.78 | 18.88 | +0.90 | 0.90 |
| **Average** | **32.75** | **31.98** | **+0.77** | **1.58 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.77 |
| MAE | 1.58 |
| RMSE | 1.96 |
| Max Abs Diff | 2.91 |

分析：

当前复现的 severity=2 Average 为 32.75，原文为 31.98，差异为 +0.77。整体略高于原文，但仍在可接受范围内。

逐 corruption 看，add_global 和 dropout_global 明显高于原文，分别为 +2.91 和 +2.83；add_local 低于原文 -2.64。也就是说，当前 24_1 的平均值略高并不是所有 corruption 均匀偏高，而是存在一定正负波动。

因此，24_1 应记录为：结果有效、平均值略高于原文、逐 corruption 存在一定波动。

---

## 9. Severity 维度分析

### 9.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 38.82 | — | 0.00 |
| S1 | 35.49 | -3.33 | -3.33 |
| S2 | 32.75 | -2.74 | -6.07 |
| S3 | 30.12 | -2.63 | -8.70 |
| S4 | 26.43 | -3.69 | -12.39 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 38.82 下降到 26.43，总下降 12.39 个百分点。整体上，severity 越高，OpenShape Zero-shot 准确率越低，说明 corruption severity 对 ScanObjNN-C hardest 性能有持续影响。

S3 到 S4 的下降最大，为 -3.69，说明最高 severity 会进一步明显破坏 OpenShape 的识别能力。

### 9.2 与 23_1 ScanObjNN clean hardest 的关系

| 设置 | Accuracy |
|---|---:|
| 23_1 ScanObjNN clean hardest Zero-shot | 41.88 |
| 24_1 ScanObjNN-C hardest S2 Average | 32.75 |
| 24_1 ScanObjNN-C hardest all35 Average | 32.72 |

对比：

| 比较 | 变化 |
|---|---:|
| S2 Average - clean hardest | -9.13 |
| all35 Average - clean hardest | -9.16 |

分析：

在 ScanObjNN clean hardest 的基础上进一步施加 corruption 后，OpenShape Zero-shot 从 41.88 下降到 all35 Avg 32.72，下降 -9.16。

这说明 ScanObjNN hardest 本身已经困难，而 corruption 会进一步加剧真实扫描域偏移。24_1 是后续 24_2 Global Cache 和 24_3 Hierarchical Cache 的必要基础对照。

---

## 10. Corruption 难度分析

### 10.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 22.76 | 平均最低，高 severity 下下降最明显 |
| 2 | dropout_local | 27.53 | 局部缺失造成明显影响 |
| 3 | add_local | 33.66 | 局部异常点也较难 |
| 4 | add_global | 35.49 | 中等难度 |
| 5 | scale | 35.83 | 相对更稳定 |
| 6 | dropout_global | 36.88 | 平均较高，但 S4 明显下降 |
| 7 | rotate | 36.91 | 当前最高，略高于 dropout_global |

分析：

jitter 是最困难的 corruption，平均准确率只有 22.76，远低于其他 corruption。尤其 jitter_4 只有 13.43，说明强坐标扰动对 OpenShape 在 ScanObjNN-C hardest 上的点云表征破坏非常明显。

dropout_local 是第二困难 corruption，平均准确率为 27.53。这说明真实扫描 hardest 数据在叠加局部缺失后，局部结构进一步受损，Zero-shot 识别明显退化。

### 10.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 37.54 | 33.31 | 4.23 | 11.27% | 35.83 |
| add_global | 39.59 | 32.93 | 6.66 | 16.82% | 35.49 |
| add_local | 38.41 | 30.71 | 7.70 | 20.05% | 33.66 |
| rotate | 41.19 | 31.09 | 10.10 | 24.52% | 36.91 |
| dropout_global | 41.12 | 24.08 | 17.04 | 41.44% | 36.88 |
| dropout_local | 36.40 | 19.47 | 16.93 | 46.51% | 27.53 |
| jitter | 37.47 | 13.43 | 24.04 | 64.16% | 22.76 |

分析：

scale 最稳定，从 S0 到 S4 只下降 4.23。jitter 的退化最强，从 37.47 下降到 13.43，绝对下降 24.04，相对下降 64.16%。

dropout_global 和 dropout_local 在 S4 也有明显退化，说明高 severity 点云缺失是 ScanObjNN-C hardest 上的重要困难来源。

---

## 11. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | dropout_local | 36.40 | rotate | 41.19 | 4.79 |
| S1 | jitter | 27.86 | dropout_global | 40.94 | 13.08 |
| S2 | jitter | 19.78 | dropout_global | 40.63 | 20.85 |
| S3 | jitter | 15.27 | dropout_global | 37.65 | 22.38 |
| S4 | jitter | 13.43 | scale | 33.31 | 19.88 |

分析：

从 S1 开始，jitter 成为明显最困难的 corruption。S2、S3、S4 中 jitter 都是最低结果。best-worst gap 在 S2 达到 20.85，在 S3 达到 22.38，说明中高 severity 下不同 corruption 的难度差异非常明显。

---

## 12. 低准确率区域分析

| 条件 | 数量 | 占比 | 主要涉及 corruption |
|---|---:|---:|---|
| Acc < 40 | 30 / 35 | 85.71% | 几乎所有 corruption |
| Acc < 35 | 17 / 35 | 48.57% | jitter, dropout_local, add_local, add_global, rotate, scale, dropout_global |
| Acc < 30 | 7 / 35 | 20.00% | jitter, dropout_local, dropout_global |
| Acc < 25 | 6 / 35 | 17.14% | jitter, dropout_local_3/4, dropout_global_4 |
| Acc < 20 | 4 / 35 | 11.43% | jitter_2/3/4, dropout_local_4 |
| Acc < 15 | 1 / 35 | 2.86% | jitter_4 |

分析：

24_1 的低准确率区域非常明显。Acc < 40 的 setting 多达 30 个，说明 OpenShape Zero-shot 在 ScanObjNN-C hardest 上整体处于较困难状态。

最严重的低准确率区域主要集中在 high-severity jitter 和 dropout_local。jitter_4 为 13.43，是唯一低于 15 的 setting。

---

## 13. 关键困难 setting

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 13.43 | 最高 severity 坐标扰动，最困难 setting |
| jitter_3 | 15.27 | 中高 severity 坐标扰动，极低 |
| dropout_local_4 | 19.47 | 高 severity 局部缺失，极低 |
| jitter_2 | 19.78 | severity=2 jitter，明显低于多数 corruption |
| dropout_local_3 | 23.21 | 中高 severity 局部缺失 |
| dropout_global_4 | 24.08 | 高 severity 全局缺失 |
| dropout_local_2 | 27.17 | severity=2 局部缺失 |
| jitter_1 | 27.86 | 低中 severity jitter 仍偏低 |

分析：

jitter_3 和 jitter_4 是最顽固失败点。它们的准确率分别只有 15.27 和 13.43。说明强坐标扰动会严重破坏真实扫描 hardest 数据上的点云表征。

dropout_local_4、dropout_local_3 和 dropout_global_4 也很困难，说明高 severity 的局部/全局缺失同样是后续方法需要重点处理的问题。

---

## 14. 与前序实验的关系

24_1 的直接前序数据设置是 23_1，即 OpenShape 在 ScanObjNN clean hardest 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | Accuracy |
|---|---|---|---:|
| 23_1_openshape_scanobjnn_clean_hardest_zs | ScanObjNN clean hardest | Zero-shot | 41.88 |
| 24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs | ScanObjNN-C hardest all35 | Zero-shot | 32.72 all35 / 32.75 S2 |

当前结果说明：从 clean hardest 到 corrupted hardest，OpenShape Zero-shot 明显下降。

| 比较 | 变化 |
|---|---:|
| 24_1 S2 Avg - 23_1 clean hardest | -9.13 |
| 24_1 all35 Avg - 23_1 clean hardest | -9.16 |

分析：

23_1 已经说明 OpenShape 在 ScanObjNN clean hardest 上远低于 ModelNet clean，24_1 进一步说明：在真实扫描 hardest 上叠加 corruption 后，性能还会进一步下降约 9 个百分点。

因此，24_1 是 OpenShape 在当前 baseline 复现实验中最困难的数据设置之一。

---

## 15. 与 22_1 ModelNet-C 的关系

22_1 是 OpenShape 在 ModelNet-C all35 上的 Zero-shot 结果；24_1 是 OpenShape 在 ScanObjNN-C hardest all35 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 22_1_openshape_modelnetc_corruptions_all35_zs | ModelNet-C all35 | Zero-shot | 73.57 | 72.57 |
| 24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs | ScanObjNN-C hardest all35 | Zero-shot | 32.75 | 32.72 |

对比：

| 比较 | S2 变化 | all35 变化 |
|---|---:|---:|
| 24_1 - 22_1 | -40.82 | -39.85 |

分析：

ScanObjNN-C hardest 比 ModelNet-C 难得多。OpenShape 在 ModelNet-C all35 上的 Zero-shot 平均为 72.57，但在 ScanObjNN-C hardest all35 上只有 32.72，下降接近 40 个百分点。

这说明真实扫描 hardest + corruption 的组合是比 synthetic ModelNet corruption 更严苛的鲁棒性测试。

---

## 16. 与 ULIP / ULIP-2 的 ScanObjNN-C hardest 关系

24_1 可以与前面 ULIP、ULIP-2 的 ScanObjNN-C hardest Zero-shot 结果进行横向比较。

| Backbone | 实验编号 | ScanObjNN-C hardest S2 Avg | ScanObjNN-C hardest all35 Avg |
|---|---|---:|---:|
| ULIP | 04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs | 23.91 | 23.65 |
| ULIP-2 | 14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs | 26.44 | 26.46 |
| OpenShape | 24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs | 32.75 | 32.72 |

Backbone 提升：

| 比较 | S2 Avg 提升 | all35 Avg 提升 |
|---|---:|---:|
| OpenShape - ULIP | +8.84 | +9.07 |
| OpenShape - ULIP-2 | +6.31 | +6.26 |

分析：

OpenShape 在 ScanObjNN-C hardest 上明显强于 ULIP 和 ULIP-2。即使不使用 cache，OpenShape 的 all35 Avg 也比 ULIP-2 高 +6.26，比 ULIP 高 +9.07。

但 OpenShape Zero-shot 的绝对值仍然只有 32.72，说明强 backbone 不能完全解决真实扫描 corrupted hardest 的域偏移问题。

---

## 17. 与后续子实验的关系

24_1 是 24 组第一个子实验，因此没有前序 24 组子实验可供对比。

它后续将作为以下实验的基线：

| 后续实验 | 对比方式 |
|---|---|
| 24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global | 与 24_1 比较，评估 Global Cache 在 OpenShape × ScanObjNN-C hardest 上的鲁棒性增益 |
| 24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local | 与 24_1 和 24_2 比较，评估完整 Point-Cache 及 Local Cache 额外影响 |

原文中 OpenShape 在 ScanObjNN-C hardest severity=2 上 cache 是有正增益的：Zero-shot Avg 为 31.98，+Global Cache Avg 为 36.80，+Hierarchical Cache Avg 为 37.70。因此，后续 24_2 和 24_3 的重点是观察：

1. Global Cache 是否提升；
2. Local Cache 是否在 Global Cache 基础上继续提升；
3. 完整 Point-Cache 是否接近原文 37.70；
4. 当前方法趋势是否保持 Zero-shot < Global < Global + Local。

---

## 18. 结果含义分析

24_1 的意义不只是给出一个 corrupted accuracy，而是说明 OpenShape 在真实扫描 hardest + corruption 场景下仍然存在显著鲁棒性问题。

| 观察 | 含义 |
|---|---|
| 24_1 all35 Avg = 32.72 | OpenShape 在 ScanObjNN-C hardest 上的 Zero-shot 总体性能 |
| 24_1 S2 Avg = 32.75 | 与原文 Table 7 对齐的复现结果 |
| 比原文 S2 高 +0.77 | 平均值略高但可接受 |
| 比 23_1 clean hardest 低 -9.16 | corruption 进一步削弱真实扫描性能 |
| 比 22_1 ModelNet-C all35 低 -39.85 | 真实扫描 corrupted hardest 远难于 synthetic corruption |
| jitter 和 dropout_local 最难 | 后续方法需重点处理强坐标扰动和局部缺失 |

因此，24_1 是后续 24_2 和 24_3 判断 cache 是否能改善真实扫描 corrupted setting 的必要基础。

---

## 19. 对后续 MCM-PC 的启发

当前 24_1 对后续 MCM-PC 方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| ScanObjNN-C hardest 是目前最困难设置之一 | 后续方法必须在真实扫描 corrupted setting 上验证 |
| OpenShape Zero-shot 仍只有 32.72 all35 | 强 backbone 仍有明显提升空间 |
| jitter 是最困难 corruption | 坐标扰动需要专门鲁棒机制 |
| dropout_local 也很困难 | 局部结构缺失需要局部证据补偿 |
| 真实扫描 corrupted setting 远难于 ModelNet-C | 方法不能只依赖 synthetic corruption 结果 |
| 24_1 与 23_1 差距明显 | corruption 会进一步放大真实扫描域偏移 |

这对 MCM-PC 很重要：后续方法如果想体现顶会级别价值，不能只在 ModelNet-C 上表现好，还必须在 ScanObjNN-C hardest 这种真实扫描 corrupted setting 上展示稳定收益。

---

## 20. 阶段性结论

本实验完成了 OpenShape × ScanObjNN-C hardest all35 的 Zero-shot baseline 复现。

主要结论如下：

1. 24_1 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 32.75，原文 Point-Cache Table 7 中 OpenShape Zero-shot Avg 为 31.98，差异 +0.77。
3. 当前 all35 Average 为 32.72，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果略高于原文，但整体可接受，可以认为 24_1 复现有效。
5. 逐 corruption 看，add_global 和 dropout_global 高于原文较多，add_local 低于原文较多，因此存在一定正负波动。
6. 相比 23_1 ScanObjNN clean hardest 的 41.88，24_1 all35 Average 下降到 32.72，下降 -9.16。
7. 相比 22_1 ModelNet-C all35 的 72.57，24_1 低 -39.85，说明 ScanObjNN-C hardest 远难于 ModelNet-C。
8. jitter 是最困难 corruption，平均准确率只有 22.76。
9. jitter_4 是全部 35 个 setting 中最低结果，只有 13.43。
10. dropout_local 是第二困难 corruption，平均准确率为 27.53。
11. 该实验是 24_2 Global Cache 和 24_3 Hierarchical Cache 的基础对照，不在本文件中展开完整 24 组方法间对比。

---

## 21. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh 1

---

## 22. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs/summary.csv
