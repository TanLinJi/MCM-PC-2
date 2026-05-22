# 24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local

## 1. 实验目的

本实验用于复现 OpenShape 在 ScanObjNN-C hardest 全部 35 个损坏设置上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local |
| Backbone | OpenShape |
| Dataset | ScanObjNN-C hardest |
| Dataset 参数 | sonn_c |
| SONN variant | hardest |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 24_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证完整 Point-Cache 对 OpenShape 在 ScanObjNN-C hardest corrupted setting 上的影响。

需要特别注意：24 组中 Global Cache 和 Local Cache 都有贡献。Global Cache 是主要提升来源，Local Cache 在 Global Cache 基础上继续带来稳定额外提升。当前 24_3 相比 24_1 Zero-shot 的 all35 Average 提升 +5.11，相比 24_2 Global Cache 额外提升 +1.13。因此，24 组与 22 组不同：22 组 OpenShape × ModelNet-C 中 Local Cache 额外贡献接近于零；24 组 OpenShape × ScanObjNN-C hardest 中 Local Cache 有明确正贡献。

本文件只记录 24_3 本身，并与前序子实验 24_1 和 24_2 进行对比。完整 24 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 24 组 summary 文档中。

---

## 2. 当前实现方式

本实验的外部命名规则如下：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local |
| 方法脚本 | Point-Cache/scripts/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/24_run_openshape_scanobjnnc_hardest_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local/ |

本实验是 all35 实验，因此使用优化 runner：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 |
| 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 |
| 模型只加载一次，每个 cor_type 重新创建 DataLoader、重新初始化 Global Cache 和 Local Cache |
| bash 通过 tee 生成单个 cor_type 的 log |
| Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv |
| summary.csv 的列结构保持不变 |

该优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、Global Cache 初始化、Local Cache 初始化、预测逻辑和输出结构。

---

## 3. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | OpenShape |
| 数据集参数 | sonn_c |
| 数据目录 | data/sonn_c |
| 数据集变体 | hardest |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 优化 runner | runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py |
| cache_type | hierarchical |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |
| Severity 编号 | 0, 1, 2, 3, 4 |
| 损坏组合数 | 7 × 5 = 35 |
| 输入点数 | 1024 |
| Global Cache shot_capacity | 3 |
| Local Cache shot_capacity | 3 |
| Global / Local alpha | 4.0 |
| Global / Local beta | 3.0 |
| KMeans 聚类数 | 3 |
| OpenShape version | vitg14 |
| OpenShape 权重 | weights/openshape/openshape-pointbert-vitg14-rgb/model.pt |
| OpenCLIP 权重 | weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 0 |

本实验使用 `sonn_c` 作为 dataset 参数，并指定：

| 参数 | 值 |
|---|---|
| sonn_variant | hardest |
| cor_type | 由 runner 内部循环 35 个 corruption setting |

实际读取文件形式为：

data/sonn_c/hardest/{corruption}_{severity}.h5

---

## 4. 方法说明

24_3 在 Zero-shot logits 的基础上同时加入 Global Cache logits 和 Local Cache logits。

| 组成部分 | 是否使用 |
|---|---:|
| Text prototype / 文本原型 | 是 |
| Point cloud global feature | 是 |
| Zero-shot logits | 是 |
| Global Cache logits | 是 |
| Local Cache logits | 是 |
| Hierarchical Cache | 是 |

完整 Point-Cache 的预测由三部分组成：

| 组成部分 | 作用 |
|---|---|
| Zero-shot logits | 来自 OpenShape 的原始文本-点云相似度预测 |
| Global Cache logits | 基于全局点云特征的测试时缓存检索结果 |
| Local Cache logits | 基于局部 patch / 局部聚类特征的测试时缓存检索结果 |

24_3 与前两个子实验的关系如下：

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 24_1 | 是 | 否 | 否 |
| 24_2 | 是 | 是 | 否 |
| 24_3 | 是 | 是 | 是 |

因此，24_3 可以用于评估完整 Point-Cache 在 OpenShape × ScanObjNN-C hardest 上的最终表现，并判断 Local Cache 是否在 Global Cache 基础上继续带来额外贡献。

---

## 5. 损坏类型

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

## 6. 输出结构

输出目录：

Point-Cache/results/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local_add_global_0_YYYYMMDD_HHMMSS.log

也就是说，优化 runner 虽然只启动一次 Python，但仍然会为 35 个 cor_type 生成 35 个独立 log。

---

## 7. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 35 | 35 | 说明 35 个 cor_type 全部完成 |
| summary 中唯一 cor_type 数 | 35 | 35 | 说明没有漏跑或重复写入 |
| summary 中唯一 log_path 数 | 35 | 35 | 说明每个 cor_type 都有独立日志路径 |
| logs 目录当前 .log 文件数 | 35 | 35 | 说明日志目录状态正常 |
| status=done 数 | 35 | 35 | 说明没有失败项 |
| severity=2 Average | 38.63 | 用于论文对齐 | 与原文 Point-Cache Table 7 对比 |
| all35 Average | 37.84 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，24_3 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ScanObjNN-C hardest severity=2 参考值进行对比。

---

## 8. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 46.09 | 43.89 | 40.94 | 41.15 | 41.08 | 42.63 |
| add_local | 44.00 | 41.40 | 40.18 | 37.44 | 36.50 | 39.90 |
| dropout_global | 45.25 | 45.28 | 43.79 | 38.06 | 28.07 | 40.09 |
| dropout_local | 42.89 | 38.27 | 35.67 | 29.35 | 24.46 | 34.08 |
| rotate | 44.66 | 42.89 | 42.30 | 37.65 | 35.84 | 40.67 |
| scale | 42.89 | 39.90 | 40.84 | 37.96 | 35.88 | 39.49 |
| jitter | 42.96 | 34.94 | 24.91 | 18.91 | 16.90 | 28.01 |
| **Average** | **44.11** | **40.09** | **38.63** | **34.51** | **31.85** | **37.84** |

整体观察：

1. all35 Average 为 37.84，表示 OpenShape 在 ScanObjNN-C hardest 全 35 个 corrupted setting 上使用完整 Point-Cache 后的整体鲁棒性水平。
2. severity=2 Average 为 38.63，用于和原文 Point-Cache Table 7 对齐。
3. add_global 的平均准确率最高，为 42.63。
4. jitter 的平均准确率最低，为 28.01。
5. jitter_4 为 16.90，是全部 35 个 setting 中最低的结果。
6. 相比 24_1 Zero-shot 和 24_2 Global Cache，24_3 取得最高的 S2 Average 和 all35 Average。

---

## 9. 与原文结果对比

原文 Point-Cache Table 7 报告的是 ScanObjNN-C hardest / S-PB T50-RS-C 在 severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 40.94 | 40.91 | +0.03 | 0.03 |
| add_local | 40.18 | 39.24 | +0.94 | 0.94 |
| dropout_global | 43.79 | 43.03 | +0.76 | 0.76 |
| dropout_local | 35.67 | 35.22 | +0.45 | 0.45 |
| rotate | 42.30 | 43.06 | -0.76 | 0.76 |
| scale | 40.84 | 37.40 | +3.44 | 3.44 |
| jitter | 24.91 | 25.05 | -0.14 | 0.14 |
| **Average** | **38.63** | **37.70** | **+0.93** | **0.93 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.93 |
| MAE | 0.93 |
| RMSE | 1.44 |
| Max Abs Diff | 3.44 |

分析：

当前复现的 severity=2 Average 为 38.63，原文为 37.70，差异为 +0.93。整体略高于原文，但仍在可接受范围内。

逐 corruption 看，scale 明显高于原文 +3.44，是主要正偏差来源；rotate 和 jitter 略低于原文。除 scale 外，其余 corruption 与原文差异都不大。

因此，24_3 应记录为：结果有效、平均值略高于原文、scale 项偏高较明显，但整体趋势和方法增益结构都与原文一致。

---

## 10. Severity 维度分析

### 10.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 44.11 | — | 0.00 |
| S1 | 40.09 | -4.02 | -4.02 |
| S2 | 38.63 | -1.46 | -5.48 |
| S3 | 34.51 | -4.12 | -9.60 |
| S4 | 31.85 | -2.66 | -12.26 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 44.11 下降到 31.85，总下降 12.26 个百分点。整体上，severity 越高，完整 Point-Cache 的准确率越低。

相比 24_1 Zero-shot，完整 Point-Cache 在所有 severity 上整体上移。相比 24_2 Global Cache，完整 Point-Cache 在所有 severity 上也都有提升，说明 Local Cache 的额外贡献在 severity 维度上较稳定。

### 10.2 与前序实验的 severity 维度对比

| Severity | 24_1 Zero-shot Avg | 24_2 ZS + Global Avg | 24_3 ZS + Global + Local Avg | Gain over 24_1 | Gain over 24_2 |
|---:|---:|---:|---:|---:|---:|
| S0 | 38.82 | 42.82 | 44.11 | +5.29 | +1.29 |
| S1 | 35.49 | 39.44 | 40.09 | +4.60 | +0.65 |
| S2 | 32.75 | 37.30 | 38.63 | +5.88 | +1.33 |
| S3 | 30.12 | 33.35 | 34.51 | +4.39 | +1.16 |
| S4 | 26.43 | 30.65 | 31.85 | +5.42 | +1.20 |
| **all35** | **32.72** | **36.71** | **37.84** | **+5.11** | **+1.13** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有显著正增益，all35 平均提升 +5.11，severity=2 提升 +5.88。

Local Cache 在 Global Cache 基础上的额外贡献也在所有 severity 上为正：S0 +1.29，S1 +0.65，S2 +1.33，S3 +1.16，S4 +1.20。说明 24 组中 Local Cache 不是偶然在某个 severity 上提升，而是比较稳定地改善了整体表现。

---

## 11. Corruption 难度分析

### 11.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 28.01 | 平均最低，高 severity 下仍然最困难 |
| 2 | dropout_local | 34.08 | 局部缺失仍然很困难 |
| 3 | scale | 39.49 | 中等偏高 |
| 4 | add_local | 39.90 | 局部异常点被完整 Point-Cache 明显改善 |
| 5 | dropout_global | 40.09 | 平均较高，但 S4 明显下降 |
| 6 | rotate | 40.67 | 整体较高 |
| 7 | add_global | 42.63 | 当前最高 |

分析：

完整 Point-Cache 后，jitter 仍然是最困难 corruption，但平均准确率从 24_1 的 22.76 提升到 24_3 的 28.01，说明完整 Point-Cache 对 jitter 有明显帮助，但无法完全解决强坐标扰动问题。

dropout_local 仍然是第二困难 corruption，平均为 34.08。说明局部缺失在真实扫描 corrupted hardest 上仍然非常困难，即使使用 Local Cache 也只能部分补偿。

### 11.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| add_global | 46.09 | 41.08 | 5.01 | 10.87% | 42.63 |
| scale | 42.89 | 35.88 | 7.01 | 16.34% | 39.49 |
| add_local | 44.00 | 36.50 | 7.50 | 17.05% | 39.90 |
| rotate | 44.66 | 35.84 | 8.82 | 19.75% | 40.67 |
| dropout_global | 45.25 | 28.07 | 17.18 | 37.97% | 40.09 |
| dropout_local | 42.89 | 24.46 | 18.43 | 42.97% | 34.08 |
| jitter | 42.96 | 16.90 | 26.06 | 60.66% | 28.01 |

分析：

add_global 最稳定，从 S0 到 S4 下降 5.01。jitter 的退化最强，从 42.96 下降到 16.90，绝对下降 26.06，相对下降 60.66%。

dropout_global 和 dropout_local 在 S4 也有明显退化，说明高 severity 点云缺失仍然是完整 Point-Cache 难以完全解决的问题。

---

## 12. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | scale / dropout_local | 42.89 | add_global | 46.09 | 3.20 |
| S1 | jitter | 34.94 | dropout_global | 45.28 | 10.34 |
| S2 | jitter | 24.91 | dropout_global | 43.79 | 18.88 |
| S3 | jitter | 18.91 | add_global | 41.15 | 22.24 |
| S4 | jitter | 16.90 | add_global | 41.08 | 24.18 |

分析：

从 S1 开始，jitter 成为明显最困难 corruption。S2、S3、S4 中 jitter 都是最低结果。best-worst gap 在 S4 达到 24.18，说明高 severity 下不同 corruption 的难度差异非常明显。

完整 Point-Cache 提高了整体准确率，但没有改变 high-severity jitter 是主要失败区域这一事实。

---

## 13. 低准确率区域分析

### 13.1 低准确率 setting 数量

| 条件 | 24_1 Zero-shot 数量 | 24_2 ZS + Global 数量 | 24_3 ZS + Global + Local 数量 |
|---|---:|---:|---:|
| Acc < 40 | 30 / 35 | 22 / 35 | 15 / 35 |
| Acc < 35 | 19 / 35 | 9 / 35 | 8 / 35 |
| Acc < 30 | 8 / 35 | 6 / 35 | 6 / 35 |
| Acc < 25 | 6 / 35 | 4 / 35 | 4 / 35 |
| Acc < 20 | 4 / 35 | 2 / 35 | 2 / 35 |
| Acc < 15 | 1 / 35 | 0 / 35 | 0 / 35 |

分析：

完整 Point-Cache 明显减少了低准确率区域。Acc < 40 的 setting 从 Zero-shot 的 30 个减少到 Global Cache 的 22 个，再减少到完整 Point-Cache 的 15 个。Acc < 35 的 setting 也从 19 个减少到 8 个。

不过，Acc < 30、Acc < 25 和 Acc < 20 的极困难 setting 数量相比 24_2 没有继续减少，说明 Local Cache 提升了总体表现，但最困难的 high-severity jitter / dropout 仍然没有完全解决。

### 13.2 当前仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 16.90 | 最高 severity 坐标扰动仍然最难 |
| jitter_3 | 18.91 | 中高 severity 坐标扰动仍然极低 |
| dropout_local_4 | 24.46 | 高 severity 局部缺失仍然困难 |
| jitter_2 | 24.91 | severity=2 jitter 仍明显低于多数 corruption |
| dropout_global_4 | 28.07 | 高 severity 全局缺失仍然困难 |
| dropout_local_3 | 29.35 | 中高 severity 局部缺失仍然困难 |

分析：

24_3 中最困难区域仍然集中在 high-severity jitter，尤其是 jitter_3 和 jitter_4。说明完整 Point-Cache 对 jitter 有提升，但不能完全解决强坐标扰动问题。

dropout_local_4、dropout_local_3 和 dropout_global_4 也仍然困难，说明高 severity 缺失仍需要后续方法进一步处理。

---

## 14. 完整 Point-Cache 相比 24_1 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +6.50 | +7.70 | +5.62 | +7.74 | +8.15 | +7.14 |
| add_local | +5.59 | +7.19 | +7.22 | +5.45 | +5.79 | +6.24 |
| dropout_global | +4.13 | +4.34 | +3.16 | +0.41 | +3.99 | +3.21 |
| dropout_local | +6.49 | +6.87 | +8.50 | +6.14 | +4.99 | +6.55 |
| rotate | +3.47 | +2.78 | +4.79 | +2.99 | +4.75 | +3.76 |
| scale | +5.35 | +2.15 | +4.93 | +3.30 | +2.57 | +3.66 |
| jitter | +5.49 | +7.08 | +5.13 | +3.64 | +3.47 | +5.25 |
| **Average** | **+5.29** | **+4.60** | **+5.88** | **+4.39** | **+5.42** | **+5.11** |

分析：

完整 Point-Cache 相比 Zero-shot 的 all35 平均提升为 +5.11，severity=2 提升为 +5.88。所有 corruption 的平均值均有明显提升。

提升最大的 corruption 是 add_global，Avg Gain 为 +7.14；其次是 dropout_local，Avg Gain 为 +6.55；add_local 也有 +6.24。说明完整 Point-Cache 对异常点和局部缺失有明显帮助。

虽然 jitter 仍然最困难，但完整 Point-Cache 对 jitter 的平均提升也达到 +5.25。

---

## 15. Local Cache 相比 24_2 Global Cache 的额外提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +2.20 | +2.18 | +1.77 | +2.32 | +0.80 | +1.85 |
| add_local | +1.95 | +2.57 | +1.39 | +2.19 | +1.98 | +2.01 |
| dropout_global | +0.35 | +0.55 | +0.07 | +0.48 | +0.94 | +0.48 |
| dropout_local | +1.74 | +0.66 | +2.15 | +1.31 | +1.84 | +1.49 |
| rotate | -0.38 | +0.80 | -0.76 | -0.41 | +0.66 | -0.01 |
| scale | +2.40 | +1.11 | +1.01 | +0.80 | -1.73 | +0.72 |
| jitter | +0.77 | +2.60 | +1.87 | +0.38 | -0.28 | +1.35 |
| **Average** | **+1.29** | **+0.65** | **+1.33** | **+1.16** | **+1.20** | **+1.13** |

分析：

Local Cache 在 Global Cache 基础上的 all35 平均额外提升为 +1.13，severity=2 额外提升 +1.33。相比 22 组 OpenShape × ModelNet-C 的 Local extra 约为 0，24 组中的 Local Cache 明显更有效。

按四舍五入后的 summary 结果看，Local Cache 在 30 / 35 个 setting 上为正增益，在 5 / 35 个 setting 上为负增益。负增益主要出现在 rotate 和少数 high-severity scale / jitter setting 上。

Local Cache 额外提升最大的 corruption 是 add_local，Avg Gain 为 +2.01；其次是 add_global +1.85、dropout_local +1.49、jitter +1.35。这说明 Local Cache 对真实扫描 corrupted setting 中的局部异常点、局部缺失和强坐标扰动都有一定补偿作用。

---

## 16. 与前序实验的关系

24_3 的前序子实验包括 24_1 和 24_2。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs | ScanObjNN-C hardest all35 | Zero-shot | 32.75 | 32.72 |
| 24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global | ScanObjNN-C hardest all35 | ZS + Global Cache | 37.30 | 36.71 |
| 24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local | ScanObjNN-C hardest all35 | ZS + Global + Local | 38.63 | 37.84 |

当前结果说明：在 OpenShape × ScanObjNN-C hardest all35 上，Global Cache 和 Local Cache 都有明确正贡献。

| 比较 | S2 Avg 变化 | all35 Avg 变化 |
|---|---:|---:|
| 24_2 - 24_1 | +4.55 | +3.99 |
| 24_3 - 24_2 | +1.33 | +1.13 |
| 24_3 - 24_1 | +5.88 | +5.11 |

分析：

Global Cache 是主要提升来源，但 Local Cache 也不是弱贡献模块。完整 Point-Cache 相比 Zero-shot 提升 +5.11 all35，其中 Global Cache 提供 +3.99，Local Cache 额外提供 +1.13。

这与 23 组和 22 组的关系很重要：

| 组别 | 数据设置 | Global extra | Local extra |
|---|---|---:|---:|
| 22 组 | OpenShape × ModelNet-C all35 | +2.56 all35 | +0.01 all35 |
| 23 组 | OpenShape × ScanObjNN clean hardest | +0.07 | +1.87 |
| 24 组 | OpenShape × ScanObjNN-C hardest all35 | +3.99 all35 | +1.13 all35 |

24 组说明：当真实扫描 hardest 进一步叠加 corruption 后，Global Cache 与 Local Cache 都有价值。

---

## 17. 与 23_3 ScanObjNN clean hardest 的关系

23_3 是 OpenShape 在 ScanObjNN clean hardest 上的完整 Point-Cache 结果；24_3 是 OpenShape 在 ScanObjNN-C hardest all35 上的完整 Point-Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy / Avg |
|---|---|---|---:|
| 23_3_openshape_scanobjnn_clean_hardest_zs_global_local | ScanObjNN clean hardest | ZS + Global + Local | 43.82 |
| 24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local | ScanObjNN-C hardest S2 Avg | ZS + Global + Local | 38.63 |
| 24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local | ScanObjNN-C hardest all35 Avg | ZS + Global + Local | 37.84 |

对比：

| 比较 | 变化 |
|---|---:|
| 24_3 S2 Avg - 23_3 clean hardest | -5.19 |
| 24_3 all35 Avg - 23_3 clean hardest | -5.98 |

分析：

在 ScanObjNN clean hardest 的基础上进一步施加 corruption 后，OpenShape 完整 Point-Cache 从 43.82 下降到 all35 Avg 37.84，下降 -5.98。

不过，相比 Zero-shot，完整 Point-Cache 缩小了 clean-to-corruption gap：

| 方法 | ScanObjNN clean hardest | ScanObjNN-C hardest all35 | 下降 |
|---|---:|---:|---:|
| Zero-shot | 41.88 | 32.72 | -9.16 |
| ZS + Global | 41.95 | 36.71 | -5.24 |
| ZS + Global + Local | 43.82 | 37.84 | -5.98 |

完整 Point-Cache 的 corrupted accuracy 最高，但因为 23_3 clean hardest 本身也更高，所以 clean-to-corruption gap 不一定最小。单纯看 corrupted setting 上的最终准确率，24_3 最好；单纯看 gap，24_2 略小。

---

## 18. 与 22_3 ModelNet-C 的关系

22_3 是 OpenShape 在 ModelNet-C all35 上的完整 Point-Cache 结果；24_3 是 OpenShape 在 ScanObjNN-C hardest all35 上的完整 Point-Cache 结果。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 22_3_openshape_modelnetc_corruptions_all35_zs_global_local | ModelNet-C all35 | ZS + Global + Local | 76.33 | 75.14 |
| 24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local | ScanObjNN-C hardest all35 | ZS + Global + Local | 38.63 | 37.84 |

对比：

| 比较 | S2 变化 | all35 变化 |
|---|---:|---:|
| 24_3 - 22_3 | -37.70 | -37.30 |

分析：

ScanObjNN-C hardest 远难于 ModelNet-C。即使使用完整 Point-Cache，OpenShape 在 ModelNet-C all35 上为 75.14，但在 ScanObjNN-C hardest all35 上只有 37.84。

这说明真实扫描 corrupted hardest 的难度远高于 synthetic ModelNet corruption。24 组是当前 baseline 复现中最关键的高难度鲁棒性设置之一。

---

## 19. 与 ULIP / ULIP-2 的 ScanObjNN-C hardest 关系

24_3 可以与前面 ULIP、ULIP-2 的 ScanObjNN-C hardest 完整 Point-Cache 结果进行横向比较。

| Backbone | 实验编号 | ScanObjNN-C hardest S2 Avg | ScanObjNN-C hardest all35 Avg |
|---|---|---:|---:|
| ULIP | 04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local | 27.94 | 27.41 |
| ULIP-2 | 14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local | 33.60 | 33.25 |
| OpenShape | 24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local | 38.63 | 37.84 |

Backbone 提升：

| 比较 | S2 Avg 提升 | all35 Avg 提升 |
|---|---:|---:|
| OpenShape - ULIP | +10.69 | +10.43 |
| OpenShape - ULIP-2 | +5.03 | +4.59 |

分析：

完整 Point-Cache 下，OpenShape 仍然明显强于 ULIP 和 ULIP-2。相比 ULIP，OpenShape all35 Avg 高 +10.43；相比 ULIP-2，高 +4.59。

这说明 OpenShape backbone 在真实扫描 corrupted hardest 上仍然具有明显优势。但 OpenShape 完整 Point-Cache 的绝对准确率仍然只有 37.84，说明该设置仍然非常困难。

---

## 20. 结果含义分析

24_3 的结果说明：完整 Point-Cache 在 OpenShape × ScanObjNN-C hardest all35 上有效，并且最终数值略高于原文。

| 观察 | 含义 |
|---|---|
| 24_3 all35 Avg = 37.84 | 当前完整 Point-Cache 总体结果 |
| 24_3 S2 Avg = 38.63 | 与原文 Table 7 对齐的复现结果 |
| 原文 S2 Avg = 37.70 | 当前高 +0.93，略高但可接受 |
| 相比 24_1 all35 提升 +5.11 | 完整 Point-Cache 有明确正增益 |
| 相比 24_2 all35 提升 +1.13 | Local Cache 有明确额外贡献 |
| jitter 仍然最低 | 高 severity 坐标扰动仍然是主要失败点 |
| dropout_local 仍然困难 | 局部缺失仍需要进一步改进 |

因此，24_3 是 24 组三个子实验中最关键的最终结果。它证明了完整 Point-Cache 在真实扫描 corrupted hardest setting 中有效，也说明 Local Cache 在该场景中具有明确价值。

---

## 21. 对后续 MCM-PC 的启发

当前 24_3 对后续 MCM-PC 方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| 完整 Point-Cache 提升 +5.11 all35 | cache 机制在真实扫描 corrupted setting 中非常重要 |
| Global Cache 提供 +3.99 all35 | 全局缓存是稳定主模块，应保留 |
| Local Cache 额外提供 +1.13 all35 | 局部缓存也有明确价值 |
| jitter 仍然最难 | 坐标扰动需要专门鲁棒机制 |
| dropout_local 仍然困难 | 局部缺失需要更可靠的局部证据建模 |
| 22 组 Local 弱，24 组 Local 强 | Local Cache 的价值依赖数据类型和域偏移类型 |
| 真实扫描 corrupted setting 远难于 ModelNet-C | 后续方法必须重视 ScanObjNN-C hardest |

这对 MCM-PC 很重要：后续方法不应简单固定 Global / Local 的作用，而应根据样本可靠性、全局-局部一致性、伪标签可信度和域偏移类型动态调节缓存贡献。

24 组显示，真实扫描 corrupted setting 中 Global Cache 和 Local Cache 都有价值。后续 MCM-PC 可以围绕如何保留全局缓存稳定增益、如何选择可靠局部证据、如何抑制 high-severity jitter / dropout 下的错误伪标签展开。

---

## 22. 阶段性结论

本实验完成了 OpenShape × ScanObjNN-C hardest all35 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 24_3 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 38.63，原文 Point-Cache Table 7 中 OpenShape +Hierarchical Cache Avg 为 37.70，差异 +0.93。
3. 当前 all35 Average 为 37.84，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果略高于原文，但整体可接受，可以认为 24_3 复现有效。
5. 逐 corruption 看，scale 明显高于原文 +3.44，是主要正偏差来源。
6. 相比 24_1 Zero-shot，24_3 的 severity=2 Average 提升 +5.88，all35 Average 提升 +5.11。
7. 相比 24_2 Global Cache，24_3 的 severity=2 Average 额外提升 +1.33，all35 Average 额外提升 +1.13。
8. Global Cache 是主要提升来源，Local Cache 也有明确额外贡献。
9. Local Cache 在 30 / 35 个 setting 上为正增益，在 5 / 35 个 setting 上为负增益。
10. jitter 仍然是最困难 corruption，完整 Point-Cache 后平均仍只有 28.01。
11. jitter_4 是全部 35 个 setting 中最低结果，只有 16.90。
12. dropout_local 是第二困难 corruption，平均准确率为 34.08。
13. 完整 Point-Cache 下，OpenShape 比 ULIP 高 +10.43 all35，比 ULIP-2 高 +4.59 all35。
14. 24_3 结果有效，不需要重跑。
15. 该实验可作为 24 组 summary 文档和后续 MCM-PC 方法改进的关键 baseline。

---

## 23. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh 1

---

## 24. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv
