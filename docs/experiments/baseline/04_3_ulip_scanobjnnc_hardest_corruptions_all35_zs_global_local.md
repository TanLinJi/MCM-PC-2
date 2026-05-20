# 04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local

## 1. 实验目的

本实验用于复现 ULIP 在 ScanObjNN-C hardest 全部 35 个损坏设置上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local |
| Backbone | ULIP |
| Dataset | ScanObjNN-C hardest |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 04_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证局部缓存是否能在全局缓存之外继续提升 ULIP 在 ScanObjNN-C hardest corrupted setting 上的鲁棒性。

本文件只记录 04_3 本身，并与前序子实验 04_1 和 04_2 进行对比。完整 04 组三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 04 组 summary 文档中。

需要特别注意：原文 Supplementary Table 7 只报告 corruption severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原文 Supplementary Table 7 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

---

## 2. 当前实现方式

本实验的外部命名规则保持不变：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local |
| 方法脚本 | Point-Cache/scripts/baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/04_run_ulip_scanobjnnc_hardest_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local/ |

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
| Backbone | ULIP |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 数据目录 | data/sonn_c |
| 实际数据目录 | data/sonn_c/hardest |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 优化 runner | runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py |
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
| 权重 | weights/ulip/pointbert_ulip1.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 0 |

本实验使用 `sonn_c` 作为 dataset 参数，`sonn_variant=hardest`，并在 Python runner 内部循环 35 个 `cor_type`。实际读取文件形式为：

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

Point-Cache/results/baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local_add_global_0_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 27.94 | 用于论文对齐 | 与原文 Supplementary Table 7 对比 |
| all35 Average | 27.41 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，04_3 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ScanObjNN-C hardest 的 severity=2 参考值进行对比。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 27.48 | 23.53 | 23.87 | 22.73 | 20.71 | 23.66 |
| add_local | 25.54 | 22.21 | 22.73 | 20.06 | 19.67 | 22.04 |
| dropout_global | 35.29 | 35.88 | 34.25 | 32.76 | 28.11 | 33.26 |
| dropout_local | 32.76 | 30.71 | 30.26 | 26.16 | 22.24 | 28.43 |
| rotate | 34.32 | 33.66 | 32.55 | 26.16 | 24.95 | 30.33 |
| scale | 31.02 | 28.35 | 28.38 | 26.75 | 28.14 | 28.53 |
| jitter | 33.55 | 28.14 | 23.53 | 23.84 | 19.15 | 25.64 |
| **Average** | **31.42** | **28.93** | **27.94** | **25.49** | **23.28** | **27.41** |

整体观察：

1. all35 Average 为 27.41，表示 ULIP 在 ScanObjNN-C hardest 全 35 个 corrupted setting 上使用完整 Point-Cache 后的整体水平。
2. severity=2 Average 为 27.94，用于和原文 Supplementary Table 7 对齐。
3. dropout_global 的平均准确率最高，为 33.26。
4. add_local 的平均准确率最低，为 22.04。
5. jitter_4 为 19.15，是当前完整 Point-Cache 下的最低结果。
6. 相比 04_2 Global Cache，04_3 在大部分 setting 上继续提升，但并非所有 setting 都正向。

---

## 8. 与原文结果对比

原文 Supplementary Table 7 报告的是 S-PB T50-RS-C severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 23.87 | 23.46 | +0.41 | 0.41 |
| add_local | 22.73 | 22.69 | +0.04 | 0.04 |
| dropout_global | 34.25 | 34.70 | -0.45 | 0.45 |
| dropout_local | 30.26 | 31.75 | -1.49 | 1.49 |
| rotate | 32.55 | 33.00 | -0.45 | 0.45 |
| scale | 28.38 | 28.28 | +0.10 | 0.10 |
| jitter | 23.53 | 25.05 | -1.52 | 1.52 |
| **Average** | **27.94** | **28.42** | **-0.48** | **0.64 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | -0.48 |
| MAE | 0.64 |
| RMSE | 0.85 |
| Max Abs Diff | 1.52 |

分析：

当前复现的 severity=2 Average 为 27.94，原文为 28.42，差异为 -0.48。整体平均结果与原文基本接近。

单项差异主要来自 dropout_local 和 jitter，分别低于原文 1.49 和 1.52。考虑到 Hierarchical Cache 涉及在线 cache、局部聚类、测试样本顺序和局部特征检索，单项结果出现小幅波动是可以接受的。

因此，04_3 不只是脚本跑通，而且 severity=2 结果也与原文 ULIP + Hierarchical Cache 在 S-PB T50-RS-C 上的结果基本对齐。

---

## 9. Severity 维度分析

### 9.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 31.42 | — | 0.00 |
| S1 | 28.93 | -2.49 | -2.49 |
| S2 | 27.94 | -0.99 | -3.48 |
| S3 | 25.49 | -2.45 | -5.93 |
| S4 | 23.28 | -2.21 | -8.14 |

分析：

随着 severity 从 S0 增大到 S4，完整 Point-Cache 的平均准确率从 31.42 下降到 23.28，总下降 8.14 个百分点。整体趋势仍然是 severity 越高，准确率越低。

相比 04_2 Global Cache，04_3 整体准确率上移，但并没有完全消除高 severity corruption 带来的退化。特别是 S4 平均只有 23.28，说明最强 corruption 下仍存在明显困难。

### 9.2 与前序实验的 severity 维度对比

| Severity | 04_1 Zero-shot Avg | 04_2 ZS + Global Avg | 04_3 ZS + Global + Local Avg | Gain over 04_1 | Gain over 04_2 |
|---:|---:|---:|---:|---:|---:|
| S0 | 26.86 | 30.20 | 31.42 | +4.55 | +1.22 |
| S1 | 25.10 | 28.21 | 28.93 | +3.83 | +0.72 |
| S2 | 23.91 | 26.84 | 27.94 | +4.03 | +1.10 |
| S3 | 22.11 | 24.74 | 25.49 | +3.38 | +0.75 |
| S4 | 20.24 | 23.02 | 23.28 | +3.04 | +0.27 |
| **all35** | **23.65** | **26.60** | **27.41** | **+3.76** | **+0.81** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有正增益，all35 平均提升 +3.76，severity=2 提升 +4.03。

Local Cache 在 Global Cache 基础上的额外提升也是正向的，但幅度小于 Global Cache 本身。all35 平均额外提升 +0.81，severity=2 额外提升 +1.10。

Local Cache 的额外提升随 severity 增大而变弱：S0 为 +1.22，S4 只有 +0.27。这说明在最高 severity 下，局部结构本身可能已经受到较严重破坏，Local Cache 的补充作用受到限制。

---

## 10. Corruption 难度分析

### 10.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | add_local | 22.04 | 平均最低，局部异常点仍然最困难 |
| 2 | add_global | 23.66 | 全局异常点仍然困难 |
| 3 | jitter | 25.64 | 高 severity 下仍然偏低 |
| 4 | dropout_local | 28.43 | Local Cache 后明显改善，但仍有下降 |
| 5 | scale | 28.53 | 较稳定 |
| 6 | rotate | 30.33 | 平均表现较高 |
| 7 | dropout_global | 33.26 | 当前最容易 |

分析：

完整 Point-Cache 后，最困难的 corruption 仍然是 add_local，其次是 add_global 和 jitter。也就是说，Global + Local 虽然能整体提升性能，但没有完全改变 ScanObjNN-C hardest 的难度结构。

dropout_global 依然是最容易的 corruption，平均达到 33.26。Local Cache 对 dropout_local、rotate 等 corruption 有较明显帮助，但对 add_local 和 jitter 的根本困难仍然没有完全解决。

### 10.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 31.02 | 28.14 | 2.88 | 9.28% | 28.53 |
| add_global | 27.48 | 20.71 | 6.77 | 24.64% | 23.66 |
| dropout_global | 35.29 | 28.11 | 7.18 | 20.35% | 33.26 |
| add_local | 25.54 | 19.67 | 5.87 | 22.98% | 22.04 |
| rotate | 34.32 | 24.95 | 9.37 | 27.30% | 30.33 |
| dropout_local | 32.76 | 22.24 | 10.52 | 32.11% | 28.43 |
| jitter | 33.55 | 19.15 | 14.40 | 42.92% | 25.64 |

分析：

scale 仍然最稳定，从 S0 到 S4 只下降 2.88。jitter 退化最大，从 33.55 下降到 19.15，绝对下降 14.40，相对下降 42.92%。

dropout_local 也有明显退化，从 32.76 下降到 22.24。这说明即使加入 Local Cache，局部结构缺失和坐标扰动仍然是 ScanObjNN-C hardest 上的重要困难来源。

---

## 11. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | add_local | 25.54 | dropout_global | 35.29 | 9.75 |
| S1 | add_local | 22.21 | dropout_global | 35.88 | 13.67 |
| S2 | add_local | 22.73 | dropout_global | 34.25 | 11.52 |
| S3 | add_local | 20.06 | dropout_global | 32.76 | 12.70 |
| S4 | jitter | 19.15 | scale | 28.14 | 8.99 |

分析：

在 S0 到 S3 中，add_local 一直是最难 corruption；到 S4 时，jitter 成为最难 corruption。这个模式与 04_2 基本一致，说明 Local Cache 没有完全改变困难 corruption 的分布。

不同 corruption 之间的 best-worst gap 约为 9 到 14 个百分点，说明完整 Point-Cache 虽然提升整体准确率，但不同 corruption 类型之间的难度差异依然明显。

---

## 12. 低准确率区域分析

### 12.1 低准确率 setting 数量

| 条件 | 04_1 Zero-shot 数量 | 04_2 ZS + Global 数量 | 04_3 ZS + Global + Local 数量 | 相比 04_1 减少 | 相比 04_2 减少 |
|---|---:|---:|---:|---:|---:|
| Acc < 30 | 30 / 35 | 25 / 35 | 23 / 35 | -7 | -2 |
| Acc < 28 | 26 / 35 | 21 / 35 | 18 / 35 | -8 | -3 |
| Acc < 25 | 19 / 35 | 14 / 35 | 11 / 35 | -8 | -3 |
| Acc < 22 | 16 / 35 | 7 / 35 | 5 / 35 | -11 | -2 |
| Acc < 20 | 10 / 35 | 3 / 35 | 2 / 35 | -8 | -1 |

分析：

完整 Point-Cache 明显减少了低准确率区域。相比 Zero-shot，Acc < 22 的 setting 从 16 个减少到 5 个，Acc < 20 的 setting 从 10 个减少到 2 个。

相比 Global Cache，Local Cache 继续减少了部分低准确率区域，例如 Acc < 25 的 setting 从 14 个减少到 11 个，Acc < 22 的 setting 从 7 个减少到 5 个。

这说明 Local Cache 不只是提高平均值，也能进一步缓解部分低性能 setting，但无法彻底消除困难区域。

### 12.2 当前仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 19.15 | 最高 severity 的坐标扰动仍然最难 |
| add_local_4 | 19.67 | 最高 severity 的局部异常点仍然困难 |
| add_local_3 | 20.06 | 中高 severity 局部异常点仍然偏低 |
| add_global_4 | 20.71 | 高 severity 全局异常点仍然偏低 |
| dropout_local_4 | 22.24 | 高 severity 局部缺失仍然有明显影响 |

分析：

完整 Point-Cache 后，最困难区域主要集中在 jitter_4、add_local_4、add_local_3、add_global_4 和 dropout_local_4。这些 setting 都和高 severity 下的局部结构破坏、异常点或坐标扰动有关。

这说明后续 MCM-PC 方法若想进一步提升 ScanObjNN-C hardest，不能只简单增加缓存，而需要判断 local cache 何时可靠、何时可能被严重扰动误导。

---

## 13. 完整 Point-Cache 相比 04_1 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +7.39 | +4.55 | +4.58 | +4.90 | +2.84 | +4.85 |
| add_local | +4.30 | +1.63 | +4.13 | +2.36 | +2.49 | +2.98 |
| dropout_global | +4.55 | +5.21 | +3.33 | +2.47 | +2.75 | +3.66 |
| dropout_local | +4.10 | +5.42 | +6.35 | +5.76 | +4.86 | +5.30 |
| rotate | +3.75 | +5.03 | +5.42 | +2.63 | +3.65 | +4.10 |
| scale | +2.50 | +0.83 | +2.32 | +0.93 | +1.42 | +1.60 |
| jitter | +5.34 | +4.09 | +2.05 | +4.62 | +3.29 | +3.88 |
| **Average** | **+4.55** | **+3.83** | **+4.03** | **+3.38** | **+3.04** | **+3.76** |

分析：

完整 Point-Cache 相比 Zero-shot 在全部 35 个 setting 上均为正增益，没有出现负增益。

平均提升最大的 corruption 是 dropout_local，Avg Gain 为 +5.30；其次是 add_global，Avg Gain 为 +4.85。说明完整 Point-Cache 对局部缺失和全局异常点都有明显修正作用。

平均提升最小的是 scale，Avg Gain 为 +1.60。scale 在 Zero-shot 下已经较稳定，因此进一步提升空间较小。

---

## 14. Local Cache 相比 04_2 Global Cache 的额外提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +2.29 | +0.10 | +0.73 | +0.14 | -0.73 | +0.50 |
| add_local | +0.87 | -0.20 | +1.84 | +0.49 | +0.83 | +0.76 |
| dropout_global | +1.22 | +1.32 | +0.87 | +0.18 | +0.84 | +0.89 |
| dropout_local | +1.12 | +1.46 | +2.92 | +2.50 | +1.46 | +1.89 |
| rotate | +1.08 | +1.04 | +1.88 | -0.25 | -0.48 | +0.65 |
| scale | -0.03 | -0.83 | -0.45 | -0.18 | -0.42 | -0.38 |
| jitter | +2.04 | +2.12 | -0.10 | +2.40 | +0.34 | +1.36 |
| **Average** | **+1.22** | **+0.72** | **+1.10** | **+0.75** | **+0.27** | **+0.81** |

分析：

Local Cache 在 Global Cache 基础上的 all35 平均额外提升为 +0.81，severity=2 额外提升为 +1.10。说明 Local Cache 在 ScanObjNN-C hardest 上有效，但贡献小于 Global Cache。

Local Cache 的额外增益不是所有 setting 都为正。scale 在所有 severity 上均为轻微负增益，平均为 -0.38；add_global_4、rotate_3、rotate_4、jitter_2 也出现负增益。这说明 Local Cache 可能在某些几何扰动或尺度变化下引入不稳定局部信息。

另一方面，dropout_local 的 Local Cache 额外提升最大，Avg Gain 为 +1.89，S2 提升 +2.92。这符合直觉：dropout_local 破坏的是局部结构，而 Local Cache 正是利用局部特征，因此能带来更明显补偿。

---

## 15. 与前序实验的关系

04_3 的前序子实验包括 04_1 和 04_2。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs | ScanObjNN-C hardest all35 | Zero-shot | 23.91 | 23.65 |
| 04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global | ScanObjNN-C hardest all35 | ZS + Global Cache | 26.84 | 26.60 |
| 04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local | ScanObjNN-C hardest all35 | ZS + Global + Local Cache | 27.94 | 27.41 |

当前结果说明：在 ScanObjNN-C hardest all35 上，完整 Point-Cache 能稳定提升 Zero-shot，也高于只使用 Global Cache 的结果。

| 比较 | S2 Avg 变化 | all35 Avg 变化 |
|---|---:|---:|
| 04_2 - 04_1 | +2.93 | +2.95 |
| 04_3 - 04_2 | +1.10 | +0.81 |
| 04_3 - 04_1 | +4.03 | +3.76 |

分析：

Global Cache 是主要提升来源，Local Cache 在此基础上继续提供额外收益。当前 severity=2 下，原文中完整 Point-Cache 相比 Zero-shot 的提升为 +4.45，当前复现为 +4.03，差异为 -0.42，整体接近。

---

## 16. 阶段性结论

本实验完成了 ULIP × ScanObjNN-C hardest all35 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 04_3 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 27.94，原文 Supplementary Table 7 中 ULIP + Hierarchical Cache Avg 为 28.42，差异为 -0.48。
3. 当前 all35 Average 为 27.41，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果与原文 severity=2 数值基本对齐，可以认为 04_3 复现成功。
5. 相比 04_1 Zero-shot，04_3 的 severity=2 Average 提升 +4.03，all35 Average 提升 +3.76。
6. 相比 04_2 Global Cache，04_3 的 severity=2 Average 额外提升 +1.10，all35 Average 额外提升 +0.81。
7. 完整 Point-Cache 在全部 35 个 setting 上相比 Zero-shot 均为正增益。
8. Local Cache 对 dropout_local 的额外提升最明显，说明局部缓存对局部结构缺失有实际帮助。
9. Local Cache 在 scale 等 setting 上存在轻微负增益，说明局部缓存并非始终可靠。
10. jitter_4、add_local_4 和 add_global_4 仍然是完整 Point-Cache 下的困难区域。
11. 本实验是 04 组最后一个子实验，完整方法间总结应放入 04 组 summary 文档中。

---

## 17. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh 1

---

## 18. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f13 | sort | uniq -c
