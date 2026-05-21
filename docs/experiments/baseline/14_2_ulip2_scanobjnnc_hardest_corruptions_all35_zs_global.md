# 14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global

## 1. 实验目的

本实验用于复现 ULIP-2 在 ScanObjNN-C hardest 全部 35 个损坏设置上使用 Zero-shot + Global Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global |
| Backbone | ULIP-2 |
| Dataset | ScanObjNN-C hardest |
| Method | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |

本实验是在 14_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存是否能够提升 ULIP-2 在 ScanObjNN-C hardest corrupted setting 上的鲁棒性。

本文件只记录 14_2 本身，并与前序子实验 14_1 进行对比。完整 14 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 14 组 summary 文档中。

需要特别注意：原文 Point-Cache Supplementary Table 7 只报告 corruption severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原文 Point-Cache Supplementary Table 7 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

---

## 2. 当前实现方式

本实验的外部命名规则如下：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global |
| 方法脚本 | Point-Cache/scripts/baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/14_run_ulip2_scanobjnnc_hardest_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global/ |

本实验是 all35 实验，因此使用优化 runner：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 |
| 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 |
| 模型只加载一次，每个 cor_type 重新创建 DataLoader、重新初始化 Global Cache |
| bash 通过 tee 生成单个 cor_type 的 log |
| Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv |
| summary.csv 的列结构保持不变 |

该优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、Global Cache 初始化、预测逻辑和输出结构。

---

## 3. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP-2 |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 数据目录 | data/sonn_c |
| 实际数据目录 | data/sonn_c/hardest |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 原始核心 runner | runners/model_with_global_cache.py |
| 优化 runner | runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py |
| cache_type | global |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |
| Severity 编号 | 0, 1, 2, 3, 4 |
| 损坏组合数 | 7 × 5 = 35 |
| 输入点数 | 1024 |
| Global Cache shot_capacity | 3 |
| Global Cache alpha | 4.0 |
| Global Cache beta | 3.0 |
| Backbone 权重 | weights/ulip/pointbert_ulip2.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| ULIP version | ulip2 |
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

Point-Cache/results/baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_add_global_0_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 31.38 | 用于论文对齐 | 与原文 Supplementary Table 7 对比 |
| all35 Average | 31.24 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，14_2 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ScanObjNN-C hardest 的 severity=2 参考值进行对比。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 37.47 | 36.99 | 36.68 | 35.91 | 35.05 | 36.42 |
| add_local | 37.40 | 34.63 | 33.73 | 31.71 | 30.50 | 33.59 |
| dropout_global | 37.86 | 35.95 | 34.56 | 29.84 | 23.32 | 32.31 |
| dropout_local | 37.54 | 34.21 | 31.33 | 27.24 | 23.77 | 30.82 |
| rotate | 40.04 | 37.02 | 33.07 | 30.26 | 27.59 | 33.60 |
| scale | 35.98 | 36.12 | 33.55 | 33.14 | 33.55 | 34.47 |
| jitter | 25.92 | 19.22 | 16.72 | 13.39 | 12.04 | 17.46 |
| **Average** | **36.03** | **33.45** | **31.38** | **28.78** | **26.55** | **31.24** |

整体观察：

1. all35 Average 为 31.24，表示 ULIP-2 在 ScanObjNN-C hardest 全 35 个 corrupted setting 上加入 Global Cache 后的整体水平。
2. severity=2 Average 为 31.38，用于和原文 Supplementary Table 7 对齐。
3. add_global 的平均准确率最高，为 36.42。
4. jitter 的平均准确率最低，为 17.46。
5. jitter_4 为 12.04，是全部 35 个 setting 中最低的结果。
6. 相比 14_1 Zero-shot，Global Cache 在全部 35 个 setting 上均带来正向提升。

---

## 8. 与原文结果对比

原文 Point-Cache Supplementary Table 7 报告的是 S-PB T50-RS-C severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 36.68 | 36.40 | +0.28 | 0.28 |
| add_local | 33.73 | 33.80 | -0.07 | 0.07 |
| dropout_global | 34.56 | 35.39 | -0.83 | 0.83 |
| dropout_local | 31.33 | 30.88 | +0.45 | 0.45 |
| rotate | 33.07 | 33.66 | -0.59 | 0.59 |
| scale | 33.55 | 35.01 | -1.46 | 1.46 |
| jitter | 16.72 | 18.36 | -1.64 | 1.64 |
| **Average** | **31.38** | **31.93** | **-0.55** | **0.76 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | -0.55 |
| MAE | 0.76 |
| RMSE | 0.91 |
| Max Abs Diff | 1.64 |

分析：

当前复现的 severity=2 Average 为 31.38，原文为 31.93，差异为 -0.55。整体仍属于可接受范围，但比 14_1 的误差更大。

单项差异主要来自 jitter、scale 和 dropout_global，其中 jitter 低于原文 1.64，scale 低于原文 1.46，dropout_global 低于原文 0.83。因此，14_2 的结论应写得更严谨：脚本执行和整体趋势正常，Global Cache 明显有效，但与原文相比，severity=2 平均值略低，主要由 scale 和 jitter 等 corruption 的偏低造成。

---

## 9. Severity 维度分析

### 9.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 36.03 | — | 0.00 |
| S1 | 33.45 | -2.58 | -2.58 |
| S2 | 31.38 | -2.07 | -4.65 |
| S3 | 28.78 | -2.60 | -7.25 |
| S4 | 26.55 | -2.23 | -9.48 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 36.03 下降到 26.55，总下降 9.48 个百分点。整体上，severity 越高，ULIP-2 + Global Cache 的准确率越低，说明 corruption severity 仍然持续影响性能。

与 14_1 Zero-shot 相比，14_2 在所有 severity 上整体上移，说明 Global Cache 并不是只改善某个特定 severity，而是在整个 severity 范围内都有效。

### 9.2 与 14_1 Zero-shot 的 severity 维度对比

| Severity | 14_1 Zero-shot Avg | 14_2 ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 30.21 | 36.03 | +5.82 |
| S1 | 28.35 | 33.45 | +5.10 |
| S2 | 26.44 | 31.38 | +4.94 |
| S3 | 24.34 | 28.78 | +4.44 |
| S4 | 22.94 | 26.55 | +3.61 |
| **all35** | **26.46** | **31.24** | **+4.78** |

分析：

Global Cache 在所有 severity 上都带来正向提升。当前 all35 平均提升为 +4.78，severity=2 提升为 +4.94。

原文 severity=2 下 Global Cache 增益为：

31.93 - 26.37 = +5.56

当前复现 severity=2 下 Global Cache 增益为：

31.38 - 26.44 = +4.94

二者差异为 -0.62。也就是说，当前 14_2 的 Global Cache 仍然明显有效，但其 severity=2 增益略低于原文。

---

## 10. Corruption 难度分析

### 10.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 17.46 | 平均最低，高 severity 下仍然非常困难 |
| 2 | dropout_local | 30.82 | 局部缺失仍有明显影响 |
| 3 | dropout_global | 32.31 | 高 severity 下降明显 |
| 4 | add_local | 33.59 | 中等偏高 |
| 5 | rotate | 33.60 | 中等偏高 |
| 6 | scale | 34.47 | 较稳定 |
| 7 | add_global | 36.42 | 当前最高 |

分析：

加入 Global Cache 后，jitter 仍然是绝对最困难的 corruption，但平均准确率从 14_1 的 13.98 提升到 17.46，说明 Global Cache 对 jitter 有帮助。

add_global、scale、rotate 和 add_local 的平均结果较高，说明 Global Cache 能有效提高这些 corruption 下的整体表现。

### 10.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| add_global | 37.47 | 35.05 | 2.42 | 6.46% | 36.42 |
| scale | 35.98 | 33.55 | 2.43 | 6.75% | 34.47 |
| add_local | 37.40 | 30.50 | 6.90 | 18.45% | 33.59 |
| rotate | 40.04 | 27.59 | 12.45 | 31.09% | 33.60 |
| dropout_local | 37.54 | 23.77 | 13.77 | 36.68% | 30.82 |
| jitter | 25.92 | 12.04 | 13.88 | 53.55% | 17.46 |
| dropout_global | 37.86 | 23.32 | 14.54 | 38.40% | 32.31 |

分析：

add_global 和 scale 最稳定，从 S0 到 S4 分别只下降 2.42 和 2.43。jitter 的相对下降最大，从 25.92 下降到 12.04，相对下降 53.55%。

dropout_global、dropout_local 和 rotate 也有明显退化，说明高 severity 下的点云缺失和旋转变化仍然会削弱 ULIP-2 + Global Cache 的识别能力。

---

## 11. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | jitter | 25.92 | rotate | 40.04 | 14.12 |
| S1 | jitter | 19.22 | rotate | 37.02 | 17.80 |
| S2 | jitter | 16.72 | add_global | 36.68 | 19.96 |
| S3 | jitter | 13.39 | add_global | 35.91 | 22.52 |
| S4 | jitter | 12.04 | add_global | 35.05 | 23.01 |

分析：

在所有 severity 下，jitter 都是最困难 corruption。随着 severity 增大，best-worst gap 从 S0 的 14.12 扩大到 S4 的 23.01。

Global Cache 提高了整体水平，但没有改变 jitter 是主要失败区域这一事实。

---

## 12. 低准确率区域分析

### 12.1 低准确率 setting 数量

| 条件 | 14_1 Zero-shot 数量 | 14_2 ZS + Global 数量 | 减少数量 |
|---|---:|---:|---:|
| Acc < 30 | 21 / 35 | 10 / 35 | -11 |
| Acc < 28 | 14 / 35 | 9 / 35 | -5 |
| Acc < 25 | 10 / 35 | 6 / 35 | -4 |
| Acc < 22 | 7 / 35 | 4 / 35 | -3 |
| Acc < 20 | 4 / 35 | 4 / 35 | 0 |
| Acc < 18 | 4 / 35 | 3 / 35 | -1 |
| Acc < 15 | 3 / 35 | 2 / 35 | -1 |

分析：

加入 Global Cache 后，低准确率区域明显减少。例如 Acc < 30 的 setting 从 21 个减少到 10 个，Acc < 25 的 setting 从 10 个减少到 6 个。

但是，严重低准确率区域仍然主要由 jitter 主导。Acc < 20 的 setting 数量仍然是 4 个，说明 Global Cache 对高 severity jitter 的修复有限。

### 12.2 当前仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 12.04 | 最高 severity 坐标扰动仍然最难 |
| jitter_3 | 13.39 | 中高 severity 坐标扰动仍然极低 |
| jitter_2 | 16.72 | severity=2 jitter 仍明显低于其他 corruption |
| jitter_1 | 19.22 | 虽有提升，但仍低于大多数 corruption |
| dropout_global_4 | 23.32 | 高 severity 全局缺失仍有明显影响 |
| dropout_local_4 | 23.77 | 高 severity 局部缺失仍有明显影响 |

分析：

14_2 中最困难区域仍然集中在 jitter，尤其是 jitter_3 和 jitter_4。说明 Global Cache 不能完全解决强坐标扰动问题。

---

## 13. Global Cache 相比 14_1 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +6.17 | +6.00 | +5.83 | +5.31 | +5.00 | +5.66 |
| add_local | +4.99 | +4.23 | +4.10 | +3.26 | +3.50 | +4.02 |
| dropout_global | +6.08 | +5.42 | +6.07 | +4.61 | +1.88 | +4.81 |
| dropout_local | +7.87 | +6.42 | +6.42 | +5.10 | +3.33 | +5.83 |
| rotate | +6.00 | +5.79 | +4.65 | +4.79 | +3.37 | +4.92 |
| scale | +5.45 | +5.31 | +3.36 | +4.86 | +4.82 | +4.76 |
| jitter | +4.20 | +2.53 | +4.12 | +3.15 | +3.37 | +3.47 |
| **Average** | **+5.82** | **+5.10** | **+4.94** | **+4.44** | **+3.61** | **+4.78** |

分析：

Global Cache 在全部 35 个 setting 上均带来正向提升，没有出现负增益。

平均提升最大的 corruption 是 dropout_local，Avg Gain 为 +5.83；其次是 add_global，Avg Gain 为 +5.66。说明 Global Cache 对局部缺失和全局异常点都有明显修正作用。

jitter 的平均提升为 +3.47，虽然也是正向，但由于 jitter 的基础准确率过低，Global Cache 后仍然是最困难 corruption。

---

## 14. 与前序实验的关系

14_2 的直接前序子实验是 14_1，即 ULIP-2 在 ScanObjNN-C hardest all35 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs | ScanObjNN-C hardest all35 | Zero-shot | 26.44 | 26.46 |
| 14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global | ScanObjNN-C hardest all35 | ZS + Global Cache | 31.38 | 31.24 |

当前结果说明：在 ScanObjNN-C hardest all35 上，Global Cache 能稳定提升 ULIP-2 Zero-shot。

| 比较 | 变化 |
|---|---:|
| 14_2 S2 Avg - 14_1 S2 Avg | +4.94 |
| 14_2 all35 Avg - 14_1 all35 Avg | +4.78 |

分析：

14_2 相比 14_1 的提升非常明显，说明 Global Cache 在真实扫描 corrupted setting 上仍然有效。虽然 14_2 的 severity=2 绝对值比原文低 0.55，但它相对 14_1 的增益仍达到 +4.94，方法有效性清楚。

---

## 15. 阶段性结论

本实验完成了 ULIP-2 × ScanObjNN-C hardest all35 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 14_2 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 31.38，原文 Supplementary Table 7 中 ULIP-2 + Global Cache Avg 为 31.93，差异为 -0.55。
3. 当前 all35 Average 为 31.24，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果整体可接受，但与原文相比略低，主要差异来自 jitter、scale 和 dropout_global。
5. 相比 14_1 Zero-shot，14_2 的 severity=2 Average 提升 +4.94，all35 Average 提升 +4.78。
6. Global Cache 在全部 35 个 setting 上均带来正向提升，没有出现负增益。
7. Global Cache 对 dropout_local 的平均提升最大，为 +5.83；对 add_global 的提升也很明显，为 +5.66。
8. jitter 仍然是最困难 corruption，Global Cache 后平均仍只有 17.46。
9. 低准确率区域明显减少，但高 severity jitter 仍然是主要失败点。
10. 本实验是 14_3 分析 Local Cache 额外贡献的直接对照，不在本文件中展开完整 14 组方法间对比。

---

## 16. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh 1

---

## 17. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f13 | sort | uniq -c
