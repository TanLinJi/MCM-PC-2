# 14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local

## 1. 实验目的

本实验用于复现 ULIP-2 在 ScanObjNN-C hardest 全部 35 个损坏设置上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local |
| Backbone | ULIP-2 |
| Dataset | ScanObjNN-C hardest |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 14_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证局部缓存是否能在全局缓存之外继续提升 ULIP-2 在 ScanObjNN-C hardest corrupted setting 上的鲁棒性。

本文件只记录 14_3 本身，并与前序子实验 14_1 和 14_2 进行对比。完整 14 组三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 14 组 summary 文档中。

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
| 实验编号 | 14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local |
| 方法脚本 | Point-Cache/scripts/baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/14_run_ulip2_scanobjnnc_hardest_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local/ |

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
| Backbone | ULIP-2 |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 数据目录 | data/sonn_c |
| 实际数据目录 | data/sonn_c/hardest |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 优化 runner | runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py |
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
| Backbone 权重 | weights/ulip/pointbert_ulip2.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| ULIP version | ulip2 |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 1 |

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

Point-Cache/results/baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local_add_global_0_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 33.36 | 用于论文对齐 | 与原文 Supplementary Table 7 对比 |
| all35 Average | 32.95 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，14_3 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ScanObjNN-C hardest 的 severity=2 参考值进行对比。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 39.07 | 36.92 | 36.50 | 36.43 | 34.63 | 36.71 |
| add_local | 40.18 | 35.11 | 36.16 | 33.31 | 30.08 | 34.97 |
| dropout_global | 40.60 | 38.13 | 37.16 | 33.10 | 24.91 | 34.78 |
| dropout_local | 41.26 | 36.81 | 35.63 | 29.63 | 25.92 | 33.85 |
| rotate | 43.37 | 38.83 | 36.33 | 32.10 | 29.84 | 36.09 |
| scale | 37.37 | 38.51 | 33.97 | 33.55 | 36.09 | 35.90 |
| jitter | 26.34 | 19.99 | 17.77 | 13.50 | 14.23 | 18.37 |
| **Average** | **38.31** | **34.90** | **33.36** | **30.23** | **27.96** | **32.95** |

整体观察：

1. all35 Average 为 32.95，表示 ULIP-2 在 ScanObjNN-C hardest 全 35 个 corrupted setting 上使用完整 Point-Cache 后的整体水平。
2. severity=2 Average 为 33.36，用于和原文 Supplementary Table 7 对齐。
3. add_global 的平均准确率最高，为 36.71。
4. jitter 的平均准确率最低，为 18.37。
5. jitter_3 为 13.50，是当前完整 Point-Cache 下的最低结果。
6. 相比 14_2 Global Cache，14_3 的 all35 Average 继续提升，说明 Local Cache 在整体上有效。

---

## 8. 与原文结果对比

原文 Point-Cache Supplementary Table 7 报告的是 S-PB T50-RS-C severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 36.50 | 35.70 | +0.80 | 0.80 |
| add_local | 36.16 | 34.42 | +1.74 | 1.74 |
| dropout_global | 37.16 | 37.75 | -0.59 | 0.59 |
| dropout_local | 35.63 | 34.21 | +1.42 | 1.42 |
| rotate | 36.33 | 36.26 | +0.07 | 0.07 |
| scale | 33.97 | 36.09 | -2.12 | 2.12 |
| jitter | 17.77 | 19.12 | -1.35 | 1.35 |
| **Average** | **33.36** | **33.36** | **+0.00** | **1.16 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | 0.00 |
| MAE | 1.16 |
| RMSE | 1.33 |
| Max Abs Diff | 2.12 |

分析：

当前复现的 severity=2 Average 为 33.36，原文也为 33.36，平均值完全一致。因此，14_3 的整体结果与原文高度对齐。

但是，逐 corruption 看存在一定抵消：add_local 和 dropout_local 高于原文，scale 和 jitter 低于原文。也就是说，14_3 的平均值非常好，但单项 corruption 的波动比 14_1 更明显。后续记录时应明确说明这一点，避免只写“完全一致”而忽略逐项差异。

---

## 9. Severity 维度分析

### 9.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 38.31 | — | 0.00 |
| S1 | 34.90 | -3.41 | -3.41 |
| S2 | 33.36 | -1.54 | -4.95 |
| S3 | 30.23 | -3.13 | -8.08 |
| S4 | 27.96 | -2.27 | -10.35 |

分析：

随着 severity 从 S0 增大到 S4，完整 Point-Cache 的平均准确率从 38.31 下降到 27.96，总下降 10.35 个百分点。整体上，severity 越高，准确率越低。

完整 Point-Cache 提升了整体水平，但没有消除 high-severity corruption 的影响。S4 仍然比 S0 低 10.35 个百分点，说明强 corruption 下仍存在明显鲁棒性缺口。

### 9.2 与前序实验的 severity 维度对比

| Severity | 14_1 Zero-shot Avg | 14_2 ZS + Global Avg | 14_3 ZS + Global + Local Avg | Gain over 14_1 | Gain over 14_2 |
|---:|---:|---:|---:|---:|---:|
| S0 | 30.21 | 36.03 | 38.31 | +8.10 | +2.28 |
| S1 | 28.35 | 33.45 | 34.90 | +6.55 | +1.45 |
| S2 | 26.44 | 31.38 | 33.36 | +6.92 | +1.98 |
| S3 | 24.34 | 28.78 | 30.23 | +5.89 | +1.45 |
| S4 | 22.94 | 26.55 | 27.96 | +5.02 | +1.41 |
| **all35** | **26.46** | **31.24** | **32.95** | **+6.50** | **+1.72** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有明显正增益，all35 平均提升 +6.50，severity=2 提升 +6.92。

Local Cache 在 Global Cache 基础上的额外提升也为正，all35 平均额外提升 +1.72，severity=2 额外提升 +1.98。

这说明在 ULIP-2 × ScanObjNN-C hardest all35 上，Global Cache 是主要提升来源，Local Cache 在此基础上继续提供稳定补充。

---

## 10. Corruption 难度分析

### 10.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 18.37 | 平均最低，高 severity 下仍然非常困难 |
| 2 | dropout_local | 33.85 | 局部缺失仍有明显影响 |
| 3 | dropout_global | 34.78 | 高 severity 下降明显 |
| 4 | add_local | 34.97 | 中等偏高 |
| 5 | scale | 35.90 | 较稳定 |
| 6 | rotate | 36.09 | 较高 |
| 7 | add_global | 36.71 | 当前最高 |

分析：

完整 Point-Cache 后，jitter 仍然是绝对最困难的 corruption，但平均准确率从 14_1 的 13.98 提升到 14_3 的 18.37，说明 Global + Local 对 jitter 有一定帮助，但无法彻底解决强坐标扰动问题。

add_global、rotate 和 scale 的表现较高，说明完整 Point-Cache 在这些 corruption 下较稳定。

### 10.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 37.37 | 36.09 | 1.28 | 3.43% | 35.90 |
| add_global | 39.07 | 34.63 | 4.44 | 11.36% | 36.71 |
| add_local | 40.18 | 30.08 | 10.10 | 25.14% | 34.97 |
| jitter | 26.34 | 14.23 | 12.11 | 45.98% | 18.37 |
| rotate | 43.37 | 29.84 | 13.53 | 31.20% | 36.09 |
| dropout_local | 41.26 | 25.92 | 15.34 | 37.18% | 33.85 |
| dropout_global | 40.60 | 24.91 | 15.69 | 38.65% | 34.78 |

分析：

scale 最稳定，从 S0 到 S4 只下降 1.28。jitter 的相对下降仍然很大，从 26.34 下降到 14.23，相对下降 45.98%。

dropout_global 和 dropout_local 的绝对下降最大，分别为 15.69 和 15.34，说明高 severity 下的点云缺失仍然会明显影响完整 Point-Cache 的表现。

---

## 11. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | jitter | 26.34 | rotate | 43.37 | 17.03 |
| S1 | jitter | 19.99 | rotate | 38.83 | 18.84 |
| S2 | jitter | 17.77 | dropout_global | 37.16 | 19.39 |
| S3 | jitter | 13.50 | add_global | 36.43 | 22.93 |
| S4 | jitter | 14.23 | scale | 36.09 | 21.86 |

分析：

在所有 severity 下，jitter 都是最困难 corruption。随着 severity 增大，best-worst gap 大多维持在 17 到 23 个百分点之间。

完整 Point-Cache 提升了整体水平，但没有改变 jitter 是主要失败区域这一事实。

---

## 12. 低准确率区域分析

### 12.1 低准确率 setting 数量

| 条件 | 14_1 Zero-shot 数量 | 14_2 ZS + Global 数量 | 14_3 ZS + Global + Local 数量 | 相比 14_1 减少 | 相比 14_2 减少 |
|---|---:|---:|---:|---:|---:|
| Acc < 30 | 21 / 35 | 10 / 35 | 9 / 35 | -12 | -1 |
| Acc < 28 | 14 / 35 | 9 / 35 | 7 / 35 | -7 | -2 |
| Acc < 25 | 10 / 35 | 6 / 35 | 5 / 35 | -5 | -1 |
| Acc < 22 | 7 / 35 | 4 / 35 | 4 / 35 | -3 | 0 |
| Acc < 20 | 4 / 35 | 4 / 35 | 4 / 35 | 0 | 0 |
| Acc < 18 | 4 / 35 | 3 / 35 | 3 / 35 | -1 | 0 |
| Acc < 15 | 3 / 35 | 2 / 35 | 2 / 35 | -1 | 0 |

分析：

完整 Point-Cache 明显减少了低准确率区域。相比 Zero-shot，Acc < 30 的 setting 从 21 个减少到 9 个，Acc < 25 的 setting 从 10 个减少到 5 个。

相比 Global Cache，Local Cache 继续减少部分低准确率区域，但幅度较小。严重低准确率区域仍然主要由 jitter 主导，Acc < 20 的 setting 数量仍为 4 个，说明强坐标扰动仍然是主要失败点。

### 12.2 当前仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_3 | 13.50 | 中高 severity 坐标扰动最难 |
| jitter_4 | 14.23 | 最高 severity 坐标扰动仍然极低 |
| jitter_2 | 17.77 | severity=2 jitter 仍明显低于其他 corruption |
| jitter_1 | 19.99 | 虽有提升，但仍低于大多数 corruption |
| dropout_global_4 | 24.91 | 高 severity 全局缺失仍然困难 |
| dropout_local_4 | 25.92 | 高 severity 局部缺失仍然困难 |
| jitter_0 | 26.34 | 即使最低 severity jitter 也偏低 |

分析：

14_3 中最困难区域仍然集中在 jitter，尤其是 jitter_3 和 jitter_4。这说明 Global + Local Cache 不能完全解决强坐标扰动问题。

---

## 13. 完整 Point-Cache 相比 14_1 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +7.77 | +5.93 | +5.65 | +5.83 | +4.58 | +5.95 |
| add_local | +7.77 | +4.71 | +6.53 | +4.86 | +3.08 | +5.39 |
| dropout_global | +8.82 | +7.60 | +8.67 | +7.87 | +3.47 | +7.29 |
| dropout_local | +11.59 | +9.02 | +10.72 | +7.49 | +5.48 | +8.86 |
| rotate | +9.33 | +7.60 | +7.91 | +6.63 | +5.62 | +7.42 |
| scale | +6.84 | +7.70 | +3.78 | +5.27 | +7.36 | +6.19 |
| jitter | +4.62 | +3.30 | +5.17 | +3.26 | +5.56 | +4.38 |
| **Average** | **+8.10** | **+6.55** | **+6.92** | **+5.89** | **+5.02** | **+6.50** |

分析：

完整 Point-Cache 相比 Zero-shot 在全部 35 个 setting 上均为正增益，没有出现负增益。

平均提升最大的 corruption 是 dropout_local，Avg Gain 为 +8.86；其次是 rotate，为 +7.42；dropout_global 为 +7.29。说明完整 Point-Cache 对局部缺失、旋转和全局缺失都有明显修正作用。

jitter 的平均提升为 +4.38，虽然也是正向，但由于基础准确率过低，最终仍然是最困难 corruption。

---

## 14. Local Cache 相比 14_2 Global Cache 的额外提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +1.60 | -0.07 | -0.18 | +0.52 | -0.42 | +0.29 |
| add_local | +2.78 | +0.48 | +2.43 | +1.60 | -0.42 | +1.37 |
| dropout_global | +2.74 | +2.18 | +2.60 | +3.26 | +1.59 | +2.47 |
| dropout_local | +3.72 | +2.60 | +4.30 | +2.39 | +2.15 | +3.03 |
| rotate | +3.33 | +1.81 | +3.26 | +1.84 | +2.25 | +2.50 |
| scale | +1.39 | +2.39 | +0.42 | +0.41 | +2.54 | +1.43 |
| jitter | +0.42 | +0.77 | +1.05 | +0.11 | +2.19 | +0.91 |
| **Average** | **+2.28** | **+1.45** | **+1.98** | **+1.45** | **+1.41** | **+1.72** |

分析：

Local Cache 在 Global Cache 基础上的 all35 平均额外提升为 +1.72，severity=2 额外提升为 +1.98。说明 Local Cache 在 ULIP-2 × ScanObjNN-C hardest all35 上有效。

Local Cache 在 31 / 35 个 setting 上为正增益，只有 4 个 setting 为负增益：

| cor_type | 14_2 | 14_3 | Local Gain |
|---|---:|---:|---:|
| add_local_4 | 30.50 | 30.08 | -0.42 |
| add_global_4 | 35.05 | 34.63 | -0.42 |
| add_global_2 | 36.68 | 36.50 | -0.18 |
| add_global_1 | 36.99 | 36.92 | -0.07 |

与 12 组不同，14 组中 Local Cache 没有在 jitter_3 或 jitter_4 上出现负增益；对 jitter 的 all35 平均仍然是正向的，虽然提升幅度较小。

Local Cache 额外提升最大的 corruption 是 dropout_local，Avg Gain 为 +3.03；其次是 rotate 和 dropout_global。这说明局部缓存对局部缺失和几何结构变化有明显帮助。

---

## 15. 与前序实验的关系

14_3 的前序子实验包括 14_1 和 14_2。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs | ScanObjNN-C hardest all35 | Zero-shot | 26.44 | 26.46 |
| 14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global | ScanObjNN-C hardest all35 | ZS + Global Cache | 31.38 | 31.24 |
| 14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local | ScanObjNN-C hardest all35 | ZS + Global + Local Cache | 33.36 | 32.95 |

当前结果说明：在 ScanObjNN-C hardest all35 上，完整 Point-Cache 能稳定提升 Zero-shot，也高于只使用 Global Cache 的结果。

| 比较 | S2 Avg 变化 | all35 Avg 变化 |
|---|---:|---:|
| 14_2 - 14_1 | +4.94 | +4.78 |
| 14_3 - 14_2 | +1.98 | +1.72 |
| 14_3 - 14_1 | +6.92 | +6.50 |

分析：

Global Cache 是主要提升来源，Local Cache 在此基础上继续提供额外收益。当前 severity=2 下，原文中完整 Point-Cache 相比 Zero-shot 的提升为 +6.99，当前复现为 +6.92，差异仅 -0.07，整体高度接近。

---

## 16. 阶段性结论

本实验完成了 ULIP-2 × ScanObjNN-C hardest all35 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 14_3 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 33.36，原文 Supplementary Table 7 中 ULIP-2 + Hierarchical Cache Avg 也为 33.36，平均值完全一致。
3. 当前 all35 Average 为 32.95，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果与原文 severity=2 平均值高度对齐，但逐 corruption 存在一定波动，主要表现为 add_local / dropout_local 偏高，scale / jitter 偏低。
5. 相比 14_1 Zero-shot，14_3 的 severity=2 Average 提升 +6.92，all35 Average 提升 +6.50。
6. 相比 14_2 Global Cache，14_3 的 severity=2 Average 额外提升 +1.98，all35 Average 额外提升 +1.72。
7. 当前方法趋势正确：Zero-shot < Global Cache < Global + Local Cache。
8. 完整 Point-Cache 在全部 35 个 setting 上相比 Zero-shot 均为正增益。
9. Local Cache 在 31 / 35 个 setting 上相比 Global Cache 为正增益，只在 4 个 setting 上有轻微负增益。
10. jitter 仍然是最困难 corruption，完整 Point-Cache 后平均仍只有 18.37。
11. jitter_3 和 jitter_4 仍然是主要失败点，准确率分别为 13.50 和 14.23。
12. 本实验是 14 组最后一个子实验，完整方法间总结应放入 14 组 summary 文档中。

---

## 17. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh 1

---

## 18. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f13 | sort | uniq -c
