# 02_3_ulip_modelnetc_corruptions_all35_zs_global_local

## 1. 实验目的

复现 ULIP 在 ModelNet-C 全部 35 个损坏设置上使用完整 Point-Cache 后的结果。

本实验属于 baseline 复现阶段的 02 组实验。02 组实验固定使用 ULIP backbone，并在 ModelNet-C 的 35 个 corrupted setting 上评估一种方法。

本实验是在 02_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证局部缓存是否能够在全局缓存之外继续提供额外增益。

具体而言，本实验回答五个问题：

| 问题 | 说明 |
|---|---|
| 完整 Point-Cache 是否提升 Zero-shot？ | 与 02_1 的 Zero-shot 结果比较 |
| Local Cache 是否在 Global Cache 上继续提升？ | 与 02_2 的 ZS + Global 结果比较 |
| Local Cache 对哪些 corruption 最有效？ | 观察 02_3 - 02_2 的 corruption × severity 增益 |
| 完整 Point-Cache 是否缓解高 severity 退化？ | 比较 S0-S4 各 severity 下的平均提升 |
| 复现结果是否与原论文对齐？ | 使用 severity=2 的结果与原论文 Table 1 对比 |

需要特别注意：原论文 Table 1 只报告 corruption severity level = 2 下的结果，而本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原论文 Table 1 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

---

## 2. 当前实现方式

本实验的外部命名规则保持不变：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 02_3_ulip_modelnetc_corruptions_all35_zs_global_local |
| 方法脚本 | Point-Cache/scripts/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/02_run_ulip_modelnetc_corruptions_all35_common.sh |
| 新增 Python runner | Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local/ |

重要变更：

| 旧实现 | 新实现 |
|---|---|
| bash 外层循环 35 次 |
| 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 |
| 模型只加载一次，每个 cor_type 重新创建 DataLoader、重新初始化 cache |
| bash 通过 tee 生成单个 cor_type 的 log |
| Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv |
| summary.csv 的列结构保持不变 |

本实验虽然改用了优化 runner，但实验定义没有改变。优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、Global Cache 初始化、Local Cache 初始化、预测逻辑和输出结构。

---

## 3. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 当前优化 runner | runners/baseline/run_ulip_modelnetc_corruptions_all35.py |
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
| GPU | 单张 Tesla T4 |

本实验使用完整 Point-Cache，即同时使用 Global Cache 和 Local Cache。Global Cache 主要基于点云整体特征，Local Cache 进一步基于局部 patch 聚类特征补充细粒度信息。

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

Point-Cache/results/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则保持不变：

02_3_ulip_modelnetc_corruptions_all35_zs_global_local_add_global_0_YYYYMMDD_HHMMSS.log

也就是说，新版 runner 虽然只启动一次 Python，但仍然会为 35 个 cor_type 生成 35 个独立 log。

---

## 6. 当前结果检查

本实验已经完成，summary.csv 和 logs 目录均为标准状态。

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 35 | 35 | 说明 35 个 cor_type 全部完成 |
| summary 中唯一 cor_type 数 | 35 | 35 | 说明没有漏跑或重复写入 |
| summary 中唯一 log_path 数 | 35 | 35 | 说明每个 cor_type 都有独立日志路径 |
| logs 目录当前 .log 文件数 | 35 | 35 | 说明日志目录状态正常 |
| status=done 数 | 35 | 35 | 说明没有失败项 |
| severity=2 Average | 54.00 | 用于论文对齐 | 与原论文 Table 1 对比 |
| all35 Average | 53.01 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

结论：02_3 实验结果完整，summary.csv、log_path 和 logs 文件数量完全一致，可以作为完整 Point-Cache baseline 的有效结果。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 55.39 | 52.76 | 47.81 | 46.19 | 39.99 | 48.43 |
| add_local | 56.40 | 51.90 | 46.68 | 46.56 | 44.25 | 49.16 |
| dropout_global | 61.67 | 58.95 | 59.20 | 57.86 | 51.46 | 57.83 |
| dropout_local | 62.16 | 58.47 | 56.69 | 53.20 | 44.89 | 55.08 |
| rotate | 62.88 | 63.37 | 62.07 | 57.54 | 49.55 | 59.08 |
| scale | 58.35 | 58.55 | 55.23 | 54.17 | 55.19 | 56.30 |
| jitter | 56.40 | 55.71 | 50.32 | 37.56 | 26.09 | 45.22 |
| **Average** | **59.04** | **57.10** | **54.00** | **50.44** | **44.49** | **53.01** |

整体观察：

1. all35 Average 为 53.01，相比 02_1 Zero-shot 的 46.85 提升 +6.16。
2. severity=2 Average 为 54.00，相比 02_1 的 47.68 提升 +6.32。
3. 相比 02_2 ZS + Global 的 all35 Average 51.62，完整 Point-Cache 进一步提升 +1.39。
4. 相比 02_2 ZS + Global 的 severity=2 Average 52.66，完整 Point-Cache 进一步提升 +1.34。
5. rotate 的平均准确率最高，为 59.08；jitter 的平均准确率最低，为 45.22。
6. jitter_4 仍然是全部 35 个 setting 中最低的结果，为 26.09，说明强坐标扰动仍然是完整 Point-Cache 下最困难的场景。

---

## 8. Severity 维度分析

### 8.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 59.04 | — | 0.00 |
| S1 | 57.10 | -1.94 | -1.94 |
| S2 | 54.00 | -3.10 | -5.04 |
| S3 | 50.44 | -3.56 | -8.60 |
| S4 | 44.49 | -5.95 | -14.55 |

分析：

随着 severity 增大，完整 Point-Cache 的平均准确率仍然单调下降。这说明 Global Cache + Local Cache 能整体提升准确率，但不能完全消除 corruption severity 增强带来的退化趋势。

S0 到 S4 总下降 14.55，略小于 02_1 Zero-shot 的 15.01，也接近 02_2 Global Cache 的 13.73。这说明完整 Point-Cache 对不同 severity 的整体上移非常明显，但在最强 severity 下仍然会遇到明显退化。

### 8.2 与 Zero-shot 和 ZS + Global 的 severity 维度对比

| Severity | Zero-shot Avg | ZS + Global Avg | ZS + Global + Local Avg | Gain over ZS | Gain over Global |
|---:|---:|---:|---:|---:|---:|
| S0 | 53.40 | 57.27 | 59.04 | +5.63 | +1.76 |
| S1 | 50.94 | 55.79 | 57.10 | +6.17 | +1.31 |
| S2 | 47.68 | 52.66 | 54.00 | +6.32 | +1.34 |
| S3 | 43.85 | 48.84 | 50.44 | +6.59 | +1.60 |
| S4 | 38.39 | 43.55 | 44.49 | +6.10 | +0.94 |
| **All35** | **46.85** | **51.62** | **53.01** | **+6.16** | **+1.39** |

分析：

完整 Point-Cache 在所有 severity 上都显著高于 Zero-shot，并且也高于只使用 Global Cache 的结果。

相对 Zero-shot，完整 Point-Cache 的提升在 S0-S4 上都超过 +5.6，最高为 S3 的 +6.59。说明完整 Point-Cache 对中高强度 corruption 有稳定补偿作用。

相对 Global Cache，Local Cache 的额外增益在 S0-S4 上均为正，但幅度较小，平均约 +1.39。其中 S4 的额外提升最小，只有 +0.94，说明在极强 corruption 下，Local Cache 的补充作用也会受到限制。

---

## 9. Corruption 难度分析

### 9.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 45.22 | 高 severity 下仍然严重退化，S4 仅 26.09 |
| 2 | add_global | 48.43 | 相比 Zero-shot 大幅提升，但仍属于困难 corruption |
| 3 | add_local | 49.16 | 局部异常点仍有明显影响 |
| 4 | dropout_local | 55.08 | Local Cache 后整体表现明显改善 |
| 5 | scale | 56.30 | 对尺度扰动较稳定 |
| 6 | dropout_global | 57.83 | 表现较好，说明全局结构缺失相对容易 |
| 7 | rotate | 59.08 | 当前方法下平均最高，最容易 |

分析：

加入完整 Point-Cache 后，整体排序与 02_2 基本一致：jitter 仍然最难，add_global 和 add_local 仍然属于困难 corruption，而 rotate、dropout_global 和 scale 表现较好。

这说明 Local Cache 虽然提升了整体性能，但没有完全改变 corruption 难度结构。也就是说，Local Cache 是在已有 Global Cache 基础上补充局部判别信息，而不是彻底重排各类 corruption 的难度。

### 9.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 58.35 | 55.19 | 3.16 | 5.42% | 56.30 |
| dropout_global | 61.67 | 51.46 | 10.21 | 16.56% | 57.83 |
| add_local | 56.40 | 44.25 | 12.15 | 21.54% | 49.16 |
| rotate | 62.88 | 49.55 | 13.33 | 21.20% | 59.08 |
| add_global | 55.39 | 39.99 | 15.40 | 27.80% | 48.43 |
| dropout_local | 62.16 | 44.89 | 17.27 | 27.78% | 55.08 |
| jitter | 56.40 | 26.09 | 30.31 | 53.74% | 45.22 |

分析：

scale 仍然最稳定，从 S0 到 S4 只下降 3.16，说明完整 Point-Cache 对尺度变化具有较强鲁棒性。

jitter 仍然退化最严重，从 S0 的 56.40 下降到 S4 的 26.09，绝对下降 30.31，相对下降 53.74%。这说明强 jitter 对全局特征和局部特征都会产生破坏，Local Cache 也难以完全恢复。

dropout_local 的平均准确率达到 55.08，相比 Zero-shot 的 50.22 和 Global Cache 的 53.62 都有提升，说明 Local Cache 对局部缺失类 corruption 有实际帮助。

---

## 10. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | add_global | 55.39 | rotate | 62.88 | 7.49 |
| S1 | add_local | 51.90 | rotate | 63.37 | 11.47 |
| S2 | add_local | 46.68 | rotate | 62.07 | 15.39 |
| S3 | jitter | 37.56 | dropout_global | 57.86 | 20.30 |
| S4 | jitter | 26.09 | scale | 55.19 | 29.10 |

分析：

随着 severity 增大，不同 corruption 之间的 best-worst gap 从 S0 的 7.49 扩大到 S4 的 29.10。完整 Point-Cache 虽然提升了整体准确率，但高 severity 下不同 corruption 之间的难度差异仍然明显扩大。

在 S0 到 S2，最困难项主要是 add_global 和 add_local；在 S3 和 S4，最困难项变为 jitter。这说明低中强度下异常点类 corruption 更困难，而高强度下坐标扰动类 corruption 最致命。

---

## 11. 低准确率区域分析

### 11.1 低准确率 setting 数量

| 条件 | Zero-shot 数量 | ZS + Global 数量 | ZS + Global + Local 数量 | 相比 ZS 减少 | 相比 Global 减少 |
|---|---:|---:|---:|---:|---:|
| Acc < 50 | 17 / 35 | 12 / 35 | 10 / 35 | -7 | -2 |
| Acc < 45 | 12 / 35 | 6 / 35 | 5 / 35 | -7 | -1 |
| Acc < 40 | 7 / 35 | 3 / 35 | 3 / 35 | -4 | 0 |
| Acc < 35 | 5 / 35 | 1 / 35 | 1 / 35 | -4 | 0 |
| Acc < 30 | 3 / 35 | 1 / 35 | 1 / 35 | -2 | 0 |
| Acc < 25 | 1 / 35 | 0 / 35 | 0 / 35 | -1 | 0 |

分析：

完整 Point-Cache 明显减少了低准确率区域。相比 Zero-shot，Acc < 50 的 setting 从 17 个减少到 10 个，Acc < 45 的 setting 从 12 个减少到 5 个。

相比 Global Cache，Local Cache 进一步减少了中等低准确率区域，例如 Acc < 50 从 12 个减少到 10 个，Acc < 45 从 6 个减少到 5 个。但对于 Acc < 40、Acc < 35 和 Acc < 30 的严重失败区域，Local Cache 没有进一步减少数量。

这说明 Local Cache 的额外作用更多体现在“中低性能 setting 的继续抬升”，而不是完全解决最极端失败案例。

### 11.2 仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 26.09 | 当前最低，强坐标扰动仍然最难 |
| jitter_3 | 37.56 | 高 severity jitter 仍然低于 40 |
| add_global_4 | 39.99 | 高 severity 全局异常点仍然困难 |
| add_local_4 | 44.25 | 高 severity 局部异常点仍然偏低 |
| dropout_local_4 | 44.89 | 高 severity 局部缺失仍有明显影响 |

分析：

完整 Point-Cache 后，最严重的低准确率区域仍集中在 jitter_3、jitter_4 和 add_global_4。尤其是 jitter_4 只有 26.09，说明强坐标扰动会同时干扰全局和局部特征，使 Global Cache 与 Local Cache 都难以可靠修正。

---

## 12. 完整 Point-Cache 相比 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +10.01 | +14.19 | +13.81 | +16.21 | +13.49 | +13.54 |
| add_local | +5.35 | +4.82 | +2.76 | +4.87 | +5.15 | +4.59 |
| dropout_global | +5.55 | +3.12 | +4.50 | +5.31 | +6.04 | +4.90 |
| dropout_local | +5.39 | +3.69 | +6.12 | +5.31 | +3.80 | +4.86 |
| rotate | +6.48 | +7.17 | +6.88 | +6.20 | +5.22 | +6.39 |
| scale | +5.27 | +6.04 | +4.34 | +4.05 | +6.77 | +5.29 |
| jitter | +1.38 | +4.13 | +5.83 | +4.21 | +2.22 | +3.55 |
| **Average** | **+5.63** | **+6.17** | **+6.32** | **+6.59** | **+6.10** | **+6.16** |

分析：

完整 Point-Cache 相比 Zero-shot 在全部 35 个 setting 上整体提升 +6.16。最大平均提升来自 add_global，Avg Gain 为 +13.54，说明完整 Point-Cache 对全局异常点有非常强的修正能力。

jitter 的平均提升最小，为 +3.55。虽然 jitter_2 和 jitter_3 分别提升 +5.83 和 +4.21，但 jitter_4 只提升 +2.22，说明强 jitter 仍然难以修复。

从 severity 维度看，完整 Point-Cache 在 S0-S4 上均有超过 +5.6 的平均提升，说明其提升并不局限于某个 severity，而是具有较稳定的总体鲁棒性。

---

## 13. Local Cache 相比 Global Cache 的额外提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +1.14 | +0.98 | +1.74 | +2.43 | +1.58 | +1.57 |
| add_local | +1.70 | +1.13 | -0.56 | +0.17 | +0.73 | +0.63 |
| dropout_global | +2.72 | +0.77 | +2.15 | +2.67 | +1.87 | +2.04 |
| dropout_local | +2.31 | +0.65 | +1.83 | +1.90 | +0.64 | +1.47 |
| rotate | +1.74 | +2.59 | +2.26 | +2.03 | +0.97 | +1.92 |
| scale | +1.87 | +2.15 | +1.26 | +1.17 | +2.23 | +1.74 |
| jitter | +0.85 | +0.89 | +0.68 | +0.81 | -1.42 | +0.36 |
| **Average** | **+1.76** | **+1.31** | **+1.34** | **+1.60** | **+0.94** | **+1.39** |

分析：

Local Cache 在 Global Cache 基础上的平均额外提升为 +1.39，说明局部缓存确实有效，但增益幅度明显小于 Global Cache 相比 Zero-shot 的提升。

Local Cache 的额外增益并不是对所有 setting 都为正。add_local_2 相比 Global Cache 下降 -0.56，jitter_4 下降 -1.42。这说明 Local Cache 在个别高难度或局部扰动场景中可能引入噪声，尤其当局部特征本身被严重破坏时，局部缓存检索可能不稳定。

平均额外增益最大的 corruption 是 dropout_global、rotate 和 scale，分别为 +2.04、+1.92 和 +1.74。dropout_local 的 Avg Gain 为 +1.47，也说明 Local Cache 对局部结构缺失有一定帮助。

jitter 的额外增益最小，只有 +0.36，且 jitter_4 为负增益。这进一步说明强坐标扰动会破坏局部 patch 表征，使 Local Cache 难以发挥稳定作用。

---

## 14. 与原论文 severity=2 结果对比

原论文 Table 1 报告的是 severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原论文 S2 | Diff | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 47.81 | 46.15 | +1.66 | 1.66 |
| add_local | 46.68 | 47.85 | -1.17 | 1.17 |
| dropout_global | 59.20 | 59.16 | +0.04 | 0.04 |
| dropout_local | 56.69 | 56.00 | +0.69 | 0.69 |
| rotate | 62.07 | 61.47 | +0.60 | 0.60 |
| scale | 55.23 | 55.35 | -0.12 | 0.12 |
| jitter | 50.32 | 49.92 | +0.40 | 0.40 |
| **Average** | **54.00** | **53.70** | **+0.30** | **0.67 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.30 |
| MAE | 0.67 |
| RMSE | 0.86 |
| Max Abs Diff | 1.66 |

分析：

当前复现与原论文的 severity=2 结果整体一致。平均差异为 +0.30，MAE 为 0.67，最大单项差异为 add_global 的 +1.66。

相较于 02_1 和 02_2，02_3 的单项差异略大，但整体平均仍然高度接近原论文的 53.70。考虑到 Hierarchical Cache 涉及 KMeans 局部聚类、cache 在线更新和测试样本顺序，局部结果存在小幅波动是可以接受的。

因此，当前 02_3 可以认为成功复现了 ULIP + Hierarchical Cache 在 ModelNet-C severity=2 下的结果。

---

## 15. 三种方法的核心对比

| 指标 | 02_1 Zero-shot | 02_2 ZS + Global | 02_3 ZS + Global + Local |
|---|---:|---:|---:|
| severity=2 Average | 47.68 | 52.66 | 54.00 |
| all35 Average | 46.85 | 51.62 | 53.01 |
| S0 Average | 53.40 | 57.27 | 59.04 |
| S1 Average | 50.94 | 55.79 | 57.10 |
| S2 Average | 47.68 | 52.66 | 54.00 |
| S3 Average | 43.85 | 48.84 | 50.44 |
| S4 Average | 38.39 | 43.55 | 44.49 |
| Acc < 40 的 setting 数 | 7 | 3 | 3 |
| Acc < 35 的 setting 数 | 5 | 1 | 1 |

分析：

三种方法呈现稳定递进关系：

Zero-shot < ZS + Global < ZS + Global + Local

Global Cache 是主要提升来源。它将 all35 Average 从 46.85 提升到 51.62，提升 +4.77。Local Cache 在此基础上进一步提升到 53.01，额外提升 +1.39。

因此，完整 Point-Cache 的总体提升可以分解为：

| 来源 | all35 Avg 提升 |
|---|---:|
| Global Cache 相比 Zero-shot | +4.77 |
| Local Cache 额外提升 | +1.39 |
| 完整 Point-Cache 相比 Zero-shot | +6.16 |

从低准确率 setting 数看，Global Cache 主要负责减少严重失败案例，而 Local Cache 主要负责进一步提高中高准确率区域。对于最极端的低准确率区域，例如 jitter_4，Local Cache 并没有解决根本问题。

---

## 16. 对后续方法设计的启发

本实验说明完整 Point-Cache 可以稳定提升 ULIP 在 ModelNet-C 上的鲁棒性，但也暴露出几个问题：

| 现象 | 启发 |
|---|---|
| Global Cache 是主要提升来源 | 后续 MCM-PC 需要保留或增强全局缓存的稳定性 |
| Local Cache 有额外收益但幅度较小 | 局部信息有价值，但需要更可靠的选择或加权机制 |
| jitter_4 仍然很低 | 强坐标扰动会同时破坏全局和局部特征 |
| Local Cache 在 jitter_4 出现负增益 | 局部缓存可能在严重扰动下引入不可靠信息 |
| add_global 仍然获得最大总体提升 | 全局异常点是 cache 方法最容易补偿的 corruption |
| 低准确率区域没有被完全消除 | 后续需要设计错误伪标签诊断、负缓存或可靠性控制机制 |

因此，后续 MCM-PC 不应只追求“增加更多 cache”，而应重点考虑：

1. cache 样本是否可靠；
2. cache logit 是否应该动态加权；
3. Local Cache 在严重扰动下是否应该被抑制；
4. 全局与局部预测冲突时，是否可以作为错误诊断信号；
5. 对 jitter 等高难度 corruption 是否需要额外的几何稳定性机制。

---

## 17. 阶段性结论

本实验完成了 ULIP × ModelNet-C 全 35 corrupted setting 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 实验完整性正常：summary.csv 有 35 行，cor_type 唯一数为 35，log_path 唯一数为 35，logs 文件数为 35。
2. severity=2 Average 为 54.00，与原论文 53.70 基本一致，说明复现结果可靠。
3. all35 Average 为 53.01，相比 02_1 Zero-shot 的 46.85 提升 +6.16。
4. all35 Average 相比 02_2 Global Cache 的 51.62 进一步提升 +1.39。
5. Global Cache 是主要提升来源，Local Cache 提供稳定但较小的额外增益。
6. Local Cache 的增益不是全局均匀的，个别 setting 如 add_local_2 和 jitter_4 出现负增益。
7. jitter 仍然是完整 Point-Cache 下最困难的 corruption，尤其 jitter_4 只有 26.09。
8. 完整 Point-Cache 明显减少低准确率区域，但没有完全解决最极端的 failure cases。
9. 本实验验证了 Point-Cache 的 hierarchical cache 确实有效，但也提示后续 MCM-PC 需要进一步做可靠性建模和错误抑制。

---

## 18. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 1

---

## 19. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f13 | sort | uniq -c
