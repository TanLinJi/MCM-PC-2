# 12_3_ulip2_modelnetc_corruptions_all35_zs_global_local

## 1. 实验目的

本实验用于复现 ULIP-2 在 ModelNet-C 全部 35 个损坏设置上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 12_3_ulip2_modelnetc_corruptions_all35_zs_global_local |
| Backbone | ULIP-2 |
| Dataset | ModelNet-C |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 12_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证局部缓存是否能在全局缓存之外继续提升 ULIP-2 在 ModelNet-C corrupted setting 上的鲁棒性。

本文件只记录 12_3 本身，并与前序子实验 12_1 和 12_2 进行对比。完整 12 组三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 12 组 summary 文档中。

需要特别注意：原文 Point-Cache Table 1 只报告 corruption severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原文 Point-Cache Table 1 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

---

## 2. 当前实现方式

本实验的外部命名规则如下：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 12_3_ulip2_modelnetc_corruptions_all35_zs_global_local |
| 方法脚本 | Point-Cache/scripts/baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/12_run_ulip2_modelnetc_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_ulip2_modelnetc_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local/ |

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
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 优化 runner | runners/baseline/run_ulip2_modelnetc_corruptions_all35.py |
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
| GPU | 单张 Tesla T4，本次运行记录为 GPU 0 |

本实验使用 `modelnet_c` 作为 dataset 参数，并在 Python runner 内部循环 35 个 `cor_type`。实际读取文件形式为：

data/modelnet_c/{corruption}_{severity}.h5

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

Point-Cache/results/baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

12_3_ulip2_modelnetc_corruptions_all35_zs_global_local_add_global_0_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 62.74 | 用于论文对齐 | 与原文 Point-Cache Table 1 对比 |
| all35 Average | 62.49 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，12_3 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ModelNet-C severity=2 参考值进行对比。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 71.03 | 69.94 | 68.40 | 68.23 | 66.65 | 68.85 |
| add_local | 66.73 | 61.95 | 61.18 | 60.21 | 57.41 | 61.50 |
| dropout_global | 74.43 | 73.38 | 72.45 | 65.76 | 52.88 | 67.78 |
| dropout_local | 72.65 | 69.21 | 63.29 | 56.93 | 49.55 | 62.33 |
| rotate | 73.99 | 74.43 | 73.46 | 71.35 | 68.68 | 72.38 |
| scale | 72.45 | 72.12 | 70.22 | 68.52 | 67.95 | 70.25 |
| jitter | 64.06 | 47.49 | 30.15 | 17.30 | 12.84 | 34.37 |
| **Average** | **70.76** | **66.93** | **62.74** | **58.33** | **53.71** | **62.49** |

整体观察：

1. all35 Average 为 62.49，表示 ULIP-2 在 ModelNet-C 全 35 个 corrupted setting 上使用完整 Point-Cache 后的整体水平。
2. severity=2 Average 为 62.74，用于和原文 Point-Cache Table 1 对齐。
3. rotate 的平均准确率最高，为 72.38。
4. jitter 的平均准确率最低，为 34.37。
5. jitter_4 为 12.84，是当前完整 Point-Cache 下的最低结果。
6. 相比 12_2 Global Cache，12_3 在大多数 setting 上继续提升，且 severity=2 的全部 7 个 corruption 都为正增益。

---

## 8. 与原文结果对比

原文 Point-Cache Table 1 报告的是 ModelNet-C severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 68.40 | 68.11 | +0.29 | 0.29 |
| add_local | 61.18 | 61.26 | -0.08 | 0.08 |
| dropout_global | 72.45 | 73.22 | -0.77 | 0.77 |
| dropout_local | 63.29 | 63.65 | -0.36 | 0.36 |
| rotate | 73.46 | 73.34 | +0.12 | 0.12 |
| scale | 70.22 | 70.42 | -0.20 | 0.20 |
| jitter | 30.15 | 29.50 | +0.65 | 0.65 |
| **Average** | **62.74** | **62.79** | **-0.05** | **0.35 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | -0.05 |
| MAE | 0.35 |
| RMSE | 0.43 |
| Max Abs Diff | 0.77 |

分析：

当前复现的 severity=2 Average 为 62.74，原文为 62.79，差异仅 -0.05。单项最大绝对差异为 dropout_global 的 0.77，整体误差很小。

因此，12_3 不只是脚本跑通，而且数值也与原文 ULIP-2 + Hierarchical Cache 在 ModelNet-C severity=2 上的结果高度对齐。

---

## 9. Severity 维度分析

### 9.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 70.76 | — | 0.00 |
| S1 | 66.93 | -3.83 | -3.83 |
| S2 | 62.74 | -4.19 | -8.02 |
| S3 | 58.33 | -4.41 | -12.43 |
| S4 | 53.71 | -4.62 | -17.05 |

分析：

随着 severity 从 S0 增大到 S4，完整 Point-Cache 的平均准确率从 70.76 下降到 53.71，总下降 17.05 个百分点。整体上，severity 越高，准确率越低。

完整 Point-Cache 明显提升了整体水平，但没有消除高 severity corruption 的影响。S4 仍然比 S0 低 17.05 个百分点，说明强 corruption 下仍存在明显鲁棒性缺口。

### 9.2 与前序实验的 severity 维度对比

| Severity | 12_1 Zero-shot Avg | 12_2 ZS + Global Avg | 12_3 ZS + Global + Local Avg | Gain over 12_1 | Gain over 12_2 |
|---:|---:|---:|---:|---:|---:|
| S0 | 67.19 | 69.28 | 70.76 | +3.57 | +1.48 |
| S1 | 61.93 | 65.37 | 66.93 | +5.00 | +1.56 |
| S2 | 58.02 | 61.22 | 62.74 | +4.71 | +1.51 |
| S3 | 53.92 | 57.08 | 58.33 | +4.41 | +1.25 |
| S4 | 49.29 | 52.52 | 53.71 | +4.41 | +1.19 |
| **all35** | **58.07** | **61.09** | **62.49** | **+4.42** | **+1.40** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有明显正增益，all35 平均提升 +4.42，severity=2 提升 +4.71。

Local Cache 在 Global Cache 基础上的额外提升也稳定为正，all35 平均额外提升 +1.40，severity=2 额外提升 +1.51。

这说明在 ULIP-2 × ModelNet-C all35 上，Local Cache 的贡献比 clean setting 中更明显。11_3 clean setting 中 Local Cache 只额外提升 +0.36，而 12_3 all35 中额外提升 +1.40，说明 corrupted setting 给 Local Cache 提供了更大的补偿空间。

---

## 10. Corruption 难度分析

### 10.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 34.37 | 平均最低，高 severity 下仍然非常困难 |
| 2 | add_local | 61.50 | 局部异常点仍然较困难 |
| 3 | dropout_local | 62.33 | 局部缺失仍有明显影响 |
| 4 | dropout_global | 67.78 | 高 severity 下降明显 |
| 5 | add_global | 68.85 | 表现中等偏高 |
| 6 | scale | 70.25 | 较稳定 |
| 7 | rotate | 72.38 | 当前最容易 |

分析：

完整 Point-Cache 后，jitter 仍然是绝对最困难的 corruption，但平均准确率从 12_1 的 27.28 提升到 12_3 的 34.37，说明 Global + Local 对 jitter 有帮助，但无法完全解决强坐标扰动问题。

rotate 和 scale 仍然最稳定，说明 ULIP-2 对旋转和尺度扰动的鲁棒性相对更强。

### 10.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 72.45 | 67.95 | 4.50 | 6.21% | 70.25 |
| rotate | 73.99 | 68.68 | 5.31 | 7.18% | 72.38 |
| add_global | 71.03 | 66.65 | 4.38 | 6.17% | 68.85 |
| add_local | 66.73 | 57.41 | 9.32 | 13.97% | 61.50 |
| dropout_global | 74.43 | 52.88 | 21.55 | 28.95% | 67.78 |
| dropout_local | 72.65 | 49.55 | 23.10 | 31.80% | 62.33 |
| jitter | 64.06 | 12.84 | 51.22 | 79.96% | 34.37 |

分析：

scale、rotate 和 add_global 在 severity 增大时较稳定；jitter 的退化仍然最严重，从 S0 的 64.06 下降到 S4 的 12.84，绝对下降 51.22，相对下降 79.96%。

dropout_local 和 dropout_global 也有明显退化，说明点云缺失类 corruption 在高 severity 下仍会严重影响完整 Point-Cache 的表现。

---

## 11. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | jitter | 64.06 | dropout_global | 74.43 | 10.37 |
| S1 | jitter | 47.49 | rotate | 74.43 | 26.94 |
| S2 | jitter | 30.15 | rotate | 73.46 | 43.31 |
| S3 | jitter | 17.30 | rotate | 71.35 | 54.05 |
| S4 | jitter | 12.84 | rotate | 68.68 | 55.84 |

分析：

在所有 severity 下，jitter 都是最困难 corruption。随着 severity 增大，best-worst gap 从 S0 的 10.37 扩大到 S4 的 55.84。

完整 Point-Cache 提升了 jitter 的绝对值，但并没有改变 jitter 是主要失败模式这一事实。

---

## 12. 低准确率区域分析

### 12.1 低准确率 setting 数量

| 条件 | 12_1 Zero-shot 数量 | 12_2 ZS + Global 数量 | 12_3 ZS + Global + Local 数量 | 相比 12_1 减少 | 相比 12_2 减少 |
|---|---:|---:|---:|---:|---:|
| Acc < 65 | 18 / 35 | 12 / 35 | 8 / 35 | -10 | -4 |
| Acc < 60 | 13 / 35 | 7 / 35 | 5 / 35 | -8 | -2 |
| Acc < 55 | 10 / 35 | 5 / 35 | 4 / 35 | -6 | -1 |
| Acc < 50 | 7 / 35 | 4 / 35 | 3 / 35 | -4 | -1 |
| Acc < 45 | 4 / 35 | 4 / 35 | 3 / 35 | -1 | -1 |
| Acc < 40 | 4 / 35 | 3 / 35 | 3 / 35 | -1 | 0 |
| Acc < 35 | 4 / 35 | 3 / 35 | 3 / 35 | -1 | 0 |
| Acc < 30 | 3 / 35 | 3 / 35 | 2 / 35 | -1 | -1 |
| Acc < 25 | 3 / 35 | 2 / 35 | 2 / 35 | -1 | 0 |

分析：

完整 Point-Cache 明显减少了低准确率区域。相比 Zero-shot，Acc < 65 的 setting 从 18 个减少到 8 个，Acc < 60 的 setting 从 13 个减少到 5 个。

相比 Global Cache，Local Cache 继续减少部分低准确率区域，例如 Acc < 65 的 setting 从 12 个减少到 8 个，Acc < 60 的 setting 从 7 个减少到 5 个。

但高 severity jitter 仍然是主要失败区域，Acc < 30 的 setting 仍有 2 个。

### 12.2 当前仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 12.84 | 最高 severity 坐标扰动仍然最难 |
| jitter_3 | 17.30 | 中高 severity 坐标扰动仍然极低 |
| jitter_2 | 30.15 | severity=2 jitter 仍明显低于其他 corruption |
| jitter_1 | 47.49 | 虽有提升，但仍低于大多数 corruption |
| dropout_local_4 | 49.55 | 高 severity 局部缺失仍然困难 |
| dropout_global_4 | 52.88 | 高 severity 全局缺失仍然困难 |

分析：

12_3 中最困难区域仍然集中在 jitter，尤其是 jitter_3 和 jitter_4。这说明 Global + Local Cache 不能完全解决强坐标扰动问题。

---

## 13. 完整 Point-Cache 相比 12_1 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +2.84 | +3.57 | +2.88 | +4.49 | +4.58 | +3.67 |
| add_local | +4.05 | +4.37 | +6.52 | +8.83 | +9.11 | +6.58 |
| dropout_global | +3.00 | +3.00 | +3.53 | +3.89 | +3.97 | +3.48 |
| dropout_local | +2.84 | +5.31 | +4.98 | +5.59 | +4.41 | +4.63 |
| rotate | +2.07 | +2.43 | +2.43 | +3.08 | +2.59 | +2.52 |
| scale | +3.41 | +3.44 | +3.49 | +1.83 | +2.76 | +2.99 |
| jitter | +6.77 | +12.89 | +9.16 | +3.12 | +3.52 | +7.09 |
| **Average** | **+3.57** | **+5.00** | **+4.71** | **+4.41** | **+4.41** | **+4.42** |

分析：

完整 Point-Cache 相比 Zero-shot 在全部 35 个 setting 上均为正增益，没有出现负增益。

平均提升最大的 corruption 是 jitter，Avg Gain 为 +7.09；其次是 add_local，Avg Gain 为 +6.58。说明完整 Point-Cache 对坐标扰动和局部异常点有明显补偿作用。

不过，由于 jitter 原始准确率极低，即使提升最大，最终仍然是最困难 corruption。

---

## 14. Local Cache 相比 12_2 Global Cache 的额外提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +0.93 | +1.42 | +1.10 | +1.29 | +1.42 | +1.23 |
| add_local | -0.57 | -0.93 | +1.74 | +3.32 | +4.67 | +1.65 |
| dropout_global | +2.19 | +1.10 | +1.87 | +1.01 | +4.62 | +2.16 |
| dropout_local | +1.18 | +2.11 | +1.74 | +0.69 | +2.10 | +1.56 |
| rotate | +0.98 | +1.05 | +1.46 | +1.33 | +0.08 | +0.98 |
| scale | +1.10 | +0.73 | +1.46 | +1.91 | +2.23 | +1.49 |
| jitter | +4.55 | +5.48 | +1.22 | -0.82 | -2.80 | +1.53 |
| **Average** | **+1.48** | **+1.56** | **+1.51** | **+1.25** | **+1.19** | **+1.40** |

分析：

Local Cache 在 Global Cache 基础上的 all35 平均额外提升为 +1.40，severity=2 额外提升为 +1.51。说明 Local Cache 在 ULIP-2 × ModelNet-C all35 上有稳定贡献。

Local Cache 不是在所有 setting 上都正向。add_local_0、add_local_1、jitter_3 和 jitter_4 出现负增益。其中 jitter 高 severity 下 Local Cache 反而下降，说明当坐标扰动非常强时，局部 patch 特征可能被严重破坏，Local Cache 可能引入不可靠信息。

另一方面，dropout_global、dropout_local、scale 和 add_global 的 Local Cache 额外增益较稳定，说明局部缓存在多数 corruption 下仍然有效。

---

## 15. 与前序实验的关系

12_3 的前序子实验包括 12_1 和 12_2。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 12_1_ulip2_modelnetc_corruptions_all35_zs | ModelNet-C all35 | Zero-shot | 58.02 | 58.07 |
| 12_2_ulip2_modelnetc_corruptions_all35_zs_global | ModelNet-C all35 | ZS + Global Cache | 61.22 | 61.09 |
| 12_3_ulip2_modelnetc_corruptions_all35_zs_global_local | ModelNet-C all35 | ZS + Global + Local Cache | 62.74 | 62.49 |

当前结果说明：在 ModelNet-C all35 上，完整 Point-Cache 能稳定提升 Zero-shot，也高于只使用 Global Cache 的结果。

| 比较 | S2 Avg 变化 | all35 Avg 变化 |
|---|---:|---:|
| 12_2 - 12_1 | +3.20 | +3.02 |
| 12_3 - 12_2 | +1.51 | +1.40 |
| 12_3 - 12_1 | +4.71 | +4.42 |

分析：

Global Cache 是主要提升来源，Local Cache 在此基础上继续提供额外收益。当前 severity=2 下，原文中完整 Point-Cache 相比 Zero-shot 的提升为 +4.84，当前复现为 +4.71，差异仅 -0.13，整体高度接近。

---

## 16. 阶段性结论

本实验完成了 ULIP-2 × ModelNet-C all35 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 12_3 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 62.74，原文 Point-Cache Table 1 中 ULIP-2 + Hierarchical Cache Avg 为 62.79，差异仅 -0.05。
3. 当前 all35 Average 为 62.49，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果与原文 severity=2 数值高度对齐，可以认为 12_3 复现成功。
5. 相比 12_1 Zero-shot，12_3 的 severity=2 Average 提升 +4.71，all35 Average 提升 +4.42。
6. 相比 12_2 Global Cache，12_3 的 severity=2 Average 额外提升 +1.51，all35 Average 额外提升 +1.40。
7. 当前方法趋势正确：Zero-shot < Global Cache < Global + Local Cache。
8. 完整 Point-Cache 在全部 35 个 setting 上相比 Zero-shot 均为正增益。
9. Local Cache 在多数 setting 上有额外正增益，但在 add_local_0、add_local_1、jitter_3 和 jitter_4 上出现负增益。
10. jitter 仍然是最困难 corruption，尤其 jitter_4 只有 12.84。
11. 本实验是 12 组最后一个子实验，完整方法间总结应放入 12 组 summary 文档中。

---

## 17. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 1

---

## 18. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f13 | sort | uniq -c
