# 22_3_openshape_modelnetc_corruptions_all35_zs_global_local

## 1. 实验目的

本实验用于复现 OpenShape 在 ModelNet-C 全部 35 个损坏设置上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 22_3_openshape_modelnetc_corruptions_all35_zs_global_local |
| Backbone | OpenShape |
| Dataset | ModelNet-C |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 22_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证完整 Point-Cache 对 OpenShape 在 ModelNet-C corrupted setting 上的影响。

需要特别注意：本实验中 Global Cache 是主要提升来源，而 Local Cache 的额外贡献接近于零。当前 22_3 相比 22_2 的 all35 Average 几乎持平，severity=2 Average 略低。因此，本文档不能把 Local Cache 描述为“明显提升”，而应如实记录为：完整 Point-Cache 仍显著优于 Zero-shot，但 Local Cache 在 OpenShape × ModelNet-C 上的额外收益很弱。

本文件只记录 22_3 本身，并与前序子实验 22_1 和 22_2 进行对比。完整 22 组三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 22 组 summary 文档中。

---

## 2. 当前实现方式

本实验的外部命名规则如下：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 22_3_openshape_modelnetc_corruptions_all35_zs_global_local |
| 方法脚本 | Point-Cache/scripts/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/22_run_openshape_modelnetc_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_openshape_modelnetc_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local/ |

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
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 优化 runner | runners/baseline/run_openshape_modelnetc_corruptions_all35.py |
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

Point-Cache/results/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

22_3_openshape_modelnetc_corruptions_all35_zs_global_local_add_global_0_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 76.33 | 用于论文对齐 | 与原文 Point-Cache Table 1 对比 |
| all35 Average | 75.14 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，22_3 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ModelNet-C severity=2 参考值进行对比。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 79.58 | 75.57 | 72.93 | 73.10 | 70.54 | 74.34 |
| add_local | 77.51 | 74.07 | 72.33 | 71.72 | 71.92 | 73.51 |
| dropout_global | 84.04 | 83.47 | 82.78 | 79.66 | 64.71 | 78.93 |
| dropout_local | 81.97 | 79.29 | 76.42 | 71.39 | 66.82 | 75.18 |
| rotate | 84.44 | 83.91 | 83.14 | 79.13 | 74.76 | 81.08 |
| scale | 79.54 | 79.21 | 78.08 | 77.51 | 75.77 | 78.02 |
| jitter | 80.67 | 74.55 | 68.64 | 55.02 | 45.83 | 64.94 |
| **Average** | **81.11** | **78.58** | **76.33** | **72.50** | **67.19** | **75.14** |

整体观察：

1. all35 Average 为 75.14，表示 OpenShape 在 ModelNet-C 全 35 个 corrupted setting 上使用完整 Point-Cache 后的整体鲁棒性水平。
2. severity=2 Average 为 76.33，用于和原文 Point-Cache Table 1 对齐。
3. rotate 的平均准确率最高，为 81.08。
4. jitter 的平均准确率最低，为 64.94。
5. jitter_4 为 45.83，是全部 35 个 setting 中最低的结果。
6. 相比 22_1 Zero-shot，22_3 有明显提升；但相比 22_2 Global Cache，22_3 基本持平。

---

## 8. 与原文结果对比

原文 Point-Cache Table 1 报告的是 ModelNet-C severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 72.93 | 74.84 | -1.91 | 1.91 |
| add_local | 72.33 | 73.70 | -1.37 | 1.37 |
| dropout_global | 82.78 | 82.21 | +0.57 | 0.57 |
| dropout_local | 76.42 | 76.26 | +0.16 | 0.16 |
| rotate | 83.14 | 82.66 | +0.48 | 0.48 |
| scale | 78.08 | 78.12 | -0.04 | 0.04 |
| jitter | 68.64 | 68.35 | +0.29 | 0.29 |
| **Average** | **76.33** | **76.59** | **-0.26** | **0.69 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | -0.26 |
| MAE | 0.69 |
| RMSE | 0.94 |
| Max Abs Diff | 1.91 |

分析：

当前复现的 severity=2 Average 为 76.33，原文为 76.59，差异为 -0.26。整体仍在可接受范围内。

不过，逐 corruption 看，22_3 的波动比 22_1 和 22_2 更明显。主要偏低项是 add_global 和 add_local，其中 add_global 低于原文 1.91，add_local 低于原文 1.37。dropout_global、dropout_local、rotate 和 jitter 则略高于原文。

因此，22_3 应记录为：平均结果基本对齐原文，但 add_global 和 add_local 两项偏低；整体仍可作为有效复现结果。

---

## 9. Severity 维度分析

### 9.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 81.11 | — | 0.00 |
| S1 | 78.58 | -2.53 | -2.53 |
| S2 | 76.33 | -2.25 | -4.78 |
| S3 | 72.50 | -3.83 | -8.60 |
| S4 | 67.19 | -5.31 | -13.91 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 81.11 下降到 67.19，总下降 13.91 个百分点。整体上，severity 越高，完整 Point-Cache 的准确率越低。

相比 22_1 Zero-shot，完整 Point-Cache 明显提升了整体水平；相比 22_2 Global Cache，不同 severity 上基本持平或小幅变化。

### 9.2 与前序实验的 severity 维度对比

| Severity | 22_1 Zero-shot Avg | 22_2 ZS + Global Avg | 22_3 ZS + Global + Local Avg | Gain over 22_1 | Gain over 22_2 |
|---:|---:|---:|---:|---:|---:|
| S0 | 80.31 | 81.28 | 81.11 | +0.80 | -0.17 |
| S1 | 76.99 | 78.59 | 78.58 | +1.59 | -0.01 |
| S2 | 73.57 | 76.46 | 76.33 | +2.76 | -0.13 |
| S3 | 69.28 | 72.56 | 72.50 | +3.22 | -0.06 |
| S4 | 62.72 | 66.79 | 67.19 | +4.47 | +0.40 |
| **all35** | **72.57** | **75.14** | **75.14** | **+2.57** | **+0.01** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有正增益，all35 平均提升 +2.57，severity=2 提升 +2.76。

但是，Local Cache 在 Global Cache 基础上的额外贡献很弱：all35 平均只提升 +0.01，severity=2 反而下降 -0.13。只有 S4 有较小正增益 +0.40。

因此，22_3 的结论应写为：完整 Point-Cache 整体优于 Zero-shot，但在 OpenShape × ModelNet-C 上，Local Cache 对 Global Cache 的额外提升并不明显。

---

## 10. Corruption 难度分析

### 10.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 64.94 | 平均最低，高 severity 下仍然最困难 |
| 2 | add_local | 73.51 | 局部异常点仍有影响 |
| 3 | add_global | 74.34 | 在 22_3 中相对偏低 |
| 4 | dropout_local | 75.18 | 局部缺失有一定影响 |
| 5 | scale | 78.02 | 相对稳定 |
| 6 | dropout_global | 78.93 | 整体较高，但 S4 下降明显 |
| 7 | rotate | 81.08 | 当前最高 |

分析：

完整 Point-Cache 后，jitter 仍然是最困难 corruption，但平均准确率从 22_1 的 57.96 提升到 22_3 的 64.94。说明完整 Point-Cache 对 jitter 有帮助，但无法完全消除 high-severity jitter 的困难。

rotate 的平均准确率最高，为 81.08；scale 和 dropout_global 也较高。

### 10.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 79.54 | 75.77 | 3.77 | 4.74% | 78.02 |
| add_local | 77.51 | 71.92 | 5.59 | 7.21% | 73.51 |
| add_global | 79.58 | 70.54 | 9.04 | 11.36% | 74.34 |
| rotate | 84.44 | 74.76 | 9.68 | 11.46% | 81.08 |
| dropout_local | 81.97 | 66.82 | 15.15 | 18.48% | 75.18 |
| dropout_global | 84.04 | 64.71 | 19.33 | 23.00% | 78.93 |
| jitter | 80.67 | 45.83 | 34.84 | 43.19% | 64.94 |

分析：

scale 最稳定，从 S0 到 S4 只下降 3.77。jitter 的退化仍然最强，从 80.67 下降到 45.83，绝对下降 34.84，相对下降 43.19%。

dropout_global 和 dropout_local 在 S4 也有明显退化，说明高 severity 下的点云缺失仍然会影响完整 Point-Cache 的表现。

---

## 11. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | add_local | 77.51 | rotate | 84.44 | 6.93 |
| S1 | add_local | 74.07 | rotate | 83.91 | 9.84 |
| S2 | jitter | 68.64 | rotate | 83.14 | 14.50 |
| S3 | jitter | 55.02 | dropout_global | 79.66 | 24.64 |
| S4 | jitter | 45.83 | scale | 75.77 | 29.94 |

分析：

在低 severity 下，add_local 是较难 corruption；从 S2 开始，jitter 成为明显最困难的 corruption。随着 severity 增大，best-worst gap 从 S0 的 6.93 扩大到 S4 的 29.94。

完整 Point-Cache 提升了整体水平，但没有改变 high-severity jitter 是主要失败区域这一事实。

---

## 12. 低准确率区域分析

### 12.1 低准确率 setting 数量

| 条件 | 22_1 Zero-shot 数量 | 22_2 ZS + Global 数量 | 22_3 ZS + Global + Local 数量 | 相比 22_1 减少 | 相比 22_2 变化 |
|---|---:|---:|---:|---:|---:|
| Acc < 80 | 24 / 35 | 19 / 35 | 27 / 35 | +3 | +8 |
| Acc < 75 | 16 / 35 | 11 / 35 | 15 / 35 | -1 | +4 |
| Acc < 70 | 11 / 35 | 6 / 35 | 5 / 35 | -6 | -1 |
| Acc < 65 | 7 / 35 | 2 / 35 | 3 / 35 | -4 | +1 |
| Acc < 60 | 2 / 35 | 2 / 35 | 2 / 35 | 0 | 0 |
| Acc < 50 | 2 / 35 | 1 / 35 | 1 / 35 | -1 | 0 |
| Acc < 40 | 1 / 35 | 0 / 35 | 0 / 35 | -1 | 0 |

分析：

完整 Point-Cache 相比 Zero-shot 明显减少了低于 70、65、50、40 的低准确率 setting。尤其 Acc < 40 的极低准确率 setting 从 1 个减少到 0 个。

但是，与 22_2 Global Cache 相比，22_3 在部分阈值下并没有继续减少低准确率区域。例如 Acc < 80 和 Acc < 75 的 setting 数量反而更多。这进一步说明 Local Cache 在 OpenShape × ModelNet-C 上没有稳定带来额外收益。

### 12.2 当前仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 45.83 | 最高 severity 坐标扰动仍然最难 |
| jitter_3 | 55.02 | 中高 severity 坐标扰动仍然较低 |
| dropout_global_4 | 64.71 | 高 severity 全局缺失仍然困难 |
| dropout_local_4 | 66.82 | 高 severity 局部缺失仍然困难 |
| jitter_2 | 68.64 | severity=2 jitter 仍明显低于多数 corruption |
| add_global_4 | 70.54 | 高 severity 全局异常点仍有影响 |
| dropout_local_3 | 71.39 | 中高 severity 局部缺失仍有影响 |
| add_local_3 | 71.72 | 中高 severity 局部异常点仍有影响 |

分析：

22_3 中最困难区域仍然集中在 high-severity jitter，尤其是 jitter_3 和 jitter_4。说明完整 Point-Cache 不能完全解决强坐标扰动问题。

---

## 13. 完整 Point-Cache 相比 22_1 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +0.69 | +1.06 | +1.78 | +3.25 | +2.10 | +1.78 |
| add_local | +2.43 | +3.81 | +4.74 | +7.01 | +8.51 | +5.30 |
| dropout_global | +0.73 | +1.18 | +1.62 | +1.09 | +1.14 | +1.15 |
| dropout_local | +1.50 | +1.41 | +3.32 | +3.20 | +6.00 | +3.09 |
| rotate | -0.37 | +0.04 | +0.36 | -0.49 | +1.99 | +0.31 |
| scale | -0.40 | +0.00 | -0.89 | -0.49 | -1.26 | -0.61 |
| jitter | +1.01 | +3.64 | +8.39 | +9.03 | +12.85 | +6.98 |
| **Average** | **+0.80** | **+1.59** | **+2.76** | **+3.22** | **+4.47** | **+2.57** |

分析：

完整 Point-Cache 相比 Zero-shot 的 all35 平均提升为 +2.57，severity=2 提升为 +2.76。整体上，完整 Point-Cache 对 corrupted setting 有明确正增益。

提升最大的 corruption 是 jitter，Avg Gain 为 +6.98；其次是 add_local，Avg Gain 为 +5.30。说明完整 Point-Cache 对坐标扰动和局部异常点有明显帮助。

但需要注意，scale 的平均变化为 -0.61，说明完整 Point-Cache 在 scale corruption 上不但没有提升，反而略低于 Zero-shot。rotate 的平均提升也很小，只有 +0.31。

---

## 14. Local Cache 相比 22_2 Global Cache 的额外提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | -0.32 | -0.77 | -2.64 | +0.25 | +0.08 | -0.68 |
| add_local | +0.00 | +1.22 | +0.61 | +1.01 | +2.71 | +1.11 |
| dropout_global | -0.08 | -0.65 | -0.20 | -0.49 | -0.69 | -0.42 |
| dropout_local | +0.89 | +0.48 | +0.57 | -0.04 | +1.71 | +0.72 |
| rotate | -0.73 | -0.41 | +0.20 | -1.26 | +0.49 | -0.34 |
| scale | -0.52 | -0.65 | -0.57 | -0.53 | -1.01 | -0.66 |
| jitter | -0.45 | +0.73 | +1.10 | +0.64 | -0.44 | +0.32 |
| **Average** | **-0.17** | **-0.01** | **-0.13** | **-0.06** | **+0.40** | **+0.01** |

分析：

Local Cache 在 Global Cache 基础上的 all35 平均额外提升只有 +0.01，几乎可以视为持平；severity=2 下为 -0.13，略低于 Global Cache。

Local Cache 在 15 / 35 个 setting 上为正增益，19 / 35 个 setting 上为负增益，1 个 setting 持平。负增益较大的 setting 包括：

| cor_type | 22_2 | 22_3 | Local Gain |
|---|---:|---:|---:|
| add_global_2 | 75.57 | 72.93 | -2.64 |
| rotate_3 | 80.39 | 79.13 | -1.26 |
| scale_4 | 76.78 | 75.77 | -1.01 |
| add_global_1 | 76.34 | 75.57 | -0.77 |
| rotate_0 | 85.17 | 84.44 | -0.73 |
| dropout_global_4 | 65.40 | 64.71 | -0.69 |

因此，22_3 的关键结论不是 Local Cache 明显提升，而是：在 OpenShape × ModelNet-C 上，Global Cache 已经提供了主要收益，Local Cache 额外贡献很弱，并且对部分 corruption 产生轻微负影响。

---

## 15. 与前序实验的关系

22_3 的前序子实验包括 22_1 和 22_2。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 22_1_openshape_modelnetc_corruptions_all35_zs | ModelNet-C all35 | Zero-shot | 73.57 | 72.57 |
| 22_2_openshape_modelnetc_corruptions_all35_zs_global | ModelNet-C all35 | ZS + Global Cache | 76.46 | 75.14 |
| 22_3_openshape_modelnetc_corruptions_all35_zs_global_local | ModelNet-C all35 | ZS + Global + Local Cache | 76.33 | 75.14 |

当前结果说明：在 OpenShape × ModelNet-C all35 上，完整 Point-Cache 能稳定提升 Zero-shot，但与只使用 Global Cache 相比基本持平。

| 比较 | S2 Avg 变化 | all35 Avg 变化 |
|---|---:|---:|
| 22_2 - 22_1 | +2.89 | +2.56 |
| 22_3 - 22_2 | -0.13 | +0.01 |
| 22_3 - 22_1 | +2.76 | +2.57 |

分析：

Global Cache 是主要提升来源。Local Cache 在 OpenShape × ModelNet-C 上没有带来稳定额外收益，severity=2 下略低于 Global Cache，all35 下几乎持平。

这与 14 组 ScanObjNN-C hardest 不同。14 组中 Local Cache 有更明确的额外贡献；22 组中 OpenShape 的基础表征已经较强，Global Cache 可能已经捕获了主要可利用信息，Local Cache 的边际收益不足。

---

## 16. 与 21_3 ModelNet clean 的关系

21_3 是 OpenShape 在 ModelNet clean 上的完整 Point-Cache 结果；22_3 是 OpenShape 在 ModelNet-C all35 上的完整 Point-Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy / Avg |
|---|---|---|---:|
| 21_3_openshape_modelnet_clean_zs_global_local | ModelNet clean | ZS + Global + Local | 84.00 |
| 22_3_openshape_modelnetc_corruptions_all35_zs_global_local | ModelNet-C S2 Avg | ZS + Global + Local | 76.33 |
| 22_3_openshape_modelnetc_corruptions_all35_zs_global_local | ModelNet-C all35 Avg | ZS + Global + Local | 75.14 |

对比：

| 比较 | 变化 |
|---|---:|
| 22_3 S2 Avg - 21_3 clean | -7.67 |
| 22_3 all35 Avg - 21_3 clean | -8.86 |

分析：

ModelNet-C corruption 使 OpenShape 完整 Point-Cache 相比 clean setting 下降约 8 到 9 个百分点。不过，相比 Zero-shot，完整 Point-Cache 缩小了 clean-to-corruption gap。

| 方法 | clean | ModelNet-C all35 | 下降 |
|---|---:|---:|---:|
| Zero-shot | 84.72 | 72.57 | -12.15 |
| ZS + Global | 84.48 | 75.14 | -9.34 |
| ZS + Global + Local | 84.00 | 75.14 | -8.86 |

这说明 OpenShape 在 clean 上 cache 略降，但在 corrupted setting 上完整 Point-Cache 仍然能减少 clean-to-corruption 性能差距。

---

## 17. 阶段性结论

本实验完成了 OpenShape × ModelNet-C all35 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 22_3 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 76.33，原文 Point-Cache Table 1 中 OpenShape +Hierarchical Cache Avg 为 76.59，差异为 -0.26。
3. 当前 all35 Average 为 75.14，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果整体接近原文，但 add_global 和 add_local 两项偏低较明显。
5. 相比 22_1 Zero-shot，22_3 的 severity=2 Average 提升 +2.76，all35 Average 提升 +2.57。
6. 相比 22_2 Global Cache，22_3 的 severity=2 Average 变化为 -0.13，all35 Average 变化为 +0.01。
7. Global Cache 是主要提升来源，Local Cache 额外贡献接近于零。
8. Local Cache 在 15 / 35 个 setting 上为正增益，19 / 35 个 setting 上为负增益，1 个 setting 持平。
9. Local Cache 对 add_local、dropout_local 和 jitter 有一定正向贡献，但对 add_global、dropout_global、rotate 和 scale 整体偏负。
10. jitter 仍然是最困难 corruption，完整 Point-Cache 后平均仍只有 64.94。
11. jitter_4 是最低 setting，准确率为 45.83。
12. 本实验是 22 组最后一个子实验，完整方法间总结应放入 22 组 summary 文档中。

---

## 18. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 1

---

## 19. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local/summary.csv
