# 22_2_openshape_modelnetc_corruptions_all35_zs_global

## 1. 实验目的

本实验用于复现 OpenShape 在 ModelNet-C 全部 35 个损坏设置上使用 Zero-shot + Global Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 22_2_openshape_modelnetc_corruptions_all35_zs_global |
| Backbone | OpenShape |
| Dataset | ModelNet-C |
| Method | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |

本实验是在 22_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存是否能够提升 OpenShape 在 ModelNet-C corrupted setting 上的鲁棒性。

需要特别注意：21 组 clean setting 中 OpenShape 使用 cache 后略有下降，但原文 Point-Cache Table 1 中 OpenShape 在 ModelNet-C severity=2 上使用 Global Cache 后是有明确正增益的。因此，22_2 是判断 OpenShape cache 鲁棒性价值的重要实验。

本文件只记录 22_2 本身，并与前序子实验 22_1 进行对比。完整 22 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 22 组 summary 文档中。

---

## 2. 当前实现方式

本实验的外部命名规则如下：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 22_2_openshape_modelnetc_corruptions_all35_zs_global |
| 方法脚本 | Point-Cache/scripts/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/22_run_openshape_modelnetc_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_openshape_modelnetc_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global/ |

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
| Backbone | OpenShape |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 原始核心 runner | runners/model_with_global_cache.py |
| 优化 runner | runners/baseline/run_openshape_modelnetc_corruptions_all35.py |
| cache_type | global |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |
| Severity 编号 | 0, 1, 2, 3, 4 |
| 损坏组合数 | 7 × 5 = 35 |
| 输入点数 | 1024 |
| Global Cache shot_capacity | 3 |
| Global Cache alpha | 4.0 |
| Global Cache beta | 3.0 |
| OpenShape version | vitg14 |
| OpenShape 权重 | weights/openshape/openshape-pointbert-vitg14-rgb/model.pt |
| OpenCLIP 权重 | weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 1 |

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

Point-Cache/results/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

22_2_openshape_modelnetc_corruptions_all35_zs_global_add_global_0_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 76.46 | 用于论文对齐 | 与原文 Point-Cache Table 1 对比 |
| all35 Average | 75.14 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，22_2 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ModelNet-C severity=2 参考值进行对比。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 79.90 | 76.34 | 75.57 | 72.85 | 70.46 | 75.02 |
| add_local | 77.51 | 72.85 | 71.72 | 70.71 | 69.21 | 72.40 |
| dropout_global | 84.12 | 84.12 | 82.98 | 80.15 | 65.40 | 79.35 |
| dropout_local | 81.08 | 78.81 | 75.85 | 71.43 | 65.11 | 74.46 |
| rotate | 85.17 | 84.32 | 82.94 | 80.39 | 74.27 | 81.42 |
| scale | 80.06 | 79.86 | 78.65 | 78.04 | 76.78 | 78.68 |
| jitter | 81.12 | 73.82 | 67.54 | 54.38 | 46.27 | 64.63 |
| **Average** | **81.28** | **78.59** | **76.46** | **72.56** | **66.79** | **75.14** |

整体观察：

1. all35 Average 为 75.14，表示 OpenShape 在 ModelNet-C 全 35 个 corrupted setting 上加入 Global Cache 后的整体鲁棒性水平。
2. severity=2 Average 为 76.46，用于和原文 Point-Cache Table 1 对齐。
3. rotate 的平均准确率最高，为 81.42。
4. jitter 的平均准确率最低，为 64.63。
5. jitter_4 为 46.27，是全部 35 个 setting 中最低的结果。
6. 相比 22_1 Zero-shot，Global Cache 在全部 35 个 setting 上均带来正向提升。

---

## 8. 与原文结果对比

原文 Point-Cache Table 1 报告的是 ModelNet-C severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 75.57 | 74.72 | +0.85 | 0.85 |
| add_local | 71.72 | 72.77 | -1.05 | 1.05 |
| dropout_global | 82.98 | 82.41 | +0.57 | 0.57 |
| dropout_local | 75.85 | 75.12 | +0.73 | 0.73 |
| rotate | 82.94 | 83.18 | -0.24 | 0.24 |
| scale | 78.65 | 78.93 | -0.28 | 0.28 |
| jitter | 67.54 | 67.91 | -0.37 | 0.37 |
| **Average** | **76.46** | **76.43** | **+0.03** | **0.58 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.03 |
| MAE | 0.58 |
| RMSE | 0.66 |
| Max Abs Diff | 1.05 |

分析：

当前复现的 severity=2 Average 为 76.46，原文为 76.43，差异仅 +0.03。虽然 add_local 单项低于原文 1.05，但整体平均几乎完全一致。

因此，22_2 不只是脚本跑通，而且数值也与原文 OpenShape 在 ModelNet-C severity=2 上的 +Global Cache 结果高度对齐。

---

## 9. Severity 维度分析

### 9.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 81.28 | — | 0.00 |
| S1 | 78.59 | -2.69 | -2.69 |
| S2 | 76.46 | -2.13 | -4.82 |
| S3 | 72.56 | -3.90 | -8.72 |
| S4 | 66.79 | -5.77 | -14.49 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 81.28 下降到 66.79，总下降 14.49 个百分点。整体上，severity 越高，OpenShape + Global Cache 的准确率越低。

与 22_1 Zero-shot 相比，22_2 在所有 severity 上整体上移，说明 Global Cache 并不是只改善某个特定 severity，而是在整个 severity 范围内都有效。

### 9.2 与 22_1 Zero-shot 的 severity 维度对比

| Severity | 22_1 Zero-shot Avg | 22_2 ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 80.31 | 81.28 | +0.97 |
| S1 | 76.99 | 78.59 | +1.60 |
| S2 | 73.57 | 76.46 | +2.89 |
| S3 | 69.28 | 72.56 | +3.28 |
| S4 | 62.72 | 66.79 | +4.07 |
| **all35** | **72.57** | **75.14** | **+2.56** |

分析：

Global Cache 在所有 severity 上都带来正向提升。当前 all35 平均提升为 +2.56，severity=2 提升为 +2.89。

更重要的是，Global Cache 的提升幅度随 severity 增大而整体增强：S0 只有 +0.97，而 S4 达到 +4.07。这说明 Global Cache 在更强 corruption 下的补偿作用更明显。

原文 severity=2 下 Global Cache 增益为：

76.43 - 73.49 = +2.94

当前复现 severity=2 下 Global Cache 增益为：

76.46 - 73.57 = +2.89

二者差异仅 -0.05。也就是说，当前 22_2 的 Global Cache 增益与原文高度一致。

---

## 10. Corruption 难度分析

### 10.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 64.63 | 平均最低，高 severity 下仍然最困难 |
| 2 | add_local | 72.40 | 局部异常点仍有明显影响 |
| 3 | dropout_local | 74.46 | 局部缺失有一定影响 |
| 4 | add_global | 75.02 | 中等难度 |
| 5 | scale | 78.68 | 较稳定 |
| 6 | dropout_global | 79.35 | 整体较高，但 S4 下降明显 |
| 7 | rotate | 81.42 | 当前最高 |

分析：

加入 Global Cache 后，jitter 仍然是最困难的 corruption，但平均准确率从 22_1 的 57.96 提升到 64.63，说明 Global Cache 对 jitter 有明显帮助。

rotate 和 dropout_global 的平均结果较高，说明 OpenShape + Global Cache 在这些 corruption 下表现较稳定。

### 10.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 80.06 | 76.78 | 3.28 | 4.10% | 78.68 |
| add_local | 77.51 | 69.21 | 8.30 | 10.71% | 72.40 |
| add_global | 79.90 | 70.46 | 9.44 | 11.81% | 75.02 |
| rotate | 85.17 | 74.27 | 10.90 | 12.80% | 81.42 |
| dropout_local | 81.08 | 65.11 | 15.97 | 19.70% | 74.46 |
| dropout_global | 84.12 | 65.40 | 18.72 | 22.25% | 79.35 |
| jitter | 81.12 | 46.27 | 34.85 | 42.96% | 64.63 |

分析：

scale 最稳定，从 S0 到 S4 只下降 3.28。jitter 的退化仍然最强，从 81.12 下降到 46.27，绝对下降 34.85，相对下降 42.96%。

dropout_global 和 dropout_local 在 S4 也有明显退化，说明高 severity 下的点云缺失仍然会削弱 OpenShape + Global Cache 的表现。

---

## 11. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | add_local | 77.51 | rotate | 85.17 | 7.66 |
| S1 | add_local | 72.85 | rotate | 84.32 | 11.47 |
| S2 | jitter | 67.54 | rotate | 82.94 | 15.40 |
| S3 | jitter | 54.38 | rotate | 80.39 | 26.01 |
| S4 | jitter | 46.27 | scale | 76.78 | 30.51 |

分析：

在低 severity 下，add_local 是较难 corruption；从 S2 开始，jitter 成为明显最困难的 corruption。随着 severity 增大，best-worst gap 从 S0 的 7.66 扩大到 S4 的 30.51。

这说明 Global Cache 提升了整体水平，但没有改变 high-severity jitter 是主要失败区域这一事实。

---

## 12. 低准确率区域分析

### 12.1 低准确率 setting 数量

| 条件 | 22_1 Zero-shot 数量 | 22_2 ZS + Global 数量 | 减少数量 |
|---|---:|---:|---:|
| Acc < 80 | 24 / 35 | 19 / 35 | -5 |
| Acc < 75 | 16 / 35 | 11 / 35 | -5 |
| Acc < 70 | 11 / 35 | 6 / 35 | -5 |
| Acc < 65 | 7 / 35 | 2 / 35 | -5 |
| Acc < 60 | 2 / 35 | 2 / 35 | 0 |
| Acc < 50 | 2 / 35 | 1 / 35 | -1 |
| Acc < 40 | 1 / 35 | 0 / 35 | -1 |

分析：

加入 Global Cache 后，低准确率区域明显减少。例如 Acc < 70 的 setting 从 11 个减少到 6 个，Acc < 65 的 setting 从 7 个减少到 2 个。

Global Cache 也消除了 Acc < 40 的极低准确率 setting。22_1 中 jitter_4 为 32.98，22_2 中 jitter_4 提升到 46.27。

### 12.2 当前仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 46.27 | 最高 severity 坐标扰动仍然最难 |
| jitter_3 | 54.38 | 中高 severity 坐标扰动仍然较低 |
| dropout_local_4 | 65.11 | 高 severity 局部缺失仍然困难 |
| dropout_global_4 | 65.40 | 高 severity 全局缺失仍然困难 |
| jitter_2 | 67.54 | severity=2 jitter 仍明显低于多数 corruption |
| add_local_4 | 69.21 | 高 severity 局部异常点仍有影响 |

分析：

22_2 中最困难区域仍然集中在 high-severity jitter，尤其是 jitter_3 和 jitter_4。说明 Global Cache 对 jitter 有明显提升，但不能完全解决强坐标扰动问题。

---

## 13. Global Cache 相比 22_1 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +1.01 | +1.83 | +4.42 | +3.00 | +2.02 | +2.45 |
| add_local | +2.43 | +2.59 | +4.13 | +6.00 | +5.80 | +4.19 |
| dropout_global | +0.81 | +1.83 | +1.82 | +1.58 | +1.83 | +1.57 |
| dropout_local | +0.61 | +0.93 | +2.75 | +3.24 | +4.29 | +2.37 |
| rotate | +0.36 | +0.45 | +0.16 | +0.77 | +1.50 | +0.65 |
| scale | +0.12 | +0.65 | -0.32 | +0.04 | -0.25 | +0.05 |
| jitter | +1.46 | +2.91 | +7.29 | +8.39 | +13.29 | +6.67 |
| **Average** | **+0.97** | **+1.60** | **+2.89** | **+3.28** | **+4.07** | **+2.56** |

分析：

Global Cache 在 33 / 35 个 setting 上为正增益，仅在 scale_2 和 scale_4 上有轻微负增益。总体上，Global Cache 的提升非常稳定。

平均提升最大的 corruption 是 jitter，Avg Gain 为 +6.67；尤其 jitter_4 从 32.98 提升到 46.27，提升 +13.29。说明 Global Cache 对 OpenShape 在强坐标扰动下的鲁棒性有明显帮助。

Global Cache 对 add_local 的提升也较大，Avg Gain 为 +4.19。相对而言，rotate 和 scale 的提升较小，因为它们在 Zero-shot 下已经较高。

---

## 14. 与前序实验的关系

22_2 的直接前序子实验是 22_1，即 OpenShape 在 ModelNet-C all35 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 22_1_openshape_modelnetc_corruptions_all35_zs | ModelNet-C all35 | Zero-shot | 73.57 | 72.57 |
| 22_2_openshape_modelnetc_corruptions_all35_zs_global | ModelNet-C all35 | ZS + Global Cache | 76.46 | 75.14 |

当前结果说明：在 ModelNet-C all35 上，Global Cache 能明显提升 OpenShape Zero-shot 的鲁棒性。

| 比较 | 变化 |
|---|---:|
| 22_2 S2 Avg - 22_1 S2 Avg | +2.89 |
| 22_2 all35 Avg - 22_1 all35 Avg | +2.56 |

分析：

22_2 相比 22_1 的提升非常明确，说明 Global Cache 在 OpenShape × ModelNet-C 上有效。这个结论与 21 组 clean setting 不同：OpenShape clean 上 Global Cache 略降，但 corrupted setting 上 Global Cache 带来明显正增益。

这说明 Point-Cache 的主要价值确实体现在 distribution shift / corruption 场景中。

---

## 15. 与 21_2 ModelNet clean 的关系

21_2 是 OpenShape 在 ModelNet clean 上的 Zero-shot + Global Cache 结果；22_2 是 OpenShape 在 ModelNet-C all35 上的 Zero-shot + Global Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy / Avg |
|---|---|---|---:|
| 21_2_openshape_modelnet_clean_zs_global | ModelNet clean | ZS + Global Cache | 84.48 |
| 22_2_openshape_modelnetc_corruptions_all35_zs_global | ModelNet-C S2 Avg | ZS + Global Cache | 76.46 |
| 22_2_openshape_modelnetc_corruptions_all35_zs_global | ModelNet-C all35 Avg | ZS + Global Cache | 75.14 |

对比：

| 比较 | 变化 |
|---|---:|
| 22_2 S2 Avg - 21_2 clean | -8.02 |
| 22_2 all35 Avg - 21_2 clean | -9.34 |

分析：

ModelNet-C corruption 使 OpenShape + Global Cache 相比 clean setting 下降约 8 到 9 个百分点。不过，下降幅度小于 Zero-shot 的 clean-to-corruption 下降。

| 方法 | clean | ModelNet-C all35 | 下降 |
|---|---:|---:|---:|
| Zero-shot | 84.72 | 72.57 | -12.15 |
| ZS + Global | 84.48 | 75.14 | -9.34 |

Global Cache 将 clean-to-corruption gap 从 -12.15 缩小到 -9.34，说明它确实提高了 OpenShape 在 corrupted setting 上的鲁棒性。

---

## 16. 阶段性结论

本实验完成了 OpenShape × ModelNet-C all35 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 22_2 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 76.46，原文 Point-Cache Table 1 中 OpenShape +Global Cache Avg 为 76.43，差异仅 +0.03。
3. 当前 all35 Average 为 75.14，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果与原文 severity=2 数值高度对齐，可以认为 22_2 复现成功。
5. 相比 22_1 Zero-shot，22_2 的 severity=2 Average 提升 +2.89，all35 Average 提升 +2.56。
6. 当前 Global Cache 的 severity=2 增益 +2.89 与原文 +2.94 高度一致。
7. Global Cache 在 33 / 35 个 setting 上为正增益，只在 scale_2 和 scale_4 上有轻微负增益。
8. Global Cache 对 jitter 的平均提升最大，为 +6.67；jitter_4 从 32.98 提升到 46.27。
9. jitter 仍然是最困难 corruption，Global Cache 后平均仍只有 64.63。
10. 本实验说明 OpenShape clean 上 cache 略降并不代表 cache 无效；在 ModelNet-C corrupted setting 上，Global Cache 有明确鲁棒性收益。
11. 本实验是 22_3 分析 Local Cache 额外影响的直接对照，不在本文件中展开完整 22 组方法间对比。

---

## 17. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global_single_gpu.sh 1

---

## 18. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global/summary.csv
