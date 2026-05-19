# 02_2_ulip_modelnetc_corruptions_all35_zs_global

## 1. 实验目的

复现 ULIP 在 ModelNet-C 全部 35 个损坏设置上使用 Zero-shot + Global Cache 后的结果。

本实验属于 baseline 复现阶段的 02 组实验。02 组实验固定使用 ULIP backbone，并在 ModelNet-C 的 35 个 corrupted setting 上评估一种方法。

本实验是在 02_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存是否能够利用测试流中的高置信度样本，改善 ULIP 在 corrupted point clouds 上的鲁棒性。

具体而言，本实验回答四个问题：

| 问题 | 说明 |
|---|---|
| Global Cache 是否提升 Zero-shot？ | 与 02_1 的 Zero-shot 结果逐项比较 |
| Global Cache 对哪些 corruption 最有效？ | 观察 corruption × severity 的增益分布 |
| Global Cache 是否缓解高 severity 退化？ | 比较 S0-S4 各 severity 下的平均提升 |
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
| 实验编号 | 02_2_ulip_modelnetc_corruptions_all35_zs_global |
| 方法脚本 | Point-Cache/scripts/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/02_run_ulip_modelnetc_corruptions_all35_common.sh |
| 新增 Python runner | Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global/ |

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

本实验虽然改用了优化 runner，但实验定义没有改变。优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、Global Cache 初始化、预测逻辑和输出结构。

---

## 3. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 原始核心 runner | runners/model_with_global_cache.py |
| 当前优化 runner | runners/baseline/run_ulip_modelnetc_corruptions_all35.py |
| cache_type | global |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |
| Severity 编号 | 0, 1, 2, 3, 4 |
| 损坏组合数 | 7 × 5 = 35 |
| 输入点数 | 1024 |
| Global Cache shot_capacity | 3 |
| Global Cache alpha | 4.0 |
| Global Cache beta | 3.0 |
| 权重 | weights/ulip/pointbert_ulip1.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| GPU | 单张 Tesla T4 |

本实验只使用 Global Cache，不使用 Local Cache。因此，本实验主要检验“全局点云特征缓存”对 corrupted point cloud recognition 的贡献。

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

Point-Cache/results/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则保持不变：

02_2_ulip_modelnetc_corruptions_all35_zs_global_add_global_0_YYYYMMDD_HHMMSS.log

也就是说，新版 runner 虽然只启动一次 Python，但仍然会为 35 个 cor_type 生成 35 个独立 log。

---

## 6. 当前结果检查

本实验最初上传的 summary 出现过重复记录：每个 cor_type 在 GPU 0 和 GPU 1 上各出现一次，共 70 行。两套结果的准确率完全一致，因此这不是算法失败，而是重复运行或重复记录导致的结果冗余。

目前已经清理完成，保留一套标准结果，并删除多余日志。当前结果目录处于标准状态。

| 检查项 | 清理前 | 清理后 | 期望值 | 说明 |
|---|---:|---:|---:|---|
| summary.csv 行数 | 70 | 35 | 35 | 清理前每个 cor_type 重复 2 次 |
| summary 中唯一 cor_type 数 | 35 | 35 | 35 | 实际 35 个设置均已完成 |
| summary 中唯一 log_path 数 | 70 | 35 | 35 | 清理后每个 cor_type 对应 1 个 log |
| logs 目录 .log 文件数 | 70 | 35 | 35 | 多余 log 已删除 |
| status=done 数 | 70 | 35 | 35 | 清理后保留一套 done 结果 |
| severity=2 Average | 52.66 | 52.66 | 用于论文对齐 | 清理不影响准确率 |
| all35 Average | 51.62 | 51.62 | 本实验扩展统计 | 清理不影响准确率 |

结论：02_2 实验结果完整，重复项已经清理，summary.csv、log_path 和 logs 文件数量完全一致。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 54.25 | 51.78 | 46.07 | 43.76 | 38.41 | 46.85 |
| add_local | 54.70 | 50.77 | 47.24 | 46.39 | 43.52 | 48.52 |
| dropout_global | 58.95 | 58.18 | 57.05 | 55.19 | 49.59 | 55.79 |
| dropout_local | 59.85 | 57.82 | 54.86 | 51.30 | 44.25 | 53.62 |
| rotate | 61.14 | 60.78 | 59.81 | 55.51 | 48.58 | 57.16 |
| scale | 56.48 | 56.40 | 53.97 | 53.00 | 52.96 | 54.56 |
| jitter | 55.55 | 54.82 | 49.64 | 36.75 | 27.51 | 44.85 |
| **Average** | **57.27** | **55.79** | **52.66** | **48.84** | **43.55** | **51.62** |

整体观察：

1. all35 Average 为 51.62，相比 02_1 Zero-shot 的 46.85 提升 4.77。
2. severity=2 Average 为 52.66，相比 02_1 的 47.68 提升 4.98。
3. Global Cache 在所有 35 个 setting 上都带来了正向提升，没有出现负增益。
4. add_global 的平均提升最大，说明 Global Cache 对全局异常点 corruption 有非常明显的修正作用。
5. jitter 仍然是高 severity 下最困难的 corruption，S4 只有 27.51，说明 Global Cache 对强坐标扰动的缓解仍然有限。

---

## 8. Severity 维度分析

### 8.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 57.27 | — | 0.00 |
| S1 | 55.79 | -1.48 | -1.48 |
| S2 | 52.66 | -3.13 | -4.61 |
| S3 | 48.84 | -3.82 | -8.43 |
| S4 | 43.55 | -5.30 | -13.73 |

分析：

随着 severity 增大，ULIP + Global Cache 的平均准确率仍然单调下降。这说明 Global Cache 能提升整体准确率，但不能完全消除 corruption severity 增强带来的退化趋势。

与 02_1 Zero-shot 相比，Global Cache 后的 S0-S4 平均准确率整体上移。尤其在 S4 高 severity 场景下，Average 从 38.39 提升到 43.55，说明 Global Cache 对强 corruption 场景仍有帮助。

### 8.2 与 Zero-shot 的 severity 维度对比

| Severity | Zero-shot Avg | ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 53.40 | 57.27 | +3.87 |
| S1 | 50.94 | 55.79 | +4.86 |
| S2 | 47.68 | 52.66 | +4.98 |
| S3 | 43.85 | 48.84 | +5.00 |
| S4 | 38.39 | 43.55 | +5.16 |
| **All35** | **46.85** | **51.62** | **+4.77** |

分析：

Global Cache 在所有 severity 上都带来稳定提升，且提升幅度从 S0 的 +3.87 增加到 S4 的 +5.16。也就是说，corruption 越严重，Global Cache 的平均增益越明显。

这说明 Global Cache 的主要价值不只是提升轻度损坏数据上的准确率，更重要的是在高强度 corruption 下提供测试流中的全局结构参考，从而缓解 Zero-shot 的退化。

---

## 9. Corruption 难度分析

### 9.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 44.85 | 高 severity 下仍然严重退化，S4 只有 27.51 |
| 2 | add_global | 46.85 | 相比 Zero-shot 大幅提升，但仍属于困难 corruption |
| 3 | add_local | 48.52 | 局部异常点仍有明显影响 |
| 4 | dropout_local | 53.62 | 中等难度，Global Cache 后整体较稳定 |
| 5 | scale | 54.56 | 对尺度扰动较稳定 |
| 6 | dropout_global | 55.79 | 表现较好，说明全局结构缺失对 Global Cache 影响较小 |
| 7 | rotate | 57.16 | 当前方法下平均最高，最容易 |

分析：

加入 Global Cache 后，Zero-shot 下最困难的 add_global 从平均 34.89 提升到 46.85，难度明显下降。但 jitter 仍然是最困难 corruption，尤其是 S4 只有 27.51。

这说明 Global Cache 对“全局异常点”特别有效，但对“强坐标扰动”仍然有限。可能原因是 jitter 会直接破坏点云局部与全局几何特征，使 query feature 本身偏离正确类别区域，即使有全局缓存，也难以完全恢复。

### 9.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 56.48 | 52.96 | 3.52 | 6.23% | 54.56 |
| dropout_global | 58.95 | 49.59 | 9.36 | 15.88% | 55.79 |
| add_local | 54.70 | 43.52 | 11.18 | 20.44% | 48.52 |
| rotate | 61.14 | 48.58 | 12.56 | 20.54% | 57.16 |
| dropout_local | 59.85 | 44.25 | 15.60 | 26.07% | 53.62 |
| add_global | 54.25 | 38.41 | 15.84 | 29.20% | 46.85 |
| jitter | 55.55 | 27.51 | 28.04 | 50.48% | 44.85 |

分析：

scale 依然最稳定，从 S0 到 S4 只下降 3.52，说明 ULIP + Global Cache 对尺度扰动具有较强鲁棒性。

jitter 仍然退化最严重，从 S0 的 55.55 下降到 S4 的 27.51，绝对下降 28.04，相对下降 50.48%。虽然相比 Zero-shot 的 S0-S4 下降 31.15 有一定缓解，但强 jitter 仍然是 Global Cache 难以解决的问题。

add_global 的平均表现虽然大幅提升，但 S0 到 S4 仍下降 15.84，说明当全局异常点强度增加时，Global Cache 的校正能力也会逐渐受限。

---

## 10. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | add_global | 54.25 | rotate | 61.14 | 6.89 |
| S1 | add_local | 50.77 | rotate | 60.78 | 10.01 |
| S2 | add_global | 46.07 | rotate | 59.81 | 13.74 |
| S3 | jitter | 36.75 | rotate | 55.51 | 18.76 |
| S4 | jitter | 27.51 | scale | 52.96 | 25.45 |

分析：

随着 severity 增大，不同 corruption 之间的 best-worst gap 从 S0 的 6.89 扩大到 S4 的 25.45。Global Cache 虽然整体提升性能，但高 severity 下不同 corruption 的难度差异仍然被放大。

Zero-shot 中 add_global 在 S0 到 S3 基本都是最困难项；加入 Global Cache 后，S3 和 S4 的最困难项变成 jitter。这说明 Global Cache 显著缓解了 add_global，但没有同等程度地解决高强度 jitter。

---

## 11. 低准确率区域分析

### 11.1 低准确率 setting 数量

| 条件 | Zero-shot 数量 | ZS + Global 数量 | 减少数量 | 主要涉及 corruption |
|---|---:|---:|---:|---|
| Acc < 50 | 17 / 35 | 12 / 35 | -5 | add_global, add_local, jitter |
| Acc < 45 | 12 / 35 | 6 / 35 | -6 | add_global, add_local, jitter, dropout_local |
| Acc < 40 | 7 / 35 | 3 / 35 | -4 | add_global, jitter |
| Acc < 35 | 5 / 35 | 1 / 35 | -4 | jitter |
| Acc < 30 | 3 / 35 | 1 / 35 | -2 | jitter |
| Acc < 25 | 1 / 35 | 0 / 35 | -1 | 无 |

分析：

加入 Global Cache 后，低准确率区域明显减少。Zero-shot 中低于 40 的 setting 有 7 个，而 ZS + Global 中减少到 3 个；低于 35 的 setting 从 5 个减少到 1 个。

这说明 Global Cache 不只是提升平均值，也确实减少了严重失败的场景。尤其是 add_global 的多个低准确率 setting 被明显拉升。

### 11.2 仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 27.51 | 当前最低，强坐标扰动仍然最难 |
| jitter_3 | 36.75 | 高 severity jitter 仍然低于 40 |
| add_global_4 | 38.41 | 高 severity 全局异常点仍然困难 |

分析：

Global Cache 之后，最严重的失败区域集中在 jitter_3、jitter_4 和 add_global_4。后续 02_3 的 Local Cache 是否能进一步缓解这些场景，是重点观察对象。

---

## 12. Global Cache 相比 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +8.87 | +13.21 | +12.07 | +13.78 | +11.91 | +11.97 |
| add_local | +3.65 | +3.69 | +3.32 | +4.70 | +4.42 | +3.96 |
| dropout_global | +2.83 | +2.35 | +2.35 | +2.64 | +4.17 | +2.87 |
| dropout_local | +3.08 | +3.04 | +4.29 | +3.41 | +3.16 | +3.40 |
| rotate | +4.74 | +4.58 | +4.62 | +4.17 | +4.25 | +4.47 |
| scale | +3.40 | +3.89 | +3.08 | +2.88 | +4.54 | +3.56 |
| jitter | +0.53 | +3.24 | +5.15 | +3.40 | +3.64 | +3.19 |
| **Average** | **+3.87** | **+4.86** | **+4.98** | **+5.00** | **+5.16** | **+4.77** |

分析：

Global Cache 对全部 35 个 setting 都带来了正向提升，最小提升为 jitter_0 的 +0.53，最大提升为 add_global_3 的 +13.78。

平均提升最大的 corruption 是 add_global，Avg Gain 达到 +11.97。这说明 Global Cache 对全局异常点 corruption 的修正能力非常强。Zero-shot 下 add_global 的平均准确率只有 34.89，而加入 Global Cache 后提升到 46.85。

平均提升较小的是 dropout_global、jitter 和 dropout_local。其中 dropout_global 本身 Zero-shot 表现较高，因此提升空间较小；jitter 虽然 Zero-shot 表现较差，但强坐标扰动会破坏 query feature 本身，因此 Global Cache 的检索和加权能力也受到限制。

---

## 13. 与原论文 severity=2 结果对比

原论文 Table 1 报告的是 severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原论文 S2 | Diff | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 46.07 | 45.79 | +0.28 | 0.28 |
| add_local | 47.24 | 47.98 | -0.74 | 0.74 |
| dropout_global | 57.05 | 56.85 | +0.20 | 0.20 |
| dropout_local | 54.86 | 53.89 | +0.97 | 0.97 |
| rotate | 59.81 | 60.25 | -0.44 | 0.44 |
| scale | 53.97 | 54.34 | -0.37 | 0.37 |
| jitter | 49.64 | 48.91 | +0.73 | 0.73 |
| **Average** | **52.66** | **52.56** | **+0.10** | **0.53 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.09 |
| MAE | 0.53 |
| RMSE | 0.59 |
| Max Abs Diff | 0.97 |

分析：

当前复现与原论文的 severity=2 结果高度一致。平均差异约 +0.10，MAE 为 0.53，最大单项差异为 dropout_local 的 +0.97，均处于可接受范围。

这说明当前 Global Cache 的配置、数据读取、模型权重、文本原型构建和推理流程整体与原始 Point-Cache baseline 基本对齐。

---

## 14. 与 02_1 Zero-shot 的核心对比

| 指标 | 02_1 Zero-shot | 02_2 ZS + Global | Gain |
|---|---:|---:|---:|
| severity=2 Average | 47.68 | 52.66 | +4.98 |
| all35 Average | 46.85 | 51.62 | +4.77 |
| S0 Average | 53.40 | 57.27 | +3.87 |
| S1 Average | 50.94 | 55.79 | +4.86 |
| S2 Average | 47.68 | 52.66 | +4.98 |
| S3 Average | 43.85 | 48.84 | +5.00 |
| S4 Average | 38.39 | 43.55 | +5.16 |
| Acc < 40 的 setting 数 | 7 | 3 | -4 |
| Acc < 35 的 setting 数 | 5 | 1 | -4 |

分析：

Global Cache 对 Zero-shot 的提升非常稳定：不仅提升了 all35 平均值，也提升了每一个 severity 的平均结果。

更重要的是，Global Cache 对高 severity 的提升略大于低 severity。S0 提升 +3.87，而 S4 提升 +5.16。这说明 Global Cache 对更强的 corruption 有更明显的补偿作用。

从低准确率区域看，Acc < 40 的 setting 从 7 个减少到 3 个，说明 Global Cache 显著减少了严重失败案例。

---

## 15. 对后续 02_3 的意义

本实验给出了 ULIP 在 ModelNet-C all35 下使用 Global Cache 后的结果：

| 指标 | 数值 |
|---|---:|
| severity=2 Average | 52.66 |
| all35 Average | 51.62 |

后续 02_3 将在此基础上进一步加入 Local Cache。需要重点观察：

| 比较 | 目的 |
|---|---|
| 02_3 - 02_2 | 评估 Local Cache 在 Global Cache 基础上的额外增益 |
| 02_3 - 02_1 | 评估完整 Point-Cache 相比 Zero-shot 的总体增益 |
| 02_3 在 jitter 上的表现 | 判断 Local Cache 是否能缓解 Global Cache 对 jitter 的不足 |
| 02_3 在 add_global 上的表现 | 判断 Local Cache 是否能继续提升已经被 Global Cache 大幅修正的 corruption |
| 02_3 在 S4 上的表现 | 判断 Local Cache 是否能进一步缓解高 severity 退化 |

特别需要关注的现象：

1. Global Cache 对 add_global 的提升非常大，Avg Gain 为 +11.97。
2. Global Cache 对 jitter 的平均提升只有 +3.19，且 jitter_4 仍然只有 27.51。
3. 如果 Local Cache 有效，02_3 应该在 jitter、dropout_local 和 add_local 等更依赖局部结构的 corruption 上带来额外增益。
4. 如果 Local Cache 主要补充局部细节，那么 02_3 相比 02_2 的提升不一定均匀，而可能集中在局部结构相关 corruption 上。

---

## 16. 阶段性结论

本实验完成了 ULIP × ModelNet-C 全 35 corrupted setting 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 实验完整性正常：清理后 summary.csv 有 35 行，cor_type 唯一数为 35，log_path 唯一数为 35，logs 文件数为 35。
2. severity=2 Average 为 52.66，与原论文 52.56 基本一致，说明复现结果可靠。
3. all35 Average 为 51.62，相比 02_1 Zero-shot 的 46.85 提升 +4.77。
4. Global Cache 在所有 35 个 setting 上都带来正向提升，没有出现负增益。
5. Global Cache 对 add_global 的提升最明显，Avg Gain 为 +11.97，说明全局缓存对全局异常点有很强的修正作用。
6. Global Cache 对 jitter 的修正有限，jitter_4 仍然是当前最低结果，说明强坐标扰动仍是难点。
7. Global Cache 减少了低准确率区域，Acc < 40 的 setting 从 Zero-shot 的 7 个减少到 3 个。
8. Global Cache 对高 severity 的平均提升略大于低 severity，说明它对更强 corruption 有一定补偿作用。
9. 本实验可作为 02_3 Global + Local Cache 的直接对照，用于评估 Local Cache 的额外贡献。

---

## 17. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global_single_gpu.sh 1

---

## 18. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global/summary.csv | cut -d',' -f13 | sort | uniq -c
