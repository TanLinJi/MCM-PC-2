# 24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global

## 1. 实验目的

本实验用于复现 OpenShape 在 ScanObjNN-C hardest 全部 35 个损坏设置上使用 Zero-shot + Global Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global |
| Backbone | OpenShape |
| Dataset | ScanObjNN-C hardest |
| Dataset 参数 | sonn_c |
| SONN variant | hardest |
| Method | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |

本实验是在 24_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存是否能够提升 OpenShape 在 ScanObjNN-C hardest corrupted setting 上的鲁棒性。

需要特别注意：24 组是 OpenShape 在真实扫描 hardest split 上进一步叠加 corruption 的实验组，是比 22 组 ModelNet-C 更困难的数据设置。当前结果显示，Global Cache 在 24 组中有明确正增益，是完整 Point-Cache 的主要提升来源之一。

本文件只记录 24_2 本身，并与前序子实验 24_1 进行对比。完整 24 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 24 组 summary 文档中。

---

## 2. 当前实现方式

本实验的外部命名规则如下：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global |
| 方法脚本 | Point-Cache/scripts/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/24_run_openshape_scanobjnnc_hardest_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global/ |

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
| 数据集参数 | sonn_c |
| 数据目录 | data/sonn_c |
| 数据集变体 | hardest |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 原始核心 runner | runners/model_with_global_cache.py |
| 优化 runner | runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py |
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

本实验使用 `sonn_c` 作为 dataset 参数，并指定：

| 参数 | 值 |
|---|---|
| sonn_variant | hardest |
| cor_type | 由 runner 内部循环 35 个 corruption setting |

实际读取文件形式为：

data/sonn_c/hardest/{corruption}_{severity}.h5

---

## 4. 方法说明

24_2 在 Zero-shot logits 的基础上加入 Global Cache logits。

| 组成部分 | 是否使用 |
|---|---:|
| Text prototype / 文本原型 | 是 |
| Point cloud global feature | 是 |
| Zero-shot logits | 是 |
| Global Cache logits | 是 |
| Local Cache logits | 否 |
| Hierarchical Cache | 否 |

Global Cache 的基本作用是：在测试过程中动态缓存高置信度样本的全局点云特征和伪标签，然后对后续样本进行全局特征检索，生成 cache logits，并与 zero-shot logits 融合。

24_2 与 24_1 的主要区别如下：

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 24_1 | 是 | 否 | 否 |
| 24_2 | 是 | 是 | 否 |

因此，24_2 可以用于单独评估 Global Cache 在 OpenShape × ScanObjNN-C hardest 上的影响。

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

Point-Cache/results/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_add_global_0_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 37.30 | 用于论文对齐 | 与原文 Point-Cache Table 7 对比 |
| all35 Average | 36.71 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，24_2 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ScanObjNN-C hardest severity=2 参考值进行对比。

---

## 8. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 43.89 | 41.71 | 39.17 | 38.83 | 40.28 | 40.78 |
| add_local | 42.05 | 38.83 | 38.79 | 35.25 | 34.52 | 37.89 |
| dropout_global | 44.90 | 44.73 | 43.72 | 37.58 | 27.13 | 39.61 |
| dropout_local | 41.15 | 37.61 | 33.52 | 28.04 | 22.62 | 32.59 |
| rotate | 45.04 | 42.09 | 43.06 | 38.06 | 35.18 | 40.69 |
| scale | 40.49 | 38.79 | 39.83 | 37.16 | 37.61 | 38.78 |
| jitter | 42.19 | 32.34 | 23.04 | 18.53 | 17.18 | 26.66 |
| **Average** | **42.82** | **39.44** | **37.30** | **33.35** | **30.65** | **36.71** |

整体观察：

1. all35 Average 为 36.71，表示 OpenShape 在 ScanObjNN-C hardest 全 35 个 corrupted setting 上加入 Global Cache 后的整体鲁棒性水平。
2. severity=2 Average 为 37.30，用于和原文 Point-Cache Table 7 对齐。
3. add_global 的平均准确率最高，为 40.78，rotate 平均为 40.69，二者非常接近。
4. jitter 的平均准确率最低，为 26.66。
5. jitter_4 为 17.18，是全部 35 个 setting 中最低的结果。
6. 相比 24_1 Zero-shot，Global Cache 在绝大多数 setting 上带来明显正向提升。

---

## 9. 与原文结果对比

原文 Point-Cache Table 7 报告的是 ScanObjNN-C hardest / S-PB T50-RS-C 在 severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 39.17 | 40.32 | -1.15 | 1.15 |
| add_local | 38.79 | 37.58 | +1.21 | 1.21 |
| dropout_global | 43.72 | 42.02 | +1.70 | 1.70 |
| dropout_local | 33.52 | 33.76 | -0.24 | 0.24 |
| rotate | 43.06 | 41.53 | +1.53 | 1.53 |
| scale | 39.83 | 38.24 | +1.59 | 1.59 |
| jitter | 23.04 | 24.12 | -1.08 | 1.08 |
| **Average** | **37.30** | **36.80** | **+0.50** | **1.21 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.50 |
| MAE | 1.21 |
| RMSE | 1.26 |
| Max Abs Diff | 1.70 |

分析：

当前复现的 severity=2 Average 为 37.30，原文为 36.80，差异为 +0.50。整体略高于原文，但对齐较好。

逐 corruption 看，dropout_global、rotate 和 scale 高于原文较多；add_global 和 jitter 低于原文。说明当前平均值略高并不是所有 corruption 均匀偏高，而是存在一定正负波动。

因此，24_2 应记录为：结果有效、平均值略高于原文、逐 corruption 有一定波动，但整体对齐较好。

---

## 10. Severity 维度分析

### 10.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 42.82 | — | 0.00 |
| S1 | 39.44 | -3.38 | -3.38 |
| S2 | 37.30 | -2.14 | -5.52 |
| S3 | 33.35 | -3.95 | -9.47 |
| S4 | 30.65 | -2.70 | -12.17 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 42.82 下降到 30.65，总下降 12.17 个百分点。整体上，severity 越高，OpenShape + Global Cache 的准确率越低。

相比 24_1 Zero-shot，24_2 在所有 severity 上整体上移，说明 Global Cache 并不是只改善某个特定 severity，而是在整个 severity 范围内都有效。

### 10.2 与 24_1 Zero-shot 的 severity 维度对比

| Severity | 24_1 Zero-shot Avg | 24_2 ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 38.82 | 42.82 | +4.00 |
| S1 | 35.49 | 39.44 | +3.95 |
| S2 | 32.75 | 37.30 | +4.55 |
| S3 | 30.12 | 33.35 | +3.23 |
| S4 | 26.43 | 30.65 | +4.21 |
| **all35** | **32.72** | **36.71** | **+3.99** |

分析：

Global Cache 在所有 severity 上都带来正向提升。当前 all35 平均提升为 +3.99，severity=2 提升为 +4.55。

与 22 组 OpenShape × ModelNet-C 相比，24 组 Global Cache 的 all35 增益更大。22 组 Global Cache all35 提升为 +2.56，而 24 组为 +3.99。这说明在更困难的真实扫描 corrupted setting 中，Global Cache 的补偿作用更明显。

原文 severity=2 下 Global Cache 增益为：

36.80 - 31.98 = +4.82

当前复现 severity=2 下 Global Cache 增益为：

37.30 - 32.75 = +4.55

二者差异为 -0.27。也就是说，当前 24_2 的 Global Cache 增益与原文较接近。

---

## 11. Corruption 难度分析

### 11.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 26.66 | 平均最低，高 severity 下仍然最困难 |
| 2 | dropout_local | 32.59 | 局部缺失仍然很困难 |
| 3 | add_local | 37.89 | 局部异常点有一定影响 |
| 4 | scale | 38.78 | 相对较稳定 |
| 5 | dropout_global | 39.61 | 平均较高，但 S4 明显下降 |
| 6 | rotate | 40.69 | 整体较高 |
| 7 | add_global | 40.78 | 当前最高 |

分析：

加入 Global Cache 后，jitter 仍然是最困难 corruption，但平均准确率从 24_1 的 22.76 提升到 26.66，说明 Global Cache 对 jitter 有帮助，但无法完全解决强坐标扰动问题。

dropout_local 仍然是第二困难 corruption，平均为 32.59。说明局部缺失在真实扫描 corrupted hardest 上仍然非常困难。

### 11.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 40.49 | 37.61 | 2.88 | 7.11% | 38.78 |
| add_global | 43.89 | 40.28 | 3.61 | 8.23% | 40.78 |
| add_local | 42.05 | 34.52 | 7.53 | 17.91% | 37.89 |
| rotate | 45.04 | 35.18 | 9.86 | 21.89% | 40.69 |
| dropout_global | 44.90 | 27.13 | 17.77 | 39.58% | 39.61 |
| dropout_local | 41.15 | 22.62 | 18.53 | 45.03% | 32.59 |
| jitter | 42.19 | 17.18 | 25.01 | 59.28% | 26.66 |

分析：

scale 最稳定，从 S0 到 S4 只下降 2.88。jitter 的退化最强，从 42.19 下降到 17.18，绝对下降 25.01，相对下降 59.28%。

dropout_global 和 dropout_local 在 S4 也有明显退化，说明高 severity 点云缺失仍然会削弱 OpenShape + Global Cache 的表现。

---

## 12. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | scale | 40.49 | rotate | 45.04 | 4.55 |
| S1 | jitter | 32.34 | dropout_global | 44.73 | 12.39 |
| S2 | jitter | 23.04 | dropout_global | 43.72 | 20.68 |
| S3 | jitter | 18.53 | add_global | 38.83 | 20.30 |
| S4 | jitter | 17.18 | add_global | 40.28 | 23.10 |

分析：

在 S0 时，scale 是最低项，但不同 corruption 之间差距不大。从 S1 开始，jitter 成为明显最困难 corruption。S2、S3、S4 中 jitter 都是最低结果。

best-worst gap 在 S4 达到 23.10，说明高 severity 下不同 corruption 的难度差异非常明显。

---

## 13. 低准确率区域分析

### 13.1 低准确率 setting 数量

| 条件 | 24_1 Zero-shot 数量 | 24_2 ZS + Global 数量 | 减少数量 |
|---|---:|---:|---:|
| Acc < 40 | 30 / 35 | 22 / 35 | -8 |
| Acc < 35 | 19 / 35 | 9 / 35 | -10 |
| Acc < 30 | 8 / 35 | 6 / 35 | -2 |
| Acc < 25 | 6 / 35 | 4 / 35 | -2 |
| Acc < 20 | 4 / 35 | 2 / 35 | -2 |
| Acc < 15 | 1 / 35 | 0 / 35 | -1 |

分析：

加入 Global Cache 后，低准确率区域明显减少。例如 Acc < 35 的 setting 从 19 个减少到 9 个，Acc < 40 的 setting 从 30 个减少到 22 个。

Global Cache 也消除了 Acc < 15 的极低准确率 setting。24_1 中 jitter_4 为 13.43，24_2 中 jitter_4 提升到 17.18。

### 13.2 当前仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 17.18 | 最高 severity 坐标扰动仍然最难 |
| jitter_3 | 18.53 | 中高 severity 坐标扰动仍然极低 |
| dropout_local_4 | 22.62 | 高 severity 局部缺失仍然困难 |
| jitter_2 | 23.04 | severity=2 jitter 仍明显低于多数 corruption |
| dropout_global_4 | 27.13 | 高 severity 全局缺失仍然困难 |
| dropout_local_3 | 28.04 | 中高 severity 局部缺失仍然困难 |
| jitter_1 | 32.34 | 低中 severity jitter 仍偏低 |
| dropout_local_2 | 33.52 | severity=2 局部缺失仍偏低 |

分析：

24_2 中最困难区域仍然集中在 high-severity jitter，尤其是 jitter_3 和 jitter_4。说明 Global Cache 对 jitter 有提升，但不能完全解决强坐标扰动问题。

dropout_local_4、dropout_local_3 和 dropout_global_4 也仍然困难，说明高 severity 缺失仍需要后续方法进一步处理。

---

## 14. Global Cache 相比 24_1 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +4.30 | +5.52 | +3.85 | +5.42 | +7.35 | +5.29 |
| add_local | +3.64 | +4.62 | +5.83 | +3.26 | +3.81 | +4.23 |
| dropout_global | +3.78 | +3.79 | +3.09 | -0.07 | +3.05 | +2.73 |
| dropout_local | +4.75 | +6.21 | +6.35 | +4.83 | +3.15 | +5.06 |
| rotate | +3.85 | +1.98 | +5.55 | +3.40 | +4.09 | +3.77 |
| scale | +2.95 | +1.04 | +3.92 | +2.50 | +4.30 | +2.94 |
| jitter | +4.72 | +4.48 | +3.26 | +3.26 | +3.75 | +3.89 |
| **Average** | **+4.00** | **+3.95** | **+4.55** | **+3.23** | **+4.21** | **+3.99** |

分析：

Global Cache 在 34 / 35 个 setting 上为正增益，仅在 dropout_global_3 上有极轻微负增益 -0.07。总体上，Global Cache 的提升非常稳定。

平均提升最大的 corruption 是 add_global，Avg Gain 为 +5.29；其次是 dropout_local，Avg Gain 为 +5.06；add_local 也有 +4.23。说明 Global Cache 对异常点、局部缺失和坐标扰动都有较强补偿作用。

尤其在 ScanObjNN-C hardest 这种真实扫描 corrupted setting 中，Global Cache 明显比在 clean hardest 上更有效。23_2 相比 23_1 只提升 +0.07，而 24_2 相比 24_1 的 all35 提升达到 +3.99。

---

## 15. 与前序实验的关系

24_2 的直接前序子实验是 24_1，即 OpenShape 在 ScanObjNN-C hardest all35 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs | ScanObjNN-C hardest all35 | Zero-shot | 32.75 | 32.72 |
| 24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global | ScanObjNN-C hardest all35 | ZS + Global Cache | 37.30 | 36.71 |

当前结果说明：在 ScanObjNN-C hardest all35 上，Global Cache 能明显提升 OpenShape Zero-shot 的鲁棒性。

| 比较 | 变化 |
|---|---:|
| 24_2 S2 Avg - 24_1 S2 Avg | +4.55 |
| 24_2 all35 Avg - 24_1 all35 Avg | +3.99 |

分析：

24_2 相比 24_1 的提升非常明确，说明 Global Cache 在 OpenShape × ScanObjNN-C hardest 上有效。这个结论与 23 组 clean hardest 不同：OpenShape clean hardest 上 Global Cache 单独提升很弱，但 corrupted hardest setting 上 Global Cache 带来明显正增益。

这说明当真实扫描 hardest split 进一步叠加 corruption 后，全局缓存的补偿作用被明显放大。

---

## 16. 与 23_2 ScanObjNN clean hardest 的关系

23_2 是 OpenShape 在 ScanObjNN clean hardest 上的 Zero-shot + Global Cache 结果；24_2 是 OpenShape 在 ScanObjNN-C hardest all35 上的 Zero-shot + Global Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy / Avg |
|---|---|---|---:|
| 23_2_openshape_scanobjnn_clean_hardest_zs_global | ScanObjNN clean hardest | ZS + Global Cache | 41.95 |
| 24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global | ScanObjNN-C hardest S2 Avg | ZS + Global Cache | 37.30 |
| 24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global | ScanObjNN-C hardest all35 Avg | ZS + Global Cache | 36.71 |

对比：

| 比较 | 变化 |
|---|---:|
| 24_2 S2 Avg - 23_2 clean hardest | -4.65 |
| 24_2 all35 Avg - 23_2 clean hardest | -5.24 |

分析：

在 ScanObjNN clean hardest 的基础上进一步施加 corruption 后，OpenShape + Global Cache 从 41.95 下降到 all35 Avg 36.71，下降 -5.24。

这个下降幅度小于 Zero-shot 的 clean-to-corruption 下降。24_1 相比 23_1 下降 -9.16，而 24_2 相比 23_2 下降 -5.24。说明 Global Cache 缩小了 clean hardest 到 corrupted hardest 的性能差距。

---

## 17. 与 22_2 ModelNet-C 的关系

22_2 是 OpenShape 在 ModelNet-C all35 上的 Zero-shot + Global Cache 结果；24_2 是 OpenShape 在 ScanObjNN-C hardest all35 上的 Zero-shot + Global Cache 结果。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 22_2_openshape_modelnetc_corruptions_all35_zs_global | ModelNet-C all35 | ZS + Global Cache | 76.46 | 75.14 |
| 24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global | ScanObjNN-C hardest all35 | ZS + Global Cache | 37.30 | 36.71 |

对比：

| 比较 | S2 变化 | all35 变化 |
|---|---:|---:|
| 24_2 - 22_2 | -39.16 | -38.43 |

分析：

ScanObjNN-C hardest 仍然远难于 ModelNet-C。即使使用 Global Cache，OpenShape 在 ModelNet-C all35 上的平均为 75.14，但在 ScanObjNN-C hardest all35 上只有 36.71。

这说明 synthetic corruption 与真实扫描 corrupted hardest 的难度差距非常大。Global Cache 能带来显著提升，但不能完全消除真实扫描域偏移。

---

## 18. 与 ULIP / ULIP-2 的 ScanObjNN-C hardest 关系

24_2 可以与前面 ULIP、ULIP-2 的 ScanObjNN-C hardest +Global Cache 结果进行横向比较。

| Backbone | 实验编号 | ScanObjNN-C hardest S2 Avg | ScanObjNN-C hardest all35 Avg |
|---|---|---:|---:|
| ULIP | 04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global | 26.84 | 26.60 |
| ULIP-2 | 14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global | 31.38 | 31.24 |
| OpenShape | 24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global | 37.30 | 36.71 |

Backbone 提升：

| 比较 | S2 Avg 提升 | all35 Avg 提升 |
|---|---:|---:|
| OpenShape - ULIP | +10.46 | +10.11 |
| OpenShape - ULIP-2 | +5.92 | +5.47 |

分析：

加入 Global Cache 后，OpenShape 仍然明显强于 ULIP 和 ULIP-2。相比 ULIP，OpenShape all35 Avg 高 +10.11；相比 ULIP-2，高 +5.47。

这说明 OpenShape backbone 在真实扫描 corrupted hardest 上依然具有明显优势。但 OpenShape + Global 的绝对准确率仍然只有 36.71，说明该设置仍然非常困难。

---

## 19. 与后续子实验的关系

24_2 是 24_3 的直接前序实验。

| 后续实验 | 对比方式 |
|---|---|
| 24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local | 与 24_2 比较，评估 Local Cache 在 Global Cache 基础上的额外影响 |

本文件不展开 24_3 的实际结果。24_3 的数值及 Local Cache 额外影响应记录在 24_3 子实验文档和 24 组 summary 文档中。

需要注意的是，当前 24_2 已经证明 Global Cache 在 ScanObjNN-C hardest 上有效，因此 24_3 的关键问题是：

| 问题 | 说明 |
|---|---|
| Local Cache 是否能在 Global Cache 基础上继续提升？ | 比较 24_3 - 24_2 |
| 完整 Point-Cache 是否接近原文 37.70？ | 比较 24_3 与原文 |
| 当前趋势是否仍是 ZS < Global < Global + Local？ | 判断整体方法趋势 |
| Local Cache 是否延续 23 组真实扫描数据上的正贡献？ | 观察 24_3 的额外增益 |

---

## 20. 结果含义分析

24_2 的结果说明：Global Cache 在 OpenShape × ScanObjNN-C hardest all35 上非常有效，能够明显改善 Zero-shot 在真实扫描 corrupted setting 中的鲁棒性。

| 观察 | 含义 |
|---|---|
| 24_2 all35 Avg = 36.71 | OpenShape + Global Cache 在 ScanObjNN-C hardest 上的总体结果 |
| 24_2 S2 Avg = 37.30 | 与原文 Table 7 对齐的复现结果 |
| 比原文 S2 高 +0.50 | 平均值略高但对齐较好 |
| 比 24_1 all35 高 +3.99 | Global Cache 有明确正增益 |
| 34 / 35 个 setting 为正增益 | Global Cache 提升稳定 |
| 比 23_2 clean hardest 低 -5.24 | corruption 仍然带来明显退化 |
| 比 22_2 ModelNet-C all35 低 -38.43 | 真实扫描 corrupted hardest 远难于 ModelNet-C |

因此，24_2 是一个非常关键的实验：它证明了在最困难的真实扫描 corrupted setting 中，Global Cache 仍然是可靠且有效的鲁棒性模块。

---

## 21. 对后续 MCM-PC 的启发

当前 24_2 对后续 MCM-PC 方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| Global Cache 在 24 组中提升 +3.99 all35 | 全局缓存仍然是重要主模块 |
| Global Cache 在 34 / 35 个 setting 上为正 | 全局缓存具有较强稳定性 |
| jitter 仍然最难 | 坐标扰动需要额外机制 |
| dropout_local 仍然很困难 | 局部缺失需要局部证据补偿 |
| clean hardest 中 Global 增益弱，但 corrupted hardest 中 Global 增益强 | cache 作用与域偏移强度有关 |
| 真实扫描 corrupted hardest 远难于 ModelNet-C | 后续方法必须在 24 组这类 setting 上展示收益 |

这对 MCM-PC 很重要：全局缓存不应被削弱或替代，而应作为稳定主干模块保留。在此基础上，后续改进可以考虑如何进一步增强 high-severity jitter、dropout_local 和 dropout_global 的鲁棒性。

---

## 22. 阶段性结论

本实验完成了 OpenShape × ScanObjNN-C hardest all35 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 24_2 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 37.30，原文 Point-Cache Table 7 中 OpenShape +Global Cache Avg 为 36.80，差异 +0.50。
3. 当前 all35 Average 为 36.71，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果略高于原文，但整体对齐较好，可以认为 24_2 复现有效。
5. 相比 24_1 Zero-shot，24_2 的 severity=2 Average 提升 +4.55，all35 Average 提升 +3.99。
6. 当前 Global Cache 的 severity=2 增益 +4.55 与原文 +4.82 较接近。
7. Global Cache 在 34 / 35 个 setting 上为正增益，仅 dropout_global_3 轻微下降 -0.07。
8. Global Cache 对 add_global、dropout_local、add_local 和 jitter 都有明显帮助。
9. jitter 仍然是最困难 corruption，Global Cache 后平均仍只有 26.66。
10. jitter_4 是全部 35 个 setting 中最低结果，只有 17.18。
11. 本实验说明 Global Cache 在 ScanObjNN-C hardest 上是明确有效的主模块。
12. 本实验是 24_3 分析 Local Cache 额外影响的直接对照，不在本文件中展开完整 24 组方法间对比。

---

## 23. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh 1

---

## 24. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv
