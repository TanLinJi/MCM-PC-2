# 32_2_uni3d_modelnetc_corruptions_all35_zs_global

## 1. 实验目的

本实验用于复现 Uni3D 在 ModelNet-C 全部 35 个损坏设置上使用 Zero-shot + Global Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 32_2_uni3d_modelnetc_corruptions_all35_zs_global |
| Backbone | Uni3D |
| Dataset | ModelNet-C |
| Dataset 参数 | modelnet_c |
| Method | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |

本实验是在 32_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存是否能够提升 Uni3D 在 ModelNet-C corrupted setting 上的鲁棒性。

需要特别注意：原文 Point-Cache Table 1 只报告 corruption severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原文 Point-Cache Table 1 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

本文件只记录 32_2 本身，并与前序子实验 32_1 进行对比。完整 32 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 32 组 summary 文档中。

---

## 2. 当前实现方式

本实验的外部命名规则如下：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 32_2_uni3d_modelnetc_corruptions_all35_zs_global |
| 方法脚本 | Point-Cache/scripts/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/32_run_uni3d_modelnetc_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_uni3d_modelnetc_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global/ |

本实验是 all35 实验，因此使用优化 runner：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 | 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 | 模型只加载一次，每个 cor_type 重新创建 DataLoader、重新初始化 Global Cache |
| bash 通过 tee 生成单个 cor_type 的 log | Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv | summary.csv 的列结构保持不变 |

该优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、Global Cache 初始化、预测逻辑和输出结构。

---

## 3. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | Uni3D |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 原始核心 runner | runners/model_with_global_cache.py |
| 优化 runner | runners/baseline/run_uni3d_modelnetc_corruptions_all35.py |
| cache_type | global |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |
| Severity 编号 | 0, 1, 2, 3, 4 |
| 损坏组合数 | 7 × 5 = 35 |
| 输入点数 | 1024 |
| Uni3D point encoder checkpoint | weights/uni3d/modelnet40/model.pt |
| Uni3D text encoder checkpoint | weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin |
| pc_model | eva_giant_patch14_560 |
| clip_model | EVA02-E-14-plus |
| pc_feat_dim | 1408 |
| num_group | 512 |
| group_size | 64 |
| pc_encoder_dim | 512 |
| embed_dim | 1024 |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 0 |

本实验使用 `modelnet_c` 作为 dataset 参数，并指定：

| 参数 | 值 |
|---|---|
| cor_type | 由 runner 内部循环 35 个 corruption setting |
| npoints | 1024 |
| sonn_variant | obj_only |

实际读取文件形式为：

data/modelnet_c/{corruption}_{severity}.h5

---

## 4. 方法说明

32_2 在 Zero-shot logits 的基础上加入 Global Cache logits。

| 组成部分 | 是否使用 |
|---|---:|
| Text prototype / 文本原型 | 是 |
| Point cloud global feature | 是 |
| Zero-shot logits | 是 |
| Global Cache logits | 是 |
| Local Cache logits | 否 |
| Hierarchical Cache | 否 |

Global Cache 的基本作用是：在测试过程中动态缓存高置信度样本的全局点云特征和伪标签，然后对后续样本进行全局特征检索，生成 cache logits，并与 zero-shot logits 融合。

32_2 与 32_1 的主要区别如下：

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 32_1 | 是 | 否 | 否 |
| 32_2 | 是 | 是 | 否 |

因此，32_2 可以用于单独评估 Global Cache 在 Uni3D × ModelNet-C 上的影响。

---

## 5. Uni3D checkpoint 说明

本实验使用的 Uni3D point encoder checkpoint 为：

weights/uni3d/modelnet40/model.pt

这是 31 / 32 组 Uni3D × ModelNet 系列实验的正式 checkpoint。

此前使用服务器原有 checkpoint：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

会导致 31 组 ModelNet clean 结果整体偏低。切换到 `weights/uni3d/modelnet40/model.pt` 后，31 组结果与原文高度对齐。因此，32 组 ModelNet-C 继续使用 `weights/uni3d/modelnet40/model.pt`，不能使用旧的 `pc_encoder/uni3d_g_ensembled_model.pt` 作为正式复现 checkpoint。

checkpoint 下载脚本已记录在：

Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh

---

## 6. 损坏类型

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

## 7. 输出结构

输出目录：

Point-Cache/results/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

32_2_uni3d_modelnetc_corruptions_all35_zs_global_add_global_0_YYYYMMDD_HHMMSS.log

本实验曾残留旧 checkpoint 运行产生的 35 个旧 log。清理后，当前 logs 目录只保留 summary.csv 中引用的 35 个最新 log。因此当前输出状态是干净的。

---

## 8. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 35 | 35 | 说明 35 个 cor_type 全部完成 |
| summary 中唯一 cor_type 数 | 35 | 35 | 说明没有漏跑或重复写入 |
| summary 中唯一 log_path 数 | 35 | 35 | 说明每个 cor_type 都有独立日志路径 |
| logs 目录当前 .log 文件数 | 35 | 35 | 已清理旧 checkpoint 残留日志 |
| status=done 数 | 35 | 35 | 说明没有失败项 |
| severity=2 Average | 72.21 | 用于论文对齐 | 与原文 Point-Cache Table 1 对比 |
| all35 Average | 71.24 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，32_2 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ModelNet-C severity=2 参考值进行对比。

---

## 9. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 80.63 | 77.67 | 76.46 | 74.47 | 73.66 | 76.58 |
| add_local | 73.14 | 69.17 | 67.10 | 62.40 | 62.28 | 66.82 |
| dropout_global | 79.34 | 76.90 | 71.92 | 63.13 | 49.72 | 68.20 |
| dropout_local | 79.94 | 74.27 | 70.42 | 62.44 | 52.27 | 67.87 |
| rotate | 82.94 | 82.70 | 81.04 | 79.78 | 75.57 | 80.41 |
| scale | 78.73 | 77.55 | 77.23 | 75.93 | 75.49 | 76.99 |
| jitter | 75.08 | 68.27 | 61.30 | 55.35 | 48.95 | 61.79 |
| **Average** | **78.54** | **75.22** | **72.21** | **67.64** | **62.56** | **71.24** |

整体观察：

1. all35 Average 为 71.24，表示 Uni3D 在 ModelNet-C 全 35 个 corrupted setting 上加入 Global Cache 后的整体鲁棒性水平。
2. severity=2 Average 为 72.21，用于和原文 Point-Cache Table 1 对齐。
3. rotate 的平均准确率最高，为 80.41。
4. jitter 的平均准确率最低，为 61.79。
5. add_local、dropout_local 和 dropout_global 仍然较难，但相比 32_1 均有明显提升。
6. jitter_4 为 48.95，是全部 35 个 setting 中最低的结果。
7. 相比 32_1 Zero-shot，Global Cache 在几乎所有 setting 上都带来正向提升。

---

## 10. 与原文结果对比

原文 Point-Cache Table 1 报告的是 ModelNet-C 在 severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

原文 Uni3D 在 ModelNet-C severity=2 下的 +Global Cache Average 为 71.81。

当前复现 severity=2 Average 为 72.21。

| 对比对象 | 原文 S2 Avg | 当前复现 S2 Avg | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Uni3D / ModelNet-C / +Global Cache | 71.81 | 72.21 | +0.40 | 0.40 |

分析：

当前复现的 severity=2 Average 为 72.21，原文为 71.81，差异为 +0.40。差异较小，可以认为 32_2 与原文高度接近。

因此，32_2 不只是脚本执行成功，而且数值也与原文对齐。该结果可以作为 32_3 分析 Local Cache 额外贡献的直接对照。

---

## 11. Severity 维度分析

### 11.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 78.54 | — | 0.00 |
| S1 | 75.22 | -3.32 | -3.32 |
| S2 | 72.21 | -3.01 | -6.33 |
| S3 | 67.64 | -4.57 | -10.90 |
| S4 | 62.56 | -5.08 | -15.98 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 78.54 下降到 62.56，总下降 15.98 个百分点。整体上，severity 越高，Uni3D + Global Cache 的准确率越低。

相比 32_1 Zero-shot，32_2 在所有 severity 上整体上移，说明 Global Cache 并不是只改善某个特定 severity，而是在整个 severity 范围内都有效。

### 11.2 与 32_1 Zero-shot 的 severity 维度对比

| Severity | 32_1 Zero-shot Avg | 32_2 ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 76.04 | 78.54 | +2.50 |
| S1 | 71.92 | 75.22 | +3.30 |
| S2 | 67.80 | 72.21 | +4.41 |
| S3 | 62.54 | 67.64 | +5.10 |
| S4 | 56.12 | 62.56 | +6.44 |
| **all35** | **66.89** | **71.24** | **+4.35** |

分析：

Global Cache 在所有 severity 上都带来正向提升。当前 all35 平均提升为 +4.35，severity=2 提升为 +4.41。

值得注意的是，Global Cache 的提升随着 severity 增大而增强：S0 提升 +2.50，S4 提升 +6.44。这说明在更强 corruption 下，全局缓存对 Uni3D 的补偿作用更明显。

原文 severity=2 下 Global Cache 增益为：

71.81 - 67.95 = +3.86

当前复现 severity=2 下 Global Cache 增益为：

72.21 - 67.80 = +4.41

当前比原文高 +0.55，但方向和幅度都合理。

---

## 12. Corruption 难度分析

### 12.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 61.79 | 平均最低，高 severity 下仍然最困难 |
| 2 | add_local | 66.82 | 局部异常点仍然较难 |
| 3 | dropout_local | 67.87 | 高 severity 下仍然困难 |
| 4 | dropout_global | 68.20 | 高 severity 下仍然困难 |
| 5 | add_global | 76.58 | 中等偏易 |
| 6 | scale | 76.99 | 相对稳定 |
| 7 | rotate | 80.41 | 当前最高 |

分析：

加入 Global Cache 后，jitter 仍然是最困难 corruption，但平均准确率从 32_1 的 54.81 提升到 61.79，说明 Global Cache 对 jitter 有明显帮助，但不能完全解决强坐标扰动问题。

add_local 仍然是第二困难 corruption，平均为 66.82。相比 32_1 的 57.00，Global Cache 对 add_local 的提升非常明显。

### 12.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 78.73 | 75.49 | 3.24 | 4.12% | 76.99 |
| rotate | 82.94 | 75.57 | 7.37 | 8.89% | 80.41 |
| add_global | 80.63 | 73.66 | 6.97 | 8.64% | 76.58 |
| add_local | 73.14 | 62.28 | 10.86 | 14.85% | 66.82 |
| dropout_local | 79.94 | 52.27 | 27.67 | 34.61% | 67.87 |
| dropout_global | 79.34 | 49.72 | 29.62 | 37.34% | 68.20 |
| jitter | 75.08 | 48.95 | 26.13 | 34.80% | 61.79 |

分析：

scale 最稳定，从 S0 到 S4 只下降 3.24。dropout_global 的退化最强，从 79.34 下降到 49.72，绝对下降 29.62。

jitter 虽然平均最低，但 S0 到 S4 的绝对下降为 26.13，略低于 dropout_global 和 dropout_local。这说明 Global Cache 后，jitter 仍然难，但 high-severity dropout 也是非常重要的失败模式。

---

## 13. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | add_local | 73.14 | rotate | 82.94 | 9.80 |
| S1 | jitter | 68.27 | rotate | 82.70 | 14.43 |
| S2 | jitter | 61.30 | rotate | 81.04 | 19.74 |
| S3 | jitter | 55.35 | rotate | 79.78 | 24.43 |
| S4 | jitter | 48.95 | rotate | 75.57 | 26.62 |

分析：

加入 Global Cache 后，S1 到 S4 中 jitter 仍然是最低项。S0 的最低项为 add_local。best-worst gap 随 severity 增大逐渐扩大，S4 达到 26.62。

这说明 Global Cache 提高了整体准确率，但没有改变 high-severity jitter 是主要困难区域这一事实。

---

## 14. 低准确率区域分析

### 14.1 低准确率 setting 数量

| 条件 | 32_1 Zero-shot 数量 | 32_2 ZS + Global 数量 | 减少数量 |
|---|---:|---:|---:|
| Acc < 70 | 17 / 35 | 10 / 35 | -7 |
| Acc < 60 | 10 / 35 | 4 / 35 | -6 |
| Acc < 50 | 5 / 35 | 2 / 35 | -3 |
| Acc < 40 | 1 / 35 | 0 / 35 | -1 |

分析：

加入 Global Cache 后，低准确率区域明显减少。例如 Acc < 70 的 setting 从 17 个减少到 10 个，Acc < 60 的 setting 从 10 个减少到 4 个。

Global Cache 也消除了 Acc < 40 的极低准确率 setting。32_1 中 jitter_4 为 35.45，32_2 中 jitter_4 提升到 48.95。

### 14.2 当前仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 48.95 | 最高 severity 坐标扰动仍然最难 |
| dropout_global_4 | 49.72 | 高 severity 全局缺失仍然困难 |
| dropout_local_4 | 52.27 | 高 severity 局部缺失仍然困难 |
| jitter_3 | 55.35 | 中高 severity 坐标扰动仍然偏低 |
| jitter_2 | 61.30 | severity=2 jitter 仍低于多数 corruption |
| add_local_4 | 62.28 | 高 severity 局部异常点仍偏低 |
| add_local_3 | 62.40 | 中高 severity 局部异常点仍偏低 |
| dropout_global_3 | 63.13 | 中高 severity 全局缺失仍偏低 |

分析：

32_2 中最困难区域仍然集中在 high-severity jitter 和 high-severity dropout。说明 Global Cache 对这些 setting 有提升，但不能完全解决强坐标扰动和点云缺失问题。

---

## 15. Global Cache 相比 32_1 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +2.83 | +3.56 | +4.34 | +4.62 | +5.10 | +4.09 |
| add_local | +3.69 | +8.19 | +11.31 | +12.20 | +13.70 | +9.82 |
| dropout_global | +1.99 | +3.20 | +3.93 | +5.31 | +4.10 | +3.70 |
| dropout_local | +2.02 | +1.13 | +3.04 | +3.97 | +4.54 | +2.94 |
| rotate | +0.89 | +1.50 | +1.10 | +1.66 | +2.31 | +1.50 |
| scale | +1.18 | +1.25 | +1.99 | +0.85 | +1.83 | +1.42 |
| jitter | +4.90 | +4.29 | +5.14 | +7.09 | +13.50 | +6.98 |
| **Average** | **+2.50** | **+3.30** | **+4.41** | **+5.10** | **+6.44** | **+4.35** |

分析：

Global Cache 在 35 / 35 个 setting 上全部为正增益，没有负增益。总体上，Global Cache 的提升非常稳定。

平均提升最大的 corruption 是 add_local，Avg Gain 为 +9.82；其次是 jitter，Avg Gain 为 +6.98。说明 Global Cache 对局部异常点和坐标扰动有明显补偿作用。

这与 32_1 的困难区域正好对应：32_1 中 add_local 和 jitter 是最难的两个 corruption，而 Global Cache 对这两个 corruption 的提升最大。

---

## 16. 与前序实验的关系

32_2 的直接前序子实验是 32_1，即 Uni3D 在 ModelNet-C all35 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 32_1_uni3d_modelnetc_corruptions_all35_zs | ModelNet-C all35 | Zero-shot | 67.80 | 66.89 |
| 32_2_uni3d_modelnetc_corruptions_all35_zs_global | ModelNet-C all35 | ZS + Global Cache | 72.21 | 71.24 |

当前结果说明：在 ModelNet-C all35 上，Global Cache 能明显提升 Uni3D Zero-shot 的鲁棒性。

| 比较 | 变化 |
|---|---:|
| 32_2 S2 Avg - 32_1 S2 Avg | +4.41 |
| 32_2 all35 Avg - 32_1 all35 Avg | +4.35 |

分析：

32_2 相比 32_1 的提升非常明确，说明 Global Cache 在 Uni3D × ModelNet-C 上有效。尤其是 all35 提升 +4.35，说明 Global Cache 对整个 corruption/severity 范围都有稳定帮助。

---

## 17. 与 31_2 ModelNet clean 的关系

31_2 是 Uni3D 在 ModelNet clean 上的 Zero-shot + Global Cache 结果；32_2 是 Uni3D 在 ModelNet-C all35 上的 Zero-shot + Global Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy / Avg |
|---|---|---|---:|
| 31_2_uni3d_modelnet_clean_zs_global | ModelNet clean | ZS + Global Cache | 83.23 |
| 32_2_uni3d_modelnetc_corruptions_all35_zs_global | ModelNet-C S2 Avg | ZS + Global Cache | 72.21 |
| 32_2_uni3d_modelnetc_corruptions_all35_zs_global | ModelNet-C all35 Avg | ZS + Global Cache | 71.24 |

对比：

| 比较 | 变化 |
|---|---:|
| 32_2 S2 Avg - 31_2 clean | -11.02 |
| 32_2 all35 Avg - 31_2 clean | -11.99 |

分析：

在 ModelNet clean 的基础上施加 corruption 后，Uni3D + Global Cache 从 83.23 下降到 all35 Avg 71.24，下降 -11.99。

这个下降幅度小于 Zero-shot 的 clean-to-corruption 下降。32_1 相比 31_1 下降 -14.96，而 32_2 相比 31_2 下降 -11.99。说明 Global Cache 缩小了 clean 到 corrupted ModelNet-C 的性能差距。

---

## 18. 与其他 backbone 的 ModelNet-C +Global Cache 关系

32_2 可以与前面 ULIP、ULIP-2、OpenShape 的 ModelNet-C +Global Cache 结果进行横向比较。

| Backbone | 实验编号 | ModelNet-C S2 Avg | ModelNet-C all35 Avg |
|---|---|---:|---:|
| ULIP | 02_2_ulip_modelnetc_corruptions_all35_zs_global | 52.66 | 51.62 |
| ULIP-2 | 12_2_ulip2_modelnetc_corruptions_all35_zs_global | 61.22 | 61.09 |
| OpenShape | 22_2_openshape_modelnetc_corruptions_all35_zs_global | 76.46 | 75.14 |
| Uni3D | 32_2_uni3d_modelnetc_corruptions_all35_zs_global | 72.21 | 71.24 |

分析：

加入 Global Cache 后，Uni3D 仍然明显强于 ULIP 和 ULIP-2，但低于 OpenShape。OpenShape 在 ModelNet-C 上的 Global Cache 结果仍然最高。

不过，Uni3D 的 Global Cache 增益非常明显，all35 从 66.89 提升到 71.24，说明 Uni3D 在 corrupted setting 上从缓存中获得了明显收益。

---

## 19. 与后续子实验的关系

32_2 是 32_3 的直接前序实验。

| 后续实验 | 对比方式 |
|---|---|
| 32_3_uni3d_modelnetc_corruptions_all35_zs_global_local | 与 32_2 比较，评估 Local Cache 在 Global Cache 基础上的额外影响 |

本文件不展开 32_3 的实际结果。32_3 的数值及 Local Cache 额外影响应记录在 32_3 子实验文档和 32 组 summary 文档中。

需要注意的是，当前 32_2 已经证明 Global Cache 在 Uni3D × ModelNet-C 上有效，因此 32_3 的关键问题是：

| 问题 | 说明 |
|---|---|
| Local Cache 是否能在 Global Cache 基础上继续提升？ | 比较 32_3 - 32_2 |
| 完整 Point-Cache 是否接近原文 73.31？ | 比较 32_3 与原文 |
| 当前趋势是否仍是 ZS < Global < Global + Local？ | 判断整体方法趋势 |
| Local Cache 对 add_local / jitter 是否继续有效？ | 观察 32_3 的 corruption 维度变化 |

---

## 20. 结果含义分析

32_2 的结果说明：Global Cache 在 Uni3D × ModelNet-C all35 上非常有效，能够明显改善 Zero-shot 在 corrupted setting 中的鲁棒性。

| 观察 | 含义 |
|---|---|
| 32_2 all35 Avg = 71.24 | Uni3D + Global Cache 在 ModelNet-C 上的总体结果 |
| 32_2 S2 Avg = 72.21 | 与原文 Table 1 对齐的复现结果 |
| 比原文 S2 高 +0.40 | 平均值略高但对齐较好 |
| 比 32_1 all35 高 +4.35 | Global Cache 有明确正增益 |
| 35 / 35 个 setting 为正增益 | Global Cache 提升非常稳定 |
| 比 31_2 clean 低 -11.99 | corruption 仍然带来明显退化 |
| add_local 和 jitter 提升最大 | Global Cache 对困难 corruption 有明显帮助 |

因此，32_2 是一个非常关键的实验：它证明了在 Uni3D × ModelNet-C corrupted setting 中，Global Cache 是可靠且有效的鲁棒性模块。

---

## 21. 对后续 MCM-PC 的启发

当前 32_2 对后续 MCM-PC 方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| Global Cache 在 32 组中提升 +4.35 all35 | 全局缓存是重要主模块 |
| Global Cache 在 35 / 35 个 setting 上为正 | 全局缓存具有很强稳定性 |
| add_local 和 jitter 提升最大 | cache 能补偿部分局部异常点和坐标扰动问题 |
| high-severity dropout 仍然困难 | 点云缺失需要进一步机制 |
| clean 到 corruption gap 被缩小 | cache 能增强 Uni3D corruption robustness |
| 32 组必须使用 modelnet40 checkpoint | checkpoint 仍是 Uni3D 实验关键设置 |

这对 MCM-PC 很重要：全局缓存不应被削弱或替代，而应作为稳定主干模块保留。在此基础上，后续改进可以考虑如何进一步增强 high-severity jitter、dropout_global、dropout_local 和 add_local 的鲁棒性。

---

## 22. 阶段性结论

本实验完成了 Uni3D × ModelNet-C all35 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 32_2 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前使用 checkpoint 为 weights/uni3d/modelnet40/model.pt。
3. 当前 severity=2 Average 为 72.21，原文 Point-Cache Table 1 中 Uni3D +Global Cache Avg 为 71.81，差异 +0.40。
4. 当前 all35 Average 为 71.24，是本实验额外统计的 35 个 corrupted setting 总平均。
5. 当前结果与原文高度接近，可以认为 32_2 复现有效。
6. 相比 32_1 Zero-shot，32_2 的 severity=2 Average 提升 +4.41，all35 Average 提升 +4.35。
7. 当前 Global Cache 的 severity=2 增益 +4.41，与原文 +3.86 接近。
8. Global Cache 在 35 / 35 个 setting 上均为正增益，没有负增益。
9. Global Cache 对 add_local 的提升最大，Avg Gain 为 +9.82。
10. Global Cache 对 jitter 的提升也很明显，Avg Gain 为 +6.98。
11. jitter 仍然是最困难 corruption，Global Cache 后平均仍只有 61.79。
12. 本实验说明 Global Cache 在 Uni3D × ModelNet-C 上是明确有效的主模块。
13. 本实验是 32_3 分析 Local Cache 额外影响的直接对照，不在本文件中展开完整 32 组方法间对比。
14. 31 / 32 组后续应统一使用 weights/uni3d/modelnet40/model.pt。

---

## 23. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global_single_gpu.sh 1

---

## 24. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global/summary.csv
