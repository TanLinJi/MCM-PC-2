# 32_3_uni3d_modelnetc_corruptions_all35_zs_global_local

## 1. 实验目的

本实验用于复现 Uni3D 在 ModelNet-C 全部 35 个损坏设置上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 32_3_uni3d_modelnetc_corruptions_all35_zs_global_local |
| Backbone | Uni3D |
| Dataset | ModelNet-C |
| Dataset 参数 | modelnet_c |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 32_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证完整 Point-Cache 对 Uni3D 在 ModelNet-C corrupted setting 上的影响。

需要特别注意：原文 Point-Cache Table 1 只报告 corruption severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原文 Point-Cache Table 1 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

本文件只记录 32_3 本身，并与前序子实验 32_1 和 32_2 进行对比。完整 32 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 32 组 summary 文档中。

---

## 2. 当前实现方式

本实验的外部命名规则如下：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 32_3_uni3d_modelnetc_corruptions_all35_zs_global_local |
| 方法脚本 | Point-Cache/scripts/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/32_run_uni3d_modelnetc_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_uni3d_modelnetc_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local/ |

本实验是 all35 实验，因此使用优化 runner：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 | 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 | 模型只加载一次，每个 cor_type 重新创建 DataLoader、重新初始化 Global Cache 和 Local Cache |
| bash 通过 tee 生成单个 cor_type 的 log | Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv | summary.csv 的列结构保持不变 |

该优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、Global Cache 初始化、Local Cache 初始化、预测逻辑和输出结构。

---

## 3. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | Uni3D |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 优化 runner | runners/baseline/run_uni3d_modelnetc_corruptions_all35.py |
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
| Uni3D point encoder checkpoint | weights/uni3d/modelnet40/model.pt |
| Uni3D text encoder checkpoint | weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin |
| pc_model | eva_giant_patch14_560 |
| clip_model | EVA02-E-14-plus |
| pc_feat_dim | 1408 |
| num_group | 512 |
| group_size | 64 |
| pc_encoder_dim | 512 |
| embed_dim | 1024 |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 1 |

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

32_3 在 Zero-shot logits 的基础上同时加入 Global Cache logits 和 Local Cache logits。

| 组成部分 | 是否使用 |
|---|---:|
| Text prototype / 文本原型 | 是 |
| Point cloud global feature | 是 |
| Zero-shot logits | 是 |
| Global Cache logits | 是 |
| Local Cache logits | 是 |
| Hierarchical Cache | 是 |

完整 Point-Cache 的预测由三部分组成：

| 组成部分 | 作用 |
|---|---|
| Zero-shot logits | 来自 Uni3D 的原始文本-点云相似度预测 |
| Global Cache logits | 基于全局点云特征的测试时缓存检索结果 |
| Local Cache logits | 基于局部 patch / 局部聚类特征的测试时缓存检索结果 |

32_3 与前两个子实验的关系如下：

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 32_1 | 是 | 否 | 否 |
| 32_2 | 是 | 是 | 否 |
| 32_3 | 是 | 是 | 是 |

因此，32_3 可以用于评估完整 Point-Cache 在 Uni3D × ModelNet-C 上的最终表现，并判断 Local Cache 是否在 Global Cache 基础上带来额外贡献。

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

Point-Cache/results/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

32_3_uni3d_modelnetc_corruptions_all35_zs_global_local_add_global_0_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 73.66 | 用于论文对齐 | 与原文 Point-Cache Table 1 对比 |
| all35 Average | 72.81 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，32_3 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ModelNet-C severity=2 参考值进行对比。

---

## 9. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 81.28 | 78.56 | 77.96 | 75.24 | 74.51 | 77.51 |
| add_local | 77.76 | 73.38 | 71.51 | 67.79 | 68.48 | 71.36 |
| dropout_global | 80.39 | 78.16 | 72.89 | 65.68 | 51.13 | 69.33 |
| dropout_local | 80.59 | 75.97 | 70.18 | 64.30 | 55.39 | 69.27 |
| rotate | 83.55 | 83.15 | 81.12 | 80.39 | 75.41 | 80.72 |
| scale | 79.34 | 78.00 | 78.44 | 76.78 | 75.37 | 77.59 |
| jitter | 75.89 | 69.98 | 62.12 | 57.90 | 52.39 | 63.86 |
| **Average** | **79.83** | **76.32** | **73.66** | **69.73** | **64.49** | **72.81** |

整体观察：

1. all35 Average 为 72.81，表示 Uni3D 在 ModelNet-C 全 35 个 corrupted setting 上使用完整 Point-Cache 后的整体鲁棒性水平。
2. severity=2 Average 为 73.66，用于和原文 Point-Cache Table 1 对齐。
3. rotate 的平均准确率最高，为 80.72。
4. jitter 的平均准确率最低，为 63.86。
5. dropout_local、dropout_global 和 add_local 仍然较难，但相比 32_1 和 32_2 均有改善。
6. dropout_global_4 为 51.13，是全部 35 个 setting 中最低的结果。
7. 相比 32_1 Zero-shot 和 32_2 Global Cache，32_3 取得最高的 S2 Average 和 all35 Average。

---

## 10. 与原文结果对比

原文 Point-Cache Table 1 报告的是 ModelNet-C 在 severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

原文 Uni3D 在 ModelNet-C severity=2 下的 +Hierarchical Cache Average 为 73.31。

当前复现 severity=2 Average 为 73.66。

| 对比对象 | 原文 S2 Avg | 当前复现 S2 Avg | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Uni3D / ModelNet-C / +Hierarchical Cache | 73.31 | 73.66 | +0.35 | 0.35 |

分析：

当前复现的 severity=2 Average 为 73.66，原文为 73.31，差异为 +0.35。差异较小，可以认为 32_3 与原文高度接近。

因此，32_3 不只是脚本执行成功，而且数值也与原文对齐。该结果可以作为 32 组完整 Point-Cache 的最终结果保留。

---

## 11. Severity 维度分析

### 11.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 79.83 | — | 0.00 |
| S1 | 76.32 | -3.51 | -3.51 |
| S2 | 73.66 | -2.66 | -6.17 |
| S3 | 69.73 | -3.93 | -10.10 |
| S4 | 64.49 | -5.24 | -15.34 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 79.83 下降到 64.49，总下降 15.34 个百分点。整体上，severity 越高，完整 Point-Cache 的准确率越低。

相比 32_1 Zero-shot，完整 Point-Cache 在所有 severity 上整体上移。相比 32_2 Global Cache，完整 Point-Cache 在所有 severity 上也都有提升，说明 Local Cache 的额外贡献在 severity 维度上比较稳定。

### 11.2 与前序实验的 severity 维度对比

| Severity | 32_1 Zero-shot Avg | 32_2 ZS + Global Avg | 32_3 ZS + Global + Local Avg | Gain over 32_1 | Gain over 32_2 |
|---:|---:|---:|---:|---:|---:|
| S0 | 76.04 | 78.54 | 79.83 | +3.79 | +1.29 |
| S1 | 71.92 | 75.22 | 76.32 | +4.40 | +1.10 |
| S2 | 67.80 | 72.21 | 73.66 | +5.86 | +1.45 |
| S3 | 62.54 | 67.64 | 69.73 | +7.19 | +2.09 |
| S4 | 56.12 | 62.56 | 64.49 | +8.37 | +1.93 |
| **all35** | **66.89** | **71.24** | **72.81** | **+5.92** | **+1.57** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有显著正增益，all35 平均提升 +5.92，severity=2 提升 +5.86。

Local Cache 在 Global Cache 基础上的额外贡献也在所有 severity 上为正：S0 +1.29，S1 +1.10，S2 +1.45，S3 +2.09，S4 +1.93。说明 32 组中 Local Cache 不是偶然在某个 severity 上提升，而是比较稳定地改善了整体表现。

---

## 12. Corruption 难度分析

### 12.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 63.86 | 平均最低，高 severity 下仍然困难 |
| 2 | dropout_local | 69.27 | 局部缺失仍然较难 |
| 3 | dropout_global | 69.33 | 高 severity 下仍然困难 |
| 4 | add_local | 71.36 | 被 Local Cache 明显改善 |
| 5 | add_global | 77.51 | 中等偏易 |
| 6 | scale | 77.59 | 相对稳定 |
| 7 | rotate | 80.72 | 当前最高 |

分析：

完整 Point-Cache 后，jitter 仍然是最困难 corruption，但平均准确率从 32_1 的 54.81 提升到 32_3 的 63.86，说明完整 Point-Cache 对 jitter 有明显帮助。

dropout_local 和 dropout_global 成为第二、第三困难 corruption，说明高 severity 点云缺失仍然是完整 Point-Cache 难以完全解决的问题。

### 12.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 79.34 | 75.37 | 3.97 | 5.00% | 77.59 |
| rotate | 83.55 | 75.41 | 8.14 | 9.74% | 80.72 |
| add_global | 81.28 | 74.51 | 6.77 | 8.33% | 77.51 |
| add_local | 77.76 | 68.48 | 9.28 | 11.93% | 71.36 |
| jitter | 75.89 | 52.39 | 23.50 | 30.97% | 63.86 |
| dropout_local | 80.59 | 55.39 | 25.20 | 31.27% | 69.27 |
| dropout_global | 80.39 | 51.13 | 29.26 | 36.40% | 69.33 |

分析：

scale 最稳定，从 S0 到 S4 只下降 3.97。dropout_global 的退化最强，从 80.39 下降到 51.13，绝对下降 29.26。

完整 Point-Cache 后，jitter 的平均准确率仍然最低，但 high-severity dropout_global 和 dropout_local 的退化更强，说明点云缺失仍然是一个重要失败模式。

---

## 13. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | jitter | 75.89 | rotate | 83.55 | 7.66 |
| S1 | jitter | 69.98 | rotate | 83.15 | 13.17 |
| S2 | jitter | 62.12 | rotate | 81.12 | 19.00 |
| S3 | jitter | 57.90 | rotate | 80.39 | 22.49 |
| S4 | dropout_global | 51.13 | rotate | 75.41 | 24.28 |

分析：

完整 Point-Cache 后，S0 到 S3 中 jitter 仍然是最低项。S4 中 dropout_global 变成最低项。best-worst gap 随 severity 增大逐渐扩大，S4 达到 24.28。

这说明完整 Point-Cache 提高了整体准确率，但没有完全消除不同 corruption 类型之间的难度差异。

---

## 14. 低准确率区域分析

### 14.1 低准确率 setting 数量

| 条件 | 32_1 Zero-shot 数量 | 32_2 ZS + Global 数量 | 32_3 ZS + Global + Local 数量 |
|---|---:|---:|---:|
| Acc < 70 | 17 / 35 | 10 / 35 | 6 / 35 |
| Acc < 60 | 10 / 35 | 4 / 35 | 3 / 35 |
| Acc < 50 | 5 / 35 | 2 / 35 | 0 / 35 |
| Acc < 40 | 1 / 35 | 0 / 35 | 0 / 35 |

分析：

完整 Point-Cache 明显减少了低准确率区域。Acc < 70 的 setting 从 Zero-shot 的 17 个减少到 Global Cache 的 10 个，再减少到完整 Point-Cache 的 6 个。Acc < 50 的 setting 从 Zero-shot 的 5 个减少到完整 Point-Cache 的 0 个。

这说明完整 Point-Cache 不仅提高平均准确率，也显著减少了低性能 setting 的数量。

### 14.2 当前仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| dropout_global_4 | 51.13 | 最高 severity 全局缺失，是当前最低 setting |
| jitter_4 | 52.39 | 最高 severity 坐标扰动仍然困难 |
| dropout_local_4 | 55.39 | 高 severity 局部缺失仍然困难 |
| jitter_3 | 57.90 | 中高 severity 坐标扰动仍然偏低 |
| jitter_2 | 62.12 | severity=2 jitter 仍低于多数 corruption |
| dropout_local_3 | 64.30 | 中高 severity 局部缺失仍偏低 |
| dropout_global_3 | 65.68 | 中高 severity 全局缺失仍偏低 |
| add_local_3 | 67.79 | 中高 severity 局部异常点仍偏低 |
| add_local_4 | 68.48 | 高 severity 局部异常点仍偏低 |

分析：

32_3 中最困难区域集中在 high-severity dropout_global、jitter 和 dropout_local。说明完整 Point-Cache 对这些 setting 有提升，但不能完全解决强坐标扰动和点云缺失问题。

---

## 15. 完整 Point-Cache 相比 32_1 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +3.48 | +4.45 | +5.84 | +5.39 | +5.95 | +5.02 |
| add_local | +8.31 | +12.40 | +15.72 | +17.59 | +19.90 | +14.36 |
| dropout_global | +3.04 | +4.46 | +4.90 | +7.86 | +5.51 | +4.83 |
| dropout_local | +2.67 | +2.83 | +2.80 | +5.83 | +7.66 | +4.34 |
| rotate | +1.50 | +1.95 | +1.18 | +2.27 | +2.15 | +1.81 |
| scale | +1.79 | +1.70 | +3.20 | +1.70 | +1.71 | +2.02 |
| jitter | +5.71 | +6.00 | +5.96 | +9.64 | +16.94 | +9.05 |
| **Average** | **+3.79** | **+4.40** | **+5.86** | **+7.19** | **+8.37** | **+5.92** |

分析：

完整 Point-Cache 相比 Zero-shot 的 all35 平均提升为 +5.92，severity=2 提升为 +5.86。所有 corruption 的平均值均有明显提升。

提升最大的 corruption 是 add_local，Avg Gain 为 +14.36；其次是 jitter，Avg Gain 为 +9.05。说明完整 Point-Cache 对局部异常点和坐标扰动有明显帮助。

---

## 16. Local Cache 相比 32_2 Global Cache 的额外提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +0.65 | +0.89 | +1.50 | +0.77 | +0.85 | +0.93 |
| add_local | +4.62 | +4.21 | +4.42 | +5.39 | +6.20 | +4.55 |
| dropout_global | +1.05 | +1.26 | +0.97 | +2.55 | +1.41 | +1.13 |
| dropout_local | +0.65 | +1.70 | -0.24 | +1.86 | +3.12 | +1.40 |
| rotate | +0.61 | +0.45 | +0.08 | +0.61 | -0.16 | +0.31 |
| scale | +0.61 | +0.45 | +1.21 | +0.85 | -0.12 | +0.60 |
| jitter | +0.81 | +1.70 | +0.81 | +2.55 | +3.44 | +2.07 |
| **Average** | **+1.29** | **+1.10** | **+1.45** | **+2.09** | **+1.93** | **+1.57** |

分析：

Local Cache 在 Global Cache 基础上的 all35 平均额外提升为 +1.57，severity=2 额外提升 +1.45。相比 22 组 OpenShape × ModelNet-C 的 Local extra 约为 0，32 组中的 Local Cache 明显更有效。

Local Cache 对 add_local 的额外帮助最大，Avg Gain 为 +4.55；对 jitter 的额外提升也明显，为 +2.07。说明局部缓存对局部异常点和坐标扰动有比较直接的补偿作用。

按四舍五入后的 summary 结果看，Local Cache 在 32 / 35 个 setting 上为正增益，在 3 / 35 个 setting 上为负增益。负增益只出现在 dropout_local_2、rotate_4 和 scale_4，幅度都很小。

---

## 17. 与前序实验的关系

32_3 的前序子实验包括 32_1 和 32_2。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 32_1_uni3d_modelnetc_corruptions_all35_zs | ModelNet-C all35 | Zero-shot | 67.80 | 66.89 |
| 32_2_uni3d_modelnetc_corruptions_all35_zs_global | ModelNet-C all35 | ZS + Global Cache | 72.21 | 71.24 |
| 32_3_uni3d_modelnetc_corruptions_all35_zs_global_local | ModelNet-C all35 | ZS + Global + Local | 73.66 | 72.81 |

当前结果说明：在 Uni3D × ModelNet-C all35 上，Global Cache 和 Local Cache 都有明确正贡献。

| 比较 | S2 Avg 变化 | all35 Avg 变化 |
|---|---:|---:|
| 32_2 - 32_1 | +4.41 | +4.35 |
| 32_3 - 32_2 | +1.45 | +1.57 |
| 32_3 - 32_1 | +5.86 | +5.92 |

分析：

Global Cache 是主要提升来源，但 Local Cache 也不是弱贡献模块。完整 Point-Cache 相比 Zero-shot 提升 +5.92 all35，其中 Global Cache 提供 +4.35，Local Cache 额外提供 +1.57。

这与 22 组 OpenShape × ModelNet-C 不同。OpenShape 在 ModelNet-C 上 Local Cache 额外贡献很弱；Uni3D 在 ModelNet-C 上 Local Cache 明确有效。

---

## 18. 与 31_3 ModelNet clean 的关系

31_3 是 Uni3D 在 ModelNet clean 上的完整 Point-Cache 结果；32_3 是 Uni3D 在 ModelNet-C all35 上的完整 Point-Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy / Avg |
|---|---|---|---:|
| 31_3_uni3d_modelnet_clean_zs_global_local | ModelNet clean | ZS + Global + Local | 83.71 |
| 32_3_uni3d_modelnetc_corruptions_all35_zs_global_local | ModelNet-C S2 Avg | ZS + Global + Local | 73.66 |
| 32_3_uni3d_modelnetc_corruptions_all35_zs_global_local | ModelNet-C all35 Avg | ZS + Global + Local | 72.81 |

对比：

| 比较 | 变化 |
|---|---:|
| 32_3 S2 Avg - 31_3 clean | -10.05 |
| 32_3 all35 Avg - 31_3 clean | -10.90 |

分析：

在 ModelNet clean 的基础上施加 corruption 后，Uni3D 完整 Point-Cache 从 83.71 下降到 all35 Avg 72.81，下降 -10.90。

相比 Zero-shot，完整 Point-Cache 缩小了 clean-to-corruption gap：

| 方法 | ModelNet clean | ModelNet-C all35 | 下降 |
|---|---:|---:|---:|
| Zero-shot | 81.85 | 66.89 | -14.96 |
| ZS + Global | 83.23 | 71.24 | -11.99 |
| ZS + Global + Local | 83.71 | 72.81 | -10.90 |

这说明完整 Point-Cache 不仅提高了 corrupted accuracy，也增强了 Uni3D 从 clean 到 corrupted setting 的鲁棒性保持能力。

---

## 19. 与其他 backbone 的 ModelNet-C 完整 Point-Cache 关系

32_3 可以与前面 ULIP、ULIP-2、OpenShape 的 ModelNet-C 完整 Point-Cache 结果进行横向比较。

| Backbone | 实验编号 | ModelNet-C S2 Avg | ModelNet-C all35 Avg |
|---|---|---:|---:|
| ULIP | 02_3_ulip_modelnetc_corruptions_all35_zs_global_local | 54.00 | 53.01 |
| ULIP-2 | 12_3_ulip2_modelnetc_corruptions_all35_zs_global_local | 62.74 | 62.49 |
| OpenShape | 22_3_openshape_modelnetc_corruptions_all35_zs_global_local | 76.33 | 75.14 |
| Uni3D | 32_3_uni3d_modelnetc_corruptions_all35_zs_global_local | 73.66 | 72.81 |

分析：

完整 Point-Cache 下，Uni3D 在 ModelNet-C 上明显强于 ULIP 和 ULIP-2，但低于 OpenShape。OpenShape 的 all35 Avg 为 75.14，Uni3D 为 72.81，差距为 -2.33。

不过，Uni3D 的 cache 增益结构比 OpenShape 更明显：OpenShape 在 ModelNet-C 上 Local Cache 额外贡献几乎为 0，而 Uni3D 的 Local Cache all35 额外提升为 +1.57。

---

## 20. 结果含义分析

32_3 的结果说明：完整 Point-Cache 在 Uni3D × ModelNet-C all35 上有效，并且最终数值与原文高度接近。

| 观察 | 含义 |
|---|---|
| 32_3 all35 Avg = 72.81 | 当前完整 Point-Cache 总体结果 |
| 32_3 S2 Avg = 73.66 | 与原文 Table 1 对齐的复现结果 |
| 原文 S2 Avg = 73.31 | 当前高 +0.35，差异较小 |
| 相比 32_1 all35 提升 +5.92 | 完整 Point-Cache 有明确正增益 |
| 相比 32_2 all35 提升 +1.57 | Local Cache 有明确额外贡献 |
| add_local 提升最大 | 局部异常点明显受益于 cache |
| jitter 仍然最低 | 坐标扰动仍然是主要困难 corruption |
| high-severity dropout 仍然困难 | 点云缺失仍需后续方法处理 |

因此，32_3 是 32 组三个子实验中最关键的最终结果。它证明了完整 Point-Cache 在 Uni3D × ModelNet-C corrupted setting 中有效，也说明 Local Cache 在该场景中具有明确价值。

---

## 21. 对后续 MCM-PC 的启发

当前 32_3 对后续 MCM-PC 方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| 完整 Point-Cache 提升 +5.92 all35 | cache 机制在 Uni3D corrupted setting 中非常重要 |
| Global Cache 提供 +4.35 all35 | 全局缓存是稳定主模块，应保留 |
| Local Cache 提供 +1.57 all35 | 局部缓存也有明确价值 |
| add_local 受益最大 | 局部异常点需要局部证据建模 |
| jitter 仍然最低 | 坐标扰动需要专门鲁棒机制 |
| high-severity dropout 仍然困难 | 点云缺失需要额外处理 |
| 32 组必须使用 modelnet40 checkpoint | checkpoint 是 Uni3D 实验关键设置 |

这对 MCM-PC 很重要：后续方法不应简单固定 Global / Local 的作用，而应根据样本可靠性、全局-局部一致性、伪标签可信度和域偏移类型动态调节缓存贡献。

32 组显示，在 Uni3D × ModelNet-C 上，Global Cache 和 Local Cache 都有价值。后续 MCM-PC 可以围绕如何保留全局缓存稳定增益、如何选择可靠局部证据、如何抑制 high-severity jitter / dropout 下的错误伪标签展开。

---

## 22. 阶段性结论

本实验完成了 Uni3D × ModelNet-C all35 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 32_3 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前使用 checkpoint 为 weights/uni3d/modelnet40/model.pt。
3. 当前 severity=2 Average 为 73.66，原文 Point-Cache Table 1 中 Uni3D +Hierarchical Cache Avg 为 73.31，差异 +0.35。
4. 当前 all35 Average 为 72.81，是本实验额外统计的 35 个 corrupted setting 总平均。
5. 当前结果与原文高度接近，可以认为 32_3 复现有效。
6. 相比 32_1 Zero-shot，32_3 的 severity=2 Average 提升 +5.86，all35 Average 提升 +5.92。
7. 相比 32_2 Global Cache，32_3 的 severity=2 Average 额外提升 +1.45，all35 Average 额外提升 +1.57。
8. Global Cache 是主要提升来源，Local Cache 也有明确额外贡献。
9. Local Cache 在 32 / 35 个 setting 上为正增益，在 3 / 35 个 setting 上为负增益，负增益幅度很小。
10. add_local 是完整 Point-Cache 提升最大的 corruption，平均提升 +14.36。
11. jitter 仍然是最困难 corruption，完整 Point-Cache 后平均仍只有 63.86。
12. dropout_global_4 是全部 35 个 setting 中最低结果，只有 51.13。
13. 完整 Point-Cache 下，Uni3D 比 ULIP 和 ULIP-2 明显更强，但低于 OpenShape。
14. 32_3 结果有效，不需要重跑。
15. 该实验可作为 32 组 summary 文档和后续 MCM-PC 方法改进的关键 baseline。
16. 31 / 32 组后续应统一使用 weights/uni3d/modelnet40/model.pt。

---

## 23. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 1

---

## 24. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local/summary.csv
