# 32_1_uni3d_modelnetc_corruptions_all35_zs

## 1. 实验目的

本实验用于复现 Uni3D 在 ModelNet-C 全部 35 个损坏设置上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 32_1_uni3d_modelnetc_corruptions_all35_zs |
| Backbone | Uni3D |
| Dataset | ModelNet-C |
| Dataset 参数 | modelnet_c |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 Uni3D 在 ModelNet-C corrupted setting 上的无缓存基础鲁棒性。该结果后续会作为 32_2 和 32_3 的对照基线，但本文件只记录 32_1 本身，不展开完整 32 组的方法间对比。

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
| 实验编号 | 32_1_uni3d_modelnetc_corruptions_all35_zs |
| 方法脚本 | Point-Cache/scripts/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/32_run_uni3d_modelnetc_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_uni3d_modelnetc_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs/ |

本实验是 all35 实验，因此使用优化 runner：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 | 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 | 模型只加载一次，每个 cor_type 重新创建 DataLoader |
| bash 通过 tee 生成单个 cor_type 的 log | Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv | summary.csv 的列结构保持不变 |

该优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、预测逻辑和输出结构。

---

## 3. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | Uni3D |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 优化 runner | runners/baseline/run_uni3d_modelnetc_corruptions_all35.py |
| 是否使用 Global Cache | 否 |
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

## 4. Uni3D checkpoint 说明

本实验使用的 Uni3D point encoder checkpoint 为：

weights/uni3d/modelnet40/model.pt

这是 31 / 32 组 Uni3D × ModelNet 系列实验的正式 checkpoint。

此前使用服务器原有 checkpoint：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

会导致 31 组 ModelNet clean 结果整体偏低。切换到 `weights/uni3d/modelnet40/model.pt` 后，31 组结果与原文高度对齐。因此，32 组 ModelNet-C 也必须继续使用 `weights/uni3d/modelnet40/model.pt`，不能再使用旧的 `pc_encoder/uni3d_g_ensembled_model.pt` 作为正式复现 checkpoint。

checkpoint 下载脚本已记录在：

Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh

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

Point-Cache/results/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

32_1_uni3d_modelnetc_corruptions_all35_zs_add_global_0_YYYYMMDD_HHMMSS.log

本实验曾残留旧 checkpoint 运行产生的 35 个旧 log。清理后，当前 logs 目录只保留 summary.csv 中引用的 35 个最新 log。因此当前输出状态是干净的。

---

## 7. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 35 | 35 | 说明 35 个 cor_type 全部完成 |
| summary 中唯一 cor_type 数 | 35 | 35 | 说明没有漏跑或重复写入 |
| summary 中唯一 log_path 数 | 35 | 35 | 说明每个 cor_type 都有独立日志路径 |
| logs 目录当前 .log 文件数 | 35 | 35 | 已清理旧 checkpoint 残留日志 |
| status=done 数 | 35 | 35 | 说明没有失败项 |
| severity=2 Average | 67.80 | 用于论文对齐 | 与原文 Point-Cache Table 1 对比 |
| all35 Average | 66.89 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，32_1 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ModelNet-C severity=2 参考值进行对比。

---

## 8. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 77.80 | 74.11 | 72.12 | 69.85 | 68.56 | 72.49 |
| add_local | 69.45 | 60.98 | 55.79 | 50.20 | 48.58 | 57.00 |
| dropout_global | 77.35 | 73.70 | 67.99 | 57.82 | 45.62 | 64.50 |
| dropout_local | 77.92 | 73.14 | 67.38 | 58.47 | 47.73 | 64.93 |
| rotate | 82.05 | 81.20 | 79.94 | 78.12 | 73.26 | 78.91 |
| scale | 77.55 | 76.30 | 75.24 | 75.08 | 73.66 | 75.57 |
| jitter | 70.18 | 63.98 | 56.16 | 48.26 | 35.45 | 54.81 |
| **Average** | **76.04** | **71.92** | **67.80** | **62.54** | **56.12** | **66.89** |

整体观察：

1. all35 Average 为 66.89，表示 Uni3D 在 ModelNet-C 全 35 个 corrupted setting 上的 Zero-shot 鲁棒性水平。
2. severity=2 Average 为 67.80，用于和原文 Point-Cache Table 1 对齐。
3. rotate 的平均准确率最高，为 78.91。
4. jitter 的平均准确率最低，为 54.81。
5. add_local 的平均准确率也较低，为 57.00。
6. jitter_4 为 35.45，是全部 35 个 setting 中最低的结果。
7. high-severity dropout_global、dropout_local、jitter 和 add_local 是主要困难区域。

---

## 9. 与原文结果对比

原文 Point-Cache Table 1 报告的是 ModelNet-C 在 severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

原文 Uni3D 在 ModelNet-C severity=2 下的 Zero-shot Average 为 67.95。

当前复现 severity=2 Average 为 67.80。

| 对比对象 | 原文 S2 Avg | 当前复现 S2 Avg | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Uni3D / ModelNet-C / Zero-shot | 67.95 | 67.80 | -0.15 | 0.15 |

分析：

当前复现的 severity=2 Average 为 67.80，原文为 67.95，差异为 -0.15。差异非常小，可以认为 32_1 与原文高度对齐。

因此，32_1 不只是脚本执行成功，而且数值也与原文对齐。该结果可以作为 32_2 和 32_3 的有效 Zero-shot baseline。

---

## 10. Severity 维度分析

### 10.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 76.04 | — | 0.00 |
| S1 | 71.92 | -4.12 | -4.12 |
| S2 | 67.80 | -4.12 | -8.24 |
| S3 | 62.54 | -5.26 | -13.50 |
| S4 | 56.12 | -6.42 | -19.92 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 76.04 下降到 56.12，总下降 19.92 个百分点。整体上，severity 越高，Uni3D Zero-shot 准确率越低。

S3 到 S4 的下降最大，为 -6.42，说明最高 severity 会进一步明显破坏 Uni3D 的识别能力。

### 10.2 与 31_1 ModelNet clean 的关系

| 设置 | Accuracy |
|---|---:|
| 31_1 ModelNet clean Zero-shot | 81.85 |
| 32_1 ModelNet-C S2 Average | 67.80 |
| 32_1 ModelNet-C all35 Average | 66.89 |

对比：

| 比较 | 变化 |
|---|---:|
| S2 Average - clean | 67.80 - 81.85 = -14.05 |
| all35 Average - clean | 66.89 - 81.85 = -14.96 |

分析：

在 ModelNet clean 的基础上施加 corruption 后，Uni3D Zero-shot 从 81.85 下降到 all35 Avg 66.89，下降 -14.96。

这说明虽然 Uni3D 在 clean setting 上较强，但 ModelNet-C corruption 仍然会显著削弱其 zero-shot 鲁棒性。32_1 是后续 32_2 Global Cache 和 32_3 Hierarchical Cache 的必要基础对照。

---

## 11. Corruption 难度分析

### 11.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 54.81 | 平均最低，高 severity 下下降最明显 |
| 2 | add_local | 57.00 | 局部异常点对 Uni3D 影响较大 |
| 3 | dropout_global | 64.50 | 高 severity 下下降明显 |
| 4 | dropout_local | 64.93 | 高 severity 下下降明显 |
| 5 | add_global | 72.49 | 中等难度 |
| 6 | scale | 75.57 | 相对稳定 |
| 7 | rotate | 78.91 | 当前最高 |

分析：

jitter 是最困难的 corruption，平均准确率只有 54.81，尤其 jitter_4 只有 35.45。add_local 是第二困难 corruption，平均为 57.00。说明局部异常点和坐标扰动是 Uni3D 在 ModelNet-C Zero-shot 下的主要困难来源。

rotate 和 scale 相对容易，平均准确率分别为 78.91 和 75.57。

### 11.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 77.55 | 73.66 | 3.89 | 5.02% | 75.57 |
| rotate | 82.05 | 73.26 | 8.79 | 10.71% | 78.91 |
| add_global | 77.80 | 68.56 | 9.24 | 11.88% | 72.49 |
| add_local | 69.45 | 48.58 | 20.87 | 30.05% | 57.00 |
| dropout_local | 77.92 | 47.73 | 30.19 | 38.74% | 64.93 |
| dropout_global | 77.35 | 45.62 | 31.73 | 41.02% | 64.50 |
| jitter | 70.18 | 35.45 | 34.73 | 49.49% | 54.81 |

分析：

scale 最稳定，从 S0 到 S4 只下降 3.89。jitter 的退化最强，从 70.18 下降到 35.45，绝对下降 34.73，相对下降 49.49%。

dropout_global 和 dropout_local 在 S4 也有明显退化，说明高 severity 点云缺失是 Uni3D 在 ModelNet-C 上的重要困难来源。

---

## 12. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | add_local | 69.45 | rotate | 82.05 | 12.60 |
| S1 | add_local | 60.98 | rotate | 81.20 | 20.22 |
| S2 | add_local | 55.79 | rotate | 79.94 | 24.15 |
| S3 | jitter | 48.26 | rotate | 78.12 | 29.86 |
| S4 | jitter | 35.45 | scale | 73.66 | 38.21 |

分析：

S0 到 S2 中，add_local 是最低项；S3 和 S4 中，jitter 成为最低项。随着 severity 增大，不同 corruption 的难度差异明显扩大，S4 的 best-worst gap 达到 38.21。

这说明高 severity 下 Uni3D 的失败不是均匀发生，而是集中在 jitter、dropout 和 add_local 等特定扰动类型上。

---

## 13. 低准确率区域分析

| 条件 | 数量 | 占比 | 主要涉及 corruption |
|---|---:|---:|---|
| Acc < 70 | 17 / 35 | 48.57% | jitter, add_local, dropout_global, dropout_local, add_global |
| Acc < 60 | 10 / 35 | 28.57% | jitter, add_local, dropout_global, dropout_local |
| Acc < 50 | 5 / 35 | 14.29% | jitter_3/4, add_local_3/4, dropout_global_4, dropout_local_4 |
| Acc < 40 | 1 / 35 | 2.86% | jitter_4 |

分析：

32_1 的低准确率区域主要集中在 high-severity jitter、add_local 和 dropout。jitter_4 是唯一低于 40 的 setting，说明最高 severity 坐标扰动是当前 Zero-shot 下最严重失败点。

---

## 14. 关键困难 setting

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 35.45 | 最高 severity 坐标扰动，最困难 setting |
| dropout_global_4 | 45.62 | 高 severity 全局缺失 |
| dropout_local_4 | 47.73 | 高 severity 局部缺失 |
| jitter_3 | 48.26 | 中高 severity 坐标扰动 |
| add_local_4 | 48.58 | 高 severity 局部异常点 |
| add_local_3 | 50.20 | 中高 severity 局部异常点 |
| add_local_2 | 55.79 | severity=2 局部异常点 |
| jitter_2 | 56.16 | severity=2 jitter |

分析：

jitter_4 是当前最困难 setting，准确率只有 35.45。高 severity dropout_global / dropout_local 也明显困难。add_local 在 S2 到 S4 中持续偏低，说明局部异常点对 Uni3D 的 zero-shot 预测影响较大。

这些困难 setting 是后续 32_2 和 32_3 判断 cache 是否有效的重要观察对象。

---

## 15. 与前序实验的关系

32_1 的直接前序数据设置是 31_1，即 Uni3D 在 ModelNet clean 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | Accuracy / Avg |
|---|---|---|---:|
| 31_1_uni3d_modelnet_clean_zs | ModelNet clean | Zero-shot | 81.85 |
| 32_1_uni3d_modelnetc_corruptions_all35_zs | ModelNet-C S2 Avg | Zero-shot | 67.80 |
| 32_1_uni3d_modelnetc_corruptions_all35_zs | ModelNet-C all35 Avg | Zero-shot | 66.89 |

当前结果说明：从 clean 到 corrupted ModelNet-C，Uni3D Zero-shot 明显下降。

| 比较 | 变化 |
|---|---:|
| 32_1 S2 Avg - 31_1 clean | -14.05 |
| 32_1 all35 Avg - 31_1 clean | -14.96 |

分析：

31_1 已经说明 Uni3D 在 ModelNet clean 上具有较强基础性能；32_1 进一步说明：即使强 backbone 在 clean 上达到 81.85，面对 ModelNet-C corruption 时仍然会下降到 all35 Avg 66.89。

因此，32_1 是后续测试 Global Cache 和 Local Cache 鲁棒性提升的关键 baseline。

---

## 16. 与其他 backbone 的 ModelNet-C Zero-shot 关系

32_1 可以与前面 ULIP、ULIP-2、OpenShape 的 ModelNet-C Zero-shot 结果进行横向比较。

| Backbone | 实验编号 | ModelNet-C S2 Avg | ModelNet-C all35 Avg |
|---|---|---:|---:|
| ULIP | 02_1_ulip_modelnetc_corruptions_all35_zs | 47.68 | 46.85 |
| ULIP-2 | 12_1_ulip2_modelnetc_corruptions_all35_zs | 58.02 | 58.07 |
| OpenShape | 22_1_openshape_modelnetc_corruptions_all35_zs | 73.57 | 72.57 |
| Uni3D | 32_1_uni3d_modelnetc_corruptions_all35_zs | 67.80 | 66.89 |

分析：

Uni3D 在 ModelNet-C Zero-shot 上明显强于 ULIP 和 ULIP-2，但低于 OpenShape。这个相对排序与 ModelNet clean 上基本一致。

OpenShape 在 ModelNet-C 上的 zero-shot 鲁棒性更强，而 Uni3D 仍然有明显提升空间。后续 32_2 和 32_3 需要观察 cache 是否能缩小 Uni3D 与 OpenShape 的差距。

---

## 17. 与后续子实验的关系

32_1 是 32 组第一个子实验，因此没有前序 32 组子实验可供对比。

它后续将作为以下实验的基线：

| 后续实验 | 对比方式 |
|---|---|
| 32_2_uni3d_modelnetc_corruptions_all35_zs_global | 与 32_1 比较，评估 Global Cache 在 Uni3D × ModelNet-C 上的鲁棒性增益 |
| 32_3_uni3d_modelnetc_corruptions_all35_zs_global_local | 与 32_1 和 32_2 比较，评估完整 Point-Cache 及 Local Cache 额外影响 |

原文中 Uni3D 在 ModelNet-C severity=2 上 cache 是有正增益的：Zero-shot Avg 为 67.95，+Global Cache Avg 为 71.81，+Hierarchical Cache Avg 为 73.31。因此，后续 32_2 和 32_3 的重点是观察：

1. Global Cache 是否提升；
2. Local Cache 是否在 Global Cache 基础上继续提升；
3. 完整 Point-Cache 是否接近原文 73.31；
4. 当前方法趋势是否保持 Zero-shot < Global < Global + Local。

---

## 18. 结果含义分析

32_1 的意义不只是给出一个 corrupted accuracy，而是说明 Uni3D 在 ModelNet-C 上仍然存在明显鲁棒性下降。

| 观察 | 含义 |
|---|---|
| 32_1 all35 Avg = 66.89 | Uni3D 在 ModelNet-C 上的 Zero-shot 总体鲁棒性 |
| 32_1 S2 Avg = 67.80 | 与原文 Table 1 对齐的复现结果 |
| 比原文 S2 低 -0.15 | 数值高度对齐 |
| 比 31_1 clean 低 -14.96 | corruption 明显削弱 Uni3D zero-shot |
| jitter 和 add_local 最难 | 坐标扰动和局部异常点是主要失败模式 |
| 使用 modelnet40 checkpoint 后对齐 | checkpoint 设置可靠 |

因此，32_1 是后续 32_2 和 32_3 判断 cache 是否能改善 Uni3D corruption robustness 的必要基础。

---

## 19. 对后续 MCM-PC 的启发

当前 32_1 对后续 MCM-PC 方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| Uni3D clean 很强，但 ModelNet-C 明显下降 | 强 backbone 仍需要测试时适应 |
| jitter 是最困难 corruption | 坐标扰动需要专门鲁棒机制 |
| add_local 也很困难 | 局部异常点可能需要局部证据或负证据处理 |
| high-severity dropout 退化明显 | 点云缺失仍是鲁棒性挑战 |
| 32 组必须使用 modelnet40 checkpoint | checkpoint 是 Uni3D 实验关键设置 |
| 32_1 与原文高度对齐 | 可作为后续方法对照 baseline |

这对 MCM-PC 很重要：如果后续方法希望证明在 Uni3D backbone 上仍有效，32 组 ModelNet-C 是一个关键实验场景。

---

## 20. 阶段性结论

本实验完成了 Uni3D × ModelNet-C all35 的 Zero-shot baseline 复现。

主要结论如下：

1. 32_1 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前使用 checkpoint 为 weights/uni3d/modelnet40/model.pt。
3. 当前 severity=2 Average 为 67.80，原文 Point-Cache Table 1 中 Uni3D Zero-shot Avg 为 67.95，差异 -0.15。
4. 当前 all35 Average 为 66.89，是本实验额外统计的 35 个 corrupted setting 总平均。
5. 当前结果与原文高度对齐，可以认为 32_1 复现有效。
6. 相比 31_1 ModelNet clean 的 81.85，32_1 all35 Average 下降到 66.89，下降 -14.96。
7. jitter 是最困难 corruption，平均准确率只有 54.81。
8. jitter_4 是全部 35 个 setting 中最低结果，只有 35.45。
9. add_local 是第二困难 corruption，平均准确率为 57.00。
10. 高 severity dropout_global 和 dropout_local 也明显困难。
11. 该实验是 32_2 Global Cache 和 32_3 Hierarchical Cache 的基础对照，不在本文件中展开完整 32 组方法间对比。
12. 31 / 32 组后续应统一使用 weights/uni3d/modelnet40/model.pt。

---

## 21. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs_single_gpu.sh 1

---

## 22. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs/summary.csv
