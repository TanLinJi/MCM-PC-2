# 22_1_openshape_modelnetc_corruptions_all35_zs

## 1. 实验目的

本实验用于复现 OpenShape 在 ModelNet-C 全部 35 个损坏设置上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 22_1_openshape_modelnetc_corruptions_all35_zs |
| Backbone | OpenShape |
| Dataset | ModelNet-C |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 OpenShape 在 ModelNet-C corrupted setting 上的无缓存基础鲁棒性。该结果后续会作为 22_2 和 22_3 的对照基线，但本文件只记录 22_1 本身，不展开完整 22 组的方法间对比。

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
| 实验编号 | 22_1_openshape_modelnetc_corruptions_all35_zs |
| 方法脚本 | Point-Cache/scripts/baseline/22_1_openshape_modelnetc_corruptions_all35_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/22_run_openshape_modelnetc_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_openshape_modelnetc_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/22_1_openshape_modelnetc_corruptions_all35_zs/ |

本实验是 all35 实验，因此使用优化 runner：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 |
| 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 |
| 模型只加载一次，每个 cor_type 重新创建 DataLoader |
| bash 通过 tee 生成单个 cor_type 的 log |
| Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv |
| summary.csv 的列结构保持不变 |

该优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、预测逻辑和输出结构。

---

## 3. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | OpenShape |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 优化 runner | runners/baseline/run_openshape_modelnetc_corruptions_all35.py |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |
| Severity 编号 | 0, 1, 2, 3, 4 |
| 损坏组合数 | 7 × 5 = 35 |
| 输入点数 | 1024 |
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

Point-Cache/results/baseline/22_1_openshape_modelnetc_corruptions_all35_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

22_1_openshape_modelnetc_corruptions_all35_zs_add_global_0_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 73.57 | 用于论文对齐 | 与原文 Point-Cache Table 1 对比 |
| all35 Average | 72.57 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，22_1 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ModelNet-C severity=2 参考值进行对比。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 78.89 | 74.51 | 71.15 | 69.85 | 68.44 | 72.57 |
| add_local | 75.08 | 70.26 | 67.59 | 64.71 | 63.41 | 68.21 |
| dropout_global | 83.31 | 82.29 | 81.16 | 78.57 | 63.57 | 77.78 |
| dropout_local | 80.47 | 77.88 | 73.10 | 68.19 | 60.82 | 72.09 |
| rotate | 84.81 | 83.87 | 82.78 | 79.62 | 72.77 | 80.77 |
| scale | 79.94 | 79.21 | 78.97 | 78.00 | 77.03 | 78.63 |
| jitter | 79.66 | 70.91 | 60.25 | 45.99 | 32.98 | 57.96 |
| **Average** | **80.31** | **76.99** | **73.57** | **69.28** | **62.72** | **72.57** |

整体观察：

1. all35 Average 为 72.57，表示 OpenShape 在 ModelNet-C 全 35 个 corrupted setting 上的 Zero-shot 鲁棒性水平。
2. severity=2 Average 为 73.57，用于和原文 Point-Cache Table 1 对齐。
3. rotate 的平均准确率最高，为 80.77。
4. jitter 的平均准确率最低，为 57.96。
5. jitter_4 为 32.98，是全部 35 个 setting 中最低的结果，说明高强度坐标扰动仍然是 OpenShape 在 ModelNet-C 上的主要困难点。

---

## 8. 与原文结果对比

原文 Point-Cache Table 1 报告的是 ModelNet-C severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 71.15 | 71.64 | -0.49 | 0.49 |
| add_local | 67.59 | 67.79 | -0.20 | 0.20 |
| dropout_global | 81.16 | 81.56 | -0.40 | 0.40 |
| dropout_local | 73.10 | 73.58 | -0.48 | 0.48 |
| rotate | 82.78 | 82.01 | +0.77 | 0.77 |
| scale | 78.97 | 78.48 | +0.49 | 0.49 |
| jitter | 60.25 | 59.36 | +0.89 | 0.89 |
| **Average** | **73.57** | **73.49** | **+0.08** | **0.53 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.08 |
| MAE | 0.53 |
| RMSE | 0.58 |
| Max Abs Diff | 0.89 |

分析：

当前复现的 severity=2 Average 为 73.57，原文为 73.49，差异仅 +0.08。单项最大绝对差异为 jitter 的 0.89，整体误差很小。

因此，22_1 不只是脚本跑通，而且数值也与原文 OpenShape 在 ModelNet-C severity=2 上的 Zero-shot 结果高度对齐。

---

## 9. Severity 维度分析

### 9.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 80.31 | — | 0.00 |
| S1 | 76.99 | -3.32 | -3.32 |
| S2 | 73.57 | -3.42 | -6.74 |
| S3 | 69.28 | -4.29 | -11.03 |
| S4 | 62.72 | -6.56 | -17.59 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 80.31 下降到 62.72，总下降 17.59 个百分点。整体上，severity 越高，OpenShape Zero-shot 准确率越低，说明 corruption severity 对性能有持续影响。

S3 到 S4 的下降最大，为 -6.56，说明最高 severity 对 OpenShape 的破坏明显增强。

### 9.2 与 21_1 ModelNet clean 的关系

| 设置 | Accuracy |
|---|---:|
| 21_1 ModelNet clean Zero-shot | 84.72 |
| 22_1 ModelNet-C S2 Average | 73.57 |
| 22_1 ModelNet-C all35 Average | 72.57 |

对比：

| 比较 | 变化 |
|---|---:|
| S2 Average - clean | -11.15 |
| all35 Average - clean | -12.15 |

分析：

ModelNet-C corruption 使 OpenShape Zero-shot 相比 ModelNet clean 下降约 11 到 12 个百分点。虽然 OpenShape 在 ModelNet clean 上非常强，但在 corrupted setting 下仍然存在明显退化。

这说明 22 组是评估 OpenShape 鲁棒性的重要实验组。21 组 clean 上 cache 略降不能说明 cache 无效；真正需要观察的是 22_2 和 22_3 在 corrupted setting 上是否能缩小 clean-to-corruption gap。

---

## 10. Corruption 难度分析

### 10.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 57.96 | 平均最低，高 severity 下下降最明显 |
| 2 | add_local | 68.21 | 局部异常点导致明显影响 |
| 3 | dropout_local | 72.09 | 局部缺失造成一定下降 |
| 4 | add_global | 72.57 | 中等难度 |
| 5 | dropout_global | 77.78 | 整体较高，但 S4 下降明显 |
| 6 | scale | 78.63 | 相对稳定 |
| 7 | rotate | 80.77 | 当前最高 |

分析：

jitter 是最困难的 corruption，平均准确率只有 57.96，远低于其他 corruption。尤其 jitter_4 只有 32.98，说明强坐标扰动对 OpenShape 的点云表征破坏非常明显。

rotate 和 scale 的结果较高，说明 OpenShape 对旋转和尺度扰动相对更稳定。

### 10.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 79.94 | 77.03 | 2.91 | 3.64% | 78.63 |
| add_global | 78.89 | 68.44 | 10.45 | 13.25% | 72.57 |
| rotate | 84.81 | 72.77 | 12.04 | 14.20% | 80.77 |
| add_local | 75.08 | 63.41 | 11.67 | 15.54% | 68.21 |
| dropout_local | 80.47 | 60.82 | 19.65 | 24.42% | 72.09 |
| dropout_global | 83.31 | 63.57 | 19.74 | 23.69% | 77.78 |
| jitter | 79.66 | 32.98 | 46.68 | 58.60% | 57.96 |

分析：

scale 最稳定，从 S0 到 S4 只下降 2.91。jitter 的退化最强，从 79.66 下降到 32.98，绝对下降 46.68，相对下降 58.60%。

dropout_global 和 dropout_local 在 S4 也有明显退化，说明高 severity 下的点云缺失仍然会显著削弱 OpenShape 的 Zero-shot 表现。

---

## 11. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | add_local | 75.08 | rotate | 84.81 | 9.73 |
| S1 | add_local | 70.26 | rotate | 83.87 | 13.61 |
| S2 | jitter | 60.25 | rotate | 82.78 | 22.53 |
| S3 | jitter | 45.99 | rotate | 79.62 | 33.63 |
| S4 | jitter | 32.98 | scale | 77.03 | 44.05 |

分析：

在低 severity 下，add_local 是较难 corruption；从 S2 开始，jitter 成为明显最困难的 corruption。随着 severity 增大，best-worst gap 从 S0 的 9.73 扩大到 S4 的 44.05。

这说明 OpenShape 在 ModelNet-C 上的困难并不是均匀分布的，而是高度集中在 high-severity jitter。

---

## 12. 低准确率区域分析

| 条件 | 数量 | 占比 | 主要涉及 corruption |
|---|---:|---:|---|
| Acc < 80 | 24 / 35 | 68.57% | jitter, add_local, add_global, dropout_local, scale, dropout_global, rotate |
| Acc < 75 | 16 / 35 | 45.71% | jitter, add_local, add_global, dropout_local, rotate, dropout_global |
| Acc < 70 | 11 / 35 | 31.43% | jitter, add_local, add_global, dropout_local, dropout_global |
| Acc < 65 | 7 / 35 | 20.00% | jitter, add_local_3/4, dropout_local_4, dropout_global_4 |
| Acc < 60 | 2 / 35 | 5.71% | jitter_3, jitter_4 |
| Acc < 50 | 2 / 35 | 5.71% | jitter_3, jitter_4 |
| Acc < 40 | 1 / 35 | 2.86% | jitter_4 |

分析：

22_1 的低准确率区域主要集中在 high-severity jitter。jitter_3 为 45.99，jitter_4 为 32.98，是最明显的失败点。

这说明 OpenShape 虽然整体鲁棒性强于 ULIP / ULIP-2，但对强坐标扰动仍然非常敏感。

---

## 13. 与前序实验的关系

22_1 的直接前序数据设置是 21_1，即 OpenShape 在 ModelNet clean 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | Accuracy |
|---|---|---|---:|
| 21_1_openshape_modelnet_clean_zs | ModelNet clean | Zero-shot | 84.72 |
| 22_1_openshape_modelnetc_corruptions_all35_zs | ModelNet-C all35 | Zero-shot | 72.57 all35 / 73.57 S2 |

当前结果说明：从 clean 到 corrupted setting，OpenShape Zero-shot 明显下降。

| 比较 | 变化 |
|---|---:|
| 22_1 S2 Avg - 21_1 clean | -11.15 |
| 22_1 all35 Avg - 21_1 clean | -12.15 |

分析：

21_1 已经说明 OpenShape 在 ModelNet clean 上非常强，22_1 进一步说明，即使是强 backbone，在 ModelNet-C corruption 下也会出现明显性能退化。因此，22_1 是后续 22_2 Global Cache 和 22_3 Hierarchical Cache 的必要基础对照。

---

## 14. 与 ULIP / ULIP-2 的 ModelNet-C 关系

22_1 可以与前面 ULIP、ULIP-2 的 ModelNet-C Zero-shot 结果进行横向比较。

| Backbone | ModelNet-C S2 Avg | ModelNet-C all35 Avg |
|---|---:|---:|
| ULIP | 47.68 左右 | 46.85 左右 |
| ULIP-2 | 58.02 | 58.07 |
| OpenShape | 73.57 | 72.57 |

分析：

OpenShape 在 ModelNet-C 上明显强于 ULIP 和 ULIP-2。即使不使用 cache，OpenShape 的 corrupted setting 准确率也显著更高。

这说明 OpenShape 本身具有更强的点云-文本对齐能力和更好的 corruption 鲁棒性。但从 21_1 到 22_1 的下降仍然超过 11 个百分点，说明 corrupted setting 仍然会显著削弱 OpenShape。

---

## 15. 与后续子实验的关系

22_1 是 22 组第一个子实验，因此没有前序 22 组子实验可供对比。

它后续将作为以下实验的基线：

| 后续实验 | 对比方式 |
|---|---|
| 22_2_openshape_modelnetc_corruptions_all35_zs_global | 与 22_1 比较，评估 Global Cache 在 OpenShape × ModelNet-C 上的鲁棒性增益 |
| 22_3_openshape_modelnetc_corruptions_all35_zs_global_local | 与 22_1 和 22_2 比较，评估完整 Point-Cache 及 Local Cache 额外影响 |

原文中 OpenShape 在 ModelNet-C severity=2 上 cache 是有正增益的：Zero-shot Avg 为 73.49，+Global Cache Avg 为 76.43，+Hierarchical Cache Avg 为 76.59。因此，后续 22_2 和 22_3 的重点不是 clean 上是否下降，而是 corrupted setting 上是否能够提升鲁棒性。

---

## 16. 阶段性结论

本实验完成了 OpenShape × ModelNet-C all35 的 Zero-shot baseline 复现。

主要结论如下：

1. 22_1 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 73.57，原文 Point-Cache Table 1 中 OpenShape Zero-shot Avg 为 73.49，差异仅 +0.08。
3. 当前 all35 Average 为 72.57，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果与原文 severity=2 数值高度对齐，可以认为 22_1 复现成功。
5. 相比 21_1 ModelNet clean 的 84.72，22_1 的 all35 Average 下降到 72.57，说明 corruption 明显降低 OpenShape 的 Zero-shot 性能。
6. jitter 是最困难 corruption，平均准确率只有 57.96。
7. jitter_4 是全部 35 个 setting 中最低结果，只有 32.98。
8. rotate 和 scale 相对稳定，其中 rotate 平均最高，为 80.77。
9. 该实验是 22_2 Global Cache 和 22_3 Hierarchical Cache 的基础对照，不在本文件中展开完整 22 组方法间对比。

---

## 17. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/22_1_openshape_modelnetc_corruptions_all35_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/22_1_openshape_modelnetc_corruptions_all35_zs_single_gpu.sh 1

---

## 18. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/22_1_openshape_modelnetc_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/22_1_openshape_modelnetc_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/22_1_openshape_modelnetc_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/22_1_openshape_modelnetc_corruptions_all35_zs/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/22_1_openshape_modelnetc_corruptions_all35_zs/summary.csv
