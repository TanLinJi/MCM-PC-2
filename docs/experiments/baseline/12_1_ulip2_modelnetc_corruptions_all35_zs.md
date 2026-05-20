# 12_1_ulip2_modelnetc_corruptions_all35_zs

## 1. 实验目的

本实验用于复现 ULIP-2 在 ModelNet-C 全部 35 个损坏设置上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 12_1_ulip2_modelnetc_corruptions_all35_zs |
| Backbone | ULIP-2 |
| Dataset | ModelNet-C |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 ULIP-2 在 ModelNet-C corrupted setting 上的无缓存基础性能。该结果后续会作为 12_2 和 12_3 的对照基线，但本文件只记录 12_1 本身，不展开整个 12 组的综合分析。

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
| 实验编号 | 12_1_ulip2_modelnetc_corruptions_all35_zs |
| 方法脚本 | Point-Cache/scripts/baseline/12_1_ulip2_modelnetc_corruptions_all35_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/12_run_ulip2_modelnetc_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_ulip2_modelnetc_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/12_1_ulip2_modelnetc_corruptions_all35_zs/ |

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
| Backbone | ULIP-2 |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 优化 runner | runners/baseline/run_ulip2_modelnetc_corruptions_all35.py |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |
| Severity 编号 | 0, 1, 2, 3, 4 |
| 损坏组合数 | 7 × 5 = 35 |
| 输入点数 | 1024 |
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

Point-Cache/results/baseline/12_1_ulip2_modelnetc_corruptions_all35_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

12_1_ulip2_modelnetc_corruptions_all35_zs_add_global_0_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 58.02 | 用于论文对齐 | 与原文 Point-Cache Table 1 对比 |
| all35 Average | 58.07 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，12_1 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ModelNet-C severity=2 参考值进行对比。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 68.19 | 66.37 | 65.52 | 63.74 | 62.07 | 65.18 |
| add_local | 62.68 | 57.58 | 54.66 | 51.38 | 48.30 | 54.92 |
| dropout_global | 71.43 | 70.38 | 68.92 | 61.87 | 48.91 | 64.30 |
| dropout_local | 69.81 | 63.90 | 58.31 | 51.34 | 45.14 | 57.70 |
| rotate | 71.92 | 72.00 | 71.03 | 68.27 | 66.09 | 69.86 |
| scale | 69.04 | 68.68 | 66.73 | 66.69 | 65.19 | 67.27 |
| jitter | 57.29 | 34.60 | 20.99 | 14.18 | 9.32 | 27.28 |
| **Average** | **67.19** | **61.93** | **58.02** | **53.92** | **49.29** | **58.07** |

整体观察：

1. all35 Average 为 58.07，表示 ULIP-2 在 ModelNet-C 全 35 个 corrupted setting 上的 Zero-shot 水平。
2. severity=2 Average 为 58.02，用于和原文 Point-Cache Table 1 对齐。
3. rotate 的平均准确率最高，为 69.86。
4. jitter 的平均准确率最低，为 27.28。
5. jitter_4 为 9.32，是全部 35 个 setting 中最低的结果，说明高强度坐标扰动对 ULIP-2 Zero-shot 影响非常严重。

---

## 8. 与原文结果对比

原文 Point-Cache Table 1 报告的是 ModelNet-C severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 65.52 | 65.15 | +0.37 | 0.37 |
| add_local | 54.66 | 54.62 | +0.04 | 0.04 |
| dropout_global | 68.92 | 68.76 | +0.16 | 0.16 |
| dropout_local | 58.31 | 57.98 | +0.33 | 0.33 |
| rotate | 71.03 | 70.30 | +0.73 | 0.73 |
| scale | 66.73 | 67.10 | -0.37 | 0.37 |
| jitter | 20.99 | 21.76 | -0.77 | 0.77 |
| **Average** | **58.02** | **57.95** | **+0.07** | **0.40 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.07 |
| MAE | 0.40 |
| RMSE | 0.47 |
| Max Abs Diff | 0.77 |

分析：

当前复现的 severity=2 Average 为 58.02，原文为 57.95，差异仅 +0.07。单项最大绝对差异为 jitter 的 0.77，整体误差较小。

因此，12_1 不只是脚本跑通，而且数值也与原文 ULIP-2 在 ModelNet-C severity=2 上的 Zero-shot 结果高度对齐。

---

## 9. Severity 维度分析

### 9.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 67.19 | — | 0.00 |
| S1 | 61.93 | -5.26 | -5.26 |
| S2 | 58.02 | -3.91 | -9.17 |
| S3 | 53.92 | -4.10 | -13.27 |
| S4 | 49.29 | -4.63 | -17.90 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 67.19 下降到 49.29，总下降 17.90 个百分点。整体上，severity 越高，ULIP-2 Zero-shot 准确率越低，说明 corruption severity 对性能有持续影响。

S0 到 S1 的下降较明显，下降 5.26；S3 到 S4 也下降 4.63。说明 ULIP-2 虽然整体比 ULIP 更强，但在高强度 corruption 下仍然会明显退化。

### 9.2 与 11_1 ModelNet clean 的关系

| 设置 | Accuracy |
|---|---:|
| 11_1 ModelNet clean Zero-shot | 72.20 |
| 12_1 ModelNet-C S2 Average | 58.02 |
| 12_1 ModelNet-C all35 Average | 58.07 |

对比：

| 比较 | 变化 |
|---|---:|
| S2 Average - clean | -14.18 |
| all35 Average - clean | -14.13 |

分析：

ModelNet-C corruption 使 ULIP-2 Zero-shot 相比 ModelNet clean 下降约 14 个百分点。这说明即使 ULIP-2 在 clean 数据上已经达到 72.20，面对 corrupted setting 时仍然存在明显鲁棒性下降。

---

## 10. Corruption 难度分析

### 10.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 27.28 | 平均最低，高 severity 下崩溃最明显 |
| 2 | add_local | 54.92 | 局部异常点较困难 |
| 3 | dropout_local | 57.70 | 局部缺失造成明显下降 |
| 4 | dropout_global | 64.30 | 高 severity 下降较大 |
| 5 | add_global | 65.18 | 表现中等偏高 |
| 6 | scale | 67.27 | 较稳定 |
| 7 | rotate | 69.86 | 当前最容易 |

分析：

jitter 是绝对最困难的 corruption，平均准确率只有 27.28，远低于其他 corruption。尤其 jitter_4 只有 9.32，说明强坐标扰动会严重破坏 ULIP-2 的点云表征。

rotate 和 scale 表现最好，说明 ULIP-2 对旋转和尺度变化相对更鲁棒。

### 10.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 69.04 | 65.19 | 3.85 | 5.58% | 67.27 |
| rotate | 71.92 | 66.09 | 5.83 | 8.11% | 69.86 |
| add_global | 68.19 | 62.07 | 6.12 | 8.97% | 65.18 |
| add_local | 62.68 | 48.30 | 14.38 | 22.94% | 54.92 |
| dropout_global | 71.43 | 48.91 | 22.52 | 31.53% | 64.30 |
| dropout_local | 69.81 | 45.14 | 24.67 | 35.34% | 57.70 |
| jitter | 57.29 | 9.32 | 47.97 | 83.73% | 27.28 |

分析：

scale、rotate 和 add_global 在 severity 增大时相对稳定；jitter 的退化最严重，从 S0 的 57.29 下降到 S4 的 9.32，绝对下降 47.97，相对下降 83.73%。

dropout_local 和 dropout_global 也有明显退化，说明点云缺失类 corruption 在高 severity 下会严重影响 ULIP-2 的识别能力。

---

## 11. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | jitter | 57.29 | rotate | 71.92 | 14.63 |
| S1 | jitter | 34.60 | rotate | 72.00 | 37.40 |
| S2 | jitter | 20.99 | rotate | 71.03 | 50.04 |
| S3 | jitter | 14.18 | rotate | 68.27 | 54.09 |
| S4 | jitter | 9.32 | rotate | 66.09 | 56.77 |

分析：

在所有 severity 下，jitter 都是最困难 corruption，而 rotate 始终是最容易 corruption。随着 severity 增大，best-worst gap 从 S0 的 14.63 扩大到 S4 的 56.77。

这说明 ULIP-2 在不同 corruption 类型之间的鲁棒性差异非常大。尤其在高 severity 下，jitter 导致模型性能严重崩溃，而 rotate 仍能保持较高准确率。

---

## 12. 低准确率区域分析

| 条件 | 数量 | 占比 | 主要涉及 corruption |
|---|---:|---:|---|
| Acc < 65 | 18 / 35 | 51.43% | jitter, add_local, dropout_local, dropout_global, add_global |
| Acc < 60 | 13 / 35 | 37.14% | jitter, add_local, dropout_local |
| Acc < 55 | 10 / 35 | 28.57% | jitter, add_local, dropout_local |
| Acc < 50 | 7 / 35 | 20.00% | jitter, add_local_4, dropout_global_4, dropout_local_4 |
| Acc < 45 | 4 / 35 | 11.43% | jitter_1 到 jitter_4 |
| Acc < 40 | 4 / 35 | 11.43% | jitter_1 到 jitter_4 |
| Acc < 35 | 4 / 35 | 11.43% | jitter_1 到 jitter_4 |
| Acc < 30 | 3 / 35 | 8.57% | jitter_2, jitter_3, jitter_4 |
| Acc < 25 | 3 / 35 | 8.57% | jitter_2, jitter_3, jitter_4 |

分析：

12_1 的严重低准确率区域高度集中在 jitter。尤其 jitter_2、jitter_3、jitter_4 均低于 30，jitter_4 只有 9.32。

这说明 ULIP-2 的 ModelNet-C Zero-shot 主要短板不是所有 corruption 都均匀退化，而是对坐标扰动类 corruption 极其敏感。后续 12_2 和 12_3 需要重点观察 Global Cache 和 Local Cache 是否能缓解 jitter。

---

## 13. 与前序实验的关系

12_1 的直接前序数据设置是 11_1，即 ULIP-2 在 ModelNet clean 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | Accuracy |
|---|---|---|---:|
| 11_1_ulip2_modelnet_clean_zs | ModelNet clean | Zero-shot | 72.20 |
| 12_1_ulip2_modelnetc_corruptions_all35_zs | ModelNet-C all35 | Zero-shot | 58.07 all35 / 58.02 S2 |

当前结果说明：从 clean ModelNet 到 corrupted ModelNet-C，ULIP-2 Zero-shot 明显下降。

| 比较 | 变化 |
|---|---:|
| 12_1 S2 Avg - 11_1 clean | -14.18 |
| 12_1 all35 Avg - 11_1 clean | -14.13 |

分析：

11_1 已经说明 ULIP-2 在 clean ModelNet 上基础性能较强，但 12_1 表明该强 backbone 在 corrupted setting 下仍存在明显鲁棒性缺口。

因此，12_1 是后续 12_2 Global Cache 和 12_3 Hierarchical Cache 的必要基础对照。

---

## 14. 阶段性结论

本实验完成了 ULIP-2 × ModelNet-C all35 的 Zero-shot baseline 复现。

主要结论如下：

1. 12_1 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 58.02，原文 Point-Cache Table 1 中 ULIP-2 Zero-shot Avg 为 57.95，差异仅 +0.07。
3. 当前 all35 Average 为 58.07，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果与原文 severity=2 数值高度对齐，可以认为 12_1 复现成功。
5. 相比 11_1 ModelNet clean 的 72.20，12_1 的 all35 Average 下降到 58.07，说明 corruption 使 ULIP-2 明显退化。
6. jitter 是最困难 corruption，平均准确率只有 27.28。
7. jitter_4 是全部 35 个 setting 中最低结果，只有 9.32。
8. rotate 和 scale 相对稳定，平均准确率分别为 69.86 和 67.27。
9. 该实验是 12_2 Global Cache 和 12_3 Hierarchical Cache 的基础对照，不在本文件中展开完整 12 组方法间对比。

---

## 15. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/12_1_ulip2_modelnetc_corruptions_all35_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/12_1_ulip2_modelnetc_corruptions_all35_zs_single_gpu.sh 1

---

## 16. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/12_1_ulip2_modelnetc_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/12_1_ulip2_modelnetc_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/12_1_ulip2_modelnetc_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/12_1_ulip2_modelnetc_corruptions_all35_zs/summary.csv | cut -d',' -f13 | sort | uniq -c
