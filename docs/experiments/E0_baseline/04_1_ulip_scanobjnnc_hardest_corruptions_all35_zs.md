# 04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs

## 1. 实验目的

本实验用于复现 ULIP 在 ScanObjNN-C hardest 全部 35 个损坏设置上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs |
| Backbone | ULIP |
| Dataset | ScanObjNN-C hardest |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 ULIP 在真实扫描 hardest split 的 corrupted setting 上的无缓存基础性能。该结果后续会作为 04_2 和 04_3 的对照基线，但本文件只记录 04_1 本身，不展开整个 04 组的综合分析。

需要特别注意：原文 Supplementary Table 7 只报告 corruption severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原文 Supplementary Table 7 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

---

## 2. 当前实现方式

本实验的外部命名规则保持不变：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs |
| 方法脚本 | Point-Cache/scripts/baseline/04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/04_run_ulip_scanobjnnc_hardest_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs/ |

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
| Backbone | ULIP |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 数据目录 | data/sonn_c |
| 实际数据目录 | data/sonn_c/hardest |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 优化 runner | runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |
| Severity 编号 | 0, 1, 2, 3, 4 |
| 损坏组合数 | 7 × 5 = 35 |
| 输入点数 | 1024 |
| 权重 | weights/ulip/pointbert_ulip1.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| GPU | 单张 Tesla T4 |

本实验使用 `sonn_c` 作为 dataset 参数，`sonn_variant=hardest`，并在 Python runner 内部循环 35 个 `cor_type`。实际读取文件形式为：

data/sonn_c/hardest/{corruption}_{severity}.h5

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

Point-Cache/results/baseline/04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs_add_global_0_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 23.91 | 用于论文对齐 | 与原文 Supplementary Table 7 对比 |
| all35 Average | 23.65 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，04_1 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ScanObjNN-C hardest 的 severity=2 参考值进行对比。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 20.09 | 18.98 | 19.29 | 17.83 | 17.87 | 18.81 |
| add_local | 21.24 | 20.58 | 18.60 | 17.70 | 17.18 | 19.06 |
| dropout_global | 30.74 | 30.67 | 30.92 | 30.29 | 25.36 | 29.60 |
| dropout_local | 28.66 | 25.29 | 23.91 | 20.40 | 17.38 | 23.13 |
| rotate | 30.57 | 28.63 | 27.13 | 23.53 | 21.30 | 26.23 |
| scale | 28.52 | 27.52 | 26.06 | 25.82 | 26.72 | 26.93 |
| jitter | 28.21 | 24.05 | 21.48 | 19.22 | 15.86 | 21.76 |
| **Average** | **26.86** | **25.10** | **23.91** | **22.11** | **20.24** | **23.65** |

整体观察：

1. all35 Average 为 23.65，表示 ULIP 在 ScanObjNN-C hardest 全 35 个 corrupted setting 上的 Zero-shot 水平。
2. severity=2 Average 为 23.91，用于和原文 Supplementary Table 7 对齐。
3. dropout_global 的平均准确率最高，为 29.60。
4. add_global 和 add_local 的平均准确率最低，分别为 18.81 和 19.06。
5. jitter_4 为 15.86，是全部 35 个 setting 中最低的结果，说明高强度坐标扰动在真实扫描 hardest split 上尤其困难。

---

## 8. 与原文结果对比

原文 Supplementary Table 7 报告的是 S-PB T50-RS-C severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 19.29 | 19.26 | +0.03 | 0.03 |
| add_local | 18.60 | 18.39 | +0.21 | 0.21 |
| dropout_global | 30.92 | 30.99 | -0.07 | 0.07 |
| dropout_local | 23.91 | 23.91 | +0.00 | 0.00 |
| rotate | 27.13 | 27.48 | -0.35 | 0.35 |
| scale | 26.06 | 26.34 | -0.28 | 0.28 |
| jitter | 21.48 | 21.44 | +0.04 | 0.04 |
| **Average** | **23.91** | **23.97** | **-0.06** | **0.14 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | -0.06 |
| MAE | 0.14 |
| RMSE | 0.19 |
| Max Abs Diff | 0.35 |

分析：

当前复现的 severity=2 Average 为 23.91，原文为 23.97，差异仅 -0.06。单项最大绝对差异为 rotate 的 0.35，整体误差很小。

因此，04_1 不只是脚本跑通，而且数值也与原文 ULIP 在 S-PB T50-RS-C 上的 Zero-shot 结果高度对齐。

---

## 9. Severity 维度分析

### 9.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 26.86 | — | 0.00 |
| S1 | 25.10 | -1.76 | -1.76 |
| S2 | 23.91 | -1.19 | -2.95 |
| S3 | 22.11 | -1.80 | -4.75 |
| S4 | 20.24 | -1.87 | -6.62 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 26.86 下降到 20.24，总下降 6.62 个百分点。整体上，severity 越高，Zero-shot 准确率越低，说明 severity 编号与损坏强度关系合理。

不过，下降幅度小于 02 组 ModelNet-C 的 Zero-shot all35 下降幅度。这并不表示 ScanObjNN-C 更容易，而是因为 ScanObjNN hardest 本身 clean 性能已经较低，所有 corrupted setting 都处在较低准确率区间，进一步下降空间较有限。

### 9.2 与 03_1 clean hardest 的关系

| 设置 | Accuracy |
|---|---:|
| 03_1 ScanObjNN clean hardest Zero-shot | 29.08 |
| 04_1 ScanObjNN-C hardest S2 Average | 23.91 |
| 04_1 ScanObjNN-C hardest all35 Average | 23.65 |

对比：

| 比较 | 变化 |
|---|---:|
| S2 Average - clean | -5.17 |
| all35 Average - clean | -5.43 |

分析：

ScanObjNN-C corruption 使 ULIP Zero-shot 相比 clean hardest 进一步下降约 5 个百分点。这说明真实扫描 hardest split 本身已经困难，而 corruption 会在真实扫描域的基础上进一步加剧 domain shift。

---

## 10. Corruption 难度分析

### 10.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | add_global | 18.81 | 平均最低，全局异常点非常困难 |
| 2 | add_local | 19.06 | 局部异常点同样困难 |
| 3 | jitter | 21.76 | 高 severity 下最低，jitter_4 只有 15.86 |
| 4 | dropout_local | 23.13 | 局部缺失造成明显下降 |
| 5 | rotate | 26.23 | 中等难度 |
| 6 | scale | 26.93 | 较稳定 |
| 7 | dropout_global | 29.60 | 当前最容易 |

分析：

在 ScanObjNN-C hardest 上，异常点类 corruption 最困难。add_global 和 add_local 的平均准确率都低于 20，说明真实扫描数据一旦引入额外异常点，ULIP 的 zero-shot 判断会明显受干扰。

dropout_global 表现最好，平均为 29.60，甚至略高于 03_1 clean hardest 的 29.08。这可能说明该 corruption 在某些 severity 下并不一定比 clean 更困难，或者部分全局 dropout 去除了干扰点，使得模型预测没有进一步恶化。

### 10.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 28.52 | 26.72 | 1.80 | 6.31% | 26.93 |
| add_global | 20.09 | 17.87 | 2.22 | 11.05% | 18.81 |
| add_local | 21.24 | 17.18 | 4.06 | 19.11% | 19.06 |
| dropout_global | 30.74 | 25.36 | 5.38 | 17.50% | 29.60 |
| rotate | 30.57 | 21.30 | 9.27 | 30.32% | 26.23 |
| dropout_local | 28.66 | 17.38 | 11.28 | 39.36% | 23.13 |
| jitter | 28.21 | 15.86 | 12.35 | 43.78% | 21.76 |

分析：

scale 最稳定，从 S0 到 S4 只下降 1.80。jitter 退化最大，从 28.21 下降到 15.86，绝对下降 12.35，相对下降 43.78%。

dropout_local 也有明显退化，从 28.66 降到 17.38。这说明在真实扫描 hardest split 上，局部结构扰动和坐标扰动都对 ULIP Zero-shot 造成严重影响。

---

## 11. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | add_global | 20.09 | dropout_global | 30.74 | 10.65 |
| S1 | add_global | 18.98 | dropout_global | 30.67 | 11.69 |
| S2 | add_local | 18.60 | dropout_global | 30.92 | 12.32 |
| S3 | add_local | 17.70 | dropout_global | 30.29 | 12.59 |
| S4 | jitter | 15.86 | scale | 26.72 | 10.86 |

分析：

在 S0 和 S1，add_global 最难；在 S2 和 S3，add_local 最难；在 S4，jitter 最难。也就是说，低中 severity 下异常点类 corruption 最困难，而最高 severity 下坐标扰动成为最严重失败点。

不同 corruption 之间的 best-worst gap 维持在约 10 到 13 个百分点之间，说明即使整体准确率较低，不同 corruption 类型之间的难度差异仍然明显。

---

## 12. 低准确率区域分析

| 条件 | 数量 | 占比 | 主要涉及 corruption |
|---|---:|---:|---|
| Acc < 30 | 30 / 35 | 85.71% | 除 dropout_global 多数 setting 外，大部分都低于 30 |
| Acc < 28 | 26 / 35 | 74.29% | add_global, add_local, jitter, dropout_local, rotate, scale |
| Acc < 25 | 19 / 35 | 54.29% | add_global, add_local, jitter, dropout_local, rotate |
| Acc < 22 | 16 / 35 | 45.71% | add_global, add_local, jitter, dropout_local, rotate |
| Acc < 20 | 10 / 35 | 28.57% | add_global, add_local, jitter, dropout_local |

分析：

04_1 的低准确率区域非常大。35 个 setting 中有 30 个低于 30，19 个低于 25，10 个低于 20。低性能主要集中在 add_global、add_local、jitter 和 dropout_local。

这说明 ULIP Zero-shot 在 ScanObjNN-C hardest 上非常脆弱，尤其对异常点、局部缺失和坐标扰动敏感。后续 04_2 和 04_3 需要重点观察 cache 是否能减少这些低准确率区域。

---

## 13. 与前序实验的关系

04_1 的直接前序数据设置是 03_1，即 ULIP 在 ScanObjNN clean hardest 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | Accuracy |
|---|---|---|---:|
| 03_1_ulip_scanobjnn_clean_hardest_zs | ScanObjNN clean hardest | Zero-shot | 29.08 |
| 04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs | ScanObjNN-C hardest all35 | Zero-shot | 23.65 all35 / 23.91 S2 |

当前结果说明：从 clean hardest 到 corrupted hardest，ULIP Zero-shot 进一步下降。

| 比较 | 变化 |
|---|---:|
| 04_1 S2 Avg - 03_1 clean | -5.17 |
| 04_1 all35 Avg - 03_1 clean | -5.43 |

分析：

03_1 已经说明 ScanObjNN clean hardest 是困难域，04_1 进一步说明在该真实扫描困难域上引入 corruption 后，ULIP Zero-shot 会继续明显退化。因此，04_1 是后续 04_2 Global Cache 和 04_3 Hierarchical Cache 的必要基础对照。

---

## 14. 阶段性结论

本实验完成了 ULIP × ScanObjNN-C hardest all35 的 Zero-shot baseline 复现。

主要结论如下：

1. 04_1 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 23.91，原文 Supplementary Table 7 中 ULIP Zero-shot Avg 为 23.97，差异仅 -0.06。
3. 当前 all35 Average 为 23.65，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果与原文 severity=2 数值高度对齐，可以认为 04_1 复现成功。
5. 相比 03_1 clean hardest 的 29.08，04_1 的 all35 Average 下降到 23.65，说明 corruption 进一步加剧 ScanObjNN hardest 上的 domain shift。
6. add_global 和 add_local 是平均意义下最困难的 corruption，分别为 18.81 和 19.06。
7. jitter_4 是全部 35 个 setting 中最低结果，只有 15.86。
8. 低准确率区域非常大，35 个 setting 中有 30 个低于 30。
9. 该实验是 04_2 Global Cache 和 04_3 Hierarchical Cache 的基础对照，不在本文件中展开完整 04 组方法间对比。

---

## 15. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh 1

---

## 16. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f13 | sort | uniq -c
