# 14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs

## 1. 实验目的

本实验用于复现 ULIP-2 在 ScanObjNN-C hardest 全部 35 个损坏设置上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs |
| Backbone | ULIP-2 |
| Dataset | ScanObjNN-C hardest |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 ULIP-2 在真实扫描 hardest split 的 corrupted setting 上的无缓存基础性能。该结果后续会作为 14_2 和 14_3 的对照基线，但本文件只记录 14_1 本身，不展开整个 14 组的综合分析。

需要特别注意：原文 Point-Cache Supplementary Table 7 只报告 corruption severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原文 Point-Cache Supplementary Table 7 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

---

## 2. 当前实现方式

本实验的外部命名规则如下：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs |
| 方法脚本 | Point-Cache/scripts/baseline/14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/14_run_ulip2_scanobjnnc_hardest_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs/ |

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
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 数据目录 | data/sonn_c |
| 实际数据目录 | data/sonn_c/hardest |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 优化 runner | runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |
| Severity 编号 | 0, 1, 2, 3, 4 |
| 损坏组合数 | 7 × 5 = 35 |
| 输入点数 | 1024 |
| Backbone 权重 | weights/ulip/pointbert_ulip2.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| ULIP version | ulip2 |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 0 |

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

Point-Cache/results/baseline/14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs_add_global_0_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 26.44 | 用于论文对齐 | 与原文 Supplementary Table 7 对比 |
| all35 Average | 26.46 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，14_1 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ScanObjNN-C hardest 的 severity=2 参考值进行对比。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 31.30 | 30.99 | 30.85 | 30.60 | 30.05 | 30.76 |
| add_local | 32.41 | 30.40 | 29.63 | 28.45 | 27.00 | 29.58 |
| dropout_global | 31.78 | 30.53 | 28.49 | 25.23 | 21.44 | 27.49 |
| dropout_local | 29.67 | 27.79 | 24.91 | 22.14 | 20.44 | 24.99 |
| rotate | 34.04 | 31.23 | 28.42 | 25.47 | 24.22 | 28.68 |
| scale | 30.53 | 30.81 | 30.19 | 28.28 | 28.73 | 29.71 |
| jitter | 21.72 | 16.69 | 12.60 | 10.24 | 8.67 | 13.98 |
| **Average** | **30.21** | **28.35** | **26.44** | **24.34** | **22.94** | **26.46** |

整体观察：

1. all35 Average 为 26.46，表示 ULIP-2 在 ScanObjNN-C hardest 全 35 个 corrupted setting 上的 Zero-shot 水平。
2. severity=2 Average 为 26.44，用于和原文 Supplementary Table 7 对齐。
3. add_global 的平均准确率最高，为 30.76。
4. jitter 的平均准确率最低，为 13.98。
5. jitter_4 为 8.67，是全部 35 个 setting 中最低的结果，说明高强度坐标扰动在真实扫描 hardest split 上极其困难。

---

## 8. 与原文结果对比

原文 Point-Cache Supplementary Table 7 报告的是 S-PB T50-RS-C severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 30.85 | 30.29 | +0.56 | 0.56 |
| add_local | 29.63 | 29.42 | +0.21 | 0.21 |
| dropout_global | 28.49 | 28.24 | +0.25 | 0.25 |
| dropout_local | 24.91 | 24.91 | +0.00 | 0.00 |
| rotate | 28.42 | 28.56 | -0.14 | 0.14 |
| scale | 30.19 | 30.22 | -0.03 | 0.03 |
| jitter | 12.60 | 12.98 | -0.38 | 0.38 |
| **Average** | **26.44** | **26.37** | **+0.07** | **0.22 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.07 |
| MAE | 0.22 |
| RMSE | 0.29 |
| Max Abs Diff | 0.56 |

分析：

当前复现的 severity=2 Average 为 26.44，原文为 26.37，差异仅 +0.07。单项最大绝对差异为 add_global 的 0.56，整体误差很小。

因此，14_1 不只是脚本跑通，而且数值也与原文 ULIP-2 在 S-PB T50-RS-C corrupted setting 上的 Zero-shot 结果高度对齐。

---

## 9. Severity 维度分析

### 9.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 30.21 | — | 0.00 |
| S1 | 28.35 | -1.86 | -1.86 |
| S2 | 26.44 | -1.91 | -3.77 |
| S3 | 24.34 | -2.10 | -5.87 |
| S4 | 22.94 | -1.40 | -7.27 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 30.21 下降到 22.94，总下降 7.27 个百分点。整体上，severity 越高，ULIP-2 Zero-shot 准确率越低，说明 corruption severity 对性能有持续影响。

不过，14_1 的整体准确率本身已经比较低，因此不同 severity 间的下降幅度没有 ModelNet-C 那样大。ScanObjNN-C hardest 的困难不仅来自 corruption，也来自真实扫描 hardest split 本身的强 domain shift。

### 9.2 与 13_1 ScanObjNN clean hardest 的关系

| 设置 | Accuracy |
|---|---:|
| 13_1 ScanObjNN clean hardest Zero-shot | 34.07 |
| 14_1 ScanObjNN-C hardest S2 Average | 26.44 |
| 14_1 ScanObjNN-C hardest all35 Average | 26.46 |

对比：

| 比较 | 变化 |
|---|---:|
| S2 Average - clean | -7.63 |
| all35 Average - clean | -7.61 |

分析：

ScanObjNN-C corruption 使 ULIP-2 Zero-shot 相比 ScanObjNN clean hardest 下降约 7.6 个百分点。这说明真实扫描 hardest split 本身已经很困难，而 corruption 会在真实扫描域偏移的基础上进一步降低性能。

---

## 10. Corruption 难度分析

### 10.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 13.98 | 平均最低，高 severity 下接近崩溃 |
| 2 | dropout_local | 24.99 | 局部缺失造成明显下降 |
| 3 | dropout_global | 27.49 | 高 severity 下降明显 |
| 4 | rotate | 28.68 | 中等偏高 |
| 5 | add_local | 29.58 | 相对稳定 |
| 6 | scale | 29.71 | 相对稳定 |
| 7 | add_global | 30.76 | 当前最高 |

分析：

jitter 是绝对最困难的 corruption，平均准确率只有 13.98，远低于其他 corruption。尤其 jitter_4 只有 8.67，说明强坐标扰动会严重破坏 ULIP-2 在真实扫描 hardest split 上的点云表征。

add_global、scale 和 add_local 的平均结果较高，说明在当前 Zero-shot setting 下，这几类 corruption 对 ULIP-2 的影响相对较小。

### 10.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| add_global | 31.30 | 30.05 | 1.25 | 3.99% | 30.76 |
| scale | 30.53 | 28.73 | 1.80 | 5.90% | 29.71 |
| add_local | 32.41 | 27.00 | 5.41 | 16.69% | 29.58 |
| dropout_local | 29.67 | 20.44 | 9.23 | 31.11% | 24.99 |
| rotate | 34.04 | 24.22 | 9.82 | 28.85% | 28.68 |
| dropout_global | 31.78 | 21.44 | 10.34 | 32.54% | 27.49 |
| jitter | 21.72 | 8.67 | 13.05 | 60.08% | 13.98 |

分析：

add_global 和 scale 最稳定，从 S0 到 S4 分别只下降 1.25 和 1.80。jitter 的退化最大，从 21.72 下降到 8.67，绝对下降 13.05，相对下降 60.08%。

dropout_global、dropout_local 和 rotate 也有明显退化，说明高 severity 下的点云缺失和旋转变化会进一步削弱 ULIP-2 的识别能力。

---

## 11. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | jitter | 21.72 | rotate | 34.04 | 12.32 |
| S1 | jitter | 16.69 | rotate | 31.23 | 14.54 |
| S2 | jitter | 12.60 | add_global | 30.85 | 18.25 |
| S3 | jitter | 10.24 | add_global | 30.60 | 20.36 |
| S4 | jitter | 8.67 | add_global | 30.05 | 21.38 |

分析：

在所有 severity 下，jitter 都是最困难 corruption。随着 severity 增大，best-worst gap 从 S0 的 12.32 扩大到 S4 的 21.38。

这说明 ULIP-2 在 ScanObjNN-C hardest 上的不同 corruption 类型之间差异很明显。尤其在高 severity 下，jitter 几乎成为单独的失败模式，而 add_global 仍然维持在约 30 的水平。

---

## 12. 低准确率区域分析

| 条件 | 数量 | 占比 | 主要涉及 corruption |
|---|---:|---:|---|
| Acc < 30 | 21 / 35 | 60.00% | jitter, dropout_local, dropout_global, rotate, add_local, scale |
| Acc < 28 | 14 / 35 | 40.00% | jitter, dropout_local, dropout_global, rotate, add_local |
| Acc < 25 | 10 / 35 | 28.57% | jitter, dropout_local, dropout_global, rotate |
| Acc < 22 | 7 / 35 | 20.00% | jitter, dropout_local_4, dropout_global_4 |
| Acc < 20 | 4 / 35 | 11.43% | jitter_1 到 jitter_4 |
| Acc < 18 | 4 / 35 | 11.43% | jitter_1 到 jitter_4 |
| Acc < 15 | 3 / 35 | 8.57% | jitter_2, jitter_3, jitter_4 |

分析：

14_1 的低准确率区域主要集中在 jitter。jitter_1 到 jitter_4 均低于 20，其中 jitter_2、jitter_3、jitter_4 均低于 15。

这说明 ULIP-2 在 ScanObjNN-C hardest 上的 Zero-shot 主要短板不是所有 corruption 均匀退化，而是对坐标扰动类 corruption 极其敏感。

---

## 13. 与前序实验的关系

14_1 的直接前序数据设置是 13_1，即 ULIP-2 在 ScanObjNN clean hardest 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | Accuracy |
|---|---|---|---:|
| 13_1_ulip2_scanobjnn_clean_hardest_zs | ScanObjNN clean hardest | Zero-shot | 34.07 |
| 14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs | ScanObjNN-C hardest all35 | Zero-shot | 26.46 all35 / 26.44 S2 |

当前结果说明：从 clean hardest 到 corrupted hardest，ULIP-2 Zero-shot 进一步下降。

| 比较 | 变化 |
|---|---:|
| 14_1 S2 Avg - 13_1 clean | -7.63 |
| 14_1 all35 Avg - 13_1 clean | -7.61 |

分析：

13_1 已经说明 ScanObjNN clean hardest 是困难域，14_1 进一步说明在该真实扫描困难域上引入 corruption 后，ULIP-2 Zero-shot 会继续明显退化。因此，14_1 是后续 14_2 Global Cache 和 14_3 Hierarchical Cache 的必要基础对照。

---

## 14. 阶段性结论

本实验完成了 ULIP-2 × ScanObjNN-C hardest all35 的 Zero-shot baseline 复现。

主要结论如下：

1. 14_1 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 26.44，原文 Supplementary Table 7 中 ULIP-2 Zero-shot Avg 为 26.37，差异仅 +0.07。
3. 当前 all35 Average 为 26.46，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果与原文 severity=2 数值高度对齐，可以认为 14_1 复现成功。
5. 相比 13_1 ScanObjNN clean hardest 的 34.07，14_1 的 all35 Average 下降到 26.46，说明 corruption 进一步加剧 ScanObjNN hardest 上的 domain shift。
6. jitter 是最困难 corruption，平均准确率只有 13.98。
7. jitter_4 是全部 35 个 setting 中最低结果，只有 8.67。
8. add_global、scale 和 add_local 相对稳定，其中 add_global 平均最高，为 30.76。
9. 该实验是 14_2 Global Cache 和 14_3 Hierarchical Cache 的基础对照，不在本文件中展开完整 14 组方法间对比。

---

## 15. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh 1

---

## 16. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f13 | sort | uniq -c
