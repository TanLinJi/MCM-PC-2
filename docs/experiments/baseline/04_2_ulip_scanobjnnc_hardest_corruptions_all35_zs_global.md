# 04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global

## 1. 实验目的

本实验用于复现 ULIP 在 ScanObjNN-C hardest 全部 35 个损坏设置上使用 Zero-shot + Global Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global |
| Backbone | ULIP |
| Dataset | ScanObjNN-C hardest |
| Method | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |

本实验是在 04_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存是否能够提升 ULIP 在 ScanObjNN-C hardest corrupted setting 上的鲁棒性。

本文件只记录 04_2 本身，并与前序子实验 04_1 进行对比。完整 04 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 04 组 summary 文档中。

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
| 实验编号 | 04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global |
| 方法脚本 | Point-Cache/scripts/baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/04_run_ulip_scanobjnnc_hardest_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global/ |

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
| Backbone | ULIP |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 数据目录 | data/sonn_c |
| 实际数据目录 | data/sonn_c/hardest |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 原始核心 runner | runners/model_with_global_cache.py |
| 优化 runner | runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py |
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
| GPU | 单张 Tesla T4，本次运行记录为 GPU 1 |

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

Point-Cache/results/baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_add_global_0_YYYYMMDD_HHMMSS.log

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
| severity=2 Average | 26.84 | 用于论文对齐 | 与原文 Supplementary Table 7 对比 |
| all35 Average | 26.60 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，04_2 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ScanObjNN-C hardest 的 severity=2 参考值进行对比。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 25.19 | 23.42 | 23.14 | 22.59 | 21.44 | 23.16 |
| add_local | 24.67 | 22.41 | 20.89 | 19.57 | 18.84 | 21.28 |
| dropout_global | 34.07 | 34.56 | 33.38 | 32.58 | 27.27 | 32.37 |
| dropout_local | 31.64 | 29.25 | 27.34 | 23.66 | 20.78 | 26.53 |
| rotate | 33.24 | 32.62 | 30.67 | 26.41 | 25.43 | 29.67 |
| scale | 31.05 | 29.18 | 28.83 | 26.93 | 28.56 | 28.91 |
| jitter | 31.51 | 26.02 | 23.63 | 21.44 | 18.81 | 24.28 |
| **Average** | **30.20** | **28.21** | **26.84** | **24.74** | **23.02** | **26.60** |

整体观察：

1. all35 Average 为 26.60，表示 ULIP 在 ScanObjNN-C hardest 全 35 个 corrupted setting 上加入 Global Cache 后的整体水平。
2. severity=2 Average 为 26.84，用于和原文 Supplementary Table 7 对齐。
3. dropout_global 的平均准确率最高，为 32.37。
4. add_local 的平均准确率最低，为 21.28。
5. jitter_4 和 add_local_4 分别为 18.81 和 18.84，是全 35 个 setting 中最低的结果区域。
6. 相比 04_1 Zero-shot，Global Cache 在所有 35 个 setting 上均带来正向提升。

---

## 8. 与原文结果对比

原文 Supplementary Table 7 报告的是 S-PB T50-RS-C severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 23.14 | 22.87 | +0.27 | 0.27 |
| add_local | 20.89 | 20.85 | +0.04 | 0.04 |
| dropout_global | 33.38 | 33.31 | +0.07 | 0.07 |
| dropout_local | 27.34 | 27.90 | -0.56 | 0.56 |
| rotate | 30.67 | 30.85 | -0.18 | 0.18 |
| scale | 28.83 | 28.63 | +0.20 | 0.20 |
| jitter | 23.63 | 24.53 | -0.90 | 0.90 |
| **Average** | **26.84** | **26.99** | **-0.15** | **0.32 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | -0.15 |
| MAE | 0.32 |
| RMSE | 0.43 |
| Max Abs Diff | 0.90 |

分析：

当前复现的 severity=2 Average 为 26.84，原文为 26.99，差异为 -0.15。单项最大绝对差异为 jitter 的 0.90，整体误差较小。

因此，04_2 不只是脚本跑通，而且数值也与原文 ULIP + Global Cache 在 S-PB T50-RS-C 上的结果基本对齐。

---

## 9. Severity 维度分析

### 9.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 30.20 | — | 0.00 |
| S1 | 28.21 | -1.98 | -1.98 |
| S2 | 26.84 | -1.37 | -3.36 |
| S3 | 24.74 | -2.10 | -5.46 |
| S4 | 23.02 | -1.72 | -7.18 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 30.20 下降到 23.02，总下降 7.18 个百分点。整体上，severity 越高，ULIP + Global Cache 的准确率越低，说明 corruption severity 仍然对模型造成持续影响。

与 04_1 Zero-shot 相比，04_2 在所有 severity 上整体上移，说明 Global Cache 并没有只改善某个特定 severity，而是在整个 severity 范围内都有效。

### 9.2 与 04_1 Zero-shot 的 severity 维度对比

| Severity | 04_1 Zero-shot Avg | 04_2 ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 26.86 | 30.20 | +3.33 |
| S1 | 25.10 | 28.21 | +3.11 |
| S2 | 23.91 | 26.84 | +2.93 |
| S3 | 22.11 | 24.74 | +2.63 |
| S4 | 20.24 | 23.02 | +2.78 |
| **all35** | **23.65** | **26.60** | **+2.95** |

分析：

Global Cache 在所有 severity 上都带来正向提升。当前 all35 平均提升为 +2.95，severity=2 提升为 +2.93。

原文 severity=2 下 Global Cache 增益为：

26.99 - 23.97 = +3.02

当前复现 severity=2 下 Global Cache 增益为：

26.84 - 23.91 = +2.93

二者差异为 -0.09，说明 Global Cache 的增益幅度与原文基本一致。

---

## 10. Corruption 难度分析

### 10.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | add_local | 21.28 | 平均最低，局部异常点仍然最困难 |
| 2 | add_global | 23.16 | 全局异常点仍然困难，但相比 Zero-shot 提升明显 |
| 3 | jitter | 24.28 | 高 severity 下仍然偏低 |
| 4 | dropout_local | 26.53 | 局部缺失仍有明显影响 |
| 5 | scale | 28.91 | 对尺度扰动相对稳定 |
| 6 | rotate | 29.67 | 平均表现较好 |
| 7 | dropout_global | 32.37 | 当前最容易 |

分析：

加入 Global Cache 后，最困难的 corruption 从 04_1 中的 add_global / add_local 变为 add_local。add_global 平均从 18.81 提升到 23.16，说明 Global Cache 对全局异常点有明显修正作用。

dropout_global 仍然是表现最好的 corruption，平均达到 32.37。jitter 虽然整体提升，但高 severity 下仍然偏低，说明强坐标扰动仍然是困难区域之一。

### 10.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 31.05 | 28.56 | 2.49 | 8.02% | 28.91 |
| add_global | 25.19 | 21.44 | 3.75 | 14.89% | 23.16 |
| add_local | 24.67 | 18.84 | 5.83 | 23.63% | 21.28 |
| dropout_global | 34.07 | 27.27 | 6.80 | 19.96% | 32.37 |
| rotate | 33.24 | 25.43 | 7.81 | 23.50% | 29.67 |
| dropout_local | 31.64 | 20.78 | 10.86 | 34.32% | 26.53 |
| jitter | 31.51 | 18.81 | 12.70 | 40.30% | 24.28 |

分析：

scale 仍然最稳定，从 S0 到 S4 只下降 2.49。jitter 退化最大，从 31.51 下降到 18.81，绝对下降 12.70，相对下降 40.30%。

dropout_local 也有明显退化，从 31.64 下降到 20.78。这说明在真实扫描 hardest split 上，局部结构缺失和坐标扰动仍然会显著影响 ULIP + Global Cache 的效果。

---

## 11. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | add_local | 24.67 | dropout_global | 34.07 | 9.40 |
| S1 | add_local | 22.41 | dropout_global | 34.56 | 12.15 |
| S2 | add_local | 20.89 | dropout_global | 33.38 | 12.49 |
| S3 | add_local | 19.57 | dropout_global | 32.58 | 13.01 |
| S4 | jitter | 18.81 | scale | 28.56 | 9.75 |

分析：

在 S0 到 S3 中，add_local 一直是最难 corruption；到 S4 时，jitter 成为最难 corruption。也就是说，低中 severity 下局部异常点最困难，而最高 severity 下坐标扰动成为主要失败点。

不同 corruption 之间的 best-worst gap 约为 9 到 13 个百分点，说明 Global Cache 提升整体准确率后，不同 corruption 类型之间的难度差异仍然存在。

---

## 12. 低准确率区域分析

### 12.1 低准确率 setting 数量

| 条件 | 04_1 Zero-shot 数量 | 04_2 ZS + Global 数量 | 减少数量 |
|---|---:|---:|---:|
| Acc < 30 | 30 / 35 | 25 / 35 | -5 |
| Acc < 28 | 26 / 35 | 21 / 35 | -5 |
| Acc < 25 | 19 / 35 | 14 / 35 | -5 |
| Acc < 22 | 16 / 35 | 7 / 35 | -9 |
| Acc < 20 | 10 / 35 | 3 / 35 | -7 |

分析：

加入 Global Cache 后，低准确率区域明显减少。尤其是 Acc < 22 的 setting 数量从 16 个减少到 7 个，Acc < 20 的 setting 数量从 10 个减少到 3 个。

这说明 Global Cache 不只是提高平均准确率，也显著减少了严重失败的 setting。

### 12.2 当前仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| add_local_4 | 18.84 | 高 severity 局部异常点仍然困难 |
| jitter_4 | 18.81 | 高 severity 坐标扰动仍然困难 |
| add_local_3 | 19.57 | 中高 severity 局部异常点仍然偏低 |

分析：

04_2 中低于 20 的 setting 只剩 3 个，集中在 add_local 和 jitter。这说明 Global Cache 对大部分严重失败区域有改善，但对局部异常点和强坐标扰动仍然有限。

---

## 13. Global Cache 相比 04_1 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +5.10 | +4.44 | +3.85 | +4.76 | +3.57 | +4.34 |
| add_local | +3.43 | +1.83 | +2.29 | +1.87 | +1.66 | +2.22 |
| dropout_global | +3.33 | +3.89 | +2.46 | +2.29 | +1.91 | +2.78 |
| dropout_local | +2.98 | +3.96 | +3.43 | +3.26 | +3.40 | +3.41 |
| rotate | +2.67 | +3.99 | +3.54 | +2.88 | +4.13 | +3.44 |
| scale | +2.53 | +1.66 | +2.77 | +1.11 | +1.84 | +1.98 |
| jitter | +3.30 | +1.97 | +2.15 | +2.22 | +2.95 | +2.52 |
| **Average** | **+3.33** | **+3.11** | **+2.93** | **+2.63** | **+2.78** | **+2.95** |

分析：

Global Cache 在全部 35 个 setting 上均带来正向提升，没有出现负增益。

平均提升最大的 corruption 是 add_global，Avg Gain 为 +4.34。说明全局缓存对全局异常点有明显修正作用。dropout_local 和 rotate 的提升也较明显，分别为 +3.41 和 +3.44。

提升最小的是 scale，Avg Gain 为 +1.98。scale 在 Zero-shot 下已经相对稳定，因此 Global Cache 的额外提升空间较小。

---

## 14. 与前序实验的关系

04_2 的直接前序子实验是 04_1，即 ULIP 在 ScanObjNN-C hardest all35 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs | ScanObjNN-C hardest all35 | Zero-shot | 23.91 | 23.65 |
| 04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global | ScanObjNN-C hardest all35 | ZS + Global Cache | 26.84 | 26.60 |

当前结果说明：在 ScanObjNN-C hardest all35 上，Global Cache 能稳定提升 Zero-shot。

| 比较 | 变化 |
|---|---:|
| 04_2 S2 Avg - 04_1 S2 Avg | +2.93 |
| 04_2 all35 Avg - 04_1 all35 Avg | +2.95 |

分析：

04_2 相比 04_1 的提升与原文 Global Cache 增益基本一致。原文 severity=2 下 Global Cache 增益为 +3.02，当前复现 severity=2 增益为 +2.93，差异仅 -0.09。

因此，04_2 不仅数值对齐原文，而且相对 04_1 的模块增益也对齐原文。

---

## 15. 阶段性结论

本实验完成了 ULIP × ScanObjNN-C hardest all35 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 04_2 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 26.84，原文 Supplementary Table 7 中 ULIP + Global Cache Avg 为 26.99，差异为 -0.15。
3. 当前 all35 Average 为 26.60，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果与原文 severity=2 数值基本对齐，可以认为 04_2 复现成功。
5. 相比 04_1 Zero-shot，04_2 的 severity=2 Average 提升 +2.93，all35 Average 提升 +2.95。
6. 当前 Global Cache 增益与原文 severity=2 增益 +3.02 基本一致。
7. Global Cache 在全部 35 个 setting 上均带来正向提升，没有出现负增益。
8. Global Cache 对 add_global 的平均提升最大，为 +4.34。
9. 低准确率区域明显减少，Acc < 20 的 setting 从 10 个减少到 3 个。
10. add_local 和 jitter 仍然是 Global Cache 后的主要困难区域。
11. 本实验是 04_3 分析 Local Cache 额外贡献的直接对照，不在本文件中展开完整 04 组方法间对比。

---

## 16. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh 1

---

## 17. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f13 | sort | uniq -c
