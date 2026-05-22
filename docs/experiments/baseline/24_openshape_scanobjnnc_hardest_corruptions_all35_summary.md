# 24_openshape_scanobjnnc_hardest_corruptions_all35_summary

## 1. 实验组目的

本总文档汇总 OpenShape 在 ScanObjNN-C hardest 全部 35 个损坏设置上的三组 baseline 复现实验。

24 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | OpenShape |
| Dataset | ScanObjNN-C hardest |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 数据目录 | data/sonn_c/hardest |
| Corruption 数量 | 7 |
| Severity 数量 | 5 |
| 总测试设置数 | 7 × 5 = 35 |
| 输入点数 | 1024 |

三组实验分别为：

| 实验编号 | 方法 | 作用 |
|---|---|---|
| 24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs | Zero-shot | 无缓存基础对照 |
| 24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global | Zero-shot + Global Cache | 验证全局缓存增益 |
| 24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 及 Local Cache 额外影响 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| OpenShape 在 ScanObjNN-C hardest 上的 Zero-shot 鲁棒性是多少？ | 由 24_1 给出 |
| Global Cache 是否有效？ | 比较 24_2 - 24_1 |
| Local Cache 是否有额外贡献？ | 比较 24_3 - 24_2 |
| 完整 Point-Cache 是否与原文对齐？ | 比较 24_3 与原文 Table 7 |
| ScanObjNN-C hardest 和 ModelNet-C 的难度差异有多大？ | 与 22 组比较 |
| corrupted hardest 相比 clean hardest 下降多少？ | 与 23 组比较 |
| 后续 MCM-PC 应重点关注哪些失败模式？ | 分析 corruption × severity 结果矩阵 |

需要特别注意：原文 Point-Cache Table 7 只报告 severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 用于和原文 Point-Cache Table 7 对齐 |
| all35 Average | 本复现实验扩展统计，表示 35 个 corrupted setting 的总体平均 |

---

## 2. 当前实现方式

24 组属于 all35 corruption 实验，因此使用优化版 Python runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/24_run_openshape_scanobjnnc_hardest_corruptions_all35_common.sh |
| 优化 runner | Point-Cache/runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |
| OpenShape 权重 | Point-Cache/weights/openshape/openshape-pointbert-vitg14-rgb/model.pt |
| OpenCLIP 权重 | Point-Cache/weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin |

优化方式如下：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 | 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 | 模型只加载一次，每个 cor_type 重新创建 DataLoader、重新初始化 cache |
| bash 通过 tee 生成单个 cor_type 的 log | Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv | summary.csv 的列结构保持不变 |

该优化只改变执行效率，不改变实验定义。每个 cor_type 仍然单独记录 log，并且每个方法的 summary.csv 仍然保持 35 行。

---

## 3. 三个实验的输出完整性检查

| 实验编号 | 方法 | summary 行数 | cor_type 唯一数 | log_path 唯一数 | logs 文件数 | status | 状态 |
|---|---|---:|---:|---:|---:|---|---|
| 24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs | Zero-shot | 35 | 35 | 35 | 35 | 35 done | 完成 |
| 24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global | ZS + Global | 35 | 35 | 35 | 35 | 35 done | 完成 |
| 24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local | ZS + Global + Local | 35 | 35 | 35 | 35 | 35 done | 完成 |

说明：

1. 三个子实验均完成 35 个 cor_type。
2. 三个子实验均有 35 个唯一 log_path。
3. 三个子实验的 logs 文件数均为 35，没有重复日志或旧日志残留。
4. 三个子实验的 status 均为 35 个 done。
5. 执行完整性正常并不等于结果正常；结果是否正常需要和原文 severity=2 参考值对比。

---

## 4. 核心结果总表

| 方法 | S0 Avg | S1 Avg | S2 Avg | S3 Avg | S4 Avg | all35 Avg |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 38.82 | 35.49 | 32.75 | 30.12 | 26.43 | 32.72 |
| ZS + Global | 42.82 | 39.44 | 37.30 | 33.35 | 30.65 | 36.71 |
| ZS + Global + Local | 44.11 | 40.09 | 38.63 | 34.51 | 31.85 | 37.84 |

核心观察：

1. Global Cache 将 all35 Avg 从 32.72 提升到 36.71，提升 +3.99。
2. Local Cache 在 Global Cache 基础上将 all35 Avg 从 36.71 提升到 37.84，额外提升 +1.13。
3. severity=2 上，Global Cache 将 Avg 从 32.75 提升到 37.30，提升 +4.55。
4. severity=2 上，Local Cache 在 Global Cache 基础上从 37.30 提升到 38.63，额外提升 +1.33。
5. 因此，24 组与 22 组不同：24 组中不仅 Global Cache 明确有效，Local Cache 也有明确额外贡献。

---

## 5. 与原文 Point-Cache Table 7 的 severity=2 对齐

原文 Point-Cache Table 7 报告的是 ScanObjNN-C hardest / S-PB T50-RS-C 在 severity level = 2 下 7 种 corruption 的结果。因此，这里只取 S2 列与原文对齐。

| 方法 | 当前复现 S2 Avg | 原文 S2 Avg | Diff |
|---|---:|---:|---:|
| Zero-shot | 32.75 | 31.98 | +0.77 |
| ZS + Global | 37.30 | 36.80 | +0.50 |
| ZS + Global + Local | 38.63 | 37.70 | +0.93 |

补充统计：

| 方法 | Mean Diff | MAE | RMSE | Max Abs Diff |
|---|---:|---:|---:|---:|
| Zero-shot | +0.77 | 1.58 | 1.96 | 2.91 |
| ZS + Global | +0.50 | 1.21 | 1.26 | 1.70 |
| ZS + Global + Local | +0.93 | 0.93 | 1.44 | 3.44 |

分析：

24_1、24_2、24_3 的 severity=2 平均值均略高于原文，但总体处于可接受范围内。24_3 的最终完整 Point-Cache 比原文高 +0.93，主要正偏差来自 scale 项，当前 scale_2 比原文高 +3.44。

因此，24 组总体可以认为是有效复现结果，但文档中应明确记录：当前 24 组整体略高于原文，尤其 24_3 的 scale 项偏高。

---

## 6. 原文增益与当前复现增益对比

| 增益来源 | 原文 S2 增益 | 当前 S2 增益 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | 36.80 - 31.98 = +4.82 | 37.30 - 32.75 = +4.55 | -0.27 |
| Local Cache extra over Global | 37.70 - 36.80 = +0.90 | 38.63 - 37.30 = +1.33 | +0.43 |
| Full Point-Cache over Zero-shot | 37.70 - 31.98 = +5.72 | 38.63 - 32.75 = +5.88 | +0.16 |

分析：

Global Cache 增益复现得较好。原文 severity=2 下 Global Cache 增益为 +4.82，当前为 +4.55，差异为 -0.27。

Local Cache 的额外贡献当前为 +1.33，高于原文 +0.90。完整 Point-Cache 相比 Zero-shot 的当前总提升为 +5.88，与原文 +5.72 非常接近，甚至略高 +0.16。

这说明 24 组的整体方法增益结构与原文一致：Global Cache 是主提升来源，Local Cache 在 Global Cache 基础上继续带来额外收益。

---

## 7. Severity 维度增益分析

### 7.1 Global Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 38.82 | 42.82 | +4.00 |
| S1 | 35.49 | 39.44 | +3.95 |
| S2 | 32.75 | 37.30 | +4.55 |
| S3 | 30.12 | 33.35 | +3.23 |
| S4 | 26.43 | 30.65 | +4.21 |
| **all35** | **32.72** | **36.71** | **+3.99** |

分析：

Global Cache 在所有 severity 上都带来正向提升。提升幅度在 S2 最大，为 +4.55；all35 平均提升为 +3.99。

这说明在 ScanObjNN-C hardest 中，Global Cache 不是只对某一个 severity 有效，而是在全 severity 范围内稳定有效。

### 7.2 Local Cache 相比 Global Cache

| Severity | ZS + Global Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 42.82 | 44.11 | +1.29 |
| S1 | 39.44 | 40.09 | +0.65 |
| S2 | 37.30 | 38.63 | +1.33 |
| S3 | 33.35 | 34.51 | +1.16 |
| S4 | 30.65 | 31.85 | +1.20 |
| **all35** | **36.71** | **37.84** | **+1.13** |

分析：

Local Cache 在 Global Cache 基础上的额外贡献在所有 severity 上均为正。S2 额外提升 +1.33，S4 额外提升 +1.20，all35 平均额外提升 +1.13。

这说明 24 组中的 Local Cache 是稳定正贡献模块，不是可忽略模块。

### 7.3 完整 Point-Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 38.82 | 44.11 | +5.29 |
| S1 | 35.49 | 40.09 | +4.60 |
| S2 | 32.75 | 38.63 | +5.88 |
| S3 | 30.12 | 34.51 | +4.39 |
| S4 | 26.43 | 31.85 | +5.42 |
| **all35** | **32.72** | **37.84** | **+5.11** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有明确正增益。all35 平均提升 +5.11，severity=2 提升 +5.88。说明完整 Point-Cache 能显著改善 OpenShape 在 ScanObjNN-C hardest 上的鲁棒性。

---

## 8. Corruption 维度结果对比

### 8.1 三种方法的 corruption 平均准确率

| Corruption | Zero-shot Avg | ZS + Global Avg | ZS + Global + Local Avg |
|---|---:|---:|---:|
| add_global | 35.49 | 40.78 | 42.63 |
| add_local | 33.66 | 37.89 | 39.90 |
| dropout_global | 36.88 | 39.61 | 40.09 |
| dropout_local | 27.53 | 32.59 | 34.08 |
| rotate | 36.91 | 40.69 | 40.67 |
| scale | 35.83 | 38.78 | 39.49 |
| jitter | 22.76 | 26.66 | 28.01 |

观察：

1. jitter 在三种方法中始终最低，是 OpenShape × ScanObjNN-C hardest 的主要困难 corruption。
2. dropout_local 是第二困难 corruption。
3. Global Cache 提升了所有 corruption 的平均准确率。
4. Local Cache 除 rotate 基本持平外，对其他 corruption 均有正向贡献。
5. 完整 Point-Cache 后，add_global 的平均准确率最高，为 42.63。

### 8.2 corruption 维度总提升

| Corruption | Global - ZS | Global + Local - Global | Global + Local - ZS |
|---|---:|---:|---:|
| add_global | +5.29 | +1.85 | +7.14 |
| add_local | +4.23 | +2.01 | +6.24 |
| dropout_global | +2.73 | +0.48 | +3.21 |
| dropout_local | +5.06 | +1.49 | +6.55 |
| rotate | +3.77 | -0.01 | +3.76 |
| scale | +2.94 | +0.72 | +3.66 |
| jitter | +3.89 | +1.35 | +5.25 |
| **Average** | **+3.99** | **+1.13** | **+5.11** |

分析：

Global Cache 对 add_global、dropout_local、add_local 和 jitter 都有明显提升。Local Cache 的额外贡献在 add_local 上最大，为 +2.01；在 add_global 上为 +1.85；在 dropout_local 上为 +1.49；在 jitter 上为 +1.35。

rotate 是唯一 Local Cache 几乎不提升的 corruption，Local extra 为 -0.01，基本可以视为持平。

因此，24 组说明 Local Cache 对真实扫描 corrupted setting 中的局部异常点、局部缺失和坐标扰动都有价值。

---

## 9. Corruption 难度排序

### 9.1 Zero-shot 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 22.76 |
| 2 | dropout_local | 27.53 |
| 3 | add_local | 33.66 |
| 4 | add_global | 35.49 |
| 5 | scale | 35.83 |
| 6 | dropout_global | 36.88 |
| 7 | rotate | 36.91 |

### 9.2 ZS + Global 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 26.66 |
| 2 | dropout_local | 32.59 |
| 3 | add_local | 37.89 |
| 4 | scale | 38.78 |
| 5 | dropout_global | 39.61 |
| 6 | rotate | 40.69 |
| 7 | add_global | 40.78 |

### 9.3 ZS + Global + Local 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 28.01 |
| 2 | dropout_local | 34.08 |
| 3 | scale | 39.49 |
| 4 | add_local | 39.90 |
| 5 | dropout_global | 40.09 |
| 6 | rotate | 40.67 |
| 7 | add_global | 42.63 |

综合分析：

三种方法下，最困难 corruption 始终是 jitter。完整 Point-Cache 能把 jitter 平均值从 22.76 提升到 28.01，但该数值仍然远低于其他 corruption。

dropout_local 始终是第二困难 corruption，完整 Point-Cache 后仍只有 34.08。这说明 high-severity 坐标扰动和局部缺失是后续 MCM-PC 方法需要重点处理的失败模式。

---

## 10. 低准确率区域分析

| 条件 | Zero-shot 数量 | ZS + Global 数量 | ZS + Global + Local 数量 |
|---|---:|---:|---:|
| Acc < 40 | 30 / 35 | 22 / 35 | 15 / 35 |
| Acc < 35 | 19 / 35 | 9 / 35 | 8 / 35 |
| Acc < 30 | 8 / 35 | 6 / 35 | 6 / 35 |
| Acc < 25 | 6 / 35 | 4 / 35 | 4 / 35 |
| Acc < 20 | 4 / 35 | 2 / 35 | 2 / 35 |
| Acc < 15 | 1 / 35 | 0 / 35 | 0 / 35 |

分析：

Global Cache 明显减少了低准确率区域。Acc < 40 的 setting 从 30 个减少到 22 个，Acc < 35 的 setting 从 19 个减少到 9 个。

Local Cache 进一步减少了 Acc < 40 的 setting，从 22 个降到 15 个。说明 Local Cache 对中低准确率区域有明显改善。

不过，对于极困难区域，Local Cache 没有进一步减少 Acc < 30、Acc < 25 和 Acc < 20 的数量。也就是说，Local Cache 提高了整体表现，但最困难的 high-severity jitter / dropout 仍然没有完全解决。

---

## 11. 关键困难 setting

| cor_type | Zero-shot | ZS + Global | ZS + Global + Local | 现象 |
|---|---:|---:|---:|---|
| jitter_4 | 13.43 | 17.18 | 16.90 | 最高 severity 坐标扰动仍然最低 |
| jitter_3 | 15.27 | 18.53 | 18.91 | 中高 severity 坐标扰动仍然极低 |
| dropout_local_4 | 19.47 | 22.62 | 24.46 | 高 severity 局部缺失仍然困难 |
| jitter_2 | 19.78 | 23.04 | 24.91 | severity=2 jitter 明显低于多数 corruption |
| dropout_global_4 | 24.08 | 27.13 | 28.07 | 高 severity 全局缺失仍然困难 |
| dropout_local_3 | 23.21 | 28.04 | 29.35 | 中高 severity 局部缺失仍然困难 |

分析：

jitter_3 和 jitter_4 是最顽固的失败点。即使完整 Point-Cache 后，jitter_4 仍只有 16.90，jitter_3 仍只有 18.91。

dropout_local_4、dropout_local_3 和 dropout_global_4 也是重要困难 setting。后续方法如果希望进一步提升 24 组结果，需要重点解决 high-severity jitter 和 high-severity dropout。

---

## 12. 方法贡献分解

以 all35 Average 为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

37.84 - 32.72 = +5.11

其中：

| 贡献来源 | all35 Avg 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +3.99 | 约 78.1% |
| Local Cache | +1.13 | 约 21.9% |
| 完整 Point-Cache | +5.11 | 100.00% |

以 severity=2 Average 为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

38.63 - 32.75 = +5.88

其中：

| 贡献来源 | S2 Avg 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +4.55 | 约 77.4% |
| Local Cache | +1.33 | 约 22.6% |
| 完整 Point-Cache | +5.88 | 100.00% |

分析：

不管按 all35 还是按 severity=2，Global Cache 都是主要贡献来源，但 Local Cache 也有明确额外贡献，占完整提升约 22%。

这与 22 组 OpenShape × ModelNet-C 不同。22 组中 Local Cache 几乎无额外贡献；24 组中 Local Cache 明确有效，说明真实扫描 corrupted hardest setting 更需要局部缓存信息。

---

## 13. 与 23 组 ScanObjNN clean hardest 的关系

23 组是 OpenShape 在 ScanObjNN clean hardest 上的结果；24 组是 OpenShape 在 ScanObjNN-C hardest all35 上的结果。

| 方法 | 23 组 clean hardest | 24 组 S2 Avg | 24 组 all35 Avg |
|---|---:|---:|---:|
| Zero-shot | 41.88 | 32.75 | 32.72 |
| ZS + Global | 41.95 | 37.30 | 36.71 |
| ZS + Global + Local | 43.82 | 38.63 | 37.84 |

从 clean hardest 到 corrupted hardest 的下降：

| 方法 | S2 Avg - clean | all35 Avg - clean |
|---|---:|---:|
| Zero-shot | -9.13 | -9.16 |
| ZS + Global | -4.65 | -5.24 |
| ZS + Global + Local | -5.19 | -5.98 |

分析：

ScanObjNN-C corruption 会在 ScanObjNN clean hardest 的基础上进一步降低 OpenShape 性能。Zero-shot 从 41.88 下降到 all35 Avg 32.72，下降 -9.16。

Global Cache 和完整 Point-Cache 都缩小了 clean-to-corruption gap。Global Cache 将 all35 gap 缩小到 -5.24，完整 Point-Cache 的 all35 gap 为 -5.98。

需要注意：完整 Point-Cache 的 corrupted accuracy 最高，但因为 23_3 clean hardest 本身也更高，所以 gap 不一定最小。单纯看 corrupted setting 的最终准确率，24_3 最好。

---

## 14. 与 22 组 ModelNet-C 的关系

22 组是 OpenShape 在 ModelNet-C all35 上的结果；24 组是 OpenShape 在 ScanObjNN-C hardest all35 上的结果。

| 方法 | 22 组 ModelNet-C all35 Avg | 24 组 ScanObjNN-C hardest all35 Avg | 变化 |
|---|---:|---:|---:|
| Zero-shot | 72.57 | 32.72 | -39.85 |
| ZS + Global | 75.14 | 36.71 | -38.43 |
| ZS + Global + Local | 75.14 | 37.84 | -37.30 |

如果用 severity=2 Average 对比：

| 方法 | 22 组 ModelNet-C S2 Avg | 24 组 ScanObjNN-C hardest S2 Avg | 变化 |
|---|---:|---:|---:|
| Zero-shot | 73.57 | 32.75 | -40.82 |
| ZS + Global | 76.46 | 37.30 | -39.16 |
| ZS + Global + Local | 76.33 | 38.63 | -37.70 |

分析：

ScanObjNN-C hardest 远难于 ModelNet-C。即使使用完整 Point-Cache，OpenShape 在 ModelNet-C all35 上为 75.14，但在 ScanObjNN-C hardest all35 上只有 37.84，低 -37.30。

这说明真实扫描 corrupted hardest 的难度远高于 synthetic ModelNet corruption。24 组是当前 baseline 复现中最关键的高难度鲁棒性设置之一。

---

## 15. 与 04 / 14 组 ScanObjNN-C hardest 的关系

04 组是 ULIP 在 ScanObjNN-C hardest all35 上的结果；14 组是 ULIP-2 在同一数据设置上的结果；24 组是 OpenShape 在同一数据设置上的结果。

| Backbone | 方法 | S2 Avg | all35 Avg |
|---|---|---:|---:|
| ULIP | Zero-shot | 23.91 | 23.65 |
| ULIP | ZS + Global | 26.84 | 26.60 |
| ULIP | ZS + Global + Local | 27.94 | 27.41 |
| ULIP-2 | Zero-shot | 26.44 | 26.46 |
| ULIP-2 | ZS + Global | 31.38 | 31.24 |
| ULIP-2 | ZS + Global + Local | 33.60 | 33.25 |
| OpenShape | Zero-shot | 32.75 | 32.72 |
| OpenShape | ZS + Global | 37.30 | 36.71 |
| OpenShape | ZS + Global + Local | 38.63 | 37.84 |

分析：

OpenShape 在 ScanObjNN-C hardest 上明显强于 ULIP 和 ULIP-2。完整 Point-Cache 下，OpenShape all35 Avg 为 37.84，比 ULIP-2 的 33.25 高 +4.59，比 ULIP 的 27.41 高 +10.43。

这说明 OpenShape backbone 在真实扫描 corrupted hardest setting 上仍然有明显优势。但 OpenShape 完整 Point-Cache 的绝对准确率仍然只有 37.84，说明该设置仍然非常困难，后续 MCM-PC 仍有较大改进空间。

---

## 16. 与 23 组 Local Cache 现象的关系

23 组 OpenShape × ScanObjNN clean hardest 中，Local Cache 在 Global Cache 基础上额外提升 +1.87。24 组 OpenShape × ScanObjNN-C hardest all35 中，Local Cache 在 Global Cache 基础上额外提升 +1.13 all35 / +1.33 S2。

| 组别 | 数据设置 | Global extra | Local extra |
|---|---|---:|---:|
| 23 组 | OpenShape × ScanObjNN clean hardest | +0.07 | +1.87 |
| 24 组 | OpenShape × ScanObjNN-C hardest all35 | +3.99 all35 | +1.13 all35 |

分析：

23 组中 Global Cache 单独增益很弱，但 Local Cache 明显有效。24 组中 Global Cache 和 Local Cache 都有效，其中 Global Cache 是主提升来源，Local Cache 提供额外提升。

这说明真实扫描数据中的 Local Cache 价值比较稳定。不论是 clean hardest 还是 corrupted hardest，Local Cache 都能提供额外帮助。区别在于：当叠加 corruption 后，Global Cache 的补偿作用也明显增强。

---

## 17. 与 22 组 Local Cache 现象的差异

22 组 OpenShape × ModelNet-C all35 中，Local Cache 在 Global Cache 基础上的额外贡献几乎为零：

| 组别 | 数据设置 | Local Cache extra |
|---|---|---:|
| 22 组 | OpenShape × ModelNet-C all35 | +0.01 all35 / -0.13 S2 |
| 24 组 | OpenShape × ScanObjNN-C hardest all35 | +1.13 all35 / +1.33 S2 |

分析：

这说明 Local Cache 的价值具有数据依赖性。

在 ModelNet-C 上，OpenShape 的全局特征本身已经较强，Global Cache 已经提供了主要鲁棒性补偿，Local Cache 边际收益很弱。

在 ScanObjNN-C hardest 上，真实扫描数据存在局部遮挡、背景残留、缺失和不完整结构，再叠加 corruption 后局部结构更不稳定，Local Cache 的局部补偿作用更明显。

这对后续 MCM-PC 很重要：Local Cache 不应简单固定启用或固定权重，而应根据数据类型、样本可靠性和全局-局部一致性动态调节。

---

## 18. 当前结果意义分析

24 组结果说明：

| 观察 | 解释 |
|---|---|
| Zero-shot all35 = 32.72 | OpenShape 在 ScanObjNN-C hardest 上的基础鲁棒性 |
| ZS + Global all35 = 36.71 | Global Cache 有明确正增益 |
| ZS + Global + Local all35 = 37.84 | 完整 Point-Cache 最好 |
| Global extra = +3.99 all35 | Global Cache 是主要提升来源 |
| Local extra = +1.13 all35 | Local Cache 也有明确额外贡献 |
| jitter 仍然最低 | 强坐标扰动仍然最困难 |
| dropout_local 第二困难 | 局部缺失仍然是重要失败模式 |
| 24 组远低于 22 组 | 真实扫描 corrupted hardest 远难于 synthetic corruption |

24 组是一个非常关键的实验组。它说明 OpenShape 在真实扫描 corrupted hardest 上仍然存在明显鲁棒性问题，而完整 Point-Cache 可以显著提升性能。

---

## 19. 对后续 MCM-PC 的启发

当前 24 组实验对后续方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| ScanObjNN-C hardest 是当前最困难设置之一 | 后续方法必须在真实扫描 corrupted setting 上验证 |
| OpenShape Zero-shot 仍只有 32.72 all35 | 强 backbone 仍有明显提升空间 |
| Global Cache 提供 +3.99 all35 | 全局缓存是稳定主模块，应保留 |
| Local Cache 提供 +1.13 all35 | 局部缓存也有明确价值 |
| jitter 仍然最难 | 坐标扰动需要专门鲁棒机制 |
| dropout_local 仍然困难 | 局部缺失需要更可靠的局部证据建模 |
| 22 组 Local 弱，24 组 Local 强 | Local Cache 价值依赖数据类型和域偏移类型 |
| 24 组远难于 22 组 | 方法不能只依赖 ModelNet-C 结果证明鲁棒性 |

这对 MCM-PC 很重要：后续方法不应简单固定 Global / Local 的作用，而应根据样本可靠性、全局-局部一致性、伪标签可信度和域偏移类型动态调节缓存贡献。

对于 24 组这种真实扫描 corrupted hardest setting，Global Cache 和 Local Cache 都有价值。后续 MCM-PC 可以围绕如何保留全局缓存稳定增益、如何选择可靠局部证据、如何抑制 high-severity jitter / dropout 下的错误伪标签展开。

---

## 20. 阶段性结论

24 组 OpenShape × ScanObjNN-C hardest all35 baseline 已完成。

主要结论如下：

1. 三个子实验均完成，并且 summary.csv、cor_type、log_path 和 logs 文件数量均为 35。
2. 24_1 Zero-shot 的 severity=2 Avg 为 32.75，原文为 31.98，差异 +0.77。
3. 24_2 ZS + Global 的 severity=2 Avg 为 37.30，原文为 36.80，差异 +0.50。
4. 24_3 ZS + Global + Local 的 severity=2 Avg 为 38.63，原文为 37.70，差异 +0.93。
5. 三个结果整体略高于原文，但均在可接受范围内。
6. all35 Avg 从 Zero-shot 的 32.72 提升到 Global 的 36.71，再提升到 Global + Local 的 37.84。
7. Global Cache 是主要提升来源，all35 Avg 提升 +3.99，severity=2 Avg 提升 +4.55。
8. Local Cache 在 Global Cache 基础上有明确额外贡献，all35 Avg 提升 +1.13，severity=2 Avg 提升 +1.33。
9. 完整 Point-Cache 相比 Zero-shot 提升明显，all35 Avg 提升 +5.11，severity=2 Avg 提升 +5.88。
10. jitter 是最困难 corruption，完整 Point-Cache 后 all35 平均仍只有 28.01。
11. jitter_3 和 jitter_4 是最顽固失败点，完整 Point-Cache 后分别为 18.91 和 16.90。
12. dropout_local 是第二困难 corruption，完整 Point-Cache 后 all35 平均为 34.08。
13. OpenShape 在 ScanObjNN-C hardest 上明显强于 ULIP 和 ULIP-2，但绝对准确率仍然较低。
14. 24 组结果说明：在真实扫描 corrupted hardest setting 中，Global Cache 和 Local Cache 都有价值。
15. 24 组完成了 C 组 OpenShape 的第四个数据设置 baseline 复现。

---

## 21. 运行命令汇总

24_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh 1

24_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh 1

24_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh 0

---

## 22. 检查命令汇总

24_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/24_1_openshape_scanobjnnc_hardest_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

24_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

24_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
