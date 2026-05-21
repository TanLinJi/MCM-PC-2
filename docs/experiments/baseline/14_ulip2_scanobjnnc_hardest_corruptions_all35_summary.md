# 14_ulip2_scanobjnnc_hardest_corruptions_all35_summary

## 1. 实验组目的

本总文档汇总 ULIP-2 在 ScanObjNN-C hardest 全部 35 个损坏设置上的三组 baseline 复现实验。

14 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | ULIP-2 |
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
| 14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs | Zero-shot | 无缓存基础对照 |
| 14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global | Zero-shot + Global Cache | 验证全局缓存增益 |
| 14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 增益 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| ULIP-2 在 ScanObjNN-C hardest 上的 Zero-shot 鲁棒性是多少？ | 由 14_1 给出 |
| Global Cache 是否有效？ | 比较 14_2 - 14_1 |
| Local Cache 是否有额外贡献？ | 比较 14_3 - 14_2 |
| 三个结果是否与原文对齐？ | 分别与原文 Supplementary Table 7 的 severity=2 结果对比 |
| ScanObjNN-C hardest 的主要困难 corruption 是哪些？ | 分析 corruption × severity 结果矩阵 |
| 后续 MCM-PC 应重点关注哪些问题？ | 从 Global / Local 的收益和失败区域中寻找方法改进方向 |

需要特别注意：原文 Supplementary Table 7 只报告 severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 用于和原文 Supplementary Table 7 对齐 |
| all35 Average | 本复现实验扩展统计，表示 35 个 corrupted setting 的总体平均 |

---

## 2. 当前实现方式

14 组属于 all35 corruption 实验，因此使用优化版 Python runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/14_run_ulip2_scanobjnnc_hardest_corruptions_all35_common.sh |
| 优化 runner | Point-Cache/runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |
| Backbone 权重 | Point-Cache/weights/ulip/pointbert_ulip2.pt |
| 文本编码器权重 | Point-Cache/weights/ulip/slip_base_100ep.pt |

优化方式如下：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 |
| 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 |
| 模型只加载一次，每个 cor_type 重新创建 DataLoader、重新初始化 cache |
| bash 通过 tee 生成单个 cor_type 的 log |
| Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv |
| summary.csv 的列结构保持不变 |

该优化只改变执行效率，不改变实验定义。每个 cor_type 仍然单独记录 log，并且每个方法的 summary.csv 仍然保持 35 行。

---

## 3. 三个实验的输出完整性检查

| 实验编号 | 方法 | summary 行数 | cor_type 唯一数 | log_path 唯一数 | logs 文件数 | status | 状态 |
|---|---|---:|---:|---:|---:|---|---|
| 14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs | Zero-shot | 35 | 35 | 35 | 35 | 35 done | 完成 |
| 14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global | ZS + Global | 35 | 35 | 35 | 35 | 35 done | 完成 |
| 14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local | ZS + Global + Local | 35 | 35 | 35 | 35 | 35 done | 完成 |

说明：

1. 三个子实验均完成 35 个 cor_type。
2. 三个子实验均有 35 个唯一 log_path。
3. 三个子实验的 logs 文件数均为 35，没有重复日志或旧日志残留。
4. 执行完整性正常并不等于结果正常；结果是否正常需要和原文 severity=2 参考值对比。

---

## 4. 核心结果总表

| 方法 | S0 Avg | S1 Avg | S2 Avg | S3 Avg | S4 Avg | all35 Avg |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 30.21 | 28.35 | 26.44 | 24.34 | 22.94 | 26.46 |
| ZS + Global | 36.03 | 33.45 | 31.38 | 28.78 | 26.55 | 31.24 |
| ZS + Global + Local | 38.31 | 34.90 | 33.36 | 30.23 | 27.96 | 32.95 |

核心观察：

1. 三种方法在所有 severity 上都呈现稳定递进关系：Zero-shot < ZS + Global < ZS + Global + Local。
2. Global Cache 将 all35 Avg 从 26.46 提升到 31.24，提升 +4.78。
3. Local Cache 在 Global Cache 基础上进一步提升到 32.95，额外提升 +1.72。
4. 完整 Point-Cache 相比 Zero-shot 的 all35 Avg 总提升为 +6.50。
5. severity=2 Avg 从 26.44 提升到 33.36，总提升为 +6.92。

---

## 5. 与原文 Supplementary Table 7 的 severity=2 对齐

原文 Supplementary Table 7 报告的是 S-PB T50-RS-C corrupted setting，也就是 ScanObjectNN hardest split corrupted version，且结果对应 severity level = 2。因此，这里只取 S2 列与原文对齐。

| 方法 | 当前复现 S2 Avg | 原文 S2 Avg | Diff |
|---|---:|---:|---:|
| Zero-shot | 26.44 | 26.37 | +0.07 |
| ZS + Global | 31.38 | 31.93 | -0.55 |
| ZS + Global + Local | 33.36 | 33.36 | +0.00 |

补充统计：

| 方法 | Mean Diff | MAE | RMSE | Max Abs Diff |
|---|---:|---:|---:|---:|
| Zero-shot | +0.07 | 0.22 | 0.29 | 0.56 |
| ZS + Global | -0.55 | 0.76 | 0.91 | 1.64 |
| ZS + Global + Local | +0.00 | 1.16 | 1.33 | 2.12 |

分析：

14_1 与原文高度对齐，severity=2 Average 只高 +0.07。14_2 的 severity=2 Average 比原文低 -0.55，整体仍可接受，但相比前面实验组误差更明显，主要来自 scale、jitter 和 dropout_global 偏低。14_3 的 severity=2 Average 与原文完全一致，但逐 corruption 存在一定抵消：add_local 和 dropout_local 偏高，scale 和 jitter 偏低。

因此，14 组总体可以认为复现有效，但文档中需要明确记录：14_2 的 Global Cache 平均值略低于原文，14_3 平均值完全对齐但单项 corruption 波动较大。

---

## 6. 原文增益与当前复现增益对比

| 增益来源 | 原文 S2 增益 | 当前 S2 增益 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | 31.93 - 26.37 = +5.56 | 31.38 - 26.44 = +4.94 | -0.62 |
| Local Cache extra over Global | 33.36 - 31.93 = +1.43 | 33.36 - 31.38 = +1.98 | +0.55 |
| Full Point-Cache over Zero-shot | 33.36 - 26.37 = +6.99 | 33.36 - 26.44 = +6.92 | -0.07 |

分析：

完整 Point-Cache 相比 Zero-shot 的总增益与原文几乎完全一致：当前为 +6.92，原文为 +6.99，差异仅 -0.07。

不过，当前增益分解与原文略有不同：当前 Global Cache 贡献略低，Local Cache 额外贡献略高。由于 14_3 的最终平均值对齐原文，而 14_2 偏低，因此 Local Cache 的当前相对增益被放大。

---

## 7. Severity 维度增益分析

### 7.1 Global Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 30.21 | 36.03 | +5.82 |
| S1 | 28.35 | 33.45 | +5.10 |
| S2 | 26.44 | 31.38 | +4.94 |
| S3 | 24.34 | 28.78 | +4.44 |
| S4 | 22.94 | 26.55 | +3.61 |
| **all35** | **26.46** | **31.24** | **+4.78** |

分析：

Global Cache 在所有 severity 上都带来正向提升。提升幅度从 S0 的 +5.82 逐步下降到 S4 的 +3.61，说明高 severity 下全局缓存仍然有效，但补偿能力有所减弱。

### 7.2 Local Cache 相比 Global Cache

| Severity | ZS + Global Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 36.03 | 38.31 | +2.28 |
| S1 | 33.45 | 34.90 | +1.45 |
| S2 | 31.38 | 33.36 | +1.98 |
| S3 | 28.78 | 30.23 | +1.45 |
| S4 | 26.55 | 27.96 | +1.41 |
| **all35** | **31.24** | **32.95** | **+1.72** |

分析：

Local Cache 在 Global Cache 基础上继续带来稳定正增益。all35 平均额外提升 +1.72，severity=2 额外提升 +1.98。

与 12 组不同，14 组中 Local Cache 没有在 jitter_3 或 jitter_4 上出现明显负增益；对 jitter 的 all35 平均仍然是正向的，尽管提升幅度不大。

### 7.3 完整 Point-Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 30.21 | 38.31 | +8.10 |
| S1 | 28.35 | 34.90 | +6.55 |
| S2 | 26.44 | 33.36 | +6.92 |
| S3 | 24.34 | 30.23 | +5.89 |
| S4 | 22.94 | 27.96 | +5.02 |
| **all35** | **26.46** | **32.95** | **+6.50** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有正向提升，all35 平均提升 +6.50。说明完整 Point-Cache 对 ULIP-2 × ScanObjNN-C hardest 的整体鲁棒性有明显帮助。

不过，即使有完整 Point-Cache，性能仍然随 severity 增大而下降。S4 Avg 为 27.96，仍比 S0 Avg 低 10.35，说明强 corruption 下仍存在明显鲁棒性缺口。

---

## 8. Corruption 维度结果对比

### 8.1 三种方法的 corruption 平均准确率

| Corruption | Zero-shot Avg | ZS + Global Avg | ZS + Global + Local Avg |
|---|---:|---:|---:|
| add_global | 30.76 | 36.42 | 36.71 |
| add_local | 29.58 | 33.59 | 34.97 |
| dropout_global | 27.49 | 32.31 | 34.78 |
| dropout_local | 24.99 | 30.82 | 33.85 |
| rotate | 28.68 | 33.60 | 36.09 |
| scale | 29.71 | 34.47 | 35.90 |
| jitter | 13.98 | 17.46 | 18.37 |

观察：

1. jitter 在三种方法中始终最低，是 ULIP-2 × ScanObjNN-C hardest 的主要失败模式。
2. add_global 在三种方法中整体较高，完整 Point-Cache 后平均为 36.71。
3. dropout_local 在 Zero-shot 下较低，但完整 Point-Cache 提升明显。
4. 完整 Point-Cache 能提升所有 corruption 的平均准确率，但没有彻底改变 jitter 最难的排序。

### 8.2 corruption 维度总提升

| Corruption | Global - ZS | Global + Local - Global | Global + Local - ZS |
|---|---:|---:|---:|
| add_global | +5.66 | +0.29 | +5.95 |
| add_local | +4.02 | +1.37 | +5.39 |
| dropout_global | +4.81 | +2.47 | +7.29 |
| dropout_local | +5.83 | +3.03 | +8.86 |
| rotate | +4.92 | +2.50 | +7.42 |
| scale | +4.76 | +1.43 | +6.19 |
| jitter | +3.47 | +0.91 | +4.38 |
| **Average** | **+4.78** | **+1.72** | **+6.50** |

分析：

Global Cache 对 dropout_local、add_global、rotate、dropout_global 和 scale 都有明显提升，其中 dropout_local 的 Global 增益最高，为 +5.83。

Local Cache 的额外提升最大的是 dropout_local，Avg Gain 为 +3.03；其次是 rotate 和 dropout_global。这说明局部缓存对局部缺失、旋转变化和全局缺失都有较明显补偿作用。

jitter 虽然也有正向提升，但提升幅度相对有限，且最终准确率仍然最低。

---

## 9. Corruption 难度排序

### 9.1 Zero-shot 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 13.98 |
| 2 | dropout_local | 24.99 |
| 3 | dropout_global | 27.49 |
| 4 | rotate | 28.68 |
| 5 | add_local | 29.58 |
| 6 | scale | 29.71 |
| 7 | add_global | 30.76 |

### 9.2 ZS + Global 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 17.46 |
| 2 | dropout_local | 30.82 |
| 3 | dropout_global | 32.31 |
| 4 | add_local | 33.59 |
| 5 | rotate | 33.60 |
| 6 | scale | 34.47 |
| 7 | add_global | 36.42 |

### 9.3 ZS + Global + Local 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 18.37 |
| 2 | dropout_local | 33.85 |
| 3 | dropout_global | 34.78 |
| 4 | add_local | 34.97 |
| 5 | scale | 35.90 |
| 6 | rotate | 36.09 |
| 7 | add_global | 36.71 |

综合分析：

三种方法下，最困难 corruption 始终是 jitter。完整 Point-Cache 能把 jitter 平均值从 13.98 提升到 18.37，但该数值仍然远低于其他 corruption。

这说明 ULIP-2 在 ScanObjNN-C hardest 上的核心鲁棒性短板不是所有 corruption 均匀退化，而是对坐标扰动类 corruption 极其敏感。

---

## 10. 低准确率区域分析

| 条件 | Zero-shot 数量 | ZS + Global 数量 | ZS + Global + Local 数量 |
|---|---:|---:|---:|
| Acc < 30 | 21 / 35 | 10 / 35 | 9 / 35 |
| Acc < 28 | 14 / 35 | 9 / 35 | 7 / 35 |
| Acc < 25 | 10 / 35 | 6 / 35 | 5 / 35 |
| Acc < 22 | 7 / 35 | 4 / 35 | 4 / 35 |
| Acc < 20 | 4 / 35 | 4 / 35 | 4 / 35 |
| Acc < 18 | 4 / 35 | 3 / 35 | 3 / 35 |
| Acc < 15 | 3 / 35 | 2 / 35 | 2 / 35 |

分析：

Global Cache 显著减少低准确率区域。例如 Acc < 30 的 setting 从 21 个减少到 10 个。

Local Cache 进一步减少部分低准确率区域，但幅度小于 Global Cache。例如 Acc < 28 的 setting 从 9 个减少到 7 个。

但是，严重低准确率区域仍然主要由 jitter 主导。完整 Point-Cache 后，Acc < 20 的 setting 仍有 4 个，全部来自 jitter_1 到 jitter_4。

---

## 11. 关键困难 setting

| cor_type | Zero-shot | ZS + Global | ZS + Global + Local | 现象 |
|---|---:|---:|---:|---|
| jitter_4 | 8.67 | 12.04 | 14.23 | 最高 severity 坐标扰动仍然极低 |
| jitter_3 | 10.24 | 13.39 | 13.50 | 中高 severity 坐标扰动最难 |
| jitter_2 | 12.60 | 16.72 | 17.77 | severity=2 jitter 明显低于其他 corruption |
| jitter_1 | 16.69 | 19.22 | 19.99 | 有提升但仍低于多数 corruption |
| jitter_0 | 21.72 | 25.92 | 26.34 | 即使最低 severity jitter 也偏低 |
| dropout_global_4 | 21.44 | 23.32 | 24.91 | 高 severity 全局缺失仍然困难 |
| dropout_local_4 | 20.44 | 23.77 | 25.92 | 高 severity 局部缺失仍然困难 |

分析：

jitter_3 和 jitter_4 是最顽固的失败点。完整 Point-Cache 后仍只有 13.50 和 14.23。说明强坐标扰动会严重破坏 ULIP-2 的点云表征，Global / Local Cache 只能部分缓解。

dropout_global_4 和 dropout_local_4 也是困难 setting，说明高 severity 点云缺失仍然是 ScanObjNN-C hardest 上的重要挑战。

---

## 12. 方法贡献分解

以 all35 Average 为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

32.95 - 26.46 = +6.49，按表格四舍五入记为 +6.50

其中：

| 贡献来源 | all35 Avg 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +4.78 | 73.54% |
| Local Cache | +1.72 | 26.46% |
| 完整 Point-Cache | +6.50 | 100.00% |

以 severity=2 Average 为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

33.36 - 26.44 = +6.92

其中：

| 贡献来源 | S2 Avg 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +4.94 | 71.39% |
| Local Cache | +1.98 | 28.61% |
| 完整 Point-Cache | +6.92 | 100.00% |

分析：

不管按 all35 还是按 severity=2，Global Cache 都是主要贡献来源。Local Cache 的贡献比例约为 26% 到 29%，是稳定的补充模块。

这与 13 组 clean hardest 的趋势一致：Global Cache 是主增益来源，Local Cache 在真实扫描数据上也有明显价值。

---

## 13. 与 13 组 ScanObjNN clean hardest 的关系

13 组是 ULIP-2 在 ScanObjNN clean hardest 上的结果；14 组是 ULIP-2 在 ScanObjNN-C hardest all35 上的结果。

| 方法 | 13 组 clean hardest | 14 组 S2 Avg | 14 组 all35 Avg |
|---|---:|---:|---:|
| Zero-shot | 34.07 | 26.44 | 26.46 |
| ZS + Global | 40.42 | 31.38 | 31.24 |
| ZS + Global + Local | 42.44 | 33.36 | 32.95 |

从 clean 到 corrupted 的下降：

| 方法 | S2 Avg - clean | all35 Avg - clean |
|---|---:|---:|
| Zero-shot | -7.63 | -7.61 |
| ZS + Global | -9.04 | -9.18 |
| ZS + Global + Local | -9.08 | -9.49 |

分析：

ScanObjNN-C corruption 会在 ScanObjNN clean hardest 的基础上进一步降低性能。完整 Point-Cache 从 clean hardest 的 42.44 下降到 all35 的 32.95，下降约 9.49 个百分点。

这说明真实扫描 hardest split 本身已经困难，而 corruption 会进一步加剧 domain shift。即使使用完整 Point-Cache，ScanObjNN-C hardest 仍然是非常困难的数据设置。

---

## 14. 与 04 组 ULIP ScanObjNN-C hardest 的关系

04 组是 ULIP 在 ScanObjNN-C hardest all35 上的结果；14 组是 ULIP-2 在同一数据设置上的结果。

| Backbone | 方法 | S2 Avg | all35 Avg |
|---|---|---:|---:|
| ULIP | Zero-shot | 23.91 | 23.65 |
| ULIP | ZS + Global | 26.84 | 26.60 |
| ULIP | ZS + Global + Local | 27.94 | 27.41 |
| ULIP-2 | Zero-shot | 26.44 | 26.46 |
| ULIP-2 | ZS + Global | 31.38 | 31.24 |
| ULIP-2 | ZS + Global + Local | 33.36 | 32.95 |

ULIP-2 相比 ULIP 的提升：

| 方法 | S2 Avg 提升 | all35 Avg 提升 |
|---|---:|---:|
| Zero-shot | +2.53 | +2.81 |
| ZS + Global | +4.54 | +4.64 |
| ZS + Global + Local | +5.42 | +5.54 |

分析：

ULIP-2 在 ScanObjNN-C hardest 上整体强于 ULIP。特别是在加入 cache 后，ULIP-2 的优势更明显。

这说明更强的 backbone 不仅提高了 Zero-shot 基础性能，也可能提供了更适合 cache 检索的特征空间，使 Global / Local Cache 能发挥更大作用。

---

## 15. 与 12 组 ModelNet-C all35 的关系

12 组是 ULIP-2 在 ModelNet-C all35 上的结果；14 组是 ULIP-2 在 ScanObjNN-C hardest all35 上的结果。

| 数据设置 | 方法 | S2 Avg | all35 Avg |
|---|---|---:|---:|
| ModelNet-C all35 | Zero-shot | 58.02 | 58.07 |
| ModelNet-C all35 | ZS + Global | 61.22 | 61.09 |
| ModelNet-C all35 | ZS + Global + Local | 62.74 | 62.49 |
| ScanObjNN-C hardest all35 | Zero-shot | 26.44 | 26.46 |
| ScanObjNN-C hardest all35 | ZS + Global | 31.38 | 31.24 |
| ScanObjNN-C hardest all35 | ZS + Global + Local | 33.36 | 32.95 |

分析：

ScanObjNN-C hardest 明显比 ModelNet-C 更困难。即使两者都是 corrupted setting，真实扫描 hardest split 带来的 domain shift 使准确率大幅降低。

以完整 Point-Cache 为例，ULIP-2 在 ModelNet-C all35 上为 62.49，而在 ScanObjNN-C hardest all35 上只有 32.95，相差 29.54 个百分点。

这说明真实扫描数据的复杂性、遮挡、背景、扫描噪声和几何缺失，比单纯 synthetic corruption 更具挑战。

---

## 16. 对后续 MCM-PC 的启发

当前 14 组实验对后续方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| ScanObjNN-C hardest 是目前最困难设置之一 | 后续方法必须在真实扫描 corrupted setting 上验证 |
| Global Cache 是主要提升来源 | 后续应优先保证全局缓存可靠性 |
| Local Cache 有稳定额外收益 | 局部缓存值得保留，但应增加可靠性判断 |
| jitter 始终最困难 | 坐标扰动需要专门的几何稳定性机制 |
| 完整 Point-Cache 仍无法解决 jitter_3 / jitter_4 | 需要针对极端扰动设计额外策略 |
| 14_2 相比原文略低，但 14_3 对齐 | Local Cache 对最终结果有重要补偿作用 |
| ULIP-2 比 ULIP 更强 | backbone 特征质量会影响 cache 检索效果 |
| ScanObjNN-C 远低于 ModelNet-C | 真实扫描域偏移比 synthetic corruption 更关键 |

因此，后续 MCM-PC 不应只关注 ModelNet-C 这类 synthetic CAD corruption，也必须强调 ScanObjNN / ScanObjNN-C 上的真实扫描域偏移。尤其是 jitter、dropout_local、dropout_global 等困难 setting，应该作为方法设计和消融分析的重点。

---

## 17. 阶段性结论

14 组 ULIP-2 × ScanObjNN-C hardest all35 baseline 已完成。

主要结论如下：

1. 三个子实验均完成，并且 summary.csv、cor_type、log_path 和 logs 文件数量均为 35。
2. 14_1 Zero-shot 的 severity=2 Avg 为 26.44，原文为 26.37，差异 +0.07。
3. 14_2 ZS + Global 的 severity=2 Avg 为 31.38，原文为 31.93，差异 -0.55。
4. 14_3 ZS + Global + Local 的 severity=2 Avg 为 33.36，原文为 33.36，平均值完全一致。
5. 14_2 比原文略低，主要来自 scale、jitter 和 dropout_global；但整体趋势和方法有效性正常。
6. 14_3 平均值完全对齐原文，但逐 corruption 存在一定抵消，主要表现为 add_local / dropout_local 偏高，scale / jitter 偏低。
7. all35 Avg 从 Zero-shot 的 26.46 提升到 Global 的 31.24，再提升到 Global + Local 的 32.95。
8. Global Cache 是主要提升来源，all35 Avg 提升 +4.78，占完整提升的 73.54%。
9. Local Cache 在 Global Cache 基础上额外提升 +1.72，占完整提升的 26.46%。
10. 完整 Point-Cache 相比 Zero-shot 的 all35 Avg 总提升为 +6.50。
11. jitter 是最困难 corruption，完整 Point-Cache 后平均仍只有 18.37。
12. jitter_3 和 jitter_4 是最顽固失败点，完整 Point-Cache 后分别只有 13.50 和 14.23。
13. 14 组完成了 ULIP-2 在四个数据设置上的 baseline 复现闭环：ModelNet clean、ModelNet-C all35、ScanObjNN clean hardest、ScanObjNN-C hardest all35。

---

## 18. 运行命令汇总

14_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh 0

14_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh 0

14_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh 0

---

## 19. 检查命令汇总

14_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/14_1_ulip2_scanobjnnc_hardest_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

14_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

14_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
