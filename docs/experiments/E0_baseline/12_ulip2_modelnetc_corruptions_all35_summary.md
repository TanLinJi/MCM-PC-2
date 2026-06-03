# 12_ulip2_modelnetc_corruptions_all35_summary

## 1. 实验组目的

本总文档汇总 ULIP-2 在 ModelNet-C 全部 35 个损坏设置上的三组 baseline 复现实验。

12 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | ULIP-2 |
| Dataset | ModelNet-C |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| Corruption 数量 | 7 |
| Severity 数量 | 5 |
| 总测试设置数 | 7 × 5 = 35 |
| 输入点数 | 1024 |

三组实验分别为：

| 实验编号 | 方法 | 作用 |
|---|---|---|
| 12_1_ulip2_modelnetc_corruptions_all35_zs | Zero-shot | 无缓存基础对照 |
| 12_2_ulip2_modelnetc_corruptions_all35_zs_global | Zero-shot + Global Cache | 验证全局缓存增益 |
| 12_3_ulip2_modelnetc_corruptions_all35_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 增益 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| ULIP-2 在 ModelNet-C 上的 Zero-shot 鲁棒性是多少？ | 由 12_1 给出 |
| Global Cache 是否有效？ | 比较 12_2 - 12_1 |
| Local Cache 是否有额外贡献？ | 比较 12_3 - 12_2 |
| 三个结果是否与原文对齐？ | 分别与原文 Point-Cache Table 1 的 severity=2 结果对比 |
| ULIP-2 在不同 corruption 下的主要失败区域是什么？ | 重点分析 corruption × severity 结果矩阵 |
| 后续 MCM-PC 应重点关注哪些问题？ | 从 Global / Local 的收益和失败区域中寻找方法改进方向 |

需要特别注意：原文 Point-Cache Table 1 只报告 severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 用于和原文 Point-Cache Table 1 对齐 |
| all35 Average | 本复现实验扩展统计，表示 35 个 corrupted setting 的总体平均 |

---

## 2. 当前实现方式

12 组属于 all35 corruption 实验，因此使用优化版 Python runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/12_run_ulip2_modelnetc_corruptions_all35_common.sh |
| 优化 runner | Point-Cache/runners/baseline/run_ulip2_modelnetc_corruptions_all35.py |
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
| 12_1_ulip2_modelnetc_corruptions_all35_zs | Zero-shot | 35 | 35 | 35 | 35 | 35 done | 完成 |
| 12_2_ulip2_modelnetc_corruptions_all35_zs_global | ZS + Global | 35 | 35 | 35 | 35 | 35 done | 完成 |
| 12_3_ulip2_modelnetc_corruptions_all35_zs_global_local | ZS + Global + Local | 35 | 35 | 35 | 35 | 35 done | 完成 |

说明：

1. 三个子实验均完成 35 个 cor_type。
2. 三个子实验均有 35 个唯一 log_path。
3. 三个子实验的 logs 文件数均为 35，没有重复日志或旧日志残留。
4. 执行完整性正常并不等于结果正常；结果是否正常需要和原文 severity=2 参考值对比。

---

## 4. 核心结果总表

| 方法 | S0 Avg | S1 Avg | S2 Avg | S3 Avg | S4 Avg | all35 Avg |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 67.19 | 61.93 | 58.02 | 53.92 | 49.29 | 58.07 |
| ZS + Global | 69.28 | 65.37 | 61.22 | 57.08 | 52.52 | 61.09 |
| ZS + Global + Local | 70.76 | 66.93 | 62.74 | 58.33 | 53.71 | 62.49 |

核心观察：

1. 三种方法在所有 severity 上都呈现稳定递进关系：Zero-shot < ZS + Global < ZS + Global + Local。
2. Global Cache 将 all35 Avg 从 58.07 提升到 61.09，提升 +3.02。
3. Local Cache 在 Global Cache 基础上进一步提升到 62.49，额外提升 +1.40。
4. 完整 Point-Cache 相比 Zero-shot 的 all35 Avg 总提升为 +4.42。
5. severity=2 Avg 从 58.02 提升到 62.74，总提升约 +4.72。

---

## 5. 与原文 Point-Cache Table 1 的 severity=2 对齐

原文 Point-Cache Table 1 报告的是 ModelNet-C severity level = 2 下 7 种 corruption 的结果。因此，这里只取 S2 列与原文对齐。

| 方法 | 当前复现 S2 Avg | 原文 S2 Avg | Diff |
|---|---:|---:|---:|
| Zero-shot | 58.02 | 57.95 | +0.07 |
| ZS + Global | 61.22 | 61.18 | +0.04 |
| ZS + Global + Local | 62.74 | 62.79 | -0.05 |

补充统计：

| 方法 | Mean Diff | MAE | RMSE | Max Abs Diff |
|---|---:|---:|---:|---:|
| Zero-shot | +0.07 | 0.40 | 0.47 | 0.77 |
| ZS + Global | +0.04 | 0.38 | 0.46 | 0.77 |
| ZS + Global + Local | -0.05 | 0.35 | 0.43 | 0.77 |

分析：

三个方法的 severity=2 结果均与原文高度对齐。三种方法的平均差异分别只有 +0.07、+0.04 和 -0.05，最大单项差异也只有 0.77。

这说明当前 12 组的数据路径、模型权重、cor_type 设置、Global Cache 和 Local Cache 推理流程整体可靠。

---

## 6. 原文增益与当前复现增益对比

| 增益来源 | 原文 S2 增益 | 当前 S2 增益 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | 61.18 - 57.95 = +3.23 | 61.22 - 58.02 = +3.20 | -0.03 |
| Local Cache extra over Global | 62.79 - 61.18 = +1.61 | 62.74 - 61.22 = +1.52 | -0.09 |
| Full Point-Cache over Zero-shot | 62.79 - 57.95 = +4.84 | 62.74 - 58.02 = +4.72 | -0.12 |

分析：

当前复现中的方法增益趋势与原文高度一致。Global Cache 是主要提升来源，当前 S2 增益为 +3.20，接近原文 +3.23。Local Cache 在 Global Cache 基础上继续提升，当前 S2 额外增益为 +1.52，接近原文 +1.61。

整体上，完整 Point-Cache 相比 Zero-shot 的当前 S2 总提升约为 +4.72，与原文 +4.84 高度接近。

---

## 7. Severity 维度增益分析

### 7.1 Global Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 67.19 | 69.28 | +2.09 |
| S1 | 61.93 | 65.37 | +3.44 |
| S2 | 58.02 | 61.22 | +3.20 |
| S3 | 53.92 | 57.08 | +3.16 |
| S4 | 49.29 | 52.52 | +3.24 |
| **all35** | **58.07** | **61.09** | **+3.02** |

分析：

Global Cache 在所有 severity 上都带来正向提升。提升幅度比较稳定，除 S0 为 +2.09 外，其余 severity 大多在 +3.1 到 +3.4 左右。说明 Global Cache 在 ULIP-2 × ModelNet-C 上不是只对某一档 severity 有效，而是整体提高了 corrupted setting 下的性能。

### 7.2 Local Cache 相比 Global Cache

| Severity | ZS + Global Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 69.28 | 70.76 | +1.48 |
| S1 | 65.37 | 66.93 | +1.56 |
| S2 | 61.22 | 62.74 | +1.52 |
| S3 | 57.08 | 58.33 | +1.25 |
| S4 | 52.52 | 53.71 | +1.19 |
| **all35** | **61.09** | **62.49** | **+1.40** |

分析：

Local Cache 在 Global Cache 基础上继续带来稳定正增益。all35 平均额外提升 +1.40，severity=2 额外提升约 +1.52。

与 11 组 ModelNet clean 相比，12 组中 Local Cache 的贡献明显更大。11 组 clean setting 中 Local Cache 只额外提升 +0.36，而 12 组 all35 中额外提升 +1.40。这说明 corrupted setting 中局部缓存有更明显的补偿空间。

### 7.3 完整 Point-Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 67.19 | 70.76 | +3.57 |
| S1 | 61.93 | 66.93 | +5.00 |
| S2 | 58.02 | 62.74 | +4.72 |
| S3 | 53.92 | 58.33 | +4.41 |
| S4 | 49.29 | 53.71 | +4.42 |
| **all35** | **58.07** | **62.49** | **+4.42** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有正向提升，all35 平均提升 +4.42。说明完整 Point-Cache 对 ULIP-2 × ModelNet-C 的整体鲁棒性有稳定帮助。

不过，即使有完整 Point-Cache，性能仍然随 severity 增大而下降。S4 Avg 为 53.71，仍比 S0 Avg 低 17.05，说明强 corruption 下仍存在明显鲁棒性缺口。

---

## 8. Corruption 维度结果对比

### 8.1 三种方法的 corruption 平均准确率

| Corruption | Zero-shot Avg | ZS + Global Avg | ZS + Global + Local Avg |
|---|---:|---:|---:|
| add_global | 65.18 | 67.62 | 68.85 |
| add_local | 54.92 | 59.85 | 61.50 |
| dropout_global | 64.30 | 65.62 | 67.78 |
| dropout_local | 57.70 | 60.76 | 62.33 |
| rotate | 69.86 | 71.40 | 72.38 |
| scale | 67.27 | 68.77 | 70.25 |
| jitter | 27.28 | 32.84 | 34.37 |

观察：

1. jitter 在三种方法中始终最低，是 ULIP-2 × ModelNet-C 的主要失败模式。
2. rotate 和 scale 在三种方法中始终较高，说明 ULIP-2 对旋转和尺度变化相对鲁棒。
3. add_local 和 dropout_local 属于中等偏难 corruption。
4. 完整 Point-Cache 能提升所有 corruption 的平均准确率，但没有彻底改变 corruption 难度排序。
5. jitter 虽然提升最大，但由于起点太低，最终仍然最低。

### 8.2 corruption 维度总提升

| Corruption | Global - ZS | Global + Local - Global | Global + Local - ZS |
|---|---:|---:|---:|
| add_global | +2.44 | +1.23 | +3.67 |
| add_local | +4.93 | +1.65 | +6.58 |
| dropout_global | +1.32 | +2.16 | +3.48 |
| dropout_local | +3.06 | +1.56 | +4.63 |
| rotate | +1.54 | +0.98 | +2.52 |
| scale | +1.50 | +1.49 | +2.99 |
| jitter | +5.56 | +1.53 | +7.09 |
| **Average** | **+3.02** | **+1.40** | **+4.42** |

分析：

Global Cache 对 jitter 和 add_local 的提升最大，分别为 +5.56 和 +4.93。说明全局缓存对坐标扰动和局部异常点有明显补偿作用。

Local Cache 的额外提升最大的是 dropout_global，Avg Gain 为 +2.16；add_local、dropout_local、jitter 也有明显额外提升。说明局部缓存不仅对局部缺失有帮助，也能在多个 corruption 类型中补充 Global Cache。

---

## 9. Corruption 难度排序

### 9.1 Zero-shot 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 27.28 |
| 2 | add_local | 54.92 |
| 3 | dropout_local | 57.70 |
| 4 | dropout_global | 64.30 |
| 5 | add_global | 65.18 |
| 6 | scale | 67.27 |
| 7 | rotate | 69.86 |

### 9.2 ZS + Global 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 32.84 |
| 2 | add_local | 59.85 |
| 3 | dropout_local | 60.76 |
| 4 | dropout_global | 65.62 |
| 5 | add_global | 67.62 |
| 6 | scale | 68.77 |
| 7 | rotate | 71.40 |

### 9.3 ZS + Global + Local 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 34.37 |
| 2 | add_local | 61.50 |
| 3 | dropout_local | 62.33 |
| 4 | dropout_global | 67.78 |
| 5 | add_global | 68.85 |
| 6 | scale | 70.25 |
| 7 | rotate | 72.38 |

综合分析：

三种方法下，最困难 corruption 始终是 jitter。完整 Point-Cache 能把 jitter 平均值从 27.28 提升到 34.37，但该数值仍然远低于其他 corruption。

这说明 ULIP-2 在 ModelNet-C 上的核心鲁棒性短板不是所有 corruption 均匀退化，而是对坐标扰动类 corruption 极其敏感。

---

## 10. 低准确率区域分析

| 条件 | Zero-shot 数量 | ZS + Global 数量 | ZS + Global + Local 数量 |
|---|---:|---:|---:|
| Acc < 65 | 18 / 35 | 12 / 35 | 8 / 35 |
| Acc < 60 | 13 / 35 | 7 / 35 | 5 / 35 |
| Acc < 55 | 10 / 35 | 5 / 35 | 4 / 35 |
| Acc < 50 | 7 / 35 | 4 / 35 | 3 / 35 |
| Acc < 45 | 4 / 35 | 4 / 35 | 3 / 35 |
| Acc < 40 | 4 / 35 | 3 / 35 | 3 / 35 |
| Acc < 35 | 4 / 35 | 3 / 35 | 3 / 35 |
| Acc < 30 | 3 / 35 | 3 / 35 | 2 / 35 |
| Acc < 25 | 3 / 35 | 2 / 35 | 2 / 35 |

分析：

Global Cache 显著减少低准确率区域。例如 Acc < 60 的 setting 从 13 个减少到 7 个。

Local Cache 进一步减少低准确率区域，例如 Acc < 65 的 setting 从 12 个减少到 8 个，Acc < 60 的 setting 从 7 个减少到 5 个。

但是，严重低准确率区域仍然主要由 jitter 主导。完整 Point-Cache 后仍有 2 个 setting 低于 25，分别是 jitter_3 和 jitter_4。

---

## 11. 关键困难 setting

| cor_type | Zero-shot | ZS + Global | ZS + Global + Local | 现象 |
|---|---:|---:|---:|---|
| jitter_4 | 9.32 | 15.64 | 12.84 | 最高 severity 坐标扰动最困难，Local Cache 反而下降 |
| jitter_3 | 14.18 | 18.12 | 17.30 | 中高 severity 坐标扰动仍然极低 |
| jitter_2 | 20.99 | 28.93 | 30.15 | severity=2 jitter 明显低于其他 corruption |
| jitter_1 | 34.60 | 42.01 | 47.49 | 有明显改善，但仍低于多数 corruption |
| dropout_local_4 | 45.14 | 47.45 | 49.55 | 高 severity 局部缺失仍然困难 |
| dropout_global_4 | 48.91 | 48.26 | 52.88 | 高 severity 全局缺失仍然困难 |

分析：

jitter_4 是最顽固的失败点。Zero-shot 为 9.32，Global Cache 提升到 15.64，但加入 Local Cache 后下降到 12.84。这说明在最高 severity 坐标扰动下，局部特征可能已经被严重破坏，Local Cache 可能引入不可靠的局部证据。

jitter_3 也有类似现象，Global Cache 为 18.12，Global + Local 为 17.30。说明 Local Cache 在强坐标扰动下并非始终可靠。

---

## 12. 方法贡献分解

以 all35 Average 为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

62.49 - 58.07 = +4.42

其中：

| 贡献来源 | all35 Avg 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +3.02 | 68.33% |
| Local Cache | +1.40 | 31.67% |
| 完整 Point-Cache | +4.42 | 100.00% |

以 severity=2 Average 为准，完整 Point-Cache 相比 Zero-shot 的总提升约为：

62.74 - 58.02 = +4.72

其中：

| 贡献来源 | S2 Avg 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +3.20 | 67.80% |
| Local Cache | +1.52 | 32.20% |
| 完整 Point-Cache | +4.72 | 100.00% |

分析：

不管按 all35 还是按 severity=2，Global Cache 都是主要贡献来源。Local Cache 的贡献比例约为 30% 左右，比 11 组 clean setting 中更明显。

这说明在 corrupted setting 中，Local Cache 的价值更高；但它仍然需要可靠性控制，因为在 jitter 高 severity 下出现了负增益。

---

## 13. 与 11 组 ModelNet clean 的关系

11 组是 ULIP-2 在 ModelNet clean 上的结果；12 组是 ULIP-2 在 ModelNet-C all35 上的结果。

| 方法 | 11 组 clean | 12 组 S2 Avg | 12 组 all35 Avg |
|---|---:|---:|---:|
| Zero-shot | 72.20 | 58.02 | 58.07 |
| ZS + Global | 73.99 | 61.22 | 61.09 |
| ZS + Global + Local | 74.35 | 62.74 | 62.49 |

从 clean 到 corrupted 的下降：

| 方法 | S2 Avg - clean | all35 Avg - clean |
|---|---:|---:|
| Zero-shot | -14.18 | -14.13 |
| ZS + Global | -12.77 | -12.90 |
| ZS + Global + Local | -11.61 | -11.86 |

分析：

ModelNet-C corruption 会在 ModelNet clean 的基础上显著降低性能。三种方法从 clean 到 corrupted 都下降 11 到 14 个百分点。

完整 Point-Cache 的下降最小，说明 Global + Local 对 corrupted setting 有一定鲁棒性优势。Zero-shot 下降最大，说明无缓存预测最容易受到 corruption 影响。

---

## 14. 与 02 组 ULIP ModelNet-C 的关系

02 组是 ULIP 在 ModelNet-C all35 上的结果；12 组是 ULIP-2 在 ModelNet-C all35 上的结果。

| Backbone | 方法 | S2 Avg | all35 Avg |
|---|---|---:|---:|
| ULIP | Zero-shot | 47.68 | 46.85 |
| ULIP | ZS + Global | 52.66 | 51.62 |
| ULIP | ZS + Global + Local | 54.00 | 53.01 |
| ULIP-2 | Zero-shot | 58.02 | 58.07 |
| ULIP-2 | ZS + Global | 61.22 | 61.09 |
| ULIP-2 | ZS + Global + Local | 62.74 | 62.49 |

分析：

ULIP-2 在 ModelNet-C 上显著强于 ULIP。以 all35 Avg 为例：

| 方法 | ULIP-2 - ULIP |
|---|---:|
| Zero-shot | +11.22 |
| ZS + Global | +9.47 |
| ZS + Global + Local | +9.48 |

这说明 backbone 能力对最终鲁棒性有很大影响。ULIP-2 的基础表示更强，因此在 corrupted setting 上整体明显优于 ULIP。

同时，ULIP-2 的 cache 增益仍然明显：

| Backbone | Global over ZS | Local over Global | Full over ZS |
|---|---:|---:|---:|
| ULIP all35 | +4.77 | +1.39 | +6.16 |
| ULIP-2 all35 | +3.02 | +1.40 | +4.42 |

ULIP 的 Global Cache 增益更大，可能因为其 Zero-shot 起点较低，cache 有更大提升空间。ULIP-2 的 Local Cache 额外增益与 ULIP 接近，说明局部缓存对两种 backbone 都有稳定补充作用。

---

## 15. 对后续 MCM-PC 的启发

当前 12 组实验对后续方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| ULIP-2 整体强于 ULIP | 更强 backbone 能显著提高鲁棒性上限 |
| Global Cache 是主要提升来源 | 后续方法应优先保证全局缓存可靠性 |
| Local Cache 在 corrupted setting 中贡献更明显 | 局部缓存值得保留，但需要动态可靠性控制 |
| jitter 是最主要失败模式 | 坐标扰动需要专门的几何稳定性或抑制机制 |
| jitter_3 / jitter_4 上 Local Cache 可能负增益 | 局部缓存不能无条件增强，需要识别不可靠局部证据 |
| add_local 也明显困难 | 局部异常点可能误导局部特征匹配 |
| 完整 Point-Cache 仍无法解决高 severity jitter | 后续方法需要针对极端扰动设计额外机制 |

因此，后续 MCM-PC 不应只是简单增加 cache 数量，而应重点考虑：

1. 哪些样本适合进入 positive cache；
2. cache logits 什么时候可信；
3. Local Cache 何时应该增强，何时应该被抑制；
4. Global 和 Local 预测冲突时如何处理；
5. 对 jitter、add_local 等困难 corruption 是否需要更保守的可靠性判断；
6. 是否可以引入 negative evidence 或 conflict-aware suppression 来避免错误 cache 传播。

---

## 16. 阶段性结论

12 组 ULIP-2 × ModelNet-C all35 baseline 已完成。

主要结论如下：

1. 三个子实验均完成，并且 summary.csv、cor_type、log_path 和 logs 文件数量均为 35。
2. 12_1 Zero-shot 的 severity=2 Avg 为 58.02，原文为 57.95，差异 +0.07。
3. 12_2 ZS + Global 的 severity=2 Avg 为 61.22，原文为 61.18，差异 +0.04。
4. 12_3 ZS + Global + Local 的 severity=2 Avg 为 62.74，原文为 62.79，差异 -0.05。
5. 三种方法的 severity=2 结果均与原文高度对齐，12 组复现可靠。
6. all35 Avg 从 Zero-shot 的 58.07 提升到 Global 的 61.09，再提升到 Global + Local 的 62.49。
7. Global Cache 是主要提升来源，all35 Avg 提升 +3.02，占完整提升的 68.33%。
8. Local Cache 在 Global Cache 基础上额外提升 +1.40，占完整提升的 31.67%。
9. 完整 Point-Cache 相比 Zero-shot 的 all35 Avg 总提升为 +4.42。
10. jitter 是最困难 corruption，完整 Point-Cache 后平均仍只有 34.37。
11. jitter_4 是最顽固失败点，完整 Point-Cache 后只有 12.84。
12. Local Cache 在多数 setting 上有正增益，但在 jitter_3 和 jitter_4 上出现负增益。
13. 12 组完成了 ULIP-2 在 ModelNet / ModelNet-C 上的 baseline 复现闭环。

---

## 17. 运行命令汇总

12_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/12_1_ulip2_modelnetc_corruptions_all35_zs_single_gpu.sh 0

12_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global_single_gpu.sh 0

12_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 0

---

## 18. 检查命令汇总

12_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/12_1_ulip2_modelnetc_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/12_1_ulip2_modelnetc_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/12_1_ulip2_modelnetc_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

12_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

12_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
