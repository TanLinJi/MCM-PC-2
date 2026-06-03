# 04_ulip_scanobjnnc_hardest_corruptions_all35_summary

## 1. 实验组目的

本总文档汇总 ULIP 在 ScanObjNN-C hardest 全部 35 个损坏设置上的三组 baseline 复现实验。

04 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | ULIP |
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
| 04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs | Zero-shot | 无缓存基础对照 |
| 04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global | Zero-shot + Global Cache | 验证全局缓存增益 |
| 04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 增益 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| ULIP 在 ScanObjNN-C hardest 上的 Zero-shot 鲁棒性是多少？ | 由 04_1 给出 |
| Global Cache 是否有效？ | 比较 04_2 - 04_1 |
| Local Cache 是否有额外贡献？ | 比较 04_3 - 04_2 |
| 三个结果是否与原文对齐？ | 分别与原文 Supplementary Table 7 的 severity=2 结果对比 |
| ScanObjNN-C hardest 的主要困难 corruption 是哪些？ | 分析 corruption × severity 结果矩阵 |
| 后续 MCM-PC 应重点改进什么？ | 从 Global / Local 的收益和失败区域中寻找方向 |

需要特别注意：原文 Supplementary Table 7 只报告 severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 用于和原文 Supplementary Table 7 对齐 |
| all35 Average | 本复现实验扩展统计，表示 35 个 corrupted setting 的总体平均 |

---

## 2. 当前实现方式

04 组属于 all35 corruption 实验，因此使用优化版 Python runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/04_run_ulip_scanobjnnc_hardest_corruptions_all35_common.sh |
| 优化 runner | Point-Cache/runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |

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
| 04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs | Zero-shot | 35 | 35 | 35 | 35 | 35 done | 完成 |
| 04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global | ZS + Global | 35 | 35 | 35 | 35 | 35 done | 完成 |
| 04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local | ZS + Global + Local | 35 | 35 | 35 | 35 | 35 done | 完成 |

说明：

1. 三个子实验均完成 35 个 cor_type。
2. 三个子实验均有 35 个唯一 log_path。
3. 三个子实验的 logs 文件数均为 35，没有重复日志或旧日志残留。
4. 执行完整性正常并不等于结果正常；结果是否正常需要和原文 severity=2 参考值对比。

---

## 4. 核心结果总表

| 方法 | S0 Avg | S1 Avg | S2 Avg | S3 Avg | S4 Avg | all35 Avg |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 26.86 | 25.10 | 23.91 | 22.11 | 20.24 | 23.65 |
| ZS + Global | 30.20 | 28.21 | 26.84 | 24.74 | 23.02 | 26.60 |
| ZS + Global + Local | 31.42 | 28.93 | 27.94 | 25.49 | 23.28 | 27.41 |

核心观察：

1. 三种方法在所有 severity 上都呈现稳定递进关系：Zero-shot < ZS + Global < ZS + Global + Local。
2. Global Cache 将 all35 Avg 从 23.65 提升到 26.60，提升 +2.95。
3. Local Cache 在 Global Cache 基础上进一步提升到 27.41，额外提升 +0.81。
4. 完整 Point-Cache 相比 Zero-shot 的 all35 Avg 总提升为 +3.76。
5. severity=2 Avg 从 23.91 提升到 27.94，总提升为 +4.03。

---

## 5. 与原文 Supplementary Table 7 的 severity=2 对齐

原文 Supplementary Table 7 报告的是 ScanObjectNN hardest split corrupted setting，也就是 S-PB T50-RS-C，且结果对应 severity level = 2。因此，这里只取 S2 列与原文对齐。

| 方法 | 当前复现 S2 Avg | 原文 S2 Avg | Diff |
|---|---:|---:|---:|
| Zero-shot | 23.91 | 23.97 | -0.06 |
| ZS + Global | 26.84 | 26.99 | -0.15 |
| ZS + Global + Local | 27.94 | 28.42 | -0.48 |

补充统计：

| 方法 | Mean Diff | MAE | RMSE | Max Abs Diff |
|---|---:|---:|---:|---:|
| Zero-shot | -0.06 | 0.14 | 0.19 | 0.35 |
| ZS + Global | -0.15 | 0.32 | 0.43 | 0.90 |
| ZS + Global + Local | -0.48 | 0.64 | 0.85 | 1.52 |

分析：

三个方法的 severity=2 结果均与原文基本对齐。Zero-shot 与 Global Cache 的平均差异都非常小；完整 Point-Cache 的平均差异稍大，主要来自 dropout_local 和 jitter 两个 corruption，但整体仍在可接受范围。

这说明当前 04 组的数据路径、模型权重、cor_type 设置、Global Cache 和 Local Cache 推理流程整体可靠。

---

## 6. 原文增益与当前复现增益对比

| 增益来源 | 原文 S2 增益 | 当前 S2 增益 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | 26.99 - 23.97 = +3.02 | 26.84 - 23.91 = +2.93 | -0.09 |
| Local Cache extra over Global | 28.42 - 26.99 = +1.43 | 27.94 - 26.84 = +1.10 | -0.33 |
| Full Point-Cache over Zero-shot | 28.42 - 23.97 = +4.45 | 27.94 - 23.91 = +4.03 | -0.42 |

分析：

当前复现中的方法增益趋势与原文一致。Global Cache 是主要提升来源，当前 S2 增益为 +2.93，接近原文 +3.02。Local Cache 在 Global Cache 基础上继续提升，当前 S2 额外增益为 +1.10，略低于原文 +1.43。

整体上，完整 Point-Cache 相比 Zero-shot 的当前 S2 总提升为 +4.03，与原文 +4.45 接近。

---

## 7. Severity 维度增益分析

### 7.1 Global Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 26.86 | 30.20 | +3.33 |
| S1 | 25.10 | 28.21 | +3.11 |
| S2 | 23.91 | 26.84 | +2.93 |
| S3 | 22.11 | 24.74 | +2.63 |
| S4 | 20.24 | 23.02 | +2.78 |
| **all35** | **23.65** | **26.60** | **+2.95** |

分析：

Global Cache 在所有 severity 上都带来正向提升。提升幅度比较稳定，约为 +2.6 到 +3.3。说明 Global Cache 在 ScanObjNN-C hardest 上不是只对某一档 severity 有效，而是整体提高了 corrupted setting 下的性能。

### 7.2 Local Cache 相比 Global Cache

| Severity | ZS + Global Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 30.20 | 31.42 | +1.22 |
| S1 | 28.21 | 28.93 | +0.72 |
| S2 | 26.84 | 27.94 | +1.10 |
| S3 | 24.74 | 25.49 | +0.75 |
| S4 | 23.02 | 23.28 | +0.27 |
| **all35** | **26.60** | **27.41** | **+0.81** |

分析：

Local Cache 在 Global Cache 基础上继续带来正向平均增益，但增益幅度小于 Global Cache 本身。尤其在 S4 上，Local Cache 只额外提升 +0.27，说明最高 severity 下局部特征可能已经受到严重破坏，Local Cache 的有效性受到限制。

### 7.3 完整 Point-Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 26.86 | 31.42 | +4.55 |
| S1 | 25.10 | 28.93 | +3.83 |
| S2 | 23.91 | 27.94 | +4.03 |
| S3 | 22.11 | 25.49 | +3.38 |
| S4 | 20.24 | 23.28 | +3.04 |
| **all35** | **23.65** | **27.41** | **+3.76** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有正向提升，all35 平均提升 +3.76。说明完整 Point-Cache 对 ScanObjNN-C hardest 的整体鲁棒性有稳定帮助。

不过，提升幅度在高 severity 上略有下降，S4 总提升为 +3.04，低于 S0 的 +4.55。这说明当 corruption 过强时，cache 方法的补偿能力也会受到限制。

---

## 8. Corruption 维度结果对比

### 8.1 三种方法的 corruption 平均准确率

| Corruption | Zero-shot Avg | ZS + Global Avg | ZS + Global + Local Avg |
|---|---:|---:|---:|
| add_global | 18.81 | 23.16 | 23.66 |
| add_local | 19.06 | 21.28 | 22.04 |
| dropout_global | 29.60 | 32.37 | 33.26 |
| dropout_local | 23.13 | 26.53 | 28.43 |
| rotate | 26.23 | 29.67 | 30.33 |
| scale | 26.93 | 28.91 | 28.53 |
| jitter | 21.76 | 24.28 | 25.64 |

观察：

1. dropout_global 在三种方法中始终最高。
2. add_global 和 add_local 是 Zero-shot 下最困难的 corruption。
3. 完整 Point-Cache 后，add_local 仍然最低，说明局部异常点仍然最难。
4. jitter 在高 severity 下仍然非常困难。
5. scale 在完整 Point-Cache 中略低于 Global Cache，说明 Local Cache 对 scale 并不稳定。

### 8.2 corruption 维度总提升

| Corruption | Global - ZS | Global + Local - Global | Global + Local - ZS |
|---|---:|---:|---:|
| add_global | +4.34 | +0.50 | +4.85 |
| add_local | +2.22 | +0.76 | +2.98 |
| dropout_global | +2.77 | +0.89 | +3.66 |
| dropout_local | +3.41 | +1.89 | +5.30 |
| rotate | +3.44 | +0.65 | +4.10 |
| scale | +1.98 | -0.38 | +1.60 |
| jitter | +2.52 | +1.36 | +3.88 |
| **Average** | **+2.95** | **+0.81** | **+3.76** |

分析：

Global Cache 对 add_global 的提升最大，Avg Gain 为 +4.34。说明全局缓存对全局异常点有明显修正作用。

Local Cache 的额外提升最大的是 dropout_local，Avg Gain 为 +1.89。这符合直觉，因为 dropout_local 直接破坏局部结构，Local Cache 更可能补充局部信息。

scale 是唯一一个 Local Cache 平均负增益的 corruption，Global + Local 相比 Global 下降 -0.38。这提示 Local Cache 并非所有场景都可靠，特别是在尺度变化类 corruption 中，局部 patch 特征可能引入不稳定信息。

---

## 9. Corruption 难度排序

### 9.1 Zero-shot 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | add_global | 18.81 |
| 2 | add_local | 19.06 |
| 3 | jitter | 21.76 |
| 4 | dropout_local | 23.13 |
| 5 | rotate | 26.23 |
| 6 | scale | 26.93 |
| 7 | dropout_global | 29.60 |

### 9.2 ZS + Global 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | add_local | 21.28 |
| 2 | add_global | 23.16 |
| 3 | jitter | 24.28 |
| 4 | dropout_local | 26.53 |
| 5 | scale | 28.91 |
| 6 | rotate | 29.67 |
| 7 | dropout_global | 32.37 |

### 9.3 ZS + Global + Local 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | add_local | 22.04 |
| 2 | add_global | 23.66 |
| 3 | jitter | 25.64 |
| 4 | dropout_local | 28.43 |
| 5 | scale | 28.53 |
| 6 | rotate | 30.33 |
| 7 | dropout_global | 33.26 |

综合分析：

三种方法下，最困难区域始终集中在 add_global、add_local 和 jitter。完整 Point-Cache 能整体提升这些 corruption，但没有彻底改变难度排序。说明 ScanObjNN-C hardest 的主要困难来自异常点和坐标扰动。

---

## 10. 低准确率区域分析

| 条件 | Zero-shot 数量 | ZS + Global 数量 | ZS + Global + Local 数量 |
|---|---:|---:|---:|
| Acc < 30 | 30 / 35 | 25 / 35 | 23 / 35 |
| Acc < 28 | 26 / 35 | 21 / 35 | 18 / 35 |
| Acc < 25 | 19 / 35 | 14 / 35 | 11 / 35 |
| Acc < 22 | 16 / 35 | 7 / 35 | 5 / 35 |
| Acc < 20 | 10 / 35 | 3 / 35 | 2 / 35 |

分析：

Global Cache 显著减少低准确率区域。例如 Acc < 20 的 setting 从 10 个减少到 3 个；Acc < 22 的 setting 从 16 个减少到 7 个。

Local Cache 进一步减少低准确率区域，但幅度较小。例如 Acc < 25 的 setting 从 14 个减少到 11 个，Acc < 22 的 setting 从 7 个减少到 5 个。

这说明 Global Cache 主要负责减少严重失败案例，Local Cache 则进一步改善部分中低性能 setting。

---

## 11. 关键困难 setting

| cor_type | Zero-shot | ZS + Global | ZS + Global + Local | 现象 |
|---|---:|---:|---:|---|
| jitter_4 | 15.86 | 18.81 | 19.15 | 始终很低，最高 severity 坐标扰动最困难 |
| add_local_4 | 17.18 | 18.84 | 19.67 | 局部异常点仍然困难 |
| add_local_3 | 17.70 | 19.57 | 20.06 | 中高 severity 局部异常点仍偏低 |
| add_global_4 | 17.87 | 21.44 | 20.71 | Local Cache 相比 Global 出现下降 |
| dropout_local_4 | 17.38 | 20.78 | 22.24 | Local Cache 有一定帮助 |

分析：

jitter_4 是当前最顽固的失败点之一。完整 Point-Cache 后仍只有 19.15，说明强坐标扰动会同时影响全局和局部特征。

add_local_4 和 add_local_3 也仍然很低，说明局部异常点在 ScanObjNN-C hardest 上很难处理。

dropout_local_4 从 Zero-shot 的 17.38 提升到完整 Point-Cache 的 22.24，说明 Local Cache 对局部缺失有帮助，但仍没有完全解决该困难区域。

---

## 12. 方法贡献分解

以 all35 Average 为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

27.41 - 23.65 = +3.76

其中：

| 贡献来源 | all35 Avg 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +2.95 | 78.46% |
| Local Cache | +0.81 | 21.54% |
| 完整 Point-Cache | +3.76 | 100.00% |

以 severity=2 Average 为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

27.94 - 23.91 = +4.03

其中：

| 贡献来源 | S2 Avg 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +2.93 | 72.70% |
| Local Cache | +1.10 | 27.30% |
| 完整 Point-Cache | +4.03 | 100.00% |

分析：

不管按 all35 还是按 severity=2，Global Cache 都是主要贡献来源。Local Cache 有稳定额外贡献，但贡献比例较小。

这与 02 组和 03 组的趋势一致：Global Cache 是 Point-Cache baseline 的主增益模块，Local Cache 是补充模块。

---

## 13. 与 03 组 clean hardest 的关系

03 组是 ULIP 在 ScanObjNN clean hardest 上的结果；04 组是 ULIP 在 ScanObjNN-C hardest all35 上的结果。

| 方法 | 03 组 clean hardest | 04 组 S2 Avg | 04 组 all35 Avg |
|---|---:|---:|---:|
| Zero-shot | 29.08 | 23.91 | 23.65 |
| ZS + Global | 32.20 | 26.84 | 26.60 |
| ZS + Global + Local | 32.48 | 27.94 | 27.41 |

从 clean 到 corrupted 的下降：

| 方法 | S2 Avg - clean | all35 Avg - clean |
|---|---:|---:|
| Zero-shot | -5.17 | -5.43 |
| ZS + Global | -5.36 | -5.60 |
| ZS + Global + Local | -4.54 | -5.07 |

分析：

ScanObjNN-C corruption 会在 ScanObjNN clean hardest 的基础上进一步降低性能。三种方法从 clean 到 corrupted 都下降约 4.5 到 5.6 个百分点。

完整 Point-Cache 的 S2 下降相对较小，为 -4.54，说明 Global + Local 对 corrupted setting 有一定鲁棒性优势。

---

## 14. 对后续 MCM-PC 的启发

当前 04 组实验对后续方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| Global Cache 是主要提升来源 | 后续方法应优先保证全局缓存可靠性 |
| Local Cache 有额外收益但幅度较小 | 局部缓存需要更精细的可靠性控制 |
| Local Cache 对 dropout_local 提升明显 | 局部缓存适合补偿局部结构缺失 |
| scale 上 Local Cache 平均负增益 | 局部缓存并非所有 corruption 下都可靠 |
| jitter_4 仍然很低 | 强坐标扰动可能需要专门的几何稳定性机制 |
| add_local 仍然最困难 | 局部异常点可能导致局部缓存被误导 |
| 低准确率区域没有完全消除 | 需要错误伪标签诊断、负缓存或动态抑制机制 |

因此，后续 MCM-PC 不应只是增加 cache 数量，而应重点考虑：

1. 哪些样本适合进入 cache；
2. cache logits 什么时候可信；
3. Local Cache 何时应该增强，何时应该被抑制；
4. Global 和 Local 预测冲突时如何处理；
5. 对 add_local、jitter 等困难 corruption 是否需要更保守的可靠性判断；
6. 是否可以引入 negative evidence 或 conflict-aware suppression 来避免错误 cache 传播。

---

## 15. 阶段性结论

04 组 ULIP × ScanObjNN-C hardest all35 baseline 已完成。

主要结论如下：

1. 三个子实验均完成，并且 summary.csv、cor_type、log_path 和 logs 文件数量均为 35。
2. 04_1 Zero-shot 的 severity=2 Avg 为 23.91，原文为 23.97，差异 -0.06。
3. 04_2 ZS + Global 的 severity=2 Avg 为 26.84，原文为 26.99，差异 -0.15。
4. 04_3 ZS + Global + Local 的 severity=2 Avg 为 27.94，原文为 28.42，差异 -0.48。
5. 三种方法的 severity=2 结果均与原文基本对齐，04 组复现可靠。
6. all35 Avg 从 Zero-shot 的 23.65 提升到 Global 的 26.60，再提升到 Global + Local 的 27.41。
7. Global Cache 是主要提升来源，all35 Avg 提升 +2.95，占完整提升的 78.46%。
8. Local Cache 在 Global Cache 基础上额外提升 +0.81，占完整提升的 21.54%。
9. 完整 Point-Cache 相比 Zero-shot 的 all35 Avg 总提升为 +3.76。
10. dropout_local 是 Local Cache 最明显受益的 corruption。
11. scale 出现 Local Cache 平均负增益，说明局部缓存并非始终可靠。
12. add_local、add_global 和 jitter 仍然是 ScanObjNN-C hardest 上的主要困难 corruption。
13. jitter_4 是最顽固的失败 setting 之一，完整 Point-Cache 后仍只有 19.15。
14. 04 组结果完成了 ULIP 在四个数据设置上的 baseline 复现闭环：ModelNet clean、ModelNet-C all35、ScanObjNN clean hardest、ScanObjNN-C hardest all35。

---

## 16. 运行命令汇总

04_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh 0

04_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh 0

04_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh 0

---

## 17. 检查命令汇总

04_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/04_1_ulip_scanobjnnc_hardest_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

04_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

04_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
