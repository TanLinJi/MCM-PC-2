# 02_ulip_modelnetc_corruptions_all35_summary

## 1. 实验组目的

本总文档汇总 ULIP 在 ModelNet-C 全部 35 个损坏设置上的三组 baseline 复现实验。

02 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | ULIP |
| Dataset | ModelNet-C |
| Corruption 数量 | 7 |
| Severity 数量 | 5 |
| 总测试设置数 | 7 × 5 = 35 |

三组实验分别为：

| 实验编号 | 方法 | 作用 |
|---|---|---|
| 02_1_ulip_modelnetc_corruptions_all35_zs | Zero-shot | 无缓存基础对照 |
| 02_2_ulip_modelnetc_corruptions_all35_zs_global | Zero-shot + Global Cache | 验证全局缓存增益 |
| 02_3_ulip_modelnetc_corruptions_all35_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 增益 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| Zero-shot baseline 是否复现成功？ | 02_1 与原论文 severity=2 结果对齐 |
| Global Cache 是否有效？ | 比较 02_2 - 02_1 |
| Local Cache 是否有额外贡献？ | 比较 02_3 - 02_2 |
| 完整 Point-Cache 是否复现成功？ | 02_3 与原论文 severity=2 结果对齐 |
| 哪些 corruption 最困难？ | 分析 corruption × severity 的失败区域 |
| 后续 MCM-PC 应该重点改哪里？ | 从 Global / Local 的增益与不足中寻找改进方向 |

需要特别注意：原论文 Table 1 只报告 severity level = 2 下 7 种 corruption 的结果；本实验组额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 用于和原论文 Table 1 对齐 |
| all35 Average | 本复现实验扩展统计，表示 35 个 corrupted setting 的总体平均 |

---

## 2. 当前实现方式

02 组实验已经从旧版 bash 外层循环改为优化版 runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/02_run_ulip_modelnetc_corruptions_all35_common.sh |
| 优化 runner | Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |

重要变更如下：

| 旧实现 | 新实现 |
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

| 实验编号 | 方法 | summary 行数 | cor_type 唯一数 | log_path 唯一数 | logs 文件数 | 状态 |
|---|---|---:|---:|---:|---:|---|
| 02_1_ulip_modelnetc_corruptions_all35_zs | Zero-shot | 35 | 35 | 35 | 35 | 完成 |
| 02_2_ulip_modelnetc_corruptions_all35_zs_global | ZS + Global | 35 | 35 | 35 | 35 | 完成并清理 |
| 02_3_ulip_modelnetc_corruptions_all35_zs_global_local | ZS + Global + Local | 35 | 35 | 35 | 35 | 完成 |

说明：

1. 02_1 已清理旧版 20260517 日志，目前只保留新版优化 runner 产生的 35 个 log。
2. 02_2 曾出现 70 行重复记录，每个 cor_type 在 GPU 0 和 GPU 1 各出现一次，准确率完全一致；目前已清理为标准的 35 行和 35 个 log。
3. 02_3 一开始就是标准的 35 行和 35 个 log。
4. 三组实验的 summary.csv、log_path 和 logs 文件数量均已对齐。

---

## 4. 核心结果总表

| 方法 | S0 Avg | S1 Avg | S2 Avg | S3 Avg | S4 Avg | all35 Avg |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 53.40 | 50.94 | 47.68 | 43.85 | 38.39 | 46.85 |
| ZS + Global | 57.27 | 55.79 | 52.66 | 48.84 | 43.55 | 51.62 |
| ZS + Global + Local | 59.04 | 57.10 | 54.00 | 50.44 | 44.49 | 53.01 |

核心结论：

1. 三种方法在所有 severity 上都呈现稳定递进关系：Zero-shot < ZS + Global < ZS + Global + Local。
2. Global Cache 是主要增益来源，将 all35 Avg 从 46.85 提升到 51.62，提升 +4.77。
3. Local Cache 在 Global Cache 基础上继续提升到 53.01，额外提升 +1.39。
4. 完整 Point-Cache 相比 Zero-shot 的 all35 Avg 总提升为 +6.16。
5. severity=2 Avg 从 47.68 提升到 54.00，总提升为 +6.32。

---

## 5. 与原论文 Table 1 的 severity=2 对齐

原论文 Table 1 报告 ModelNet-C severity level = 2 下的 7 种 corruption 平均结果。因此，这里只取 S2 列进行对齐。

| 方法 | 当前复现 S2 Avg | 原论文 S2 Avg | Diff |
|---|---:|---:|---:|
| Zero-shot | 47.68 | 47.52 | +0.16 |
| ZS + Global | 52.66 | 52.56 | +0.10 |
| ZS + Global + Local | 54.00 | 53.70 | +0.30 |

分析：

三种方法的 severity=2 平均结果均与原论文高度接近。最大平均差异为完整 Point-Cache 的 +0.30，仍处于可接受范围。说明当前数据路径、模型权重、文本原型构建、Global Cache、Local Cache 和推理流程整体与原始 Point-Cache baseline 对齐。

---

## 6. severity 维度增益分析

### 6.1 Global Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 53.40 | 57.27 | +3.87 |
| S1 | 50.94 | 55.79 | +4.86 |
| S2 | 47.68 | 52.66 | +4.98 |
| S3 | 43.85 | 48.84 | +5.00 |
| S4 | 38.39 | 43.55 | +5.16 |
| **all35** | **46.85** | **51.62** | **+4.77** |

分析：

Global Cache 对所有 severity 均带来正向提升，且高 severity 的提升略大于低 severity。S0 提升 +3.87，而 S4 提升 +5.16。这说明 Global Cache 对更强 corruption 有一定补偿作用。

### 6.2 Local Cache 相比 Global Cache

| Severity | ZS + Global Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 57.27 | 59.04 | +1.76 |
| S1 | 55.79 | 57.10 | +1.31 |
| S2 | 52.66 | 54.00 | +1.34 |
| S3 | 48.84 | 50.44 | +1.60 |
| S4 | 43.55 | 44.49 | +0.94 |
| **all35** | **51.62** | **53.01** | **+1.39** |

分析：

Local Cache 在 Global Cache 基础上继续带来稳定正增益，但增益幅度小于 Global Cache 本身。S4 的额外提升最小，只有 +0.94，说明极强 corruption 下局部特征本身可能已经被破坏，Local Cache 的作用受到限制。

### 6.3 完整 Point-Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 53.40 | 59.04 | +5.63 |
| S1 | 50.94 | 57.10 | +6.17 |
| S2 | 47.68 | 54.00 | +6.32 |
| S3 | 43.85 | 50.44 | +6.59 |
| S4 | 38.39 | 44.49 | +6.10 |
| **all35** | **46.85** | **53.01** | **+6.16** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有超过 +5.6 的提升，说明其鲁棒性增益较稳定。S3 的提升最大，为 +6.59；S4 虽然仍然提升 +6.10，但绝对准确率仍只有 44.49，说明强 corruption 仍然是难点。

---

## 7. corruption 维度结果对比

### 7.1 三种方法的 corruption 平均准确率

| Corruption | Zero-shot Avg | ZS + Global Avg | ZS + Global + Local Avg |
|---|---:|---:|---:|
| add_global | 34.89 | 46.85 | 48.43 |
| add_local | 44.57 | 48.52 | 49.16 |
| dropout_global | 52.92 | 55.79 | 57.83 |
| dropout_local | 50.22 | 53.62 | 55.08 |
| rotate | 52.69 | 57.16 | 59.08 |
| scale | 51.00 | 54.56 | 56.30 |
| jitter | 41.66 | 44.85 | 45.22 |

观察：

1. add_global 的提升最显著，从 34.89 提升到 48.43，总提升 +13.54。
2. rotate 在完整 Point-Cache 下最高，达到 59.08。
3. jitter 在完整 Point-Cache 下仍最低，只有 45.22。
4. Global Cache 与 Local Cache 都没有彻底改变 corruption 难度排序：jitter、add_global、add_local 仍然是主要困难项。

### 7.2 corruption 维度总提升

| Corruption | Global - ZS | Global + Local - Global | Global + Local - ZS |
|---|---:|---:|---:|
| add_global | +11.97 | +1.57 | +13.54 |
| add_local | +3.96 | +0.63 | +4.59 |
| dropout_global | +2.87 | +2.04 | +4.90 |
| dropout_local | +3.40 | +1.47 | +4.86 |
| rotate | +4.47 | +1.92 | +6.39 |
| scale | +3.56 | +1.74 | +5.29 |
| jitter | +3.19 | +0.36 | +3.55 |
| **Average** | **+4.77** | **+1.39** | **+6.16** |

分析：

Global Cache 对 add_global 的提升极大，说明全局缓存对全局异常点有很强的修正能力。Local Cache 的额外增益主要体现在 dropout_global、rotate、scale、add_global 和 dropout_local 上，而对 jitter 的平均额外增益只有 +0.36。

这说明 Local Cache 并不是对所有 corruption 都同等有效。它能补充局部结构信息，但在局部几何本身被强烈扰动时，可能无法稳定发挥作用。

---

## 8. corruption 难度排序

### 8.1 Zero-shot 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | add_global | 34.89 |
| 2 | jitter | 41.66 |
| 3 | add_local | 44.57 |
| 4 | dropout_local | 50.22 |
| 5 | scale | 51.00 |
| 6 | rotate | 52.69 |
| 7 | dropout_global | 52.92 |

### 8.2 ZS + Global 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 44.85 |
| 2 | add_global | 46.85 |
| 3 | add_local | 48.52 |
| 4 | dropout_local | 53.62 |
| 5 | scale | 54.56 |
| 6 | dropout_global | 55.79 |
| 7 | rotate | 57.16 |

### 8.3 ZS + Global + Local 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 45.22 |
| 2 | add_global | 48.43 |
| 3 | add_local | 49.16 |
| 4 | dropout_local | 55.08 |
| 5 | scale | 56.30 |
| 6 | dropout_global | 57.83 |
| 7 | rotate | 59.08 |

综合分析：

Zero-shot 下最困难的是 add_global。加入 Global Cache 后，add_global 被显著修正，jitter 成为最困难 corruption。完整 Point-Cache 下，jitter 仍然最困难。这说明 Point-Cache 的全局/局部缓存机制更擅长处理异常点类 corruption，而对强坐标扰动的补偿不足。

---

## 9. 低准确率区域分析

| 条件 | Zero-shot 数量 | ZS + Global 数量 | ZS + Global + Local 数量 |
|---|---:|---:|---:|
| Acc < 50 | 17 / 35 | 12 / 35 | 10 / 35 |
| Acc < 45 | 12 / 35 | 6 / 35 | 5 / 35 |
| Acc < 40 | 7 / 35 | 3 / 35 | 3 / 35 |
| Acc < 35 | 5 / 35 | 1 / 35 | 1 / 35 |
| Acc < 30 | 3 / 35 | 1 / 35 | 1 / 35 |
| Acc < 25 | 1 / 35 | 0 / 35 | 0 / 35 |

分析：

Global Cache 明显减少低准确率区域。例如 Acc < 40 的 setting 从 7 个减少到 3 个，Acc < 35 的 setting 从 5 个减少到 1 个。

Local Cache 进一步减少中等低准确率区域，例如 Acc < 50 从 12 个减少到 10 个，Acc < 45 从 6 个减少到 5 个。但对于 Acc < 40、Acc < 35 和 Acc < 30 的极端失败区域，Local Cache 没有进一步减少数量。

这说明 Global Cache 主要负责减少严重失败案例；Local Cache 更像是在 Global Cache 基础上进一步抬高中等难度 setting 的准确率，而不是彻底解决最困难场景。

---

## 10. 关键困难 setting

| cor_type | Zero-shot | ZS + Global | ZS + Global + Local | 现象 |
|---|---:|---:|---:|---|
| jitter_4 | 23.87 | 27.51 | 26.09 | 始终最低，Local Cache 反而下降 |
| jitter_3 | 33.35 | 36.75 | 37.56 | 有提升，但仍低于 40 |
| add_global_4 | 26.50 | 38.41 | 39.99 | Global Cache 大幅提升，但仍低于 40 |
| add_global_3 | 29.98 | 43.76 | 46.19 | Point-Cache 明显修正 |
| add_local_4 | 39.10 | 43.52 | 44.25 | 有持续提升，但仍偏低 |
| dropout_local_4 | 41.09 | 44.25 | 44.89 | Local Cache 小幅提升 |

分析：

jitter_4 是当前最顽固的失败点。Zero-shot 为 23.87，Global Cache 提升到 27.51，但加入 Local Cache 后下降到 26.09。这说明强 jitter 会严重破坏全局和局部特征，Local Cache 在该 setting 下可能引入不稳定的局部检索信号。

add_global_4 从 26.50 提升到 39.99，说明 Point-Cache 对全局异常点有明显修正能力，但高 severity 下仍未完全解决。

---

## 11. 原论文对齐总表

| 方法 | 当前 S2 Avg | 原论文 S2 Avg | Mean Diff | MAE | RMSE | Max Abs Diff |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 47.68 | 47.52 | +0.16 | 0.28 | 0.37 | 0.69 |
| ZS + Global | 52.66 | 52.56 | +0.09 | 0.53 | 0.59 | 0.97 |
| ZS + Global + Local | 54.00 | 53.70 | +0.30 | 0.67 | 0.86 | 1.66 |

分析：

三种方法的 severity=2 结果均与原论文对齐。随着方法从 Zero-shot 到 Global Cache 再到 Global + Local，单项差异略有增大，这可能与 cache 在线更新、测试样本顺序、KMeans 局部聚类等因素有关。但平均结果仍然非常接近，说明整体复现实验可靠。

---

## 12. 方法贡献分解

| 贡献来源 | all35 Avg 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +4.77 | 77.44% |
| Local Cache | +1.39 | 22.56% |
| 完整 Point-Cache | +6.16 | 100.00% |

分析：

Global Cache 贡献了完整 Point-Cache 提升的大部分，约占 77.44%。Local Cache 贡献约 22.56%。这说明当前 ULIP × ModelNet-C 设定下，全局缓存是性能提升的主因，局部缓存是有效但相对较小的补充模块。

---

## 13. 对后续 MCM-PC 的启发

当前 02 组实验对后续方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| Global Cache 是主要提升来源 | 后续方法应优先保证全局缓存的可靠性 |
| Local Cache 有额外收益但幅度较小 | 局部缓存需要更精细的筛选、加权或冲突检测 |
| jitter 始终最难 | 强坐标扰动可能需要几何稳定性或可靠性约束 |
| jitter_4 中 Local Cache 出现负增益 | 局部缓存并非总是可靠，严重扰动下可能需要抑制 |
| add_global 被显著修正 | Cache 对全局异常点有明显价值，可作为方法有效性的支撑 |
| 低准确率区域没有完全消失 | 后续需要错误伪标签诊断、负缓存或不确定性抑制 |
| Global / Local 增益不均匀 | 可以考虑 corruption-aware 或 reliability-aware 的 cache fusion |

因此，后续 MCM-PC 不应只是简单增加 cache 数量，而应重点解决：

1. 哪些测试样本适合进入 cache；
2. cache logits 什么时候可信；
3. Global 和 Local 预测冲突时如何处理；
4. 对 jitter 等几何扰动强的 setting，是否应该降低 Local Cache 权重；
5. 是否可以用不确定性、能量、global-local conflict 或 negative evidence 来抑制错误 cache 传播。

---

## 14. 阶段性结论

02 组 ULIP × ModelNet-C all35 baseline 已完成。

主要结论如下：

1. 三个实验均完成，并且 summary.csv、log_path 和 logs 文件数量均为 35。
2. Zero-shot 的 all35 Avg 为 46.85，severity=2 Avg 为 47.68。
3. ZS + Global 的 all35 Avg 为 51.62，severity=2 Avg 为 52.66。
4. ZS + Global + Local 的 all35 Avg 为 53.01，severity=2 Avg 为 54.00。
5. 三种方法呈现稳定递进关系：Zero-shot < ZS + Global < ZS + Global + Local。
6. Global Cache 是主要提升来源，all35 Avg 提升 +4.77，占完整提升的约 77.44%。
7. Local Cache 在 Global Cache 基础上额外提升 +1.39，占完整提升的约 22.56%。
8. add_global 是 Global Cache 最明显修正的 corruption，完整 Point-Cache 相比 Zero-shot 提升 +13.54。
9. jitter 是完整 Point-Cache 下仍然最困难的 corruption，jitter_4 只有 26.09。
10. Local Cache 在个别 setting 上可能出现负增益，说明局部缓存需要可靠性控制。
11. 当前结果与原论文 severity=2 结果基本对齐，说明 baseline 复现可靠。
12. 该实验组可以作为后续 MCM-PC 方法设计和消融实验的基础对照。

---

## 15. 运行命令汇总

02_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/02_1_ulip_modelnetc_corruptions_all35_zs_single_gpu.sh 0

02_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global_single_gpu.sh 0

02_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 0

---

## 16. 检查命令汇总

02_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/02_1_ulip_modelnetc_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

02_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

02_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
