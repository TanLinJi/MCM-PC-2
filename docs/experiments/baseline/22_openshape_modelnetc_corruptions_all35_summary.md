# 22_openshape_modelnetc_corruptions_all35_summary

## 1. 实验组目的

本总文档汇总 OpenShape 在 ModelNet-C 全部 35 个损坏设置上的三组 baseline 复现实验。

22 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | OpenShape |
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
| 22_1_openshape_modelnetc_corruptions_all35_zs | Zero-shot | 无缓存基础对照 |
| 22_2_openshape_modelnetc_corruptions_all35_zs_global | Zero-shot + Global Cache | 验证全局缓存增益 |
| 22_3_openshape_modelnetc_corruptions_all35_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 及 Local Cache 额外影响 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| OpenShape 在 ModelNet-C 上的 Zero-shot 鲁棒性是多少？ | 由 22_1 给出 |
| Global Cache 是否有效？ | 比较 22_2 - 22_1 |
| Local Cache 是否有额外贡献？ | 比较 22_3 - 22_2 |
| 三个结果是否与原文对齐？ | 分别与原文 Point-Cache Table 1 的 severity=2 结果对比 |
| OpenShape clean 上 cache 略降是否影响 corrupted setting？ | 与 21 组 clean 结果对比 |
| ModelNet-C 中主要困难 corruption 是哪些？ | 分析 corruption × severity 结果矩阵 |
| 后续 MCM-PC 应重点关注哪些问题？ | 从 Global / Local 的收益和失败区域中寻找方法改进方向 |

需要特别注意：原文 Point-Cache Table 1 只报告 severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 用于和原文 Point-Cache Table 1 对齐 |
| all35 Average | 本复现实验扩展统计，表示 35 个 corrupted setting 的总体平均 |

---

## 2. 当前实现方式

22 组属于 all35 corruption 实验，因此使用优化版 Python runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/22_run_openshape_modelnetc_corruptions_all35_common.sh |
| 优化 runner | Point-Cache/runners/baseline/run_openshape_modelnetc_corruptions_all35.py |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |
| OpenShape 权重 | Point-Cache/weights/openshape/openshape-pointbert-vitg14-rgb/model.pt |
| OpenCLIP 权重 | Point-Cache/weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin |

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
| 22_1_openshape_modelnetc_corruptions_all35_zs | Zero-shot | 35 | 35 | 35 | 35 | 35 done | 完成 |
| 22_2_openshape_modelnetc_corruptions_all35_zs_global | ZS + Global | 35 | 35 | 35 | 35 | 35 done | 完成 |
| 22_3_openshape_modelnetc_corruptions_all35_zs_global_local | ZS + Global + Local | 35 | 35 | 35 | 35 | 35 done | 完成 |

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
| Zero-shot | 80.31 | 76.99 | 73.57 | 69.28 | 62.72 | 72.57 |
| ZS + Global | 81.28 | 78.59 | 76.46 | 72.56 | 66.79 | 75.14 |
| ZS + Global + Local | 81.11 | 78.58 | 76.33 | 72.50 | 67.19 | 75.14 |

核心观察：

1. Global Cache 将 all35 Avg 从 72.57 提升到 75.14，提升 +2.56。
2. Local Cache 在 Global Cache 基础上的 all35 Avg 只从 75.14 变化到 75.14，四舍五入后几乎不变，精确变化约 +0.01。
3. severity=2 上，Global Cache 将 Avg 从 73.57 提升到 76.46，提升 +2.89。
4. severity=2 上，Local Cache 在 Global Cache 基础上从 76.46 变为 76.33，变化 -0.13。
5. 因此，22 组不能写成 “Local Cache 明显提升”；更准确的结论是：Global Cache 是主要提升来源，Local Cache 额外贡献接近于零。

---

## 5. 与原文 Point-Cache Table 1 的 severity=2 对齐

原文 Point-Cache Table 1 报告的是 ModelNet-C severity level = 2 下 7 种 corruption 的结果。因此，这里只取 S2 列与原文对齐。

| 方法 | 当前复现 S2 Avg | 原文 S2 Avg | Diff |
|---|---:|---:|---:|
| Zero-shot | 73.57 | 73.49 | +0.08 |
| ZS + Global | 76.46 | 76.43 | +0.03 |
| ZS + Global + Local | 76.33 | 76.59 | -0.26 |

补充统计：

| 方法 | Mean Diff | MAE | RMSE | Max Abs Diff |
|---|---:|---:|---:|---:|
| Zero-shot | +0.08 | 0.53 | 0.58 | 0.89 |
| ZS + Global | +0.03 | 0.58 | 0.66 | 1.05 |
| ZS + Global + Local | -0.26 | 0.69 | 0.94 | 1.91 |

分析：

22_1 与原文高度对齐，severity=2 Average 只高 +0.08。22_2 也与原文高度对齐，severity=2 Average 只高 +0.03。22_3 比原文低 -0.26，仍在可接受范围内，但逐 corruption 波动更明显，主要是 add_global 和 add_local 偏低。

因此，22 组总体可以认为复现有效，但文档中必须明确记录：22_3 的 Local Cache 额外贡献较弱，且 22_3 的 add_global / add_local 两项比原文偏低。

---

## 6. 原文增益与当前复现增益对比

| 增益来源 | 原文 S2 增益 | 当前 S2 增益 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | 76.43 - 73.49 = +2.94 | 76.46 - 73.57 = +2.89 | -0.05 |
| Local Cache extra over Global | 76.59 - 76.43 = +0.16 | 76.33 - 76.46 = -0.13 | -0.29 |
| Full Point-Cache over Zero-shot | 76.59 - 73.49 = +3.10 | 76.33 - 73.57 = +2.76 | -0.34 |

分析：

Global Cache 增益复现得非常好。原文 severity=2 下 Global Cache 增益为 +2.94，当前为 +2.89，只差 -0.05。

Local Cache 的额外贡献没有复现出正增益。原文中 Local Cache 在 Global Cache 基础上额外提升 +0.16，而当前为 -0.13。这个差异绝对值只有 0.29，但说明在当前复现中 Local Cache 不是主要收益来源。

完整 Point-Cache 相比 Zero-shot 仍有明确正增益，severity=2 提升 +2.76，all35 提升 +2.57。

---

## 7. Severity 维度增益分析

### 7.1 Global Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 80.31 | 81.28 | +0.97 |
| S1 | 76.99 | 78.59 | +1.60 |
| S2 | 73.57 | 76.46 | +2.89 |
| S3 | 69.28 | 72.56 | +3.28 |
| S4 | 62.72 | 66.79 | +4.07 |
| **all35** | **72.57** | **75.14** | **+2.56** |

分析：

Global Cache 在所有 severity 上都带来正向提升。提升幅度从 S0 的 +0.97 增大到 S4 的 +4.07，说明 corruption 越强，Global Cache 的补偿作用越明显。

这与 21 组 clean 现象形成对比：OpenShape 在 clean setting 上 Global Cache 略降，但在 corrupted setting 上 Global Cache 明显提升鲁棒性。

### 7.2 Local Cache 相比 Global Cache

| Severity | ZS + Global Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 81.28 | 81.11 | -0.17 |
| S1 | 78.59 | 78.58 | -0.01 |
| S2 | 76.46 | 76.33 | -0.13 |
| S3 | 72.56 | 72.50 | -0.06 |
| S4 | 66.79 | 67.19 | +0.40 |
| **all35** | **75.14** | **75.14** | **+0.01** |

分析：

Local Cache 在 Global Cache 基础上的额外贡献很弱。S0、S1、S2、S3 均略低于 Global Cache，仅 S4 有 +0.40 的正增益。all35 平均几乎持平。

因此，22 组的 Local Cache 结论应写为：Local Cache 在 OpenShape × ModelNet-C 上没有稳定带来额外提升；其主要价值不是这组实验的核心来源。

### 7.3 完整 Point-Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 80.31 | 81.11 | +0.80 |
| S1 | 76.99 | 78.58 | +1.59 |
| S2 | 73.57 | 76.33 | +2.76 |
| S3 | 69.28 | 72.50 | +3.22 |
| S4 | 62.72 | 67.19 | +4.47 |
| **all35** | **72.57** | **75.14** | **+2.57** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有正向提升。提升幅度随着 severity 增大而增强，S4 提升最大，为 +4.47。

这说明即使 Local Cache 的额外贡献很弱，完整 Point-Cache 相比 Zero-shot 仍然能明显改善 OpenShape 在 ModelNet-C 上的鲁棒性。

---

## 8. Corruption 维度结果对比

### 8.1 三种方法的 corruption 平均准确率

| Corruption | Zero-shot Avg | ZS + Global Avg | ZS + Global + Local Avg |
|---|---:|---:|---:|
| add_global | 72.57 | 75.02 | 74.34 |
| add_local | 68.21 | 72.40 | 73.51 |
| dropout_global | 77.78 | 79.35 | 78.93 |
| dropout_local | 72.09 | 74.46 | 75.18 |
| rotate | 80.77 | 81.42 | 81.08 |
| scale | 78.63 | 78.68 | 78.02 |
| jitter | 57.96 | 64.63 | 64.94 |

观察：

1. jitter 在三种方法中始终最低，是 OpenShape × ModelNet-C 的主要困难 corruption。
2. rotate 在三种方法中整体最高，完整 Point-Cache 后平均为 81.08。
3. Global Cache 提升了所有 corruption 的平均准确率。
4. Local Cache 对 add_local、dropout_local 和 jitter 有一定正向作用。
5. Local Cache 对 add_global、dropout_global、rotate 和 scale 整体偏负。
6. 因此，Local Cache 的效果具有 corruption 依赖性，而不是稳定全局提升。

### 8.2 corruption 维度总提升

| Corruption | Global - ZS | Global + Local - Global | Global + Local - ZS |
|---|---:|---:|---:|
| add_global | +2.45 | -0.68 | +1.78 |
| add_local | +4.19 | +1.11 | +5.30 |
| dropout_global | +1.57 | -0.42 | +1.15 |
| dropout_local | +2.37 | +0.72 | +3.09 |
| rotate | +0.65 | -0.34 | +0.31 |
| scale | +0.05 | -0.66 | -0.61 |
| jitter | +6.67 | +0.32 | +6.98 |
| **Average** | **+2.56** | **+0.01** | **+2.57** |

分析：

Global Cache 对 jitter 的提升最大，Avg Gain 为 +6.67；对 add_local 的提升也很明显，为 +4.19。说明 Global Cache 对坐标扰动和局部异常点有较强补偿作用。

Local Cache 的额外贡献非常不均衡。它对 add_local、dropout_local 和 jitter 有正向作用，但对 add_global、dropout_global、rotate 和 scale 为负。最终正负抵消，all35 Average 只剩 +0.01。

这说明 OpenShape × ModelNet-C 上，Local Cache 不应被视为稳定增益模块，而应被视为一个需要可靠性判断的辅助信号。

---

## 9. Corruption 难度排序

### 9.1 Zero-shot 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 57.96 |
| 2 | add_local | 68.21 |
| 3 | dropout_local | 72.09 |
| 4 | add_global | 72.57 |
| 5 | dropout_global | 77.78 |
| 6 | scale | 78.63 |
| 7 | rotate | 80.77 |

### 9.2 ZS + Global 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 64.63 |
| 2 | add_local | 72.40 |
| 3 | dropout_local | 74.46 |
| 4 | add_global | 75.02 |
| 5 | scale | 78.68 |
| 6 | dropout_global | 79.35 |
| 7 | rotate | 81.42 |

### 9.3 ZS + Global + Local 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 64.94 |
| 2 | add_local | 73.51 |
| 3 | add_global | 74.34 |
| 4 | dropout_local | 75.18 |
| 5 | scale | 78.02 |
| 6 | dropout_global | 78.93 |
| 7 | rotate | 81.08 |

综合分析：

三种方法下，最困难 corruption 始终是 jitter。完整 Point-Cache 能把 jitter 平均值从 57.96 提升到 64.94，但该数值仍然远低于其他 corruption。

这说明 OpenShape 虽然整体鲁棒性较强，但对高 severity 坐标扰动仍然敏感。jitter 是后续 MCM-PC 方法设计中必须重点关注的失败模式。

---

## 10. 低准确率区域分析

| 条件 | Zero-shot 数量 | ZS + Global 数量 | ZS + Global + Local 数量 |
|---|---:|---:|---:|
| Acc < 80 | 24 / 35 | 19 / 35 | 27 / 35 |
| Acc < 75 | 16 / 35 | 11 / 35 | 15 / 35 |
| Acc < 70 | 11 / 35 | 6 / 35 | 5 / 35 |
| Acc < 65 | 7 / 35 | 2 / 35 | 3 / 35 |
| Acc < 60 | 2 / 35 | 2 / 35 | 2 / 35 |
| Acc < 50 | 2 / 35 | 1 / 35 | 1 / 35 |
| Acc < 40 | 1 / 35 | 0 / 35 | 0 / 35 |

分析：

Global Cache 明显减少了低准确率区域。例如 Acc < 70 的 setting 从 11 个减少到 6 个，Acc < 65 的 setting 从 7 个减少到 2 个。

Local Cache 的低准确率区域变化不稳定。相比 Global Cache，完整 Point-Cache 在 Acc < 70 上进一步减少 1 个 setting，但在 Acc < 80 和 Acc < 75 上反而更多。这与 Local Cache 的平均增益接近 0 一致。

因此，22 组的低准确率区域分析也支持同一结论：Global Cache 是主要有效模块，Local Cache 在该 setting 下不稳定。

---

## 11. 关键困难 setting

| cor_type | Zero-shot | ZS + Global | ZS + Global + Local | 现象 |
|---|---:|---:|---:|---|
| jitter_4 | 32.98 | 46.27 | 45.83 | 最高 severity 坐标扰动仍然最低 |
| jitter_3 | 45.99 | 54.38 | 55.02 | 中高 severity 坐标扰动仍然困难 |
| dropout_global_4 | 63.57 | 65.40 | 64.71 | 高 severity 全局缺失仍然困难 |
| dropout_local_4 | 60.82 | 65.11 | 66.82 | 高 severity 局部缺失仍然困难 |
| jitter_2 | 60.25 | 67.54 | 68.64 | severity=2 jitter 明显低于多数 corruption |
| add_global_4 | 68.44 | 70.46 | 70.54 | 高 severity 全局异常点仍有影响 |
| add_local_3 | 64.71 | 70.71 | 71.72 | 局部异常点被 cache 明显改善 |

分析：

jitter_3 和 jitter_4 是最顽固的失败点。Global Cache 明显提升了它们，但完整 Point-Cache 后仍然处于全组最低区域。

dropout_global_4 和 dropout_local_4 也是高 severity 下的重要困难 setting。后续方法如果希望提升 OpenShape × ModelNet-C 的鲁棒性，需要重点解决高 severity jitter 和高 severity dropout。

---

## 12. 方法贡献分解

以 all35 Average 为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

75.14 - 72.57 = +2.57

其中：

| 贡献来源 | all35 Avg 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +2.56 | 约 99.6% |
| Local Cache | +0.01 | 约 0.4% |
| 完整 Point-Cache | +2.57 | 100.00% |

以 severity=2 Average 为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

76.33 - 73.57 = +2.76

其中：

| 贡献来源 | S2 Avg 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +2.89 | 超过完整提升 |
| Local Cache | -0.13 | 负贡献 |
| 完整 Point-Cache | +2.76 | 100.00% |

分析：

不管按 all35 还是按 severity=2，Global Cache 都是主要贡献来源。Local Cache 在 all35 下几乎无贡献，在 severity=2 下甚至略微负贡献。

这与 14 组 ScanObjNN-C hardest 不同。14 组中 Local Cache 有明确额外收益；22 组中 OpenShape × ModelNet-C 的 Local Cache 效果较弱。

---

## 13. 与 21 组 ModelNet clean 的关系

21 组是 OpenShape 在 ModelNet clean 上的结果；22 组是 OpenShape 在 ModelNet-C all35 上的结果。

| 方法 | 21 组 clean | 22 组 S2 Avg | 22 组 all35 Avg |
|---|---:|---:|---:|
| Zero-shot | 84.72 | 73.57 | 72.57 |
| ZS + Global | 84.48 | 76.46 | 75.14 |
| ZS + Global + Local | 84.00 | 76.33 | 75.14 |

从 clean 到 corrupted 的下降：

| 方法 | S2 Avg - clean | all35 Avg - clean |
|---|---:|---:|
| Zero-shot | -11.15 | -12.15 |
| ZS + Global | -8.02 | -9.34 |
| ZS + Global + Local | -7.67 | -8.86 |

分析：

ModelNet-C corruption 会在 ModelNet clean 的基础上明显降低 OpenShape 性能。Zero-shot 从 84.72 下降到 all35 Avg 72.57，下降 -12.15。

Global Cache 和完整 Point-Cache 都缩小了 clean-to-corruption gap。Global Cache 将 gap 缩小到 -9.34，完整 Point-Cache 缩小到 -8.86。

这说明 21 组 clean 上 cache 略降并不代表 cache 无效。在 corrupted setting 中，cache 仍然有明确鲁棒性收益。

---

## 14. 与 02 / 12 组 ModelNet-C 的关系

02 组是 ULIP 在 ModelNet-C all35 上的结果；12 组是 ULIP-2 在 ModelNet-C all35 上的结果；22 组是 OpenShape 在同一数据设置上的结果。

| Backbone | 方法 | S2 Avg | all35 Avg |
|---|---|---:|---:|
| ULIP | Zero-shot | 47.68 | 46.85 |
| ULIP | ZS + Global | 52.66 | 51.62 |
| ULIP | ZS + Global + Local | 54.00 | 53.01 |
| ULIP-2 | Zero-shot | 58.02 | 58.07 |
| ULIP-2 | ZS + Global | 61.22 | 61.09 |
| ULIP-2 | ZS + Global + Local | 62.74 | 62.49 |
| OpenShape | Zero-shot | 73.57 | 72.57 |
| OpenShape | ZS + Global | 76.46 | 75.14 |
| OpenShape | ZS + Global + Local | 76.33 | 75.14 |

分析：

OpenShape 在 ModelNet-C 上明显强于 ULIP 和 ULIP-2。即使不使用 cache，OpenShape Zero-shot 的 all35 Avg 也达到 72.57，明显高于 ULIP-2 完整 Point-Cache 的 62.49。

这说明 OpenShape backbone 的基础鲁棒性更强。但也正因为 OpenShape 本身很强，Local Cache 的边际收益变小。Global Cache 仍然有效，但 Local Cache 没有表现出稳定额外增益。

---

## 15. 对后续 MCM-PC 的启发

当前 22 组实验对后续方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| OpenShape 在 ModelNet-C 上基础鲁棒性很强 | 强 backbone 上改进空间更小 |
| Global Cache 有明确正增益 | 全局缓存仍是可靠主模块 |
| Local Cache 额外贡献接近于零 | 局部缓存需要可靠性选择或条件启用 |
| Local Cache 对不同 corruption 表现不一致 | 需要 corruption-aware 或 uncertainty-aware 融合策略 |
| jitter 仍然最困难 | 坐标扰动需要专门处理 |
| scale 上完整 Point-Cache 低于 Zero-shot | cache 融合可能对某些稳定 corruption 产生扰动 |
| clean 上 cache 下降，corruption 上 Global 提升 | cache 应该根据 domain shift 强度自适应介入 |
| OpenShape 比 ULIP / ULIP-2 强很多 | MCM-PC 的贡献应避免只依赖弱 backbone 上的提升 |

因此，后续 MCM-PC 如果要在 OpenShape 这种强 backbone 上取得稳定收益，需要重点解决两个问题：

1. 如何保留 Global Cache 的鲁棒性收益；
2. 如何避免 Local Cache 在某些 corruption 或高质量预测场景中产生不必要扰动。

---

## 16. 阶段性结论

22 组 OpenShape × ModelNet-C all35 baseline 已完成。

主要结论如下：

1. 三个子实验均完成，并且 summary.csv、cor_type、log_path 和 logs 文件数量均为 35。
2. 22_1 Zero-shot 的 severity=2 Avg 为 73.57，原文为 73.49，差异 +0.08。
3. 22_2 ZS + Global 的 severity=2 Avg 为 76.46，原文为 76.43，差异 +0.03。
4. 22_3 ZS + Global + Local 的 severity=2 Avg 为 76.33，原文为 76.59，差异 -0.26。
5. 22_1 和 22_2 与原文高度对齐，22_3 略低于原文但仍可接受。
6. all35 Avg 从 Zero-shot 的 72.57 提升到 Global 的 75.14，再到 Global + Local 的 75.14，Local Cache 几乎不改变整体均值。
7. Global Cache 是主要提升来源，all35 Avg 提升 +2.56，severity=2 Avg 提升 +2.89。
8. Local Cache 在 Global Cache 基础上的额外贡献接近于零，all35 Avg 为 +0.01，severity=2 Avg 为 -0.13。
9. 完整 Point-Cache 相比 Zero-shot 仍然有明确提升，all35 Avg 提升 +2.57，severity=2 Avg 提升 +2.76。
10. jitter 是最困难 corruption，完整 Point-Cache 后 all35 平均仍只有 64.94。
11. jitter_3 和 jitter_4 是最顽固失败点，完整 Point-Cache 后分别为 55.02 和 45.83。
12. OpenShape 在 ModelNet-C 上明显强于 ULIP 和 ULIP-2，但强 backbone 上 Local Cache 边际收益更弱。
13. 22 组结果说明：OpenShape clean 上 cache 略降，但 corrupted setting 上 Global Cache 有明确鲁棒性收益。

---

## 17. 运行命令汇总

22_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/22_1_openshape_modelnetc_corruptions_all35_zs_single_gpu.sh 0

22_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global_single_gpu.sh 1

22_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 0

---

## 18. 检查命令汇总

22_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/22_1_openshape_modelnetc_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/22_1_openshape_modelnetc_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/22_1_openshape_modelnetc_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

22_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

22_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
