# 32_uni3d_modelnetc_corruptions_all35_summary

## 1. 实验组目的

本总文档汇总 Uni3D 在 ModelNet-C 全部 35 个损坏设置上的三组 baseline 复现实验。

32 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | Uni3D |
| Dataset | ModelNet-C |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| Corruption 数量 | 7 |
| Severity 数量 | 5 |
| 总测试设置数 | 7 × 5 = 35 |
| 输入点数 | 1024 |
| Uni3D point encoder checkpoint | weights/uni3d/modelnet40/model.pt |
| Uni3D text encoder checkpoint | weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin |

三组实验分别为：

| 实验编号 | 方法 | 作用 |
|---|---|---|
| 32_1_uni3d_modelnetc_corruptions_all35_zs | Zero-shot | 无缓存基础对照 |
| 32_2_uni3d_modelnetc_corruptions_all35_zs_global | Zero-shot + Global Cache | 验证全局缓存增益 |
| 32_3_uni3d_modelnetc_corruptions_all35_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 及 Local Cache 额外影响 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| Uni3D 在 ModelNet-C 上的 Zero-shot 鲁棒性是多少？ | 由 32_1 给出 |
| Global Cache 是否有效？ | 比较 32_2 - 32_1 |
| Local Cache 是否有额外贡献？ | 比较 32_3 - 32_2 |
| 完整 Point-Cache 是否与原文对齐？ | 比较 32_3 与原文 Table 1 |
| ModelNet-C 相比 ModelNet clean 下降多少？ | 与 31 组比较 |
| Uni3D 和其他 backbone 在 ModelNet-C 上关系如何？ | 与 02 / 12 / 22 组比较 |
| 后续 MCM-PC 应重点关注哪些失败模式？ | 分析 corruption × severity 结果矩阵 |

需要特别注意：原文 Point-Cache Table 1 只报告 severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 用于和原文 Point-Cache Table 1 对齐 |
| all35 Average | 本复现实验扩展统计，表示 35 个 corrupted setting 的总体平均 |

---

## 2. 当前实现方式

32 组属于 all35 corruption 实验，因此使用优化版 Python runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/32_run_uni3d_modelnetc_corruptions_all35_common.sh |
| 优化 runner | Point-Cache/runners/baseline/run_uni3d_modelnetc_corruptions_all35.py |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |
| checkpoint 下载脚本 | Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh |

优化方式如下：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 | 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 | 模型只加载一次，每个 cor_type 重新创建 DataLoader、重新初始化 cache |
| bash 通过 tee 生成单个 cor_type 的 log | Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv | summary.csv 的列结构保持不变 |

该优化只改变执行效率，不改变实验定义。每个 cor_type 仍然单独记录 log，并且每个方法的 summary.csv 仍然保持 35 行。

---

## 3. Uni3D checkpoint 设置

32 组正式使用：

weights/uni3d/modelnet40/model.pt

这是 31 / 32 组 ModelNet 系列实验必须使用的 Uni3D checkpoint。

此前使用服务器原有的：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

会导致 31 组 ModelNet clean 结果整体偏低。切换到 ModelNet40 checkpoint 后，31 组 clean 结果与原文高度对齐。因此 32 组 ModelNet-C 继续使用 `weights/uni3d/modelnet40/model.pt`。

后续规则：

| 实验组 | 数据设置 | 应使用 checkpoint |
|---|---|---|
| 31 | Uni3D × ModelNet clean | weights/uni3d/modelnet40/model.pt |
| 32 | Uni3D × ModelNet-C all35 | weights/uni3d/modelnet40/model.pt |
| 33 | Uni3D × ScanObjNN clean hardest | weights/uni3d/scanobjnn/model.pt |
| 34 | Uni3D × ScanObjNN-C hardest all35 | weights/uni3d/scanobjnn/model.pt |

`weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt` 不应作为正式复现 checkpoint，除非专门做 checkpoint ablation。

---

## 4. 日志清理记录

32 组运行过程中，logs 目录曾残留旧 checkpoint 运行产生的日志。

清理前，三个子实验均出现：

| 项目 | 数量 |
|---|---:|
| summary.csv 行数 | 35 |
| summary 中唯一 log_path 数 | 35 |
| logs 目录 .log 文件数 | 70 |
| status=done 数 | 35 |

这说明当前 summary 已经是 35 个有效结果，但 logs 目录保留了旧 checkpoint 运行产生的 35 个旧 log。

清理方式为：只保留 summary.csv 中 `log_path` 字段实际引用的 35 个 log，删除 logs 目录中未被 summary 引用的旧 log。

清理后，三个子实验均为：

| 项目 | 数量 |
|---|---:|
| summary.csv 行数 | 35 |
| summary 中唯一 log_path 数 | 35 |
| logs 目录 .log 文件数 | 35 |
| status=done 数 | 35 |

因此当前 32 组结果目录已经干净。

---

## 5. 三个实验的输出完整性检查

| 实验编号 | 方法 | summary 行数 | cor_type 唯一数 | log_path 唯一数 | logs 文件数 | status | 状态 |
|---|---|---:|---:|---:|---:|---|---|
| 32_1_uni3d_modelnetc_corruptions_all35_zs | Zero-shot | 35 | 35 | 35 | 35 | 35 done | 完成 |
| 32_2_uni3d_modelnetc_corruptions_all35_zs_global | ZS + Global | 35 | 35 | 35 | 35 | 35 done | 完成 |
| 32_3_uni3d_modelnetc_corruptions_all35_zs_global_local | ZS + Global + Local | 35 | 35 | 35 | 35 | 35 done | 完成 |

说明：

1. 三个子实验均完成 35 个 cor_type。
2. 三个子实验均有 35 个唯一 log_path。
3. 三个子实验的 logs 文件数均为 35，旧 checkpoint 残留日志已清理。
4. 三个子实验的 status 均为 35 个 done。
5. 执行完整性正常并不等于结果正常；结果是否正常需要和原文 severity=2 参考值对比。

---

## 6. 核心结果总表

| 方法 | S0 Avg | S1 Avg | S2 Avg | S3 Avg | S4 Avg | all35 Avg |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 76.04 | 71.92 | 67.80 | 62.54 | 56.12 | 66.89 |
| ZS + Global | 78.54 | 75.22 | 72.21 | 67.64 | 62.56 | 71.24 |
| ZS + Global + Local | 79.83 | 76.32 | 73.66 | 69.73 | 64.49 | 72.81 |

核心观察：

1. Global Cache 将 all35 Avg 从 66.89 提升到 71.24，提升 +4.35。
2. Local Cache 在 Global Cache 基础上将 all35 Avg 从 71.24 提升到 72.81，额外提升 +1.57。
3. severity=2 上，Global Cache 将 Avg 从 67.80 提升到 72.21，提升 +4.41。
4. severity=2 上，Local Cache 在 Global Cache 基础上从 72.21 提升到 73.66，额外提升 +1.45。
5. 因此，32 组中 Global Cache 和 Local Cache 都明确有效。

---

## 7. 与原文 Point-Cache Table 1 的 severity=2 对齐

原文 Point-Cache Table 1 报告的是 ModelNet-C severity level = 2 下 7 种 corruption 的结果。因此，这里只取 S2 列与原文对齐。

| 方法 | 当前复现 S2 Avg | 原文 S2 Avg | Diff |
|---|---:|---:|---:|
| Zero-shot | 67.80 | 67.95 | -0.15 |
| ZS + Global | 72.21 | 71.81 | +0.40 |
| ZS + Global + Local | 73.66 | 73.31 | +0.35 |

分析：

32_1、32_2、32_3 的 severity=2 平均值均与原文高度接近。最大差异为 +0.40，说明 32 组三个子实验都可以认为是有效复现结果。

---

## 8. 原文增益与当前复现增益对比

| 增益来源 | 原文 S2 增益 | 当前 S2 增益 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | 71.81 - 67.95 = +3.86 | 72.21 - 67.80 = +4.41 | +0.55 |
| Local Cache extra over Global | 73.31 - 71.81 = +1.50 | 73.66 - 72.21 = +1.45 | -0.05 |
| Full Point-Cache over Zero-shot | 73.31 - 67.95 = +5.36 | 73.66 - 67.80 = +5.86 | +0.50 |

分析：

Global Cache 是主要提升来源，当前 S2 增益 +4.41，略高于原文 +3.86。Local Cache 的额外贡献当前 +1.45，与原文 +1.50 几乎一致。完整 Point-Cache 相比 Zero-shot 的总提升当前 +5.86，略高于原文 +5.36。

因此，32 组的方法增益结构与原文一致：Global Cache 是主提升来源，Local Cache 在 Global Cache 基础上继续带来稳定额外收益。

---

## 9. Severity 维度增益分析

### 9.1 Global Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 76.04 | 78.54 | +2.50 |
| S1 | 71.92 | 75.22 | +3.30 |
| S2 | 67.80 | 72.21 | +4.41 |
| S3 | 62.54 | 67.64 | +5.10 |
| S4 | 56.12 | 62.56 | +6.44 |
| **all35** | **66.89** | **71.24** | **+4.35** |

分析：

Global Cache 在所有 severity 上都带来正向提升。提升幅度随 severity 增大而增强，S4 提升最大，为 +6.44。这说明在更强 corruption 下，全局缓存对 Uni3D 的补偿作用更明显。

### 9.2 Local Cache 相比 Global Cache

| Severity | ZS + Global Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 78.54 | 79.83 | +1.29 |
| S1 | 75.22 | 76.32 | +1.10 |
| S2 | 72.21 | 73.66 | +1.45 |
| S3 | 67.64 | 69.73 | +2.09 |
| S4 | 62.56 | 64.49 | +1.93 |
| **all35** | **71.24** | **72.81** | **+1.57** |

分析：

Local Cache 在 Global Cache 基础上的额外贡献在所有 severity 上均为正。S3 额外提升 +2.09，S4 额外提升 +1.93，说明 Local Cache 在中高 severity 上尤其有价值。

### 9.3 完整 Point-Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 76.04 | 79.83 | +3.79 |
| S1 | 71.92 | 76.32 | +4.40 |
| S2 | 67.80 | 73.66 | +5.86 |
| S3 | 62.54 | 69.73 | +7.19 |
| S4 | 56.12 | 64.49 | +8.37 |
| **all35** | **66.89** | **72.81** | **+5.92** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有明确正增益。增益随 severity 增大而增强，S4 提升 +8.37。说明完整 Point-Cache 能显著改善 Uni3D 在 ModelNet-C 高 severity corruption 下的鲁棒性。

---

## 10. Corruption 维度结果对比

### 10.1 三种方法的 corruption 平均准确率

| Corruption | Zero-shot Avg | ZS + Global Avg | ZS + Global + Local Avg |
|---|---:|---:|---:|
| add_global | 72.49 | 76.58 | 77.51 |
| add_local | 57.00 | 66.82 | 71.36 |
| dropout_global | 64.50 | 68.20 | 69.33 |
| dropout_local | 64.93 | 67.87 | 69.27 |
| rotate | 78.91 | 80.41 | 80.72 |
| scale | 75.57 | 76.99 | 77.59 |
| jitter | 54.81 | 61.79 | 63.86 |

观察：

1. jitter 在三种方法中始终最低，是 Uni3D × ModelNet-C 的主要困难 corruption。
2. add_local 在 Zero-shot 下第二困难，但完整 Point-Cache 对其提升最大。
3. Global Cache 提升了所有 corruption 的平均准确率。
4. Local Cache 对 add_local 和 jitter 的额外贡献最明显。
5. 完整 Point-Cache 后，rotate 的平均准确率最高，为 80.72。

### 10.2 Corruption 维度总提升

| Corruption | Global - ZS | Global + Local - Global | Global + Local - ZS |
|---|---:|---:|---:|
| add_global | +4.09 | +0.93 | +5.02 |
| add_local | +9.82 | +4.55 | +14.36 |
| dropout_global | +3.70 | +1.13 | +4.83 |
| dropout_local | +2.94 | +1.40 | +4.34 |
| rotate | +1.50 | +0.31 | +1.81 |
| scale | +1.42 | +0.60 | +2.02 |
| jitter | +6.98 | +2.07 | +9.05 |
| **Average** | **+4.35** | **+1.57** | **+5.92** |

分析：

完整 Point-Cache 对 add_local 提升最大，all35 平均提升 +14.36；对 jitter 提升第二大，all35 平均提升 +9.05。说明 cache 对局部异常点和坐标扰动具有明显补偿作用。

Local Cache 在 add_local 上额外提升 +4.55，在 jitter 上额外提升 +2.07，说明局部缓存对局部异常点和坐标扰动非常重要。

---

## 11. Corruption 难度排序

### 11.1 Zero-shot 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 54.81 |
| 2 | add_local | 57.00 |
| 3 | dropout_global | 64.50 |
| 4 | dropout_local | 64.93 |
| 5 | add_global | 72.49 |
| 6 | scale | 75.57 |
| 7 | rotate | 78.91 |

### 11.2 ZS + Global 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 61.79 |
| 2 | add_local | 66.82 |
| 3 | dropout_local | 67.87 |
| 4 | dropout_global | 68.20 |
| 5 | add_global | 76.58 |
| 6 | scale | 76.99 |
| 7 | rotate | 80.41 |

### 11.3 ZS + Global + Local 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 63.86 |
| 2 | dropout_local | 69.27 |
| 3 | dropout_global | 69.33 |
| 4 | add_local | 71.36 |
| 5 | add_global | 77.51 |
| 6 | scale | 77.59 |
| 7 | rotate | 80.72 |

综合分析：

三种方法下，最困难 corruption 始终是 jitter。完整 Point-Cache 能把 jitter 平均值从 54.81 提升到 63.86，但该数值仍然低于其他 corruption。

完整 Point-Cache 后，dropout_local 和 dropout_global 成为第二、第三困难 corruption，说明点云缺失仍然是后续方法需要重点处理的问题。

---

## 12. 低准确率区域分析

| 条件 | Zero-shot 数量 | ZS + Global 数量 | ZS + Global + Local 数量 |
|---|---:|---:|---:|
| Acc < 70 | 17 / 35 | 10 / 35 | 6 / 35 |
| Acc < 60 | 10 / 35 | 4 / 35 | 3 / 35 |
| Acc < 50 | 5 / 35 | 2 / 35 | 0 / 35 |
| Acc < 40 | 1 / 35 | 0 / 35 | 0 / 35 |

分析：

Global Cache 明显减少了低准确率区域。Acc < 70 的 setting 从 17 个减少到 10 个，Acc < 60 的 setting 从 10 个减少到 4 个。

Local Cache 进一步减少了低准确率区域。Acc < 70 的 setting 从 10 个降到 6 个，Acc < 50 的 setting 从 2 个降到 0 个。

这说明完整 Point-Cache 不仅提高平均准确率，也显著减少了低性能 setting 的数量。

---

## 13. 关键困难 setting

| cor_type | Zero-shot | ZS + Global | ZS + Global + Local | 现象 |
|---|---:|---:|---:|---|
| dropout_global_4 | 45.62 | 49.72 | 51.13 | 最高 severity 全局缺失仍然最低 |
| jitter_4 | 35.45 | 48.95 | 52.39 | 最高 severity 坐标扰动提升明显，但仍困难 |
| dropout_local_4 | 47.73 | 52.27 | 55.39 | 高 severity 局部缺失仍困难 |
| jitter_3 | 48.26 | 55.35 | 57.90 | 中高 severity 坐标扰动仍偏低 |
| jitter_2 | 56.16 | 61.30 | 62.12 | severity=2 jitter 仍较低 |
| dropout_local_3 | 58.47 | 62.44 | 64.30 | 中高 severity 局部缺失仍偏低 |
| dropout_global_3 | 57.82 | 63.13 | 65.68 | 中高 severity 全局缺失仍偏低 |
| add_local_3 | 50.20 | 62.40 | 67.79 | 局部异常点被 cache 明显改善 |
| add_local_4 | 48.58 | 62.28 | 68.48 | 高 severity 局部异常点被明显改善 |

分析：

jitter_4 和 add_local_4 在完整 Point-Cache 后有大幅提升，但仍低于较容易的 corruption。dropout_global_4 成为完整 Point-Cache 下最低 setting，说明高 severity 全局缺失是完整方法仍难以解决的问题。

---

## 14. 方法贡献分解

以 all35 Average 为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

72.81 - 66.89 = +5.92

其中：

| 贡献来源 | all35 Avg 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +4.35 | 约 73.5% |
| Local Cache | +1.57 | 约 26.5% |
| 完整 Point-Cache | +5.92 | 100.00% |

以 severity=2 Average 为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

73.66 - 67.80 = +5.86

其中：

| 贡献来源 | S2 Avg 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +4.41 | 约 75.3% |
| Local Cache | +1.45 | 约 24.7% |
| 完整 Point-Cache | +5.86 | 100.00% |

分析：

不管按 all35 还是按 severity=2，Global Cache 都是主要贡献来源，但 Local Cache 也有明确额外贡献，占完整提升约 25%。

这与 22 组 OpenShape × ModelNet-C 不同。22 组中 Local Cache 几乎无额外贡献；32 组中 Local Cache 明确有效，说明不同 backbone 对 Local Cache 的响应不同。

---

## 15. 与 31 组 ModelNet clean 的关系

31 组是 Uni3D 在 ModelNet clean 上的结果；32 组是 Uni3D 在 ModelNet-C all35 上的结果。

| 方法 | 31 组 clean | 32 组 S2 Avg | 32 组 all35 Avg |
|---|---:|---:|---:|
| Zero-shot | 81.85 | 67.80 | 66.89 |
| ZS + Global | 83.23 | 72.21 | 71.24 |
| ZS + Global + Local | 83.71 | 73.66 | 72.81 |

从 clean 到 corrupted ModelNet-C 的下降：

| 方法 | S2 Avg - clean | all35 Avg - clean |
|---|---:|---:|
| Zero-shot | -14.05 | -14.96 |
| ZS + Global | -11.02 | -11.99 |
| ZS + Global + Local | -10.05 | -10.90 |

分析：

ModelNet-C corruption 会在 ModelNet clean 的基础上明显降低 Uni3D 性能。Zero-shot 从 81.85 下降到 all35 Avg 66.89，下降 -14.96。

Global Cache 和完整 Point-Cache 都缩小了 clean-to-corruption gap。Global Cache 将 all35 gap 缩小到 -11.99，完整 Point-Cache 将 all35 gap 进一步缩小到 -10.90。

这说明完整 Point-Cache 不仅提高了 corrupted accuracy，也增强了 Uni3D 对 corruption 的鲁棒性保持能力。

---

## 16. 与 02 / 12 / 22 组 ModelNet-C 的关系

02 组是 ULIP 在 ModelNet-C all35 上的结果；12 组是 ULIP-2；22 组是 OpenShape；32 组是 Uni3D。

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
| Uni3D | Zero-shot | 67.80 | 66.89 |
| Uni3D | ZS + Global | 72.21 | 71.24 |
| Uni3D | ZS + Global + Local | 73.66 | 72.81 |

分析：

Uni3D 在 ModelNet-C 上明显强于 ULIP 和 ULIP-2，但低于 OpenShape。完整 Point-Cache 下，Uni3D all35 Avg 为 72.81，OpenShape 为 75.14，差距为 -2.33。

不过，Uni3D 的 cache 增益比 OpenShape 更明显。OpenShape 在 ModelNet-C 上主要依赖 Global Cache，Local Cache 额外贡献接近 0；Uni3D 中 Global 和 Local 都有明显贡献。

---

## 17. 与 22 组 Local Cache 现象的差异

22 组 OpenShape × ModelNet-C all35 中，Local Cache 在 Global Cache 基础上的额外贡献几乎为零：

| 组别 | 数据设置 | Local Cache extra |
|---|---|---:|
| 22 组 | OpenShape × ModelNet-C all35 | +0.01 all35 / -0.13 S2 |
| 32 组 | Uni3D × ModelNet-C all35 | +1.57 all35 / +1.45 S2 |

分析：

这说明 Local Cache 的价值具有 backbone 依赖性。

在 OpenShape × ModelNet-C 上，OpenShape 的全局特征已经非常强，Global Cache 提供了主要鲁棒性补偿，Local Cache 边际收益很弱。

在 Uni3D × ModelNet-C 上，Local Cache 对 add_local 和 jitter 有明显额外提升，说明 Uni3D 在 corrupted setting 中更能从局部缓存中获益。

这对后续 MCM-PC 很重要：不能简单假设 Local Cache 对所有 backbone 都同样有效或无效，应根据 backbone、样本可靠性和全局-局部一致性动态调节。

---

## 18. 当前结果意义分析

32 组结果说明：

| 观察 | 解释 |
|---|---|
| Zero-shot all35 = 66.89 | Uni3D 在 ModelNet-C 上的基础鲁棒性 |
| ZS + Global all35 = 71.24 | Global Cache 有明确正增益 |
| ZS + Global + Local all35 = 72.81 | 完整 Point-Cache 最好 |
| Global extra = +4.35 all35 | Global Cache 是主要提升来源 |
| Local extra = +1.57 all35 | Local Cache 也有明确额外贡献 |
| add_local 提升最大 | 局部异常点从 cache 中明显受益 |
| jitter 仍然最低 | 坐标扰动仍然是主要困难 corruption |
| high-severity dropout 仍然困难 | 点云缺失仍需要进一步机制 |

32 组是一个非常关键的实验组。它说明 Uni3D 在 ModelNet-C 上仍然存在明显鲁棒性下降，而完整 Point-Cache 可以显著提升性能。

---

## 19. 对后续 MCM-PC 的启发

当前 32 组实验对后续方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| ModelNet-C corruption 会显著降低 Uni3D 表现 | 强 backbone 仍需要测试时适应 |
| Global Cache 提供 +4.35 all35 | 全局缓存是稳定主模块，应保留 |
| Local Cache 提供 +1.57 all35 | 局部缓存也有明确价值 |
| add_local 提升最大 | 局部异常点需要局部证据建模 |
| jitter 仍然最低 | 坐标扰动需要专门鲁棒机制 |
| high-severity dropout 仍然困难 | 点云缺失需要额外处理 |
| Uni3D 与 OpenShape 的 Local Cache 现象不同 | cache 贡献应考虑 backbone 依赖性 |
| 32 组必须使用 modelnet40 checkpoint | checkpoint 是 Uni3D 实验关键设置 |

这对 MCM-PC 很重要：后续方法不应简单固定 Global / Local 的作用，而应根据 backbone、样本可靠性、全局-局部一致性、伪标签可信度和域偏移类型动态调节缓存贡献。

---

## 20. 阶段性结论

32 组 Uni3D × ModelNet-C all35 baseline 已完成。

主要结论如下：

1. 三个子实验均完成，并且 summary.csv、cor_type、log_path 和 logs 文件数量均为 35。
2. 当前正式结果均使用 weights/uni3d/modelnet40/model.pt。
3. 32_1 Zero-shot 的 severity=2 Avg 为 67.80，原文为 67.95，差异 -0.15。
4. 32_2 ZS + Global 的 severity=2 Avg 为 72.21，原文为 71.81，差异 +0.40。
5. 32_3 ZS + Global + Local 的 severity=2 Avg 为 73.66，原文为 73.31，差异 +0.35。
6. 三个结果整体与原文高度接近，均可作为有效复现结果保留。
7. all35 Avg 从 Zero-shot 的 66.89 提升到 Global 的 71.24，再提升到 Global + Local 的 72.81。
8. Global Cache 是主要提升来源，all35 Avg 提升 +4.35，severity=2 Avg 提升 +4.41。
9. Local Cache 在 Global Cache 基础上有明确额外贡献，all35 Avg 提升 +1.57，severity=2 Avg 提升 +1.45。
10. 完整 Point-Cache 相比 Zero-shot 提升明显，all35 Avg 提升 +5.92，severity=2 Avg 提升 +5.86。
11. add_local 是完整 Point-Cache 提升最大的 corruption，平均提升 +14.36。
12. jitter 是最困难 corruption，完整 Point-Cache 后 all35 平均仍只有 63.86。
13. dropout_global_4 是完整 Point-Cache 后最低 setting，准确率为 51.13。
14. 32 组结果说明：在 Uni3D × ModelNet-C 中，Global Cache 和 Local Cache 都有价值。
15. 32 组完成了 D 组 Uni3D 的第二个数据设置 baseline 复现。
16. 31 / 32 组后续应统一使用 weights/uni3d/modelnet40/model.pt。

---

## 21. 运行命令汇总

32_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs_single_gpu.sh 0

32_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global_single_gpu.sh 0

32_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 1

---

## 22. 检查命令汇总

32_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/32_1_uni3d_modelnetc_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

32_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

32_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
