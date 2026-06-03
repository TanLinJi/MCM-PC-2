# 03_ulip_scanobjnn_clean_hardest_summary

## 1. 实验组目的

本总文档汇总 ULIP 在 ScanObjNN clean hardest 上的三组 baseline 复现实验。

03 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | ULIP |
| Dataset | ScanObjNN clean hardest |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 输入点数 | 1024 |
| 测试设置数 | 1 个 clean setting |

三组实验分别为：

| 实验编号 | 方法 | 作用 |
|---|---|---|
| 03_1_ulip_scanobjnn_clean_hardest_zs | Zero-shot | 无缓存基础对照 |
| 03_2_ulip_scanobjnn_clean_hardest_zs_global | Zero-shot + Global Cache | 验证全局缓存增益 |
| 03_3_ulip_scanobjnn_clean_hardest_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 增益 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| ULIP 在 ScanObjNN clean hardest 上的 Zero-shot 基础性能是多少？ | 由 03_1 给出 |
| Global Cache 是否有效？ | 比较 03_2 - 03_1 |
| Local Cache 是否有额外贡献？ | 比较 03_3 - 03_2 |
| 三个结果是否与原文对齐？ | 分别与原文 Supplementary Table 7 对比 |
| ScanObjNN clean hardest 与 ModelNet clean 有什么不同？ | ScanObjNN hardest 是真实扫描数据，更接近 real-world domain shift |

需要特别注意：03 组是 clean 单文件实验，不是 all35 corruption 实验。因此本文档不包含 corruption × severity 矩阵，而是围绕 clean hardest 的单点 accuracy、原文对齐和方法间增益展开分析。

---

## 2. 当前实现方式

03 组使用普通 bash 脚本执行，不使用 02 组的 Python 内部 all35 优化 runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/03_run_ulip_scanobjnn_clean_hardest_common.sh |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |
| 数据文件 | Point-Cache/data/sonn_c/hardest/clean.h5 |

之所以不使用 all35 优化 runner，是因为 03 组只运行一个 clean 文件，不存在 35 次重复加载模型的问题。保持普通脚本结构更简单，也更容易和 01 组 clean 实验保持一致。

---

## 3. 三个实验的输出完整性检查

| 实验编号 | 方法 | summary 行数 | cor_type 唯一数 | log_path 唯一数 | status | 状态 |
|---|---|---:|---:|---:|---|---|
| 03_1_ulip_scanobjnn_clean_hardest_zs | Zero-shot | 1 | 1 | 1 | done | 完成 |
| 03_2_ulip_scanobjnn_clean_hardest_zs_global | ZS + Global | 1 | 1 | 1 | done | 完成 |
| 03_3_ulip_scanobjnn_clean_hardest_zs_global_local | ZS + Global + Local | 1 | 1 | 1 | done | 完成 |

说明：

1. 03 组每个子实验都只对应 `clean.h5` 一个测试文件，因此 summary 行数应为 1。
2. 三个子实验均为 `status=done`。
3. 每个子实验都有唯一 log_path，说明结果和日志可以一一对应。
4. 执行完整性正常并不等于结果正常；结果是否正常还需要与原文参考值对比。

---

## 4. 核心结果总表

| 实验编号 | 方法 | 当前复现值 | 原文参考值 | Diff = 当前 - 原文 | 是否对齐 |
|---|---|---:|---:|---:|---|
| 03_1_ulip_scanobjnn_clean_hardest_zs | Zero-shot | 29.08 | 29.29 | -0.21 | 是 |
| 03_2_ulip_scanobjnn_clean_hardest_zs_global | ZS + Global Cache | 32.20 | 32.37 | -0.17 | 是 |
| 03_3_ulip_scanobjnn_clean_hardest_zs_global_local | ZS + Global + Local Cache | 32.48 | 32.48 | +0.00 | 是 |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | -0.13 |
| MAE | 0.13 |
| Max Abs Diff | 0.21 |

分析：

三个子实验均与原文 Supplementary Table 7 基本对齐。最大绝对差异为 03_1 的 0.21，属于很小误差；03_3 与原文完全一致。

因此，03 组不仅脚本执行成功，而且数值复现也可靠。

---

## 5. 方法间增益分析

### 5.1 当前复现增益

| 比较 | 当前复现值 | 增益 |
|---|---:|---:|
| 03_1 Zero-shot | 29.08 | — |
| 03_2 ZS + Global | 32.20 | +3.12 over 03_1 |
| 03_3 ZS + Global + Local | 32.48 | +0.28 over 03_2 |
| 03_3 ZS + Global + Local | 32.48 | +3.40 over 03_1 |

### 5.2 原文增益

| 比较 | 原文值 | 增益 |
|---|---:|---:|
| Original Data | 29.29 | — |
| + Global Cache | 32.37 | +3.08 over Original Data |
| + Hierarchical Cache | 32.48 | +0.11 over Global Cache |
| + Hierarchical Cache | 32.48 | +3.19 over Original Data |

### 5.3 当前增益与原文增益对比

| 增益来源 | 原文增益 | 当前复现增益 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | +3.08 | +3.12 | +0.04 |
| Local Cache extra over Global | +0.11 | +0.28 | +0.17 |
| Full Point-Cache over Zero-shot | +3.19 | +3.40 | +0.21 |

分析：

当前复现中，Global Cache 将准确率从 29.08 提升到 32.20，提升 +3.12；原文对应提升为 +3.08。二者几乎一致，说明 Global Cache 的主要增益复现成功。

Local Cache 在 Global Cache 基础上进一步提升 +0.28，原文对应提升为 +0.11。当前复现的 Local Cache 额外增益略高于原文，但总体仍然是小幅增益。

完整 Point-Cache 相比 Zero-shot 的总提升为 +3.40，和原文 +3.19 接近。

---

## 6. 方法贡献分解

以当前复现结果为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

32.48 - 29.08 = +3.40

其中：

| 贡献来源 | 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +3.12 | 91.76% |
| Local Cache | +0.28 | 8.24% |
| 完整 Point-Cache | +3.40 | 100.00% |

分析：

在 ScanObjNN clean hardest 上，Global Cache 是绝对主要的提升来源，占完整提升的约 91.76%。Local Cache 有正向贡献，但占比只有约 8.24%。

这说明在 clean hardest 场景下，全局缓存能够捕获主要的测试分布信息，而局部缓存只提供少量补充。该现象和 02 组 ModelNet-C all35 的结论方向一致：Global Cache 是主贡献，Local Cache 是补充增益。

---

## 7. 与 01 / 02 组的关系

### 7.1 与 01 组 ModelNet clean 的关系

01 组是 ULIP 在 ModelNet clean 上的结果；03 组是 ULIP 在 ScanObjNN clean hardest 上的结果。

| 组号 | Dataset | 数据性质 | 分析重点 |
|---|---|---|---|
| 01 | ModelNet clean | CAD / synthetic object 数据 | clean synthetic domain |
| 03 | ScanObjNN clean hardest | real-world scanned object 数据 | clean real-scan hardest domain |

ScanObjNN hardest split 比 ModelNet clean 更困难。03_1 的 Zero-shot 只有 29.08，说明真实扫描数据即使没有 synthetic corruption，也存在明显 domain gap。

因此，03 组不应简单和 01 组数值直接比较“高低是否正常”，而应优先与原文 ScanObjNN hardest clean 的参考值对齐。

### 7.2 与 02 组 ModelNet-C all35 的关系

02 组是 ModelNet-C all35 corruption 实验，03 组是 ScanObjNN clean hardest 实验。

| 组号 | Dataset setting | 是否 corruption | 是否 all35 |
|---|---|---:|---:|
| 02 | ModelNet-C all35 | 是 | 是 |
| 03 | ScanObjNN clean hardest | 否 | 否 |

02 组关注 synthetic corruption robustness；03 组关注 real-world scanned clean domain。两者都属于 ULIP baseline 复现的一部分，但分析重点不同。

02 组中可以分析 corruption × severity；03 组中只能分析单点 clean hardest performance 和 cache 增益。

---

## 8. 结果意义分析

03 组结果说明：

| 观察 | 解释 |
|---|---|
| Zero-shot = 29.08 | ULIP 在真实扫描 hardest clean 上基础性能较低 |
| Global Cache = 32.20 | 测试流全局缓存能够缓解真实扫描域偏移 |
| Global + Local = 32.48 | 局部缓存提供小幅额外收益 |
| Global 增益远大于 Local 增益 | clean hardest 中全局分布信息比局部缓存贡献更明显 |
| 三个结果均与原文对齐 | 03 组复现可靠 |

ScanObjNN clean hardest 的特殊性在于：即使没有 synthetic corruption，它仍然是一个困难域。真实扫描数据可能包含遮挡、背景干扰、扫描噪声、点云不完整和类别间结构混淆。因此，Point-Cache 在这里的意义是缓解真实扫描域偏移，而不仅仅是处理人工 corruption。

---

## 9. 对后续 04 组的意义

03 组完成后，ULIP 在 ScanObjNN clean hardest 上的 baseline 已经明确：

| 方法 | clean hardest Accuracy |
|---|---:|
| Zero-shot | 29.08 |
| ZS + Global | 32.20 |
| ZS + Global + Local | 32.48 |

下一步 04 组将进入：

ULIP × ScanObjNN-C hardest all35

也就是在 ScanObjNN hardest split 上评估 7 种 corruption × 5 个 severity。

03 组将作为 04 组的 clean 参考：

| 后续比较 | 目的 |
|---|---|
| 04_1 vs 03_1 | 观察 corruption 相比 clean 对 Zero-shot 的影响 |
| 04_2 vs 03_2 | 观察 corruption 相比 clean 对 Global Cache 的影响 |
| 04_3 vs 03_3 | 观察 corruption 相比 clean 对完整 Point-Cache 的影响 |
| 04_2 - 04_1 | 评估 Global Cache 在 ScanObjNN-C 上的增益 |
| 04_3 - 04_2 | 评估 Local Cache 在 ScanObjNN-C 上的额外增益 |

尤其需要关注：Local Cache 在 clean hardest 上只提供 +0.28 的小幅收益，那么在 corrupted ScanObjNN-C 上，Local Cache 是否会更有价值，还是会因为局部结构被破坏而变得不稳定。这是 04 组需要重点观察的问题。

---

## 10. 阶段性结论

03 组 ULIP × ScanObjNN clean hardest baseline 已完成。

主要结论如下：

1. 三个子实验均完成，summary.csv 行数均为 1，status 均为 done。
2. 03_1 Zero-shot 当前复现值为 29.08，原文参考值为 29.29，差异 -0.21。
3. 03_2 ZS + Global 当前复现值为 32.20，原文参考值为 32.37，差异 -0.17。
4. 03_3 ZS + Global + Local 当前复现值为 32.48，原文参考值为 32.48，差异 +0.00。
5. 三个结果均与原文基本对齐，03 组复现可靠。
6. 当前复现中 Global Cache 带来 +3.12 提升，是主要增益来源。
7. Local Cache 在 Global Cache 基础上额外提升 +0.28，贡献较小但为正。
8. 完整 Point-Cache 相比 Zero-shot 提升 +3.40。
9. ScanObjNN clean hardest 即使没有 synthetic corruption，Zero-shot 性能也较低，说明真实扫描域本身存在明显 domain gap。
10. 03 组结果将作为后续 04 组 ScanObjNN-C hardest all35 的 clean 参考。

---

## 11. 运行命令汇总

03_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/03_1_ulip_scanobjnn_clean_hardest_zs_single_gpu.sh 0

03_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global_single_gpu.sh 0

03_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh 0

---

## 12. 检查命令汇总

03_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/03_1_ulip_scanobjnn_clean_hardest_zs/summary.csv | wc -l

tail -n +2 results/baseline/03_1_ulip_scanobjnn_clean_hardest_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/03_1_ulip_scanobjnn_clean_hardest_zs/logs -maxdepth 1 -name '*.log' | wc -l

03_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

03_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
