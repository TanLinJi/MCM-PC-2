# 13_ulip2_scanobjnn_clean_hardest_summary

## 1. 实验组目的

本总文档汇总 ULIP-2 在 ScanObjNN clean hardest 上的三组 baseline 复现实验。

13 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | ULIP-2 |
| Dataset | ScanObjNN clean hardest |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 输入点数 | 1024 |
| 测试设置数 | 1 个 clean hardest setting |

三组实验分别为：

| 实验编号 | 方法 | 作用 |
|---|---|---|
| 13_1_ulip2_scanobjnn_clean_hardest_zs | Zero-shot | 无缓存基础对照 |
| 13_2_ulip2_scanobjnn_clean_hardest_zs_global | Zero-shot + Global Cache | 验证全局缓存增益 |
| 13_3_ulip2_scanobjnn_clean_hardest_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 增益 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| ULIP-2 在 ScanObjNN clean hardest 上的 Zero-shot 基础性能是多少？ | 由 13_1 给出 |
| Global Cache 是否有效？ | 比较 13_2 - 13_1 |
| Local Cache 是否有额外贡献？ | 比较 13_3 - 13_2 |
| 三个结果是否与原文对齐？ | 分别与原文 Supplementary Table 7 对比 |
| ScanObjNN clean hardest 与 ModelNet clean 有什么差异？ | 反映真实扫描域偏移 |
| ULIP-2 相比 ULIP 在 ScanObjNN clean hardest 上是否更强？ | 与 03 组结果对比 |

需要特别注意：13 组是 clean 单文件实验，不是 all35 corruption 实验。因此本文档不包含 corruption × severity 矩阵，而是围绕 ScanObjNN clean hardest 的单点 accuracy、原文对齐、方法间增益和跨数据集 / 跨 backbone 对比展开分析。

---

## 2. 当前实现方式

13 组使用普通 bash 脚本执行，不使用 all35 优化 runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/13_run_ulip2_scanobjnn_clean_hardest_common.sh |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |
| 数据文件 | Point-Cache/data/sonn_c/hardest/clean.h5 |
| Backbone 权重 | Point-Cache/weights/ulip/pointbert_ulip2.pt |
| 文本编码器权重 | Point-Cache/weights/ulip/slip_base_100ep.pt |

之所以不使用 all35 优化 runner，是因为 13 组只运行一个 clean 文件，不存在 35 次重复加载模型的问题。保持普通脚本结构更简单，也与 03 组 ScanObjNN clean hardest 的实验组织方式一致。

---

## 3. 三个实验的输出完整性检查

| 实验编号 | 方法 | summary 行数 | cor_type 唯一数 | log_path 唯一数 | logs 文件数 | status | 状态 |
|---|---|---:|---:|---:|---:|---|---|
| 13_1_ulip2_scanobjnn_clean_hardest_zs | Zero-shot | 1 | 1 | 1 | 1 | done | 完成 |
| 13_2_ulip2_scanobjnn_clean_hardest_zs_global | ZS + Global | 1 | 1 | 1 | 1 | done | 完成 |
| 13_3_ulip2_scanobjnn_clean_hardest_zs_global_local | ZS + Global + Local | 1 | 1 | 1 | 1 | done | 完成 |

说明：

1. 13 组每个子实验都只对应 `clean.h5` 一个测试文件，因此 summary 行数应为 1。
2. 三个子实验均为 `status=done`。
3. 每个子实验都有唯一 log_path，说明结果和日志可以一一对应。
4. 每个 logs 目录均只有 1 个 log 文件，没有旧日志或重复日志残留。
5. 执行完整性正常并不等于结果正常；结果是否正常还需要与原文参考值对比。

---

## 4. 核心结果总表

| 实验编号 | 方法 | 当前复现值 | 原文参考值 | Diff = 当前 - 原文 | 是否对齐 |
|---|---|---:|---:|---:|---|
| 13_1_ulip2_scanobjnn_clean_hardest_zs | Zero-shot | 34.07 | 33.38 | +0.69 | 基本对齐，略高 |
| 13_2_ulip2_scanobjnn_clean_hardest_zs_global | ZS + Global Cache | 40.42 | 40.28 | +0.14 | 高度对齐 |
| 13_3_ulip2_scanobjnn_clean_hardest_zs_global_local | ZS + Global + Local Cache | 42.44 | 42.40 | +0.04 | 高度对齐 |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.29 |
| MAE | 0.29 |
| Max Abs Diff | 0.69 |

分析：

13 组三个子实验均完成，并且三种方法的绝对结果整体与原文 Supplementary Table 7 对齐。

其中 13_2 和 13_3 与原文非常接近，差异分别只有 +0.14 和 +0.04。13_1 Zero-shot 比原文高 +0.69，略高但仍可接受。由于 13_1 略高，后续基于当前复现值计算的 cache 总增益会略小于原文增益。

---

## 5. 方法间增益分析

### 5.1 当前复现增益

| 比较 | 当前复现值 | 增益 |
|---|---:|---:|
| 13_1 Zero-shot | 34.07 | — |
| 13_2 ZS + Global | 40.42 | +6.35 over 13_1 |
| 13_3 ZS + Global + Local | 42.44 | +2.02 over 13_2 |
| 13_3 ZS + Global + Local | 42.44 | +8.37 over 13_1 |

### 5.2 原文增益

| 比较 | 原文值 | 增益 |
|---|---:|---:|
| Zero-shot | 33.38 | — |
| + Global Cache | 40.28 | +6.90 over Zero-shot |
| + Hierarchical Cache | 42.40 | +2.12 over Global Cache |
| + Hierarchical Cache | 42.40 | +9.02 over Zero-shot |

### 5.3 当前增益与原文增益对比

| 增益来源 | 原文增益 | 当前复现增益 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | +6.90 | +6.35 | -0.55 |
| Local Cache extra over Global | +2.12 | +2.02 | -0.10 |
| Full Point-Cache over Zero-shot | +9.02 | +8.37 | -0.65 |

分析：

当前复现中的方法趋势正确：

Zero-shot < ZS + Global Cache < ZS + Global Cache + Local Cache

Global Cache 和 Local Cache 都带来明显正向提升。当前 Global Cache 增益为 +6.35，Local Cache 额外增益为 +2.02，完整 Point-Cache 总增益为 +8.37。

当前方法间增益略低于原文，主要原因是当前 13_1 Zero-shot 复现值 34.07 高于原文 33.38，导致以当前 13_1 为基线计算的增益被轻微压缩。尤其 Local Cache 的额外增益与原文非常接近，仅相差 -0.10，说明局部缓存贡献复现良好。

---

## 6. 方法贡献分解

以当前复现结果为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

42.44 - 34.07 = +8.37

其中：

| 贡献来源 | 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +6.35 | 75.87% |
| Local Cache | +2.02 | 24.13% |
| 完整 Point-Cache | +8.37 | 100.00% |

分析：

在 ULIP-2 × ScanObjNN clean hardest 上，Global Cache 是主要提升来源，占完整提升的约 75.87%。Local Cache 也有明显贡献，占约 24.13%。

这与 11 组 ModelNet clean 不同。11 组中 Local Cache 只带来 +0.36，而 13 组中 Local Cache 带来 +2.02。说明真实扫描 hardest split 中局部结构更复杂、更不完整，Local Cache 有更大的补偿空间。

---

## 7. 与 11 组 ModelNet clean 的关系

11 组是 ULIP-2 在 ModelNet clean 上的结果；13 组是 ULIP-2 在 ScanObjNN clean hardest 上的结果。

| 方法 | 11 组 ModelNet clean | 13 组 ScanObjNN clean hardest | 差值 |
|---|---:|---:|---:|
| Zero-shot | 72.20 | 34.07 | -38.13 |
| ZS + Global | 73.99 | 40.42 | -33.57 |
| ZS + Global + Local | 74.35 | 42.44 | -31.91 |

分析：

ScanObjNN clean hardest 比 ModelNet clean 难很多。即使两者都是 clean 数据，ScanObjNN hardest 是真实扫描点云，包含遮挡、缺失、背景干扰、扫描噪声和真实几何复杂性；ModelNet clean 则更接近规则 CAD 数据。

从 clean ModelNet 到 clean ScanObjNN hardest 的性能下降为：

| 方法 | 下降 |
|---|---:|
| Zero-shot | -38.13 |
| ZS + Global | -33.57 |
| ZS + Global + Local | -31.91 |

可以看到，cache 机制能够缓解一部分跨数据域下降。完整 Point-Cache 将下降幅度从 Zero-shot 的 -38.13 缩小到 -31.91，说明 Global + Local Cache 对真实扫描域偏移有实际帮助。

---

## 8. 13 组与 11 组的 cache 增益对比

| 数据设置 | Global 增益 | Local 额外增益 | 完整 Point-Cache 总增益 |
|---|---:|---:|---:|
| 11 组 ModelNet clean | +1.79 | +0.36 | +2.15 |
| 13 组 ScanObjNN clean hardest | +6.35 | +2.02 | +8.37 |

分析：

Point-Cache 在 ScanObjNN clean hardest 上的提升远大于在 ModelNet clean 上的提升。

这说明：

1. ModelNet clean 中 ULIP-2 的基础性能已经很高，cache 边际收益有限。
2. ScanObjNN clean hardest 中存在明显真实域偏移，Zero-shot 起点较低。
3. 测试时缓存机制能够利用测试流中的分布信息，缓解真实扫描数据中的 domain gap。
4. Local Cache 在真实扫描数据上比在规则 CAD 数据上更有价值。

因此，13 组结果说明 Point-Cache 的真正价值不仅在 synthetic corruption，也体现在真实扫描数据的 clean hardest setting 中。

---

## 9. 与 03 组 ULIP ScanObjNN clean hardest 的关系

03 组是 ULIP 在 ScanObjNN clean hardest 上的结果；13 组是 ULIP-2 在同一数据设置上的结果。

| Backbone | 方法 | Accuracy |
|---|---|---:|
| ULIP | Zero-shot | 29.08 |
| ULIP | ZS + Global | 32.20 |
| ULIP | ZS + Global + Local | 32.48 |
| ULIP-2 | Zero-shot | 34.07 |
| ULIP-2 | ZS + Global | 40.42 |
| ULIP-2 | ZS + Global + Local | 42.44 |

ULIP-2 相比 ULIP 的提升：

| 方法 | ULIP-2 - ULIP |
|---|---:|
| Zero-shot | +4.99 |
| ZS + Global | +8.22 |
| ZS + Global + Local | +9.96 |

分析：

在 ScanObjNN clean hardest 上，ULIP-2 明显优于 ULIP。特别是在加入 cache 后，ULIP-2 的优势更大：

1. Zero-shot 阶段，ULIP-2 比 ULIP 高 +4.99。
2. Global Cache 阶段，ULIP-2 比 ULIP 高 +8.22。
3. 完整 Point-Cache 阶段，ULIP-2 比 ULIP 高 +9.96。

这说明 ULIP-2 不只是基础表征更强，而且其特征空间也可能更适合测试时缓存检索，使 Global / Local Cache 能发挥更大的作用。

---

## 10. 与 12 组 ModelNet-C all35 的关系

12 组是 ULIP-2 在 ModelNet-C all35 上的结果；13 组是 ULIP-2 在 ScanObjNN clean hardest 上的结果。

| 数据设置 | 方法 | Accuracy / Avg |
|---|---|---:|
| ModelNet-C all35 | Zero-shot | 58.07 |
| ModelNet-C all35 | ZS + Global | 61.09 |
| ModelNet-C all35 | ZS + Global + Local | 62.49 |
| ScanObjNN clean hardest | Zero-shot | 34.07 |
| ScanObjNN clean hardest | ZS + Global | 40.42 |
| ScanObjNN clean hardest | ZS + Global + Local | 42.44 |

分析：

虽然 ModelNet-C 是 corrupted setting，ScanObjNN clean hardest 是 clean setting，但 ScanObjNN clean hardest 的数值仍显著更低。这说明真实扫描 hardest split 本身带来的 domain shift 非常强，甚至比 synthetic corruption 后的 ModelNet-C 更难。

因此后续 14 组 ScanObjNN-C hardest all35 很可能是更具挑战的数据设置，需要重点观察：

| 后续观察点 | 说明 |
|---|---|
| 14_1 vs 13_1 | corruption 在真实扫描 hardest 上造成多大额外下降 |
| 14_2 vs 14_1 | Global Cache 在真实扫描 corrupted setting 上是否仍然有效 |
| 14_3 vs 14_2 | Local Cache 是否继续有效，还是会在强 corruption 下变得不稳定 |
| 14 组困难 corruption | 尤其关注 jitter、add_local、dropout_local 等 |
| Local Cache 可靠性 | 观察是否存在类似 12 组 jitter 高 severity 的负增益现象 |

---

## 11. 结果意义分析

13 组结果说明：

| 观察 | 解释 |
|---|---|
| Zero-shot = 34.07 | ULIP-2 在真实扫描 hardest clean 上基础性能较低 |
| ZS + Global = 40.42 | Global Cache 能显著缓解真实扫描域偏移 |
| ZS + Global + Local = 42.44 | Local Cache 在 Global Cache 基础上继续明显提升 |
| 13_2 与原文差 +0.14 | Global Cache 绝对结果高度复现 |
| 13_3 与原文差 +0.04 | 完整 Point-Cache 绝对结果高度复现 |
| 完整 Point-Cache 总增益 +8.37 | 测试时缓存机制在真实扫描数据上非常有效 |

ScanObjNN clean hardest 的特殊性在于：即使没有 synthetic corruption，它仍然是一个困难域。真实扫描数据可能包含遮挡、背景干扰、扫描噪声、点云不完整和类别间结构混淆。因此，Point-Cache 在这里的意义是缓解真实扫描域偏移，而不仅仅是处理人工 corruption。

---

## 12. 对后续 14 组的意义

13 组完成后，ULIP-2 在 ScanObjNN clean hardest 上的 baseline 已经明确：

| 方法 | ScanObjNN clean hardest Accuracy |
|---|---:|
| Zero-shot | 34.07 |
| ZS + Global | 40.42 |
| ZS + Global + Local | 42.44 |

下一步 14 组将进入：

ULIP-2 × ScanObjNN-C hardest all35

也就是在 ScanObjNN hardest split 上评估 7 种 corruption × 5 个 severity。

13 组将作为 14 组的 clean 参考：

| 后续比较 | 目的 |
|---|---|
| 14_1 vs 13_1 | 观察 corruption 相比 clean 对 Zero-shot 的影响 |
| 14_2 vs 13_2 | 观察 corruption 相比 clean 对 Global Cache 的影响 |
| 14_3 vs 13_3 | 观察 corruption 相比 clean 对完整 Point-Cache 的影响 |
| 14_2 - 14_1 | 评估 Global Cache 在 ScanObjNN-C 上的增益 |
| 14_3 - 14_2 | 评估 Local Cache 在 ScanObjNN-C 上的额外增益 |

尤其需要关注：13 组中 Local Cache 额外提升 +2.02，说明局部缓存对真实扫描 clean hardest 有明显作用；但在 14 组 corrupted setting 中，局部结构可能被进一步破坏，Local Cache 是否仍然稳定，需要通过实验确认。

---

## 13. 对后续 MCM-PC 的启发

当前 13 组实验对后续方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| ScanObjNN clean hardest 的 Zero-shot 很低 | 真实扫描域偏移是重要问题 |
| Global Cache 提升 +6.35 | 全局缓存是缓解真实域偏移的关键模块 |
| Local Cache 额外提升 +2.02 | 局部缓存对真实扫描数据有明显价值 |
| ULIP-2 比 ULIP 提升明显 | 更强 backbone 能提升缓存检索质量 |
| 完整 Point-Cache 仍只有 42.44 | 真实扫描 hardest 仍有很大提升空间 |
| 真实扫描 clean 比 ModelNet-C all35 更低 | 真实域偏移比部分 synthetic corruption 更具挑战 |

因此，后续 MCM-PC 方法设计不应只关注 synthetic corruption，也应强调真实扫描域偏移。ScanObjNN clean hardest 和 ScanObjNN-C hardest 是验证方法是否真正适用于 real-world point cloud 的关键数据设置。

---

## 14. 阶段性结论

13 组 ULIP-2 × ScanObjNN clean hardest baseline 已完成。

主要结论如下：

1. 三个子实验均完成，summary.csv 行数均为 1，status 均为 done。
2. 13_1 Zero-shot 当前复现值为 34.07，原文参考值为 33.38，差异 +0.69。
3. 13_2 ZS + Global 当前复现值为 40.42，原文参考值为 40.28，差异 +0.14。
4. 13_3 ZS + Global + Local 当前复现值为 42.44，原文参考值为 42.40，差异 +0.04。
5. 三个结果整体与原文基本对齐，其中 13_2 和 13_3 高度对齐。
6. 当前复现中 Global Cache 带来 +6.35 提升，是主要增益来源。
7. Local Cache 在 Global Cache 基础上额外提升 +2.02，贡献明显且与原文接近。
8. 完整 Point-Cache 相比 Zero-shot 提升 +8.37。
9. 当前方法趋势正确：Zero-shot < Global Cache < Global + Local Cache。
10. ScanObjNN clean hardest 比 ModelNet clean 困难很多，体现真实扫描域偏移。
11. ULIP-2 在 ScanObjNN clean hardest 上明显强于 ULIP，完整 Point-Cache 高出 +9.96。
12. 13 组结果将作为后续 14 组 ULIP-2 × ScanObjNN-C hardest all35 的 clean 参考。

---

## 15. 运行命令汇总

13_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/13_1_ulip2_scanobjnn_clean_hardest_zs_single_gpu.sh 0

13_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global_single_gpu.sh 0

13_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh 0

---

## 16. 检查命令汇总

13_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/13_1_ulip2_scanobjnn_clean_hardest_zs/summary.csv | wc -l

tail -n +2 results/baseline/13_1_ulip2_scanobjnn_clean_hardest_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/13_1_ulip2_scanobjnn_clean_hardest_zs/logs -maxdepth 1 -name '*.log' | wc -l

13_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

13_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
