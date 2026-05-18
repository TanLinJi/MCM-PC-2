# 02_3_ulip_modelnetc_corruptions_all35_zs_global_local

## 1. 实验名称

ULIP × ModelNet-C corruptions_all35 × Zero-shot + Global Cache + Local Cache。

本实验是 `02_ulip_modelnetc_corruptions_all35` 实验组中的第三个子实验，也是本组实验中的完整 Point-Cache 方法。

| 项目 | 内容 |
|---|---|
| 实验编号 | 02_3_ulip_modelnetc_corruptions_all35_zs_global_local |
| 实验组 | 02_ulip_modelnetc_corruptions_all35 |
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |
| 方法类型 | Hierarchical Cache / 完整 Point-Cache |
| 运行范围 | 7 种 corruption × 5 个 severity = 35 个 corrupted subsets |

---

## 2. 实验目的

本实验用于复现 ULIP 在 ModelNet-C 全部 35 个损坏设置上使用完整 Point-Cache 后的结果。

它的作用有三个：

| 目的 | 说明 |
|---|---|
| 复现原论文 Hierarchical Cache baseline | 对齐原论文 Table 1 中 ULIP + Hierarchical Cache 的 ModelNet-C 结果 |
| 验证完整 Point-Cache 的效果 | 与 `02_1` Zero-shot 对比，观察完整方法的总提升 |
| 分析 Local Cache 的额外贡献 | 与 `02_2` Global Cache 对比，观察 Local Cache 是否进一步提升 |

---

## 3. 与原论文指标的关系

原论文 Table 1 中，ModelNet-C 的设置是：

| 项目 | 原论文设置 |
|---|---|
| 数据集 | ModelNet-C |
| corruption 类型 | 7 种 |
| 点数 | 1024 |
| severity level | 2 |
| 报告方式 | 7 种 corruption 的平均准确率 |

因此，本实验中用于和原论文直接对齐的指标是：

| 指标 | 含义 |
|---|---|
| S2 Avg | severity=2 时 7 种 corruption 的平均准确率 |
| All35 Avg | severity=0,1,2,3,4 全部 35 个 corrupted subsets 的平均准确率 |

注意：`All35 Avg` 是本项目额外统计的完整鲁棒性指标，不是原论文 Table 1 的直接报告指标。

---

## 4. 实验脚本与结果路径

| 项目 | 路径 |
|---|---|
| 方法脚本 | Point-Cache/scripts/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/02_run_ulip_modelnetc_corruptions_all35_common.sh |
| 结果目录 | Point-Cache/results/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local |
| 结果汇总 | Point-Cache/results/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local/summary.csv |
| 日志目录 | Point-Cache/results/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local/logs |

---

## 5. 运行命令

使用第一张 T4：

| 命令 |
|---|
| cd /root/autodl-tmp/MCM-PC-2/Point-Cache |
| bash scripts/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 0 |

使用第二张 T4：

| 命令 |
|---|
| cd /root/autodl-tmp/MCM-PC-2/Point-Cache |
| bash scripts/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh 1 |

---

## 6. 方法说明：Zero-shot + Global Cache + Local Cache

本实验使用完整 Point-Cache，即 Hierarchical Cache。

完整方法由三部分组成：

| 组成部分 | 说明 |
|---|---|
| Zero-shot logits | ULIP 原始点云-文本相似度预测 |
| Global Cache logits | 基于全局点云特征的缓存检索结果 |
| Local Cache logits | 基于局部 patch 特征聚类后的缓存检索结果 |

最终预测可以理解为：

| 预测来源 | 作用 |
|---|---|
| Zero-shot | 提供原始语义分类能力 |
| Global Cache | 利用测试流中可靠样本的整体形状信息 |
| Local Cache | 进一步利用局部几何细节，补充 Global Cache 的不足 |

---

## 7. 与 02_2 的区别

| 实验 | 使用 Global Cache | 使用 Local Cache | 说明 |
|---|---|---|---|
| 02_2 | 是 | 否 | 只使用全局点云特征缓存 |
| 02_3 | 是 | 是 | 同时使用全局缓存和局部缓存 |

因此，`02_3 - 02_2` 的差值可以理解为 Local Cache 在 Global Cache 基础上的额外贡献。

---

## 8. 关键参数

| 参数 | 数值 | 说明 |
|---|---:|---|
| shot_capacity | 3 | 每个类别最多缓存 3 个样本 |
| n_cluster | 3 | 每个点云的局部 patch 聚类数 |
| alpha | 4.0 | cache logits 的融合权重 |
| beta | 3.0 | cache attention / affinity 的锐度系数 |
| npoints | 1024 | 每个点云输入点数 |
| severity 编号 | 0,1,2,3,4 | 文件后缀从 0 开始 |

---

## 9. 实验完成情况

| 项目 | 结果 |
|---|---:|
| 应运行 corrupted subsets | 35 |
| 实际完成 corrupted subsets | 35 |
| failed 数量 | 0 |
| missing_file 数量 | 0 |
| failed_parse_acc 数量 | 0 |
| 最终状态 | 完成 |

---

## 10. severity=2 与原论文对齐结果

该表用于直接对齐原论文 Table 1。

| Corruption | 当前复现 S2 | 原论文 S2 | Diff |
|---|---:|---:|---:|
| add_global | 45.62 | 46.15 | -0.53 |
| add_local | 48.10 | 47.85 | +0.25 |
| dropout_global | 59.12 | 59.16 | -0.04 |
| dropout_local | 55.96 | 56.00 | -0.04 |
| rotate | 61.59 | 61.47 | +0.12 |
| scale | 54.98 | 55.35 | -0.37 |
| jitter | 49.80 | 49.92 | -0.12 |
| Average | 53.60 | 53.70 | -0.10 |

分析：

1. 当前复现的 S2 Avg 为 53.60。
2. 原论文 Table 1 中 ULIP + Hierarchical Cache 的 ModelNet-C Avg 为 53.70。
3. 两者差值为 -0.10，说明本实验与原论文结果高度一致。
4. 逐 corruption 看，最大差异出现在 `add_global`，当前复现比原论文低 -0.53。
5. 整体误差很小，可以认为本实验完成了对原论文 ULIP + Hierarchical Cache 的成功复现。

---

## 11. All35 总体结果

| 指标 | 数值 |
|---|---:|
| S0 平均 | 58.39 |
| S1 平均 | 56.49 |
| S2 平均 | 53.60 |
| S3 平均 | 49.60 |
| S4 平均 | 44.46 |
| All35 平均 | 52.51 |

分析：

1. 从 S0 到 S4，准确率从 58.39 下降到 44.46。
2. 损坏等级越高，完整 Point-Cache 的准确率仍然下降。
3. 但相比 Zero-shot 和 Global Cache，Hierarchical Cache 在每个 severity 下都是最高的。
4. All35 平均值为 52.51，高于 Zero-shot 的 46.86，也高于 Global Cache 的 51.58。
5. 因此，完整 Point-Cache 在 all35 设置下具有最好的整体鲁棒性。

---

## 12. ZS + Global + Local Cache 结果矩阵

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 52.76 | 49.96 | 45.62 | 44.89 | 42.26 | 47.10 |
| add_local | 55.02 | 50.24 | 48.10 | 47.69 | 42.63 | 48.74 |
| dropout_global | 60.25 | 60.98 | 59.12 | 57.82 | 50.61 | 57.76 |
| dropout_local | 61.43 | 58.39 | 55.96 | 51.58 | 44.37 | 54.35 |
| rotate | 63.94 | 62.48 | 61.59 | 54.98 | 49.64 | 58.53 |
| scale | 58.59 | 58.10 | 54.98 | 53.77 | 54.13 | 55.91 |
| jitter | 56.77 | 55.31 | 49.80 | 36.47 | 27.55 | 45.18 |
| Average | 58.39 | 56.49 | 53.60 | 49.60 | 44.46 | 52.51 |

---

## 13. 与 Zero-shot 的按 severity 对比

| Severity | Zero-shot | Hierarchical | Hier - ZS |
|---:|---:|---:|---:|
| S0 | 53.36 | 58.39 | +5.03 |
| S1 | 50.98 | 56.49 | +5.51 |
| S2 | 47.64 | 53.60 | +5.96 |
| S3 | 43.82 | 49.60 | +5.78 |
| S4 | 38.50 | 44.46 | +5.96 |
| Average | 46.86 | 52.51 | +5.65 |

分析：

1. 完整 Point-Cache 在所有 severity 下均明显优于 Zero-shot。
2. All35 平均提升为 +5.65。
3. S2 和 S4 的提升都接近 +6，说明完整缓存结构在中等和高强度损坏下都有效。
4. 相比 Zero-shot，Hierarchical Cache 的鲁棒性提升非常稳定。

---

## 14. 与 Global Cache 的按 severity 对比

| Severity | Global | Hierarchical | Hier - Global |
|---:|---:|---:|---:|
| S0 | 57.26 | 58.39 | +1.13 |
| S1 | 55.45 | 56.49 | +1.04 |
| S2 | 52.53 | 53.60 | +1.07 |
| S3 | 48.72 | 49.60 | +0.88 |
| S4 | 43.95 | 44.46 | +0.51 |
| Average | 51.58 | 52.51 | +0.93 |

分析：

1. Local Cache 在所有 severity 下都带来正提升。
2. All35 平均额外提升为 +0.93。
3. 从 S0 到 S4，Local Cache 的额外收益整体变小。
4. S4 下只提升 +0.51，说明在极严重损坏下，局部结构本身也可能被破坏，Local Cache 的可靠性下降。

---

## 15. 与 Zero-shot 和 Global Cache 的按 corruption 对比

| Corruption | Zero-shot | Global | Hierarchical | Hier - ZS | Hier - Global |
|---|---:|---:|---:|---:|---:|
| add_global | 34.89 | 46.02 | 47.10 | +12.21 | +1.08 |
| add_local | 44.71 | 48.65 | 48.74 | +4.03 | +0.09 |
| dropout_global | 52.79 | 56.01 | 57.76 | +4.97 | +1.75 |
| dropout_local | 50.31 | 53.42 | 54.35 | +4.04 | +0.92 |
| rotate | 52.83 | 57.47 | 58.53 | +5.70 | +1.05 |
| scale | 50.92 | 54.46 | 55.91 | +4.99 | +1.46 |
| jitter | 41.57 | 45.05 | 45.18 | +3.61 | +0.13 |
| Average | 46.86 | 51.58 | 52.51 | +5.65 | +0.93 |

分析：

1. `add_global` 是完整 Point-Cache 总提升最大的 corruption，Hier - ZS = +12.21。
2. `dropout_global` 是 Local Cache 额外收益最大的 corruption，Hier - Global = +1.75。
3. `scale` 中 Local Cache 额外提升也较明显，Hier - Global = +1.46。
4. `add_local` 和 `jitter` 中 Local Cache 额外提升很小，分别只有 +0.09 和 +0.13。
5. 这说明 Local Cache 并非对所有 corruption 都同样有效，其收益具有明显的 corruption-specific 特征。

---

## 16. 关键观察

| 观察 | 说明 |
|---|---|
| 完整 Point-Cache 表现最好 | All35 平均为 52.51，高于 Zero-shot 和 Global Cache |
| Global Cache 是主要提升来源 | 从 46.86 提升到 51.58，增加 +4.72 |
| Local Cache 是额外补充 | 从 51.58 提升到 52.51，增加 +0.93 |
| dropout_global 中 Local Cache 最有效 | Hier - Global = +1.75 |
| add_local 和 jitter 中 Local Cache 收益很小 | 分别只有 +0.09 和 +0.13 |
| 高 severity 下 Local Cache 收益下降 | S4 下 Hier - Global 只有 +0.51 |

---

## 17. 与方法设计的关系

Point-Cache 的 Hierarchical Cache 设计目标是同时利用：

| 缓存 | 捕捉的信息 |
|---|---|
| Global Cache | 点云整体形状和全局结构 |
| Local Cache | 点云局部 patch 和细粒度几何细节 |

从本实验结果看：

1. Global Cache 已经提供了主要鲁棒性提升。
2. Local Cache 可以进一步提升，但提升幅度依赖 corruption 类型。
3. 当 corruption 破坏全局结构时，例如 dropout_global，Local Cache 更容易发挥作用。
4. 当 corruption 本身强烈破坏局部结构时，例如 jitter，Local Cache 的额外收益较小。
5. 这为后续 MCM-PC 的改进提供了方向：Local Cache 不应盲目加权，而应根据 corruption 或样本可靠性动态调整。

---

## 18. 与后续 MCM-PC 改进的关系

本实验暴露出几个值得改进的问题：

| 问题 | 现象 | 后续可能方向 |
|---|---|---|
| Local Cache 贡献不稳定 | add_local 和 jitter 中额外提升很小 | 设计 local reliability / local confidence 机制 |
| 高 severity 下局部收益下降 | S4 中 Hier - Global 只有 +0.51 | 对高损坏样本降低局部缓存权重 |
| 不同 corruption 的最优缓存策略不同 | dropout_global 收益大，jitter 收益小 | 使用 corruption-aware 或 conflict-aware 的缓存融合策略 |
| 当前融合权重固定 | alpha 和 beta 固定 | 设计动态权重或矩阵化缓存融合 |

这些观察可以作为后续 MCM-PC 方法设计和消融实验的重要依据。

---

## 19. 阶段性结论

本实验可以记录为：

ULIP × ModelNet-C corruptions_all35 × ZS + Global + Local Cache 复现成功。

主要依据如下：

1. 35 个 corrupted subsets 全部运行完成。
2. 所有结果均成功写入 summary.csv。
3. severity=2 的平均准确率为 53.60，与原论文 53.70 仅差 -0.10。
4. all35 平均准确率为 52.51，高于 Zero-shot 的 46.86 和 Global Cache 的 51.58。
5. 完整 Point-Cache 在所有 severity 下都优于 Global Cache。
6. Local Cache 的平均额外提升为 +0.93，但收益随 corruption 类型明显变化。
7. dropout_global、scale、add_global 中 Local Cache 较有帮助；add_local 和 jitter 中 Local Cache 贡献较弱。

---

## 20. 后续记录

`02` 组文档已经基本补全：

| 文档 | 状态 |
|---|---|
| 02_ulip_modelnetc_corruptions_all35_summary.md | 已补充 |
| 02_1_ulip_modelnetc_corruptions_all35_zs.md | 已补充 |
| 02_2_ulip_modelnetc_corruptions_all35_zs_global.md | 已补充 |
| 02_3_ulip_modelnetc_corruptions_all35_zs_global_local.md | 当前补充 |

下一步建议回到 `01` 组，补充 clean ModelNet 文档：

| 顺序 | 文档 |
|---:|---|
| 1 | 01_ulip_modelnet_clean_summary.md |
| 2 | 01_1_ulip_modelnet_clean_zs.md |
| 3 | 01_2_ulip_modelnet_clean_zs_global.md |
| 4 | 01_3_ulip_modelnet_clean_zs_global_local.md |

