# 02_1_ulip_modelnetc_corruptions_all35_zs

## 1. 实验名称

ULIP × ModelNet-C corruptions_all35 × Zero-shot。

本实验是 `02_ulip_modelnetc_corruptions_all35` 实验组中的第一个子实验。

| 项目 | 内容 |
|---|---|
| 实验编号 | 02_1_ulip_modelnetc_corruptions_all35_zs |
| 实验组 | 02_ulip_modelnetc_corruptions_all35 |
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 方法 | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |
| 是否使用 test-time adaptation | 否 |
| 运行范围 | 7 种 corruption × 5 个 severity = 35 个 corrupted subsets |

---

## 2. 实验目的

本实验用于复现 ULIP 在 ModelNet-C 全部 35 个损坏设置上的原始 Zero-shot baseline。

它的作用有三个：

| 目的 | 说明 |
|---|---|
| 建立原始 baseline | 后续 `02_2` 和 `02_3` 的所有提升都要和本实验比较 |
| 对齐原论文 Table 1 | 原论文只报告 severity=2 的 7 类 corruption 平均值 |
| 得到完整 all35 鲁棒性矩阵 | 本实验额外统计 severity=0,1,2,3,4 全部 35 个 corrupted subsets |

---

## 3. 与原论文指标的关系

原论文 Table 1 中，ModelNet-C 的结果设置如下：

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

注意：`All35 Avg` 是我们额外统计的完整鲁棒性指标，不是原论文 Table 1 的直接报告指标。

---

## 4. 实验脚本与结果路径

| 项目 | 路径 |
|---|---|
| 方法脚本 | Point-Cache/scripts/baseline/02_1_ulip_modelnetc_corruptions_all35_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/02_run_ulip_modelnetc_corruptions_all35_common.sh |
| 结果目录 | Point-Cache/results/baseline/02_1_ulip_modelnetc_corruptions_all35_zs |
| 结果汇总 | Point-Cache/results/baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv |
| 日志目录 | Point-Cache/results/baseline/02_1_ulip_modelnetc_corruptions_all35_zs/logs |

---

## 5. 运行命令

使用第一张 T4：

| 命令 |
|---|
| cd /root/autodl-tmp/MCM-PC-2/Point-Cache |
| bash scripts/baseline/02_1_ulip_modelnetc_corruptions_all35_zs_single_gpu.sh 0 |

使用第二张 T4：

| 命令 |
|---|
| cd /root/autodl-tmp/MCM-PC-2/Point-Cache |
| bash scripts/baseline/02_1_ulip_modelnetc_corruptions_all35_zs_single_gpu.sh 1 |

本次实际运行使用的是单张 T4。

---

## 6. 数据文件说明

本实验读取 `data/modelnet_c` 下的 35 个 corrupted h5 文件。

| corruption | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| add_global | add_global_0.h5 | add_global_1.h5 | add_global_2.h5 | add_global_3.h5 | add_global_4.h5 |
| add_local | add_local_0.h5 | add_local_1.h5 | add_local_2.h5 | add_local_3.h5 | add_local_4.h5 |
| dropout_global | dropout_global_0.h5 | dropout_global_1.h5 | dropout_global_2.h5 | dropout_global_3.h5 | dropout_global_4.h5 |
| dropout_local | dropout_local_0.h5 | dropout_local_1.h5 | dropout_local_2.h5 | dropout_local_3.h5 | dropout_local_4.h5 |
| rotate | rotate_0.h5 | rotate_1.h5 | rotate_2.h5 | rotate_3.h5 | rotate_4.h5 |
| scale | scale_0.h5 | scale_1.h5 | scale_2.h5 | scale_3.h5 | scale_4.h5 |
| jitter | jitter_0.h5 | jitter_1.h5 | jitter_2.h5 | jitter_3.h5 | jitter_4.h5 |

重要说明：

| 项目 | 说明 |
|---|---|
| severity 编号 | 从 0 到 4 |
| 原论文对齐等级 | S2，即文件后缀 `_2` |
| 不应使用的编号 | 1 到 5 |

---

## 7. 方法说明：Zero-shot

本实验使用的是 ULIP 原始 Zero-shot 推理。

Zero-shot 的流程可以简化理解为：

| 步骤 | 说明 |
|---|---|
| 1 | 使用 ULIP 点云编码器提取点云全局特征 |
| 2 | 使用文本编码器提取 40 个 ModelNet 类别的文本特征 |
| 3 | 计算点云特征和文本特征之间的相似度 |
| 4 | 选择相似度最高的类别作为预测结果 |
| 5 | 在整个测试集上统计 Top-1 accuracy |

本实验不进行缓存构建，也不会利用之前测试样本的信息。

因此，它是最原始的 baseline，也是后续 cache 方法的对照组。

---

## 8. 实验完成情况

| 项目 | 结果 |
|---|---:|
| 应运行 corrupted subsets | 35 |
| 实际完成 corrupted subsets | 35 |
| failed 数量 | 0 |
| missing_file 数量 | 0 |
| failed_parse_acc 数量 | 0 |
| 最终状态 | 完成 |

---

## 9. severity=2 与原论文对齐结果

该表用于直接对齐原论文 Table 1。

| Corruption | 当前复现 S2 | 原论文 S2 | Diff |
|---|---:|---:|---:|
| add_global | 33.55 | 33.55 | +0.00 |
| add_local | 44.08 | 43.92 | +0.16 |
| dropout_global | 54.70 | 54.70 | +0.00 |
| dropout_local | 50.89 | 50.89 | +0.00 |
| rotate | 55.39 | 55.27 | +0.12 |
| scale | 51.05 | 50.20 | +0.85 |
| jitter | 43.80 | 44.08 | -0.28 |
| Average | 47.64 | 47.52 | +0.12 |

分析：

1. 当前复现的 S2 Avg 为 47.64。
2. 原论文 Table 1 中 ULIP Zero-shot 的 ModelNet-C Avg 为 47.52。
3. 两者差值为 +0.12，说明本实验与原论文结果高度一致。
4. 逐 corruption 看，最大差异出现在 `scale`，当前复现比原论文高 +0.85。
5. 其余 corruption 的差异都很小，整体可以认为复现成功。

---

## 10. All35 总体结果

| 指标 | 数值 |
|---|---:|
| S0 平均 | 53.36 |
| S1 平均 | 50.98 |
| S2 平均 | 47.64 |
| S3 平均 | 43.82 |
| S4 平均 | 38.50 |
| All35 平均 | 46.86 |

分析：

1. 从 S0 到 S4，准确率从 53.36 下降到 38.50。
2. 损坏等级越高，ULIP Zero-shot 的性能越差。
3. All35 平均值为 46.86，这是完整 35 个 corrupted subsets 上的总体表现。
4. All35 平均值低于 S2 平均值，说明高 severity 的 S3 和 S4 拉低了整体鲁棒性。

---

## 11. Zero-shot 结果矩阵

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 45.71 | 38.65 | 33.55 | 29.86 | 26.66 | 34.89 |
| add_local | 51.09 | 47.53 | 44.08 | 42.02 | 38.82 | 44.71 |
| dropout_global | 55.83 | 55.47 | 54.70 | 52.51 | 45.42 | 52.79 |
| dropout_local | 56.85 | 55.02 | 50.89 | 47.65 | 41.13 | 50.31 |
| rotate | 56.40 | 56.04 | 55.39 | 51.34 | 44.98 | 52.83 |
| scale | 53.00 | 52.51 | 51.05 | 49.51 | 48.54 | 50.92 |
| jitter | 54.66 | 51.62 | 43.80 | 33.83 | 23.95 | 41.57 |
| Average | 53.36 | 50.98 | 47.64 | 43.82 | 38.50 | 46.86 |

---

## 12. 按 severity 分析

| Severity | Avg Acc | 相比 S0 的下降 |
|---:|---:|---:|
| S0 | 53.36 | 0.00 |
| S1 | 50.98 | -2.38 |
| S2 | 47.64 | -5.72 |
| S3 | 43.82 | -9.54 |
| S4 | 38.50 | -14.86 |

分析：

1. S0 是最轻损坏等级，平均准确率为 53.36。
2. S4 是最重损坏等级，平均准确率为 38.50。
3. 从 S0 到 S4，总下降为 14.86。
4. 说明 ULIP 原始 zero-shot 对强损坏较敏感，尤其在高 severity 下性能下降明显。
5. 这一现象为后续 cache 方法提供了改进空间。

---

## 13. 按 corruption 分析

| Corruption | Avg Acc | 排名 |
|---|---:|---:|
| rotate | 52.83 | 1 |
| dropout_global | 52.79 | 2 |
| scale | 50.92 | 3 |
| dropout_local | 50.31 | 4 |
| add_local | 44.71 | 5 |
| jitter | 41.57 | 6 |
| add_global | 34.89 | 7 |

分析：

1. `rotate` 和 `dropout_global` 对 ULIP Zero-shot 的破坏相对较小，平均准确率分别为 52.83 和 52.79。
2. `scale` 和 `dropout_local` 居中，说明尺度扰动和局部丢弃会带来一定性能下降，但不是最严重。
3. `jitter` 和 `add_global` 是最困难的两类 corruption。
4. `add_global` 的平均准确率最低，仅为 34.89，说明全局离群点对 ULIP 原始 zero-shot 影响最大。
5. `jitter` 在 S4 下下降到 23.95，说明高强度噪声抖动会严重破坏点云几何结构。

---

## 14. 关键观察

| 观察 | 说明 |
|---|---|
| severity 越高，准确率越低 | 结果符合 corrupted benchmark 的基本预期 |
| S2 平均与原论文高度一致 | 说明复现实验配置基本正确 |
| add_global 最困难 | 全局离群点严重干扰 ULIP 的全局几何表示 |
| jitter 在高 severity 下极不稳定 | 局部噪声会显著破坏点云几何结构 |
| Zero-shot 无法利用测试流信息 | 后续 Global Cache 和 Local Cache 的提升空间明确 |

---

## 15. 与后续实验的关系

本实验是 `02` 组的基础对照。

| 后续实验 | 比较方式 | 目的 |
|---|---|---|
| 02_2_ulip_modelnetc_corruptions_all35_zs_global | Global - ZS | 观察 Global Cache 是否提升 corrupted robustness |
| 02_3_ulip_modelnetc_corruptions_all35_zs_global_local | Hier - ZS 和 Hier - Global | 观察完整 Point-Cache 是否进一步提升 |

后续在总文档中已经完成三种方法的综合对比。  
本子文档只保留 Zero-shot 单方法的详细记录。

---

## 16. 阶段性结论

本实验可以记录为：

ULIP × ModelNet-C corruptions_all35 × Zero-shot 复现成功。

主要依据如下：

1. 35 个 corrupted subsets 全部运行完成。
2. 所有结果均成功写入 summary.csv。
3. severity=2 的平均准确率为 47.64，与原论文 47.52 仅差 +0.12。
4. severity 从 S0 到 S4 时，准确率呈稳定下降趋势。
5. all35 平均准确率为 46.86，可作为后续方法增益计算的 zero-shot 基准。

---

## 17. 后续记录

下一步应补充：

| 顺序 | 文档 |
|---:|---|
| 1 | 02_2_ulip_modelnetc_corruptions_all35_zs_global.md |
| 2 | 02_3_ulip_modelnetc_corruptions_all35_zs_global_local.md |
| 3 | 01_ulip_modelnet_clean_summary.md |
| 4 | 01_1_ulip_modelnet_clean_zs.md |
| 5 | 01_2_ulip_modelnet_clean_zs_global.md |
| 6 | 01_3_ulip_modelnet_clean_zs_global_local.md |

