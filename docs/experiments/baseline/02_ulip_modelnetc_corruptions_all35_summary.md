# 02_ulip_modelnetc_corruptions_all35_summary

## 1. 实验名称

ULIP × ModelNet-C corruptions_all35 baseline 复现总结。

本实验组用于复现 Point-Cache 在 **ULIP backbone + ModelNet-C** 上的 baseline 结果。

本组包含三个子实验：

| 实验编号 | 方法 | 说明 |
|---|---|---|
| 02_1_ulip_modelnetc_corruptions_all35_zs | Zero-shot | 不使用缓存，仅使用 ULIP 原始 zero-shot 推理 |
| 02_2_ulip_modelnetc_corruptions_all35_zs_global | ZS + Global Cache | 在 zero-shot 基础上加入全局缓存 |
| 02_3_ulip_modelnetc_corruptions_all35_zs_global_local | ZS + Global + Local Cache | 在 Global Cache 基础上进一步加入 Local Cache，即 Point-Cache 的 hierarchical cache |

---

## 2. 实验目的

原论文 Table 1 中，ModelNet-C 的结果只报告 **corruption severity level = 2** 下 7 种 corruption 的平均值。

而本实验额外完整跑了：

| 维度 | 数量 |
|---|---:|
| corruption 类型 | 7 |
| severity 等级 | 5 |
| corrupted subsets | 35 |
| 方法数 | 3 |
| 总运行数 | 105 |

因此本文档同时记录两类指标：

| 指标 | 含义 | 用途 |
|---|---|---|
| S2 Avg | severity=2 时 7 种 corruption 的平均准确率 | 与原论文 Table 1 直接对齐 |
| All35 Avg | severity=0,1,2,3,4 全部 35 个 corrupted subsets 的平均准确率 | 作为更完整的鲁棒性评估指标 |

---

## 3. 实验环境与路径

| 项目 | 内容 |
|---|---|
| 项目根目录 | /root/autodl-tmp/MCM-PC-2 |
| Point-Cache 根目录 | /root/autodl-tmp/MCM-PC-2/Point-Cache |
| 数据集参数 | modelnet_c |
| 数据目录 | Point-Cache/data/modelnet_c |
| Backbone | ULIP |
| 点数 | 1024 |
| severity 编号 | 0, 1, 2, 3, 4 |
| corruption 数量 | 7 |
| 总运行数 | 3 methods × 7 corruptions × 5 severities = 105 runs |

---

## 4. 脚本与结果目录

| 实验编号 | 脚本 | 结果目录 |
|---|---|---|
| 02_1 | Point-Cache/scripts/baseline/02_1_ulip_modelnetc_corruptions_all35_zs_single_gpu.sh | Point-Cache/results/baseline/02_1_ulip_modelnetc_corruptions_all35_zs |
| 02_2 | Point-Cache/scripts/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global_single_gpu.sh | Point-Cache/results/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global |
| 02_3 | Point-Cache/scripts/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh | Point-Cache/results/baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local |

公共脚本：

| 脚本 | 作用 |
|---|---|
| Point-Cache/scripts/baseline/02_run_ulip_modelnetc_corruptions_all35_common.sh | 被 02_1、02_2、02_3 调用，负责循环运行 7 种 corruption × 5 个 severity |

---

## 5. corruption 与 severity 说明

### 5.1 corruption 类型

| corruption | 中文解释 | severity=2 文件示例 |
|---|---|---|
| add_global | 全局离群点添加 | data/modelnet_c/add_global_2.h5 |
| add_local | 局部离群点添加 | data/modelnet_c/add_local_2.h5 |
| dropout_global | 全局结构丢弃 | data/modelnet_c/dropout_global_2.h5 |
| dropout_local | 局部结构丢弃 | data/modelnet_c/dropout_local_2.h5 |
| rotate | 旋转扰动 | data/modelnet_c/rotate_2.h5 |
| scale | 尺度扰动 | data/modelnet_c/scale_2.h5 |
| jitter | 抖动噪声 | data/modelnet_c/jitter_2.h5 |

### 5.2 severity 编号

| 文件后缀 | 本文档记法 | 说明 |
|---|---|---|
| _0 | S0 | 最低损坏等级 |
| _1 | S1 | 较低损坏等级 |
| _2 | S2 | 原论文 Table 1 使用的等级 |
| _3 | S3 | 较高损坏等级 |
| _4 | S4 | 最高损坏等级 |

注意：severity 是 5 档，但文件编号从 0 开始，因此不能写成 1 到 5。

---

## 6. 实验完成情况

| 实验编号 | 方法 | 运行数 | 状态 |
|---|---|---:|---|
| 02_1 | Zero-shot | 35 / 35 | 完成 |
| 02_2 | ZS + Global Cache | 35 / 35 | 完成 |
| 02_3 | ZS + Global + Local Cache | 35 / 35 | 完成 |
| 合计 | 三种方法 | 105 / 105 | 完成 |

---

## 7. 与原论文 Table 1 对齐：severity=2

原论文 Table 1 报告的是 ModelNet-C 在 severity=2 下的结果，因此下面这张表是最重要的复现对齐表。

| 方法 | 当前复现 S2 Avg | 原论文 S2 Avg | Diff |
|---|---:|---:|---:|
| Zero-shot | 47.64 | 47.52 | +0.12 |
| ZS + Global Cache | 52.53 | 52.56 | -0.03 |
| ZS + Global + Local Cache | 53.60 | 53.70 | -0.10 |

结论：三个方法的复现结果都与原论文高度一致，平均误差均小于 0.15。

---

## 8. severity=2 逐 corruption 对比

| Corruption | ZS 当前 | ZS 原文 | Global 当前 | Global 原文 | Hier 当前 | Hier 原文 |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 33.55 | 33.55 | 45.38 | 45.79 | 45.62 | 46.15 |
| add_local | 44.08 | 43.92 | 47.97 | 47.98 | 48.10 | 47.85 |
| dropout_global | 54.70 | 54.70 | 56.93 | 56.85 | 59.12 | 59.16 |
| dropout_local | 50.89 | 50.89 | 53.85 | 53.89 | 55.96 | 56.00 |
| rotate | 55.39 | 55.27 | 60.25 | 60.25 | 61.59 | 61.47 |
| scale | 51.05 | 50.20 | 54.25 | 54.34 | 54.98 | 55.35 |
| jitter | 43.80 | 44.08 | 49.07 | 48.91 | 49.80 | 49.92 |
| Average | 47.64 | 47.52 | 52.53 | 52.56 | 53.60 | 53.70 |

分析：

1. Zero-shot 的 S2 平均为 47.64，与原文 47.52 只差 +0.12。
2. ZS + Global Cache 的 S2 平均为 52.53，与原文 52.56 只差 -0.03。
3. ZS + Global + Local Cache 的 S2 平均为 53.60，与原文 53.70 只差 -0.10。
4. 因此，该组实验可以认为完成了对原论文 Table 1 中 ULIP × ModelNet-C 部分的成功复现。

---

## 9. All35 总体结果

完整 35 个 corrupted subsets 上的平均结果如下。这个指标不是原论文 Table 1 的直接指标，而是我们为了后续鲁棒性分析额外统计的完整结果。

| 方法 | All35 Avg | 相比 ZS 提升 | 相比 Global 提升 |
|---|---:|---:|---:|
| Zero-shot | 46.86 | - | - |
| ZS + Global Cache | 51.58 | +4.72 | - |
| ZS + Global + Local Cache | 52.51 | +5.65 | +0.93 |

结论：

| 结论 | 说明 |
|---|---|
| Zero-shot < Global Cache | Global Cache 是主要提升来源 |
| Global Cache < Global + Local Cache | Local Cache 在 Global Cache 基础上仍有额外收益 |
| All35 平均提升稳定 | 完整 35 个损坏设置下趋势成立 |

---

## 10. 按 severity 统计

| Severity | Zero-shot | Global | Hierarchical | Global - ZS | Hier - ZS | Hier - Global |
|---:|---:|---:|---:|---:|---:|---:|
| S0 | 53.36 | 57.26 | 58.39 | +3.90 | +5.03 | +1.13 |
| S1 | 50.98 | 55.45 | 56.49 | +4.47 | +5.52 | +1.05 |
| S2 | 47.64 | 52.53 | 53.60 | +4.89 | +5.96 | +1.07 |
| S3 | 43.82 | 48.72 | 49.60 | +4.90 | +5.78 | +0.88 |
| S4 | 38.50 | 43.95 | 44.46 | +5.45 | +5.96 | +0.50 |
| Average | 46.86 | 51.58 | 52.51 | +4.72 | +5.65 | +0.93 |

分析：

1. 随着 severity 从 S0 增加到 S4，三种方法的准确率整体下降，说明损坏强度越高，任务越困难。
2. Global Cache 在所有 severity 下都稳定优于 Zero-shot。
3. Hierarchical Cache 在所有 severity 下都进一步优于 Global Cache。
4. Local Cache 的额外提升在 S4 上变小，说明在极严重损坏下，局部特征可能也会受到较强破坏。

---

## 11. 按 corruption 统计

| Corruption | Zero-shot | Global | Hierarchical | Global - ZS | Hier - ZS | Hier - Global |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 34.89 | 46.02 | 47.10 | +11.14 | +12.21 | +1.08 |
| add_local | 44.71 | 48.65 | 48.74 | +3.94 | +4.03 | +0.09 |
| dropout_global | 52.79 | 56.01 | 57.76 | +3.22 | +4.97 | +1.75 |
| dropout_local | 50.31 | 53.42 | 54.35 | +3.11 | +4.04 | +0.92 |
| rotate | 52.83 | 57.47 | 58.53 | +4.64 | +5.70 | +1.05 |
| scale | 50.92 | 54.46 | 55.91 | +3.53 | +4.99 | +1.46 |
| jitter | 41.57 | 45.05 | 45.18 | +3.48 | +3.61 | +0.13 |
| Average | 46.86 | 51.58 | 52.51 | +4.72 | +5.65 | +0.93 |

分析：

1. add_global 是 Global Cache 提升最大的 corruption，Global - ZS = +11.14，Hier - ZS = +12.21。
2. dropout_global 中 Local Cache 的额外提升最大，Hier - Global = +1.75。
3. add_local 和 jitter 中 Local Cache 的额外提升很小，分别只有 +0.09 和 +0.13。
4. add_local 和 jitter 后续可以作为 MCM-PC 改进的重点分析对象。

---

## 12. Zero-shot 结果矩阵

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

## 13. ZS + Global Cache 结果矩阵

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 52.51 | 49.27 | 45.38 | 42.63 | 40.32 | 46.02 |
| add_local | 54.70 | 50.53 | 47.97 | 46.88 | 43.15 | 48.65 |
| dropout_global | 59.36 | 58.79 | 56.93 | 55.88 | 49.07 | 56.01 |
| dropout_local | 60.58 | 57.70 | 53.85 | 50.81 | 44.17 | 53.42 |
| rotate | 61.35 | 60.78 | 60.25 | 55.43 | 49.55 | 57.47 |
| scale | 56.08 | 56.40 | 54.25 | 52.71 | 52.84 | 54.46 |
| jitter | 56.24 | 54.66 | 49.07 | 36.71 | 28.57 | 45.05 |
| Average | 57.26 | 55.45 | 52.53 | 48.72 | 43.95 | 51.58 |

---

## 14. ZS + Global + Local Cache 结果矩阵

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

## 15. 阶段性结论

本组实验可以正式记录为：

**ULIP × ModelNet-C corruptions_all35 baseline 复现成功。**

主要结论如下：

1. severity=2 结果与原论文 Table 1 高度一致，说明复现实验配置基本正确。
2. 完整 all35 结果显示，Global Cache 和 Local Cache 的提升趋势在不同 severity 下都成立。
3. Global Cache 是主要提升来源，All35 平均提升 +4.72。
4. Local Cache 在 Global Cache 基础上继续提升 +0.93，但提升幅度随 corruption 类型变化明显。
5. add_global 是缓存收益最大的 corruption。
6. add_local 和 jitter 中 Local Cache 的额外收益较小，后续 MCM-PC 可以重点分析这两类场景。
7. dropout_global 中 Local Cache 额外收益较明显，说明局部缓存对结构缺失类扰动可能更有效。

---

## 16. 后续计划

建议后续补全文档顺序如下：

| 顺序 | 文档 |
|---:|---|
| 1 | 02_1_ulip_modelnetc_corruptions_all35_zs.md |
| 2 | 02_2_ulip_modelnetc_corruptions_all35_zs_global.md |
| 3 | 02_3_ulip_modelnetc_corruptions_all35_zs_global_local.md |
| 4 | 01_ulip_modelnet_clean_summary.md |
| 5 | 01_1_ulip_modelnet_clean_zs.md |
| 6 | 01_2_ulip_modelnet_clean_zs_global.md |
| 7 | 01_3_ulip_modelnet_clean_zs_global_local.md |

下一步先补单实验文档：

**02_1_ulip_modelnetc_corruptions_all35_zs.md**

