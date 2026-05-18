# 02_2_ulip_modelnetc_corruptions_all35_zs_global

## 1. 实验名称

ULIP × ModelNet-C corruptions_all35 × Zero-shot + Global Cache。

本实验是 `02_ulip_modelnetc_corruptions_all35` 实验组中的第二个子实验。

| 项目 | 内容 |
|---|---|
| 实验编号 | 02_2_ulip_modelnetc_corruptions_all35_zs_global |
| 实验组 | 02_ulip_modelnetc_corruptions_all35 |
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 方法 | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |
| 运行范围 | 7 种 corruption × 5 个 severity = 35 个 corrupted subsets |

---

## 2. 实验目的

本实验用于复现 ULIP 在 ModelNet-C 全部 35 个损坏设置上加入 Global Cache 后的结果。

它的作用有三个：

| 目的 | 说明 |
|---|---|
| 复现原论文 Global Cache baseline | 对齐原论文 Table 1 中 ULIP + Global Cache 的 ModelNet-C 结果 |
| 比较 Global Cache 的效果 | 与 `02_1` Zero-shot 对比，观察全局缓存带来的提升 |
| 得到 all35 Global Cache 鲁棒性矩阵 | 记录 severity=0,1,2,3,4 全部 35 个 corrupted subsets 上的 Global Cache 表现 |

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
| 方法脚本 | Point-Cache/scripts/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/02_run_ulip_modelnetc_corruptions_all35_common.sh |
| 结果目录 | Point-Cache/results/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global |
| 结果汇总 | Point-Cache/results/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global/summary.csv |
| 日志目录 | Point-Cache/results/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global/logs |

---

## 5. 运行命令

使用第一张 T4：

| 命令 |
|---|
| cd /root/autodl-tmp/MCM-PC-2/Point-Cache |
| bash scripts/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global_single_gpu.sh 0 |

使用第二张 T4：

| 命令 |
|---|
| cd /root/autodl-tmp/MCM-PC-2/Point-Cache |
| bash scripts/baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global_single_gpu.sh 1 |

---

## 6. 方法说明：Zero-shot + Global Cache

本实验在 ULIP zero-shot 推理基础上加入 Global Cache。

Global Cache 的基本流程是：

| 步骤 | 说明 |
|---|---|
| 1 | 对每个测试样本先进行 zero-shot 预测 |
| 2 | 计算预测分布的熵，用熵衡量预测置信度 |
| 3 | 将低熵、高置信度样本的全局点云特征存入缓存 |
| 4 | 每个类别最多保留固定数量的高质量缓存样本 |
| 5 | 新测试样本到来时，检索与其相似的全局缓存特征 |
| 6 | 将 Global Cache logits 与 zero-shot logits 融合，得到最终预测 |

本实验不使用 Local Cache，因此它只验证全局缓存对 ModelNet-C corruption 的影响。

---

## 7. 关键参数

| 参数 | 数值 | 说明 |
|---|---:|---|
| shot_capacity | 3 | 每个类别最多缓存 3 个样本 |
| alpha | 4.0 | cache logits 的融合权重 |
| beta | 3.0 | cache attention / affinity 的锐度系数 |
| npoints | 1024 | 每个点云输入点数 |
| severity 编号 | 0,1,2,3,4 | 文件后缀从 0 开始 |

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
| add_global | 45.38 | 45.79 | -0.41 |
| add_local | 47.97 | 47.98 | -0.01 |
| dropout_global | 56.93 | 56.85 | +0.08 |
| dropout_local | 53.85 | 53.89 | -0.04 |
| rotate | 60.25 | 60.25 | +0.00 |
| scale | 54.25 | 54.34 | -0.09 |
| jitter | 49.07 | 48.91 | +0.16 |
| Average | 52.53 | 52.56 | -0.03 |

分析：

1. 当前复现的 S2 Avg 为 52.53。
2. 原论文 Table 1 中 ULIP + Global Cache 的 ModelNet-C Avg 为 52.56。
3. 两者差值为 -0.03，说明本实验与原论文结果高度一致。
4. 逐 corruption 看，最大差异出现在 `add_global`，当前复现比原论文低 -0.41。
5. 其余 corruption 的差异都很小，整体可以认为复现成功。

---

## 10. All35 总体结果

| 指标 | 数值 |
|---|---:|
| S0 平均 | 57.26 |
| S1 平均 | 55.45 |
| S2 平均 | 52.53 |
| S3 平均 | 48.72 |
| S4 平均 | 43.95 |
| All35 平均 | 51.58 |

分析：

1. 从 S0 到 S4，准确率从 57.26 下降到 43.95。
2. 损坏等级越高，Global Cache 方法的准确率仍然下降。
3. 但相比 Zero-shot，Global Cache 在每个 severity 上都有明显提升。
4. All35 平均值为 51.58，高于 Zero-shot 的 46.86。
5. 因此，Global Cache 是这一组实验中最主要的提升来源。

---

## 11. ZS + Global Cache 结果矩阵

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

## 12. 与 Zero-shot 的按 severity 对比

| Severity | Zero-shot | Global | Global - ZS |
|---:|---:|---:|---:|
| S0 | 53.36 | 57.26 | +3.90 |
| S1 | 50.98 | 55.45 | +4.47 |
| S2 | 47.64 | 52.53 | +4.89 |
| S3 | 43.82 | 48.72 | +4.90 |
| S4 | 38.50 | 43.95 | +5.45 |
| Average | 46.86 | 51.58 | +4.72 |

分析：

1. Global Cache 在所有 severity 下均优于 Zero-shot。
2. 在 S4 下提升最大，Global - ZS = +5.45。
3. 说明当 corruption 强度升高时，全局缓存仍然能提供有效的测试流信息。
4. All35 平均提升为 +4.72，说明 Global Cache 对鲁棒性提升稳定。

---

## 13. 与 Zero-shot 的按 corruption 对比

| Corruption | Zero-shot | Global | Global - ZS |
|---|---:|---:|---:|
| add_global | 34.89 | 46.02 | +11.14 |
| add_local | 44.71 | 48.65 | +3.94 |
| dropout_global | 52.79 | 56.01 | +3.22 |
| dropout_local | 50.31 | 53.42 | +3.11 |
| rotate | 52.83 | 57.47 | +4.64 |
| scale | 50.92 | 54.46 | +3.53 |
| jitter | 41.57 | 45.05 | +3.48 |
| Average | 46.86 | 51.58 | +4.72 |

分析：

1. `add_global` 是 Global Cache 提升最大的 corruption，提升达到 +11.14。
2. 这说明全局缓存对全局离群点扰动特别有效。
3. `rotate` 的提升也较明显，为 +4.64。
4. `dropout_local` 的提升相对较小，为 +3.11。
5. 所有 corruption 上 Global Cache 都带来正提升，没有负增益项。

---

## 14. 关键观察

| 观察 | 说明 |
|---|---|
| Global Cache 稳定提升 | 7 种 corruption 和 5 个 severity 上均优于 Zero-shot |
| add_global 收益最大 | 全局缓存可以显著缓解全局离群点干扰 |
| 高 severity 下仍有提升 | S4 提升 +5.45，说明 Global Cache 对强损坏也有效 |
| 仍然不包含局部信息 | 本实验没有使用 Local Cache，因此无法验证局部几何细节的贡献 |
| jitter 仍然困难 | jitter 的 Global 平均为 45.05，S4 仅为 28.57，说明强噪声下仍有明显不足 |

---

## 15. 与后续实验的关系

本实验是 `02_3` 的直接前置对照。

| 实验 | 方法 | 用途 |
|---|---|---|
| 02_1 | Zero-shot | 原始 baseline |
| 02_2 | ZS + Global Cache | 验证全局缓存收益 |
| 02_3 | ZS + Global + Local Cache | 验证局部缓存是否带来额外收益 |

后续需要重点比较：

| 比较 | 含义 |
|---|---|
| Global - ZS | Global Cache 的贡献 |
| Hier - Global | Local Cache 在 Global Cache 基础上的额外贡献 |
| Hier - ZS | 完整 Point-Cache 的总贡献 |

---

## 16. 阶段性结论

本实验可以记录为：

ULIP × ModelNet-C corruptions_all35 × ZS + Global Cache 复现成功。

主要依据如下：

1. 35 个 corrupted subsets 全部运行完成。
2. 所有结果均成功写入 summary.csv。
3. severity=2 的平均准确率为 52.53，与原论文 52.56 仅差 -0.03。
4. all35 平均准确率为 51.58，比 Zero-shot 的 46.86 高 +4.72。
5. Global Cache 在所有 corruption 和所有 severity 上均带来正提升。
6. add_global 上提升最大，说明全局缓存对全局离群点扰动特别有效。

---

## 17. 后续记录

下一步应补充：

| 顺序 | 文档 |
|---:|---|
| 1 | 02_3_ulip_modelnetc_corruptions_all35_zs_global_local.md |
| 2 | 01_ulip_modelnet_clean_summary.md |
| 3 | 01_1_ulip_modelnet_clean_zs.md |
| 4 | 01_2_ulip_modelnet_clean_zs_global.md |
| 5 | 01_3_ulip_modelnet_clean_zs_global_local.md |

