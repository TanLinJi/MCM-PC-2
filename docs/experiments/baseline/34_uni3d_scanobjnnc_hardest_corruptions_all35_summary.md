# 34_uni3d_scanobjnnc_hardest_corruptions_all35_summary

## 1. 实验组目的

本总文档汇总 Uni3D 在 ScanObjNN-C hardest 全部 35 个损坏设置上的三组 baseline 复现实验。

34 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | Uni3D |
| Dataset | ScanObjNN-C hardest |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| h5 文件目录 | data/sonn_c/hardest |
| 输入点数 | 1024 |
| Uni3D point encoder checkpoint | weights/uni3d/scanobjnn/model.pt |
| Uni3D text encoder checkpoint | weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin |

三组实验分别为：

| 实验编号 | 方法 | 作用 |
|---|---|---|
| 34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs | Zero-shot | 无缓存基础对照 |
| 34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global | Zero-shot + Global Cache | 验证全局缓存增益 |
| 34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 及 Local Cache 额外影响 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| Uni3D 在 ScanObjNN-C hardest 上的 Zero-shot corrupted 鲁棒性是多少？ | 由 34_1 给出 |
| Global Cache 是否有效？ | 比较 34_2 - 34_1 |
| Local Cache 是否有额外贡献？ | 比较 34_3 - 34_2 |
| 完整 Point-Cache 是否与原文对齐？ | 比较 34_3 与原文 Table 7 |
| ScanObjNN-C hardest 相比 ScanObjNN clean hardest 下降多少？ | 与 33 组比较 |
| 后续 MCM-PC 应重点关注哪些失败模式？ | 分析 corruption × severity 结果矩阵 |
| 整个 baseline 大复现阶段是否完成？ | 34 组是最后一组 baseline 实验 |

需要特别注意：原文 Point-Cache Table 7 只报告 ScanObjNN-C hardest 在 severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 用于和原文 Point-Cache Table 7 对齐 |
| all35 Average | 本复现实验扩展统计，表示 35 个 corrupted setting 的总体平均 |

---

## 2. 当前实现方式

34 组属于 all35 corruption 实验，因此使用优化版 Python runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/34_run_uni3d_scanobjnnc_hardest_corruptions_all35_common.sh |
| 优化 runner | Point-Cache/runners/baseline/run_uni3d_scanobjnnc_hardest_corruptions_all35.py |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |
| checkpoint 下载脚本 | Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh |

优化方式如下：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 | 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 | 模型只加载一次，每个 cor_type 重新创建 DataLoader、重新初始化 cache |
| bash 通过 tee 生成单个 cor_type 的 log | Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv | summary.csv 的列结构保持不变，每个 cor_type 一行 |

该优化只改变执行效率，不改变实验定义。每个 cor_type 仍然单独记录 log，并且每个方法的 summary.csv 仍然保持 35 行。

---

## 3. 34 runner 修正记录

34 组是最后一组 baseline，也是在复现过程中最容易出错的一组。原因是它同时满足三个条件：

| 条件 | 风险 |
|---|---|
| 从 32 组 ModelNet-C all35 runner 复制而来 | 容易残留 `data/modelnet_c`、`modelnet_c_root`、`sonn_variant="-"` |
| 属于 ScanObjNN-C hardest | 需要记录 `sonn_variant=hardest` |
| 使用 SONN_C 数据集类 | loader root 和 summary h5 root 不完全相同 |

最初 runner 中残留了 32 组 ModelNet-C metadata：

| 错误字段 | 问题 |
|---|---|
| args.modelnet_c_root = "data/modelnet_c" | 继承自 32 组 ModelNet-C |
| data_root = args.modelnet_c_root | summary 中写成 ModelNet-C 路径 |
| sonn_variant = "-" | 没有记录 hardest split |
| file = data/modelnet_c/{cor_type}.h5 | summary 文件路径错误 |

当前已修正为：

| 字段 | 当前正确值 | 用途 |
|---|---|---|
| args.dataset | sonn_c | 数据集类型 |
| args.sonn_variant | hardest | SONN-C hardest split |
| args.sonn_c_root | data/sonn_c | SONN_C loader root |
| args.data_root | data/sonn_c | build_test_data_loader 使用 |
| summary data_root | data/sonn_c/hardest | summary 中记录实际 h5 目录 |
| summary file | data/sonn_c/hardest/{cor_type}.h5 | summary 中记录实际 h5 文件 |
| summary sonn_variant | hardest | summary 中记录 split |
| expected checkpoint | weights/uni3d/scanobjnn/model.pt | Uni3D ScanObjNN checkpoint |

这里最重要的一点是：

| root 类型 | 正确值 | 原因 |
|---|---|---|
| loader root | data/sonn_c | SONN_C 数据集类需要从 data/sonn_c/shape_names.txt 读取类别名 |
| actual h5 root / summary root | data/sonn_c/hardest | corrupted h5 文件实际位于 data/sonn_c/hardest/{cor_type}.h5 |

因此不能把 `args.data_root` 直接设置为 `data/sonn_c/hardest`，否则数据集类会错误寻找：

data/sonn_c/hardest/shape_names.txt

当前 runner 已加入 fail-fast 检查，确保 34 组不会再静默落回 ModelNet-C metadata 或错误 checkpoint。

---

## 4. Uni3D checkpoint 设置

34 组正式使用：

weights/uni3d/scanobjnn/model.pt

这是 33 / 34 组 ScanObjNN 系列实验必须使用的 Uni3D checkpoint。

后续规则固定如下：

| 实验组 | 数据设置 | 应使用 checkpoint |
|---|---|---|
| 31 | Uni3D × ModelNet clean | weights/uni3d/modelnet40/model.pt |
| 32 | Uni3D × ModelNet-C all35 | weights/uni3d/modelnet40/model.pt |
| 33 | Uni3D × ScanObjNN clean hardest | weights/uni3d/scanobjnn/model.pt |
| 34 | Uni3D × ScanObjNN-C hardest all35 | weights/uni3d/scanobjnn/model.pt |

不能使用：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

也不能使用：

weights/uni3d/modelnet40/model.pt

原因是 Uni3D 对 checkpoint 选择非常敏感。此前 31 / 33 组已经证明，错误 checkpoint 会导致结果稳定偏低，并影响是否能与原文对齐。

---

## 5. 三个实验的输出完整性检查

| 实验编号 | 方法 | summary 行数 | cor_type 唯一数 | log_path 唯一数 | logs 文件数 | status | metadata |
|---|---|---:|---:|---:|---:|---|---|
| 34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs | Zero-shot | 35 | 35 | 35 | 35 | 35 done | 正确 |
| 34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global | ZS + Global | 35 | 35 | 35 | 35 | 35 done | 正确 |
| 34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local | ZS + Global + Local | 35 | 35 | 35 | 35 | 35 done | 正确 |

当前三个 summary 均确认：

| 字段 | 当前值 |
|---|---|
| dataset | sonn_c |
| data_root | data/sonn_c/hardest |
| file | data/sonn_c/hardest/{cor_type}.h5 |
| sonn_variant | hardest |
| backbone | Uni3D |

说明：

1. 三个子实验均完成 35 个 cor_type。
2. 三个子实验均有 35 个唯一 log_path。
3. 三个子实验的 logs 文件数均为 35。
4. 三个子实验的 status 均为 35 个 done。
5. 三个子实验的 metadata 已修正并确认正确。
6. 执行完整性正常并不等于结果正常；结果是否正常还需要和原文 severity=2 参考值对比。

---

## 6. 核心结果总表

| 方法 | S0 Avg | S1 Avg | S2 Avg | S3 Avg | S4 Avg | all35 Avg |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 42.67 | 40.46 | 37.75 | 35.06 | 32.37 | 37.66 |
| ZS + Global | 47.10 | 44.22 | 42.31 | 39.52 | 36.74 | 41.98 |
| ZS + Global + Local | 48.29 | 45.43 | 44.13 | 40.20 | 37.82 | 43.17 |

核心观察：

1. Global Cache 将 all35 Avg 从 37.66 提升到 41.98，提升 +4.32。
2. Local Cache 在 Global Cache 基础上将 all35 Avg 从 41.98 提升到 43.17，额外提升 +1.19。
3. severity=2 上，Global Cache 将 Avg 从 37.75 提升到 42.31，提升 +4.56。
4. severity=2 上，Local Cache 在 Global Cache 基础上从 42.31 提升到 44.13，额外提升 +1.82。
5. 因此，34 组中 Global Cache 和 Local Cache 都明确有效。

---

## 7. 与原文 Point-Cache Table 7 的 severity=2 对齐

原文 Point-Cache Table 7 报告的是 ScanObjNN-C hardest 在 severity level = 2 下 7 种 corruption 的结果。因此，这里只取 S2 列与原文对齐。

| 方法 | 当前复现 S2 Avg | 原文 S2 Avg | Diff |
|---|---:|---:|---:|
| Zero-shot | 37.75 | 37.38 | +0.37 |
| ZS + Global | 42.31 | 42.03 | +0.28 |
| ZS + Global + Local | 44.13 | 43.10 | +1.03 |

分析：

34_1 和 34_2 与原文高度接近，差异分别为 +0.37 和 +0.28。34_3 比原文高 +1.03，略高但方法趋势和增益结构合理。

因此，34 组三个子实验都可以认为是有效复现结果。34_3 需要明确记录为“略高于原文”。

---

## 8. 原文增益与当前复现增益对比

| 增益来源 | 原文 S2 增益 | 当前 S2 增益 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | 42.03 - 37.38 = +4.65 | 42.31 - 37.75 = +4.56 | -0.09 |
| Local Cache extra over Global | 43.10 - 42.03 = +1.07 | 44.13 - 42.31 = +1.82 | +0.75 |
| Full Point-Cache over Zero-shot | 43.10 - 37.38 = +5.72 | 44.13 - 37.75 = +6.38 | +0.66 |

分析：

Global Cache 的相对增益几乎完全复现原文，原文 +4.65，当前 +4.56，差异只有 -0.09。

Local Cache 当前额外增益比原文更强，原文 +1.07，当前 +1.82，差异 +0.75。这也是 34_3 比原文高 +1.03 的主要原因。

整体上，34 组的方法趋势和增益结构合理：

Zero-shot < ZS + Global Cache < ZS + Global Cache + Local Cache

---

## 9. Severity 维度增益分析

### 9.1 Global Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 42.67 | 47.10 | +4.43 |
| S1 | 40.46 | 44.22 | +3.76 |
| S2 | 37.75 | 42.31 | +4.56 |
| S3 | 35.06 | 39.52 | +4.46 |
| S4 | 32.37 | 36.74 | +4.37 |
| **all35** | **37.66** | **41.98** | **+4.32** |

分析：

Global Cache 在所有 severity 上都带来稳定正增益。增益幅度在 +3.76 到 +4.56 之间，说明 Global Cache 不只改善某一个 severity，而是对全 severity 范围都有效。

### 9.2 Local Cache 相比 Global Cache

| Severity | ZS + Global Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 47.10 | 48.29 | +1.19 |
| S1 | 44.22 | 45.43 | +1.21 |
| S2 | 42.31 | 44.13 | +1.82 |
| S3 | 39.52 | 40.20 | +0.68 |
| S4 | 36.74 | 37.82 | +1.08 |
| **all35** | **41.98** | **43.17** | **+1.19** |

分析：

Local Cache 在 Global Cache 基础上的额外贡献在所有 severity 上均为正。S2 的额外提升最大，为 +1.82；S3 的额外提升较小，为 +0.68。

### 9.3 完整 Point-Cache 相比 Zero-shot

| Severity | Zero-shot Avg | ZS + Global + Local Avg | Gain |
|---:|---:|---:|---:|
| S0 | 42.67 | 48.29 | +5.62 |
| S1 | 40.46 | 45.43 | +4.97 |
| S2 | 37.75 | 44.13 | +6.38 |
| S3 | 35.06 | 40.20 | +5.14 |
| S4 | 32.37 | 37.82 | +5.45 |
| **all35** | **37.66** | **43.17** | **+5.51** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有明显正增益。S2 增益最大，为 +6.38；all35 增益为 +5.51。

---

## 10. Corruption 维度结果对比

### 10.1 三种方法的 corruption 平均准确率

| Corruption | Zero-shot Avg | ZS + Global Avg | ZS + Global + Local Avg |
|---|---:|---:|---:|
| add_global | 47.39 | 52.37 | 52.89 |
| add_local | 38.69 | 41.69 | 41.19 |
| dropout_global | 35.07 | 40.76 | 42.27 |
| dropout_local | 32.67 | 37.22 | 39.96 |
| rotate | 42.84 | 46.72 | 47.68 |
| scale | 38.14 | 41.66 | 42.93 |
| jitter | 28.85 | 33.43 | 35.30 |

观察：

1. jitter 在三种方法中始终最低，是 Uni3D × ScanObjNN-C hardest 的主要困难 corruption。
2. dropout_local 和 dropout_global 也始终较低，是第二类主要困难区域。
3. Global Cache 提升了所有 corruption 的平均准确率。
4. Local Cache 对 dropout_local、jitter 和 dropout_global 的额外贡献明显。
5. Local Cache 对 add_local 的 all35 平均略降。

### 10.2 Corruption 维度总提升

| Corruption | Global - ZS | Global + Local - Global | Global + Local - ZS |
|---|---:|---:|---:|
| add_global | +4.98 | +0.52 | +5.50 |
| add_local | +3.00 | -0.50 | +2.50 |
| dropout_global | +5.69 | +1.51 | +7.20 |
| dropout_local | +4.55 | +2.74 | +7.29 |
| rotate | +3.88 | +0.96 | +4.84 |
| scale | +3.52 | +1.27 | +4.79 |
| jitter | +4.58 | +1.87 | +6.45 |
| **Average** | **+4.32** | **+1.19** | **+5.51** |

分析：

完整 Point-Cache 对 dropout_local、dropout_global 和 jitter 的提升最大，说明 cache 机制对点云缺失和坐标扰动有明显帮助。

但 Local Cache 对 add_local 的平均额外贡献为 -0.50，说明局部缓存并非对所有 corruption 都稳定正向。后续 MCM-PC 可以考虑对 Local Cache 进行 reliability-aware 或 corruption-aware 调节。

---

## 11. Corruption 难度排序

### 11.1 Zero-shot 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 28.85 |
| 2 | dropout_local | 32.67 |
| 3 | dropout_global | 35.07 |
| 4 | scale | 38.14 |
| 5 | add_local | 38.69 |
| 6 | rotate | 42.84 |
| 7 | add_global | 47.39 |

### 11.2 ZS + Global 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 33.43 |
| 2 | dropout_local | 37.22 |
| 3 | dropout_global | 40.76 |
| 4 | scale | 41.66 |
| 5 | add_local | 41.69 |
| 6 | rotate | 46.72 |
| 7 | add_global | 52.37 |

### 11.3 ZS + Global + Local 难度排序

| 排名 | Corruption | Avg |
|---:|---|---:|
| 1 | jitter | 35.30 |
| 2 | dropout_local | 39.96 |
| 3 | add_local | 41.19 |
| 4 | dropout_global | 42.27 |
| 5 | scale | 42.93 |
| 6 | rotate | 47.68 |
| 7 | add_global | 52.89 |

综合分析：

三种方法下，最困难 corruption 始终是 jitter。完整 Point-Cache 能把 jitter 平均值从 28.85 提升到 35.30，但该数值仍然低于其他 corruption。

dropout_local 和 dropout_global 也是主要困难 corruption。完整 Point-Cache 对它们有明显提升，但仍然无法完全解决高 severity 下的点云缺失问题。

---

## 12. 低准确率区域分析

| 条件 | Zero-shot 数量 | ZS + Global 数量 | ZS + Global + Local 数量 |
|---|---:|---:|---:|
| Acc < 40 | 22 / 35 | 11 / 35 | 8 / 35 |
| Acc < 35 | 11 / 35 | 7 / 35 | 5 / 35 |
| Acc < 30 | 6 / 35 | 3 / 35 | 2 / 35 |
| Acc < 25 | 3 / 35 | 1 / 35 | 0 / 35 |

分析：

Global Cache 明显减少了低准确率区域。Acc < 40 的 setting 从 22 个减少到 11 个。

Local Cache 进一步减少低准确率区域。Acc < 40 的 setting 从 11 个减少到 8 个，Acc < 25 的 setting 从 1 个减少到 0 个。

这说明完整 Point-Cache 不仅提高平均准确率，也减少了极低性能 setting 的数量。

---

## 13. 关键困难 setting

完整 Point-Cache 后仍然最低的 setting 是：

| cor_type | Zero-shot | ZS + Global | ZS + Global + Local | 现象 |
|---|---:|---:|---:|---|
| jitter_4 | 18.53 | 23.32 | 25.29 | 最高 severity 坐标扰动，仍最低 |
| jitter_3 | 23.66 | 28.11 | 29.32 | 中高 severity 坐标扰动 |
| dropout_local_4 | 23.91 | 28.21 | 30.08 | 高 severity 局部缺失 |
| dropout_global_4 | 25.19 | 32.65 | 34.04 | 高 severity 全局缺失 |
| dropout_local_3 | 27.97 | 32.55 | 34.42 | 中高 severity 局部缺失 |
| jitter_2 | 28.66 | 34.42 | 35.60 | severity=2 坐标扰动 |
| jitter_1 | 33.34 | 36.75 | 38.76 | severity=1 坐标扰动 |
| dropout_global_3 | 30.88 | 37.89 | 39.00 | 中高 severity 全局缺失 |

分析：

这些 setting 显示 34 组的主要困难区域非常集中：high-severity jitter、dropout_local、dropout_global。完整 Point-Cache 对它们均有提升，但最终值仍然偏低。

这为后续 MCM-PC 的方法设计提供了明确目标。

---

## 14. 与 33 组 ScanObjNN clean hardest 的关系

33 组是 Uni3D 在 ScanObjNN clean hardest 上的结果；34 组是 Uni3D 在 ScanObjNN-C hardest all35 上的结果。

| 方法 | 33 组 clean hardest | 34 组 S2 Avg | 34 组 all35 Avg |
|---|---:|---:|---:|
| Zero-shot | 45.63 | 37.75 | 37.66 |
| ZS + Global | 50.03 | 42.31 | 41.98 |
| ZS + Global + Local | 51.98 | 44.13 | 43.17 |

从 clean hardest 到 corrupted ScanObjNN-C hardest 的下降：

| 方法 | S2 Avg - clean | all35 Avg - clean |
|---|---:|---:|
| Zero-shot | -7.88 | -7.97 |
| ZS + Global | -7.72 | -8.05 |
| ZS + Global + Local | -7.85 | -8.81 |

分析：

ScanObjNN-C corruption 会在 ScanObjNN clean hardest 的基础上明显降低 Uni3D 性能。即使使用完整 Point-Cache，all35 Avg 仍从 clean 的 51.98 下降到 43.17，下降 -8.81。

这说明 cache 能提高 corrupted accuracy，但不能完全消除 clean-to-corruption gap。

---

## 15. 与其他 backbone 的 ScanObjNN-C hardest 关系

34 组可以与 ULIP、OpenShape 的 ScanObjNN-C hardest 结果进行横向比较。

| Backbone | 方法 | S2 Avg | all35 Avg |
|---|---|---:|---:|
| ULIP | Zero-shot | 23.91 | 23.65 |
| ULIP | ZS + Global | 26.84 | 26.60 |
| ULIP | ZS + Global + Local | 27.94 | 27.41 |
| OpenShape | Zero-shot | 32.75 | 32.72 |
| OpenShape | ZS + Global | 37.30 | 36.71 |
| OpenShape | ZS + Global + Local | 38.63 | 37.84 |
| Uni3D | Zero-shot | 37.75 | 37.66 |
| Uni3D | ZS + Global | 42.31 | 41.98 |
| Uni3D | ZS + Global + Local | 44.13 | 43.17 |

分析：

Uni3D 在 ScanObjNN-C hardest 上明显强于 ULIP 和 OpenShape。完整 Point-Cache 下，Uni3D all35 Avg 为 43.17，OpenShape 为 37.84，差距为 +5.33。

这说明在真实扫描 hardest corruption 场景中，Uni3D 是当前最强 backbone，但其绝对准确率仍然不高，因此该 setting 仍有很大提升空间。

---

## 16. 当前结果意义分析

34 组结果说明：

| 观察 | 解释 |
|---|---|
| Zero-shot all35 = 37.66 | Uni3D 在 ScanObjNN-C hardest 上的基础鲁棒性 |
| ZS + Global all35 = 41.98 | Global Cache 有明确正增益 |
| ZS + Global + Local all35 = 43.17 | 完整 Point-Cache 最好 |
| Global extra = +4.32 all35 | Global Cache 是主要提升来源 |
| Local extra = +1.19 all35 | Local Cache 也有额外贡献 |
| Full gain = +5.51 all35 | 完整 Point-Cache 显著优于 Zero-shot |
| jitter 始终最低 | 坐标扰动是最主要失败模式 |
| high-severity dropout 仍然困难 | 点云缺失仍需要进一步机制 |
| 34_3 略高于原文 | Local Cache 额外贡献强于原文 |

34 组是整个 baseline 复现阶段的最后一组，也是最困难的一组。它说明 Uni3D 在 ScanObjNN-C hardest 上能够通过 cache 获得显著提升，但 high-severity jitter 和 dropout 仍然是明显短板。

---

## 17. 对后续 MCM-PC 的启发

当前 34 组实验对后续方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| ScanObjNN-C hardest all35 难度极高 | 后续方法应重点验证真实扫描 corruption setting |
| Global Cache 是主要提升来源 | 全局缓存是稳定主模块，应保留 |
| Local Cache 有额外贡献 | 局部缓存对 dropout / jitter 有价值 |
| Local Cache 对 add_local 略负 | 需要 reliability-aware 或 corruption-aware 的局部缓存调节 |
| jitter_4 仍只有 25.29 | 高 severity 坐标扰动需要专门鲁棒机制 |
| dropout_local_4 只有 30.08 | 局部缺失仍是强失败模式 |
| dropout_global_4 只有 34.04 | 全局缺失仍是强失败模式 |
| Uni3D checkpoint 必须与数据集匹配 | 实验脚本必须显式记录 checkpoint |
| all35 runner 复制容易残留 metadata | 后续复制 runner 必须逐项检查 Python 内部字段 |

这对 MCM-PC 很重要：后续方法不应简单固定 Global / Local 的作用，而应根据样本可靠性、全局-局部一致性、伪标签可信度和 corruption 类型动态调节缓存贡献。

---

## 18. Baseline 大复现阶段收尾说明

34 组完成后，当前 Point-Cache baseline 大复现阶段已经覆盖四个 backbone 和四类数据设置。

| Backbone | ModelNet clean | ModelNet-C all35 | ScanObjNN clean hardest | ScanObjNN-C hardest all35 |
|---|---|---|---|---|
| ULIP | 01 | 02 | 03 | 04 |
| ULIP-2 | 11 | 12 | 13 | 14 |
| OpenShape | 21 | 22 | 23 | 24 |
| Uni3D | 31 | 32 | 33 | 34 |

因此，34 组是当前 baseline 大复现阶段的最后一组实验。

当前 34 组完成后，可以进入以下后续工作：

| 后续方向 | 说明 |
|---|---|
| 汇总所有 baseline 结果 | 整理 01–34 的总表 |
| 横向比较 backbone | ULIP、ULIP-2、OpenShape、Uni3D |
| 分析 cache 贡献模式 | Global / Local 在不同 backbone 和 dataset 上的差异 |
| 识别 MCM-PC 切入点 | high-severity jitter、dropout、local/global conflict |
| 准备方法实验 | 在确认 baseline 后开始 MCM-PC 新模块实验 |

---

## 19. 阶段性结论

34 组 Uni3D × ScanObjNN-C hardest all35 baseline 已完成。

主要结论如下：

1. 三个子实验均完成，并且 summary.csv、cor_type、log_path 和 logs 文件数量均为 35。
2. 当前正式结果均使用 weights/uni3d/scanobjnn/model.pt。
3. 当前 metadata 已修正并确认正确：dataset=sonn_c，data_root=data/sonn_c/hardest，file=data/sonn_c/hardest/{cor_type}.h5，sonn_variant=hardest。
4. 34_1 Zero-shot 的 severity=2 Avg 为 37.75，原文为 37.38，差异 +0.37。
5. 34_2 ZS + Global 的 severity=2 Avg 为 42.31，原文为 42.03，差异 +0.28。
6. 34_3 ZS + Global + Local 的 severity=2 Avg 为 44.13，原文为 43.10，差异 +1.03。
7. 三个结果整体与原文接近，均可作为有效复现结果保留。
8. all35 Avg 从 Zero-shot 的 37.66 提升到 Global 的 41.98，再提升到 Global + Local 的 43.17。
9. Global Cache 是主要提升来源，all35 Avg 提升 +4.32，severity=2 Avg 提升 +4.56。
10. Local Cache 在 Global Cache 基础上有明确额外贡献，all35 Avg 提升 +1.19，severity=2 Avg 提升 +1.82。
11. 完整 Point-Cache 相比 Zero-shot 提升明显，all35 Avg 提升 +5.51，severity=2 Avg 提升 +6.38。
12. jitter 是最困难 corruption，完整 Point-Cache 后 all35 平均仍只有 35.30。
13. dropout_local 和 dropout_global 也是主要困难 corruption。
14. jitter_4、jitter_3、dropout_local_4 和 dropout_global_4 是当前最困难 setting。
15. 34 组说明：在 Uni3D × ScanObjNN-C hardest 中，Global Cache 和 Local Cache 都有价值。
16. 34 组是当前 Point-Cache baseline 大复现阶段的最后一组实验。
17. 后续可以进入所有 baseline 结果总表整理和 MCM-PC 方法实验设计阶段。

---

## 20. 运行命令汇总

34_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs_single_gpu.sh 0

34_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh 0

34_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh 1

---

## 21. 检查命令汇总

34_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f6 | sort -u | wc -l

tail -n +2 results/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

34_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f6 | sort -u | wc -l

tail -n +2 results/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

34_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f6 | sort -u | wc -l

tail -n +2 results/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
