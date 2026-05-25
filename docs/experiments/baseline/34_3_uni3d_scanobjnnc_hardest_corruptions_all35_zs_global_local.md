# 34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local

## 1. 实验目的

本实验用于复现 Uni3D 在 ScanObjNN-C hardest 全部 35 个损坏设置上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local |
| Backbone | Uni3D |
| Dataset | ScanObjNN-C hardest |
| Dataset 参数 | sonn_c |
| SONN variant | hardest |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 34_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证完整 Point-Cache 对 Uni3D 在 ScanObjNN-C hardest corrupted setting 上的最终影响。

需要特别注意：原文 Point-Cache Table 7 报告的是 ScanObjNN-C hardest 在 severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原文 Point-Cache Table 7 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

本文件只记录 34_3 本身，并与前序子实验 34_1 和 34_2 进行对比。完整 34 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 34 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | Uni3D |
| 数据集参数 | sonn_c |
| 数据目录 | data/sonn_c |
| 实际 h5 文件目录 | data/sonn_c/hardest |
| SONN variant | hardest |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 优化 runner | Point-Cache/runners/baseline/run_uni3d_scanobjnnc_hardest_corruptions_all35.py |
| 方法脚本 | Point-Cache/scripts/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/34_run_uni3d_scanobjnnc_hardest_corruptions_all35_common.sh |
| cache_type | hierarchical |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |
| 输入点数 | 1024 |
| Global Cache shot_capacity | 3 |
| Local Cache shot_capacity | 3 |
| Global / Local alpha | 4.0 |
| Global / Local beta | 3.0 |
| KMeans 聚类数 | 3 |
| Uni3D point encoder checkpoint | weights/uni3d/scanobjnn/model.pt |
| Uni3D text encoder checkpoint | weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin |
| pc_model | eva_giant_patch14_560 |
| clip_model | EVA02-E-14-plus |
| pc_feat_dim | 1408 |
| num_group | 512 |
| group_size | 64 |
| pc_encoder_dim | 512 |
| embed_dim | 1024 |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 1 |

本实验使用 `sonn_c` 作为 dataset 参数，并指定：

| 参数 | 值 |
|---|---|
| sonn_variant | hardest |
| cor_type | 由 runner 内部循环 35 个 corruption setting |
| npoints | 1024 |
| sim2real_type | so_obj_only_9 |

实际 h5 文件形式为：

data/sonn_c/hardest/{corruption}_{severity}.h5

---

## 3. 当前实现方式

34 组属于 ScanObjNN-C hardest all35 实验，因此使用优化版 Python runner：

Point-Cache/runners/baseline/run_uni3d_scanobjnnc_hardest_corruptions_all35.py

该 runner 的设计是：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 | 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 | 模型只加载一次，每个 cor_type 重新创建 DataLoader、重新初始化 Global Cache 和 Local Cache |
| bash 通过 tee 生成单个 cor_type 的 log | Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv | summary.csv 的列结构保持不变，每个 cor_type 一行 |

该优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、Global Cache 初始化、Local Cache 初始化、预测逻辑和输出结构。

---

## 4. 34 runner 修正记录

34 组最初从 32 组 ModelNet-C all35 runner 复制而来，因此曾出现过 runner 内部 metadata 残留问题。旧 runner 中存在类似字段：

| 错误字段 | 问题 |
|---|---|
| args.modelnet_c_root = "data/modelnet_c" | 继承自 32 组 ModelNet-C |
| data_root = args.modelnet_c_root | summary 中写成 ModelNet-C 路径 |
| sonn_variant = "-" | 没有记录 hardest split |
| file = data/modelnet_c/{cor_type}.h5 | summary 文件路径错误 |

修正后，当前 runner 已确认使用以下设置：

| 字段 | 当前正确值 |
|---|---|
| args.dataset | sonn_c |
| args.sonn_variant | hardest |
| args.sonn_c_root | data/sonn_c |
| args.data_root | data/sonn_c |
| summary data_root | data/sonn_c/hardest |
| summary file | data/sonn_c/hardest/{cor_type}.h5 |
| summary sonn_variant | hardest |
| expected checkpoint | weights/uni3d/scanobjnn/model.pt |

需要注意：`SONN_C` 数据集类需要在 `data/sonn_c` 下读取 `shape_names.txt`，因此 loader root 应为 `data/sonn_c`；但实际 h5 文件在 `data/sonn_c/hardest`，因此 summary 中记录的实际文件路径应为 `data/sonn_c/hardest/{cor_type}.h5`。

当前 34_3 summary 已确认 metadata 正确。

---

## 5. 方法说明

34_3 在 Zero-shot logits 的基础上同时加入 Global Cache logits 和 Local Cache logits。

| 组成部分 | 是否使用 |
|---|---:|
| Text prototype / 文本原型 | 是 |
| Point cloud global feature | 是 |
| Zero-shot logits | 是 |
| Global Cache logits | 是 |
| Local Cache logits | 是 |
| Hierarchical Cache | 是 |

完整 Point-Cache 的预测由三部分组成：

| 组成部分 | 作用 |
|---|---|
| Zero-shot logits | 来自 Uni3D 的原始文本-点云相似度预测 |
| Global Cache logits | 基于全局点云特征的测试时缓存检索结果 |
| Local Cache logits | 基于局部 patch / 局部聚类特征的测试时缓存检索结果 |

34_3 与前两个子实验的关系如下：

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 34_1 | 是 | 否 | 否 |
| 34_2 | 是 | 是 | 否 |
| 34_3 | 是 | 是 | 是 |

因此，34_3 可以用于评估完整 Point-Cache 在 Uni3D × ScanObjNN-C hardest 上的最终表现，并判断 Local Cache 是否在 Global Cache 基础上带来额外贡献。

---

## 6. Uni3D checkpoint 说明

本实验使用的 Uni3D point encoder checkpoint 为：

weights/uni3d/scanobjnn/model.pt

这是 33 / 34 组 Uni3D × ScanObjNN 系列实验的正式 checkpoint。

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

该 checkpoint 的下载脚本已记录在：

Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh

---

## 7. 损坏类型

本实验包含 7 种 corruption，每种 corruption 包含 5 个 severity level：

| corruption | severity 编号 |
|---|---|
| add_global | 0, 1, 2, 3, 4 |
| add_local | 0, 1, 2, 3, 4 |
| dropout_global | 0, 1, 2, 3, 4 |
| dropout_local | 0, 1, 2, 3, 4 |
| rotate | 0, 1, 2, 3, 4 |
| scale | 0, 1, 2, 3, 4 |
| jitter | 0, 1, 2, 3, 4 |

注意：severity 文件编号从 0 开始，到 4 结束。不要写成 1 到 5。

文件命名形式为：

| 示例 | 含义 |
|---|---|
| add_global_0.h5 | add_global corruption，severity=0 |
| add_global_2.h5 | add_global corruption，severity=2 |
| jitter_4.h5 | jitter corruption，severity=4 |

---

## 8. 输出结构

输出目录：

Point-Cache/results/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local_{cor_type}_YYYYMMDD_HHMMSS.log

---

## 9. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 35 | 35 | 说明 35 个 cor_type 全部完成 |
| summary 中唯一 cor_type 数 | 35 | 35 | 说明没有漏跑或重复写入 |
| summary 中唯一 log_path 数 | 35 | 35 | 说明每个 cor_type 都有独立日志路径 |
| logs 目录当前 .log 文件数 | 35 | 35 | 说明没有旧日志或重复日志残留 |
| status=done 数 | 35 | 35 | 说明没有失败项 |
| dataset | sonn_c | sonn_c | 正确 |
| data_root | data/sonn_c/hardest | data/sonn_c/hardest | 正确 |
| file | data/sonn_c/hardest/{cor_type}.h5 | data/sonn_c/hardest/{cor_type}.h5 | 正确 |
| sonn_variant | hardest | hardest | 正确 |
| severity=2 Average | 44.13 | 用于论文对齐 | 与原文 Point-Cache Table 7 对比 |
| all35 Average | 43.17 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性和 metadata 看，34_3 现在是干净结果。

---

## 10. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 51.91 | 54.58 | 55.31 | 50.94 | 51.73 | 52.89 |
| add_local | 45.21 | 39.35 | 41.67 | 39.52 | 40.18 | 41.19 |
| dropout_global | 47.92 | 45.98 | 44.41 | 39.00 | 34.04 | 42.27 |
| dropout_local | 48.44 | 46.08 | 40.77 | 34.42 | 30.08 | 39.96 |
| rotate | 50.38 | 49.41 | 48.85 | 46.81 | 42.96 | 47.68 |
| scale | 46.63 | 43.82 | 42.33 | 41.39 | 40.46 | 42.93 |
| jitter | 47.54 | 38.76 | 35.60 | 29.32 | 25.29 | 35.30 |
| **Average** | **48.29** | **45.43** | **44.13** | **40.20** | **37.82** | **43.17** |

整体观察：

1. all35 Average 为 43.17，表示 Uni3D 在 ScanObjNN-C hardest 全 35 个 corrupted setting 上使用完整 Point-Cache 后的总体鲁棒性水平。
2. severity=2 Average 为 44.13，用于和原文 Point-Cache Table 7 对齐。
3. add_global 的平均准确率最高，为 52.89。
4. jitter 的平均准确率最低，为 35.30。
5. jitter_4 为 25.29，是全部 35 个 setting 中最低结果。
6. 相比 34_1 Zero-shot 和 34_2 Global Cache，34_3 取得最高的 S2 Average 和 all35 Average。

---

## 11. 与原文结果对比

原文 Point-Cache Table 7 报告的是 ScanObjNN-C hardest 在 severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

原文 Uni3D 在 ScanObjNN-C hardest severity=2 下的 +Hierarchical Cache Average 为 43.10。

当前复现 severity=2 Average 为 44.13。

| 对比对象 | 原文 S2 Avg | 当前复现 S2 Avg | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Uni3D / ScanObjNN-C hardest / +Hierarchical Cache | 43.10 | 44.13 | +1.03 | 1.03 |

分析：

当前复现结果 44.13 比原文 43.10 高 +1.03。相较 34_1 和 34_2，34_3 的正偏差更明显，但方法趋势和增益结构合理。

因此，34_3 可以作为有效复现结果保留，但文档中应明确记录：当前完整 Point-Cache 结果略高于原文。

---

## 12. 与前序实验 34_1 和 34_2 的对比

34_1 是无缓存 Zero-shot，34_2 是 Zero-shot + Global Cache，34_3 是完整 Point-Cache。

| 实验编号 | 方法 | S2 Avg | all35 Avg |
|---|---|---:|---:|
| 34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs | Zero-shot | 37.75 | 37.66 |
| 34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global | Zero-shot + Global Cache | 42.31 | 41.98 |
| 34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local | Zero-shot + Global Cache + Local Cache | 44.13 | 43.17 |

当前复现变化：

| 比较 | S2 Gain | all35 Gain | 含义 |
|---|---:|---:|---|
| 34_2 - 34_1 | +4.56 | +4.32 | Global Cache 相比 Zero-shot 的变化 |
| 34_3 - 34_2 | +1.82 | +1.19 | Local Cache 在 Global Cache 基础上的额外变化 |
| 34_3 - 34_1 | +6.38 | +5.51 | 完整 Point-Cache 相比 Zero-shot 的总体变化 |

原文对应变化：

| 比较 | 原文 S2 变化 |
|---|---:|
| Global Cache - Zero-shot | 42.03 - 37.38 = +4.65 |
| Hierarchical Cache - Global Cache | 43.10 - 42.03 = +1.07 |
| Hierarchical Cache - Zero-shot | 43.10 - 37.38 = +5.72 |

变化对齐情况：

| 指标 | 原文 S2 | 当前复现 S2 | Diff |
|---|---:|---:|---:|
| Global Cache 相对 Zero-shot 的变化 | +4.65 | +4.56 | -0.09 |
| Local Cache 额外变化 | +1.07 | +1.82 | +0.75 |
| 完整 Point-Cache 相对 Zero-shot 的变化 | +5.72 | +6.38 | +0.66 |

分析：

当前复现中，完整 Point-Cache 将 severity=2 Average 从 34_1 的 37.75 提升到 34_3 的 44.13，总提升 +6.38。原文总提升为 +5.72，当前高 +0.66。

Global Cache 的增益与原文几乎完全一致：原文 +4.65，当前 +4.56。Local Cache 的额外增益当前更强：原文 +1.07，当前 +1.82。这也是 34_3 比原文高 +1.03 的主要原因。

因此，34_3 可以认为复现有效，但需要在 summary 文档中记录：Local Cache 额外增益高于原文，导致完整 Point-Cache 最终值略高于原文。

---

## 13. Severity 维度分析

### 13.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 48.29 | — | 0.00 |
| S1 | 45.43 | -2.86 | -2.86 |
| S2 | 44.13 | -1.30 | -4.16 |
| S3 | 40.20 | -3.93 | -8.09 |
| S4 | 37.82 | -2.38 | -10.47 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 48.29 下降到 37.82，总下降 10.47 个百分点。整体上，severity 越高，完整 Point-Cache 的准确率越低。

### 13.2 与 34_1 / 34_2 的 severity 维度对比

| Severity | 34_1 Zero-shot Avg | 34_2 ZS + Global Avg | 34_3 ZS + Global + Local Avg | Gain over 34_1 | Gain over 34_2 |
|---:|---:|---:|---:|---:|---:|
| S0 | 42.67 | 47.10 | 48.29 | +5.62 | +1.19 |
| S1 | 40.46 | 44.22 | 45.43 | +4.97 | +1.21 |
| S2 | 37.75 | 42.31 | 44.13 | +6.38 | +1.82 |
| S3 | 35.06 | 39.52 | 40.20 | +5.14 | +0.68 |
| S4 | 32.37 | 36.74 | 37.82 | +5.45 | +1.08 |
| **all35** | **37.66** | **41.98** | **43.17** | **+5.51** | **+1.19** |

分析：

完整 Point-Cache 相比 Zero-shot 在所有 severity 上均有明显正增益。Local Cache 在 Global Cache 基础上也在所有 severity 上为正增益，其中 S2 的额外提升最大，为 +1.82。

---

## 14. Corruption 维度分析

### 14.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 35.30 | 平均最低，高 severity 下仍然困难 |
| 2 | dropout_local | 39.96 | 点云局部缺失仍然困难 |
| 3 | add_local | 41.19 | Local Cache 后略低于 Global Cache |
| 4 | dropout_global | 42.27 | 点云全局缺失仍然困难 |
| 5 | scale | 42.93 | 中等水平 |
| 6 | rotate | 47.68 | 相对较高 |
| 7 | add_global | 52.89 | 当前最高 |

分析：

完整 Point-Cache 后，jitter 仍然是最困难 corruption，但平均准确率从 34_1 的 28.85 提升到 34_3 的 35.30。dropout_local 和 dropout_global 仍然较难，说明点云缺失仍是 ScanObjNN-C hardest 的主要困难类型。

### 14.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|
| add_global | 51.91 | 51.73 | -0.18 | 52.89 |
| add_local | 45.21 | 40.18 | -5.03 | 41.19 |
| scale | 46.63 | 40.46 | -6.17 | 42.93 |
| rotate | 50.38 | 42.96 | -7.42 | 47.68 |
| dropout_global | 47.92 | 34.04 | -13.88 | 42.27 |
| dropout_local | 48.44 | 30.08 | -18.36 | 39.96 |
| jitter | 47.54 | 25.29 | -22.25 | 35.30 |

分析：

jitter 的退化最强，从 S0 的 47.54 下降到 S4 的 25.29，下降 -22.25。dropout_local 的退化也很强，从 48.44 下降到 30.08。说明完整 Point-Cache 虽然提高了整体准确率，但不能完全解决 high-severity jitter 和 dropout_local 的困难。

---

## 15. 完整 Point-Cache 相比 34_1 Zero-shot 的逐项提升

| Corruption | 34_1 ZS Avg | 34_3 Full Avg | Full Gain |
|---|---:|---:|---:|
| add_global | 47.39 | 52.89 | +5.50 |
| add_local | 38.69 | 41.19 | +2.50 |
| dropout_global | 35.07 | 42.27 | +7.20 |
| dropout_local | 32.67 | 39.96 | +7.29 |
| rotate | 42.84 | 47.68 | +4.84 |
| scale | 38.14 | 42.93 | +4.79 |
| jitter | 28.85 | 35.30 | +6.45 |
| **Average** | **37.66** | **43.17** | **+5.51** |

分析：

完整 Point-Cache 对全部 7 种 corruption 均为正增益。提升最大的 corruption 是 dropout_local，平均提升 +7.29；其次是 dropout_global，平均提升 +7.20；jitter 也有明显提升，平均提升 +6.45。

这说明完整 Point-Cache 对点云缺失和坐标扰动有明显帮助，但这些 corruption 仍然是最终最困难区域。

---

## 16. Local Cache 相比 34_2 Global Cache 的额外贡献

| Corruption | 34_2 Global Avg | 34_3 Full Avg | Local Extra |
|---|---:|---:|---:|
| add_global | 52.37 | 52.89 | +0.52 |
| add_local | 41.69 | 41.19 | -0.50 |
| dropout_global | 40.76 | 42.27 | +1.51 |
| dropout_local | 37.22 | 39.96 | +2.74 |
| rotate | 46.72 | 47.68 | +0.96 |
| scale | 41.66 | 42.93 | +1.27 |
| jitter | 33.43 | 35.30 | +1.87 |
| **Average** | **41.98** | **43.17** | **+1.19** |

分析：

Local Cache 在 Global Cache 基础上的 all35 平均额外提升为 +1.19，severity=2 额外提升为 +1.82。Local Cache 对 dropout_local、jitter 和 dropout_global 的额外帮助最明显，分别为 +2.74、+1.87 和 +1.51。

不过，Local Cache 对 add_local 的 all35 平均为 -0.50，说明局部缓存并不是对所有 corruption 都稳定正向。与 32 组 Uni3D × ModelNet-C 不同，34 组中 Local Cache 主要改善 dropout 和 jitter，而不是 add_local。

---

## 17. 低准确率区域分析

| 条件 | 34_1 Zero-shot 数量 | 34_2 ZS + Global 数量 | 34_3 ZS + Global + Local 数量 |
|---|---:|---:|---:|
| Acc < 40 | 22 / 35 | 11 / 35 | 8 / 35 |
| Acc < 35 | 11 / 35 | 7 / 35 | 5 / 35 |
| Acc < 30 | 6 / 35 | 3 / 35 | 2 / 35 |
| Acc < 25 | 3 / 35 | 1 / 35 | 0 / 35 |

分析：

完整 Point-Cache 明显减少了低准确率区域。Acc < 40 的 setting 从 Zero-shot 的 22 个减少到 Global Cache 的 11 个，再减少到完整 Point-Cache 的 8 个。Acc < 25 的 setting 从 Zero-shot 的 3 个减少到完整 Point-Cache 的 0 个。

这说明完整 Point-Cache 不仅提高平均准确率，也减少了极低性能 setting 的数量。

---

## 18. 当前仍然困难的低准确率 setting

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 25.29 | 最高 severity 坐标扰动，当前最低 setting |
| jitter_3 | 29.32 | 中高 severity 坐标扰动 |
| dropout_local_4 | 30.08 | 高 severity 局部缺失 |
| dropout_global_4 | 34.04 | 高 severity 全局缺失 |
| dropout_local_3 | 34.42 | 中高 severity 局部缺失 |
| jitter_2 | 35.60 | severity=2 坐标扰动 |
| jitter_1 | 38.76 | severity=1 坐标扰动 |
| dropout_global_3 | 39.00 | 中高 severity 全局缺失 |

分析：

34_3 中最困难区域仍然集中在 high-severity jitter、dropout_local 和 dropout_global。说明完整 Point-Cache 对这些 setting 有明显提升，但不能完全解决强坐标扰动和点云缺失问题。

---

## 19. 与 33_3 ScanObjNN clean hardest 的关系

33_3 是 Uni3D 在 ScanObjNN clean hardest 上的完整 Point-Cache 结果；34_3 是 Uni3D 在 ScanObjNN-C hardest all35 上的完整 Point-Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy / Avg |
|---|---|---|---:|
| 33_3_uni3d_scanobjnn_clean_hardest_zs_global_local | ScanObjNN clean hardest | ZS + Global + Local | 51.98 |
| 34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local | ScanObjNN-C hardest S2 Avg | ZS + Global + Local | 44.13 |
| 34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local | ScanObjNN-C hardest all35 Avg | ZS + Global + Local | 43.17 |

当前结果说明：从 ScanObjNN clean hardest 到 ScanObjNN-C hardest corruption，Uni3D 完整 Point-Cache 明显下降。

| 比较 | 变化 |
|---|---:|
| 34_3 S2 Avg - 33_3 clean | 44.13 - 51.98 = -7.85 |
| 34_3 all35 Avg - 33_3 clean | 43.17 - 51.98 = -8.81 |

分析：

ScanObjNN-C hardest corruption 在 clean hardest 的基础上进一步降低 Uni3D 完整 Point-Cache 性能。完整 Point-Cache 提高了 corrupted accuracy，但 corruption gap 仍然存在。

---

## 20. 与其他 backbone 的 ScanObjNN-C hardest 完整 Point-Cache 关系

34_3 可以与前面 ULIP、OpenShape 等 ScanObjNN-C hardest 完整 Point-Cache 结果进行横向比较。

| Backbone | 实验编号 | ScanObjNN-C hardest S2 Avg | ScanObjNN-C hardest all35 Avg |
|---|---|---:|---:|
| ULIP | 04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local | 27.94 | 27.41 |
| OpenShape | 24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local | 38.63 | 37.84 |
| Uni3D | 34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local | 44.13 | 43.17 |

分析：

完整 Point-Cache 下，Uni3D 在 ScanObjNN-C hardest 上明显强于 ULIP 和 OpenShape。当前 Uni3D all35 Avg 为 43.17，比 OpenShape 的 37.84 高 +5.33。

这说明在真实扫描 hardest corruption 场景中，Uni3D 是当前最强 backbone，但其 corrupted accuracy 仍然存在明显提升空间。

---

## 21. 结果含义分析

34_3 的结果说明：完整 Point-Cache 在 Uni3D × ScanObjNN-C hardest all35 上有效，并且最终数值略高于原文。

| 观察 | 含义 |
|---|---|
| 34_3 all35 Avg = 43.17 | 当前完整 Point-Cache 总体结果 |
| 34_3 S2 Avg = 44.13 | 与原文 Table 7 对齐的复现结果 |
| 原文 +Hierarchical Avg = 43.10 | 当前高 +1.03，略高但可接受 |
| 相比 34_1 S2 提升 +6.38 | 完整 Point-Cache 有明确正增益 |
| 相比 34_2 S2 提升 +1.82 | Local Cache 有额外贡献 |
| Global Cache 是主要提升来源 | 34_2 - 34_1 的 all35 增益为 +4.32 |
| Local Cache 主要改善 dropout 和 jitter | 对 dropout_local、jitter、dropout_global 额外提升明显 |
| high-severity jitter / dropout 仍然困难 | 仍是后续 MCM-PC 的关键失败模式 |

因此，34_3 是 34 组三个子实验中最关键的最终结果。它证明了完整 Point-Cache 在 Uni3D × ScanObjNN-C hardest corrupted setting 中有效，也说明 Local Cache 在该场景中具有明确价值。

---

## 22. 对后续 MCM-PC 的启发

当前 34_3 对后续 MCM-PC 方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| 完整 Point-Cache 相比 Zero-shot all35 提升 +5.51 | cache 机制在最困难真实扫描 corruption 场景中有效 |
| Global Cache 提供主要提升 | 全局缓存是稳定主模块，应保留 |
| Local Cache 仍有额外贡献 | 局部缓存对 dropout / jitter 有价值 |
| Local Cache 对 add_local 略负 | 局部缓存贡献需要 corruption-aware 或 reliability-aware 调节 |
| jitter_4 仍只有 25.29 | 高 severity 坐标扰动需要专门鲁棒机制 |
| dropout_local_4 只有 30.08 | 点云局部缺失仍是难点 |
| dropout_global_4 只有 34.04 | 点云全局缺失仍是难点 |
| ScanObjNN-C hardest 远难于 clean hardest | 后续方法要关注真实扫描 corruption gap |

这对 MCM-PC 很重要：后续方法不应简单固定 Global / Local 的作用，而应根据样本可靠性、全局-局部一致性、伪标签可信度和 corruption 类型动态调节缓存贡献。

---

## 23. 阶段性结论

本实验完成了 Uni3D × ScanObjNN-C hardest all35 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 34_3 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前使用 checkpoint 为 weights/uni3d/scanobjnn/model.pt。
3. 当前 metadata 已修正并确认正确：dataset=sonn_c，data_root=data/sonn_c/hardest，sonn_variant=hardest。
4. 当前 severity=2 Average 为 44.13，原文 Point-Cache Table 7 中 Uni3D +Hierarchical Cache Avg 为 43.10，差异 +1.03。
5. 当前 all35 Average 为 43.17，是本实验额外统计的 35 个 corrupted setting 总平均。
6. 当前结果略高于原文，但趋势和增益结构合理，可以认为 34_3 复现有效。
7. 相比 34_1 Zero-shot，34_3 的 severity=2 Average 提升 +6.38，all35 Average 提升 +5.51。
8. 相比 34_2 Global Cache，34_3 的 severity=2 Average 额外提升 +1.82，all35 Average 额外提升 +1.19。
9. Global Cache 是主要提升来源，Local Cache 也有明确额外贡献。
10. Local Cache 对 dropout_local、jitter 和 dropout_global 的额外贡献最明显。
11. Local Cache 对 add_local 的 all35 平均略降 -0.50，说明局部缓存并非对所有 corruption 都稳定正向。
12. jitter 仍然是最困难 corruption，完整 Point-Cache 后平均仍只有 35.30。
13. jitter_4 是全部 35 个 setting 中最低结果，只有 25.29。
14. high-severity jitter、dropout_local 和 dropout_global 是后续 MCM-PC 的主要困难目标。
15. 该实验可作为 34 组 summary 文档和后续 MCM-PC 方法改进的重要 baseline。
16. 33 / 34 组后续应统一使用 weights/uni3d/scanobjnn/model.pt。

---

## 24. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh 1

---

## 25. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f6 | sort -u | wc -l

tail -n +2 results/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv | cut -d',' -f13 | sort | uniq -c

head -2 results/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv

cat results/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local/summary.csv
