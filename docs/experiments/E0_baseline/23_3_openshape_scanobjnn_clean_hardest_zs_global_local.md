# 23_3_openshape_scanobjnn_clean_hardest_zs_global_local

## 1. 实验目的

本实验用于复现 OpenShape 在 ScanObjNN clean hardest split 上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 23_3_openshape_scanobjnn_clean_hardest_zs_global_local |
| Backbone | OpenShape |
| Dataset | ScanObjNN clean hardest |
| Dataset 参数 | sonn_c |
| SONN variant | hardest |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 23_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证完整 Point-Cache 对 OpenShape 在 ScanObjNN clean hardest 上的影响。

需要特别注意：当前 23 组中，23_2 Global Cache 单独增益很弱，只比 23_1 Zero-shot 高 +0.07；而 23_3 在加入 Local Cache 后达到 43.82，比 23_2 高 +1.87。因此，23 组的主要提升来源不是 Global Cache，而是 Local Cache 在 Global Cache 基础上的额外贡献。

本文件只记录 23_3 本身，并与前序子实验 23_1 和 23_2 进行对比。完整 23 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 23 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | OpenShape |
| 数据集参数 | sonn_c |
| 数据目录 | data/sonn_c |
| 数据集变体 | hardest |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 方法脚本 | Point-Cache/scripts/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/23_run_openshape_scanobjnn_clean_hardest_common.sh |
| cache_type | hierarchical |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |
| 输入点数 | 1024 |
| Global Cache shot_capacity | 3 |
| Local Cache shot_capacity | 3 |
| Global / Local alpha | 4.0 |
| Global / Local beta | 3.0 |
| KMeans 聚类数 | 3 |
| OpenShape version | vitg14 |
| OpenShape 权重 | weights/openshape/openshape-pointbert-vitg14-rgb/model.pt |
| OpenCLIP 权重 | weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 0 |

本实验使用 `sonn_c` 作为 dataset 参数，并指定：

| 参数 | 值 |
|---|---|
| sonn_variant | hardest |
| cor_type | clean |

实际读取文件为：

data/sonn_c/hardest/clean.h5

---

## 3. 当前实现方式

23 组是 clean 单文件实验，因此不需要 all35 优化 runner。

| 项目 | 说明 |
|---|---|
| 是否需要 35 个 cor_type 循环 | 否 |
| 是否需要 Python 内部循环 | 否 |
| 是否需要 all35 runner | 否 |
| 是否每个子实验只生成 1 行 summary | 是 |
| 是否每个子实验只生成 1 个 log | 是 |

23_3 的执行路径为：

| 层级 | 文件 |
|---|---|
| 单方法脚本 | Point-Cache/scripts/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/23_run_openshape_scanobjnn_clean_hardest_common.sh |
| 原始推理 runner | Point-Cache/runners/model_with_hierarchical_caches.py |

该结构与 03、13、21、23_1、23_2 等 clean 单文件实验保持一致。

---

## 4. 方法说明

23_3 在 Zero-shot logits 的基础上同时加入 Global Cache logits 和 Local Cache logits。

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
| Zero-shot logits | 来自 OpenShape 的原始文本-点云相似度预测 |
| Global Cache logits | 基于全局点云特征的测试时缓存检索结果 |
| Local Cache logits | 基于局部 patch / 局部聚类特征的测试时缓存检索结果 |

23_3 与前两个子实验的关系如下：

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 23_1 | 是 | 否 | 否 |
| 23_2 | 是 | 是 | 否 |
| 23_3 | 是 | 是 | 是 |

因此，23_3 可以用于评估完整 Point-Cache 在 OpenShape × ScanObjNN clean hardest 上的最终表现，并判断 Local Cache 是否在 Global Cache 基础上带来额外贡献。

---

## 5. 输出结构

输出目录：

Point-Cache/results/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

23_3_openshape_scanobjnn_clean_hardest_zs_global_local_clean_YYYYMMDD_HHMMSS.log

因为本实验只测试 `clean.h5`，所以期望只有 1 行 summary 记录和 1 个对应 log。

---

## 6. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 1 | 1 | clean 实验只包含 1 个测试文件 |
| summary 中唯一 cor_type 数 | 1 | 1 | 只测试 clean |
| summary 中唯一 log_path 数 | 1 | 1 | 每次运行对应 1 个 log |
| logs 目录 .log 文件数 | 1 | 1 | 没有旧日志或重复日志残留 |
| status=done 数 | 1 | 1 | 脚本执行成功 |
| 当前复现 accuracy | 43.82 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，23_3 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 7. 当前结果表

| 实验编号 | Dataset | Variant | File | Method | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 23_3_openshape_scanobjnn_clean_hardest_zs_global_local | sonn_c | hardest | data/sonn_c/hardest/clean.h5 | Zero-shot + Global Cache + Local Cache | 43.82 | done |

该结果表示：OpenShape 在 ScanObjNN clean hardest split 上使用完整 Point-Cache 后，准确率为 43.82。

---

## 8. 与原文结果对比

原文 Point-Cache 的 ScanObjectNN hardest / S-PB T50-RS-C 相关表中，OpenShape / O-Shape 在 Original Data，也就是 ScanObjNN clean hardest 上的 +Hierarchical Cache 结果为 43.72。

当前复现结果为 43.82。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache OpenShape / ScanObjNN clean hardest / +Hierarchical Cache | 43.72 | 43.82 | +0.10 | 0.10 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与原文差异 | +0.10 |
| 绝对差异 | 0.10 |

分析：

当前复现结果 43.82 比原文 43.72 高 +0.10，差异非常小。可以认为 23_3 的 OpenShape × ScanObjNN clean hardest +Hierarchical Cache 结果与原文高度对齐。

因此，23_3 不只是脚本执行成功，而且最终完整 Point-Cache 数值与原文基本一致。

---

## 9. 与前序实验 23_1 和 23_2 的对比

23_1 是无缓存 Zero-shot，23_2 是 Zero-shot + Global Cache，23_3 是完整 Point-Cache。

| 实验编号 | 方法 | 当前复现值 |
|---|---|---:|
| 23_1_openshape_scanobjnn_clean_hardest_zs | Zero-shot | 41.88 |
| 23_2_openshape_scanobjnn_clean_hardest_zs_global | Zero-shot + Global Cache | 41.95 |
| 23_3_openshape_scanobjnn_clean_hardest_zs_global_local | Zero-shot + Global Cache + Local Cache | 43.82 |

当前复现变化：

| 比较 | 当前复现变化 | 含义 |
|---|---:|---|
| 23_2 - 23_1 | +0.07 | Global Cache 相比 Zero-shot 的变化 |
| 23_3 - 23_2 | +1.87 | Local Cache 在 Global Cache 基础上的额外变化 |
| 23_3 - 23_1 | +1.94 | 完整 Point-Cache 相比 Zero-shot 的总体变化 |

原文对应变化：

| 比较 | 原文变化 |
|---|---:|
| Global Cache - Zero-shot | 42.16 - 41.12 = +1.04 |
| Hierarchical Cache - Global Cache | 43.72 - 42.16 = +1.56 |
| Hierarchical Cache - Zero-shot | 43.72 - 41.12 = +2.60 |

变化对齐情况：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 相对 Zero-shot 的变化 | +1.04 | +0.07 | -0.97 |
| Local Cache 额外变化 | +1.56 | +1.87 | +0.31 |
| 完整 Point-Cache 相对 Zero-shot 的变化 | +2.60 | +1.94 | -0.66 |

分析：

当前复现中，完整 Point-Cache 将准确率从 23_1 的 41.88 提升到 23_3 的 43.82，总提升 +1.94。虽然这个总提升低于原文 +2.60，但最终绝对值 43.82 与原文 43.72 几乎一致。

当前 23 组最重要的现象是贡献来源发生了偏移：

| 贡献来源 | 当前现象 |
|---|---|
| Global Cache | 单独增益很弱，只提升 +0.07 |
| Local Cache | 额外提升明显，贡献 +1.87 |
| 完整 Point-Cache | 最终结果高度对齐原文 |

因此，23 组不能简单写成 “Global Cache 是主要提升来源”。更准确的写法是：在 OpenShape × ScanObjNN clean hardest 上，当前复现实验中 Local Cache 是主要提升来源，完整 Point-Cache 的最终结果与原文高度对齐。

---

## 10. 为什么 23 组中 Local Cache 更重要

ScanObjNN clean hardest 是真实扫描数据，虽然文件名是 clean，但它不同于 ModelNet clean。ScanObjNN hardest 中常见问题包括：

| 真实扫描挑战 | 对点云识别的影响 |
|---|---|
| 局部遮挡 | 物体局部结构不完整 |
| 背景残留 | 全局特征可能混入非目标区域 |
| 扫描噪声 | 点分布不规则 |
| 形状缺失 | 全局轮廓不稳定 |
| 真实物体姿态变化 | 同类样本内部差异更大 |

Global Cache 主要使用整 object 的全局特征进行缓存检索。当真实扫描数据中全局形状受到遮挡、背景、缺失和噪声影响时，仅依赖全局特征可能不足以稳定补偿 zero-shot logits。

Local Cache 则进一步利用局部 patch / 局部聚类特征。对于真实扫描 hardest split，局部结构可能更能捕捉可辨别部件或残留有效几何信息。因此，23_3 中 Local Cache 在 Global Cache 基础上带来 +1.87 的额外提升，是符合数据特点的。

这也解释了为什么 23 组和 22 组不同：

| 组别 | 数据设置 | 主要增益来源 |
|---|---|---|
| 22 组 | OpenShape × ModelNet-C all35 | Global Cache 是主要提升来源，Local Cache 额外贡献接近 0 |
| 23 组 | OpenShape × ScanObjNN clean hardest | Local Cache 在 Global Cache 基础上贡献明显 |

这个差异说明 Local Cache 的价值具有数据依赖性：在 synthetic corruption 上不一定稳定提升，但在真实扫描 hardest split 上可能更有价值。

---

## 11. 与 21_3 ModelNet clean 的关系

21_3 是 OpenShape 在 ModelNet clean 上的完整 Point-Cache 结果；23_3 是 OpenShape 在 ScanObjNN clean hardest 上的完整 Point-Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy |
|---|---|---|---:|
| 21_3_openshape_modelnet_clean_zs_global_local | ModelNet clean | ZS + Global + Local | 84.00 |
| 23_3_openshape_scanobjnn_clean_hardest_zs_global_local | ScanObjNN clean hardest | ZS + Global + Local | 43.82 |

从 ModelNet clean 到 ScanObjNN clean hardest 的下降为：

| 比较 | 变化 |
|---|---:|
| 23_3 - 21_3 | 43.82 - 84.00 = -40.18 |

分析：

即使使用完整 Point-Cache，OpenShape 在 ScanObjNN clean hardest 上仍然远低于 ModelNet clean，下降 -40.18。这说明真实扫描 hardest split 的域偏移非常强，完整 Point-Cache 只能部分补偿，但不能完全消除 clean synthetic 与 real scan 之间的差距。

不过，相比 Zero-shot，完整 Point-Cache 缩小了一部分差距：

| 方法 | ModelNet clean | ScanObjNN clean hardest | 下降 |
|---|---:|---:|---:|
| Zero-shot | 84.72 | 41.88 | -42.84 |
| ZS + Global | 84.48 | 41.95 | -42.53 |
| ZS + Global + Local | 84.00 | 43.82 | -40.18 |

完整 Point-Cache 的下降幅度最小，说明它对真实扫描 hardest split 有一定补偿作用。

---

## 12. 与 22_3 ModelNet-C 的关系

22_3 是 OpenShape 在 ModelNet-C all35 上的完整 Point-Cache 结果；23_3 是 OpenShape 在 ScanObjNN clean hardest 上的完整 Point-Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy / Avg |
|---|---|---|---:|
| 22_3_openshape_modelnetc_corruptions_all35_zs_global_local | ModelNet-C all35 | ZS + Global + Local | 75.14 all35 / 76.33 S2 |
| 23_3_openshape_scanobjnn_clean_hardest_zs_global_local | ScanObjNN clean hardest | ZS + Global + Local | 43.82 |

对比：

| 比较 | 变化 |
|---|---:|
| 23_3 - 22_3 all35 | 43.82 - 75.14 = -31.32 |
| 23_3 - 22_3 S2 Avg | 43.82 - 76.33 = -32.51 |

分析：

即使与 corrupted ModelNet-C 相比，ScanObjNN clean hardest 仍然明显更难。OpenShape 完整 Point-Cache 在 ModelNet-C all35 上为 75.14，但在 ScanObjNN clean hardest 上只有 43.82。

这说明真实扫描域偏移比 synthetic corruption 更复杂。ModelNet-C 的 corruption 是在标准 CAD-like 数据上施加的受控扰动，而 ScanObjNN hardest 本身具有真实扫描中的背景、遮挡、缺失和采样不规则问题。

---

## 13. 与 ULIP / ULIP-2 的 ScanObjNN clean hardest 关系

23_3 可以与前面 ULIP、ULIP-2 的 ScanObjNN clean hardest 完整 Point-Cache 结果进行横向比较。

| Backbone | 实验编号 | ScanObjNN clean hardest ZS + Global + Local |
|---|---|---:|
| ULIP | 03_3_ulip_scanobjnn_clean_hardest_zs_global_local | 32.48 |
| ULIP-2 | 13_3_ulip2_scanobjnn_clean_hardest_zs_global_local | 42.44 |
| OpenShape | 23_3_openshape_scanobjnn_clean_hardest_zs_global_local | 43.82 |

Backbone 提升：

| 比较 | 提升 |
|---|---:|
| OpenShape - ULIP | 43.82 - 32.48 = +11.34 |
| OpenShape - ULIP-2 | 43.82 - 42.44 = +1.38 |

分析：

完整 Point-Cache 下，OpenShape 仍然高于 ULIP 和 ULIP-2。相比 ULIP，OpenShape 高 +11.34；相比 ULIP-2，高 +1.38。

不过，OpenShape 相对 ULIP-2 的优势并不大。23_1 Zero-shot 时，OpenShape 比 ULIP-2 高 +7.81；23_3 完整 Point-Cache 时，OpenShape 只高 +1.38。这说明 ULIP-2 在 ScanObjNN clean hardest 上从 cache 中获得了更大相对收益，而 OpenShape 的优势主要来自更强的 zero-shot 起点。

---

## 14. 与后续实验的关系

23_3 是 23 组最后一个子实验，因此它本身可以作为后续 23 组 summary 文档的输入。

本文件只记录 23_3 自身及其与前序子实验的关系。完整 23 组总结应在单独 summary 文档中完成，包括：

| 后续 summary 应包含的内容 | 说明 |
|---|---|
| 23_1 / 23_2 / 23_3 总表 | 横向比较三种方法 |
| 与原文三种方法对齐 | 分别比较 41.12、42.16、43.72 |
| Global Cache 影响 | 分析 23_2 - 23_1 |
| Local Cache 影响 | 分析 23_3 - 23_2 |
| 与 21 / 22 组关系 | 分析 ModelNet clean、ModelNet-C、ScanObjNN clean hardest 的差异 |
| 与 ULIP / ULIP-2 关系 | 分析 backbone 差异 |
| 对 24 组 ScanObjNN-C hardest 的意义 | 作为 clean hardest 参考 |

23_3 的结果也为后续 24 组 OpenShape × ScanObjNN-C hardest all35 提供 clean reference。后续需要观察 corruption 进一步叠加到 ScanObjNN hardest 后，完整 Point-Cache 是否仍然保持提升。

---

## 15. 结果含义分析

23_3 的结果说明：完整 Point-Cache 在 OpenShape × ScanObjNN clean hardest 上有效，并且最终数值高度对齐原文。

| 观察 | 含义 |
|---|---|
| 23_3 = 43.82 | 当前完整 Point-Cache 最终结果 |
| 原文为 43.72 | 当前只高 +0.10，复现高度对齐 |
| 相比 23_1 提升 +1.94 | 完整 Point-Cache 有明确正增益 |
| 相比 23_2 提升 +1.87 | Local Cache 是当前主要提升来源 |
| 23_2 相比 23_1 只提升 +0.07 | Global Cache 单独作用很弱 |
| 趋势为 ZS < Global < Global + Local | 与原文趋势一致 |

因此，23_3 是 23 组三个子实验中最关键的结果。它证明了完整 Point-Cache 的最终效果是可靠的，也说明 Local Cache 在真实扫描 clean hardest 上具有明显价值。

---

## 16. 对后续 MCM-PC 的启发

当前 23_3 对后续 MCM-PC 方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| Local Cache 在 23 组中贡献明显 | 真实扫描数据上局部证据很重要 |
| Global Cache 单独增益很弱 | 仅依赖全局特征可能不足 |
| 完整结果高度对齐原文 | 当前复现实验可靠 |
| ScanObjNN hardest 远难于 ModelNet / ModelNet-C | 后续必须重视真实扫描域偏移 |
| OpenShape 相对 ULIP-2 的完整方法优势很小 | 强 backbone 上方法增益空间更有限 |
| 23 组与 22 组 Local Cache 现象不同 | Local Cache 是否有效具有数据依赖性 |

这对 MCM-PC 很重要：后续方法不应把 Global Cache 或 Local Cache 的作用简单固定下来，而应根据样本可靠性、全局-局部一致性、伪标签可信度和域偏移类型动态决定缓存贡献。

对于真实扫描 hardest split，局部结构信号可能比 synthetic corruption 场景更重要。后续 MCM-PC 的改进可以围绕如何筛选可靠局部证据、如何抑制错误局部伪标签、如何平衡文本原型、全局缓存和局部缓存展开。

---

## 17. 阶段性结论

本实验完成了 OpenShape × ScanObjNN clean hardest 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 23_3 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 43.82。
3. 原文 OpenShape / O-Shape 在 ScanObjNN clean hardest 上的 +Hierarchical Cache 结果为 43.72。
4. 当前复现结果比原文高 +0.10，差异很小，可以认为高度对齐。
5. 相比 23_1 Zero-shot 的 41.88，23_3 提升到 43.82，总提升 +1.94。
6. 相比 23_2 Global Cache 的 41.95，23_3 提升 +1.87。
7. 当前 23 组中，Global Cache 单独提升只有 +0.07，Local Cache 是主要提升来源。
8. 当前趋势为 Zero-shot < Global Cache < Global + Local Cache，与原文趋势一致。
9. 与 21_3 ModelNet clean 相比，23_3 低 -40.18，说明真实扫描 hardest split 仍然极难。
10. 与 22_3 ModelNet-C all35 相比，23_3 低 -31.32，说明真实扫描域偏移比 synthetic corruption 更具挑战。
11. 完整 Point-Cache 下，OpenShape 比 ULIP 高 +11.34，比 ULIP-2 高 +1.38。
12. 23_3 结果有效，不需要重跑。
13. 该实验可作为 23 组 summary 文档和后续 24 组 ScanObjNN-C hardest all35 的 clean 参考。

---

## 18. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh 1

---

## 19. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local/summary.csv
