# 23_2_openshape_scanobjnn_clean_hardest_zs_global

## 1. 实验目的

本实验用于复现 OpenShape 在 ScanObjNN clean hardest split 上使用 Zero-shot + Global Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 23_2_openshape_scanobjnn_clean_hardest_zs_global |
| Backbone | OpenShape |
| Dataset | ScanObjNN clean hardest |
| Dataset 参数 | sonn_c |
| SONN variant | hardest |
| Method | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |

本实验是在 23_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存是否能够提升 OpenShape 在 ScanObjNN clean hardest 上的表现。

需要特别注意：当前 23_2 相比 23_1 的提升非常小，只有 +0.07。也就是说，在当前复现实验中，Global Cache 单独并不是 23 组的主要提升来源。后续 23_3 才能判断 Local Cache 是否带来明显额外贡献。

本文件只记录 23_2 本身，并与前序子实验 23_1 进行对比。完整 23 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 23 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | OpenShape |
| 数据集参数 | sonn_c |
| 数据目录 | data/sonn_c |
| 数据集变体 | hardest |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 原始核心 runner | runners/model_with_global_cache.py |
| 方法脚本 | Point-Cache/scripts/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/23_run_openshape_scanobjnn_clean_hardest_common.sh |
| cache_type | global |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |
| 输入点数 | 1024 |
| Global Cache shot_capacity | 3 |
| Global Cache alpha | 4.0 |
| Global Cache beta | 3.0 |
| OpenShape version | vitg14 |
| OpenShape 权重 | weights/openshape/openshape-pointbert-vitg14-rgb/model.pt |
| OpenCLIP 权重 | weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 1 |

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

23_2 的执行路径为：

| 层级 | 文件 |
|---|---|
| 单方法脚本 | Point-Cache/scripts/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/23_run_openshape_scanobjnn_clean_hardest_common.sh |
| 原始推理 runner | Point-Cache/runners/model_with_global_cache.py |

该结构与 03、13、21、23_1 等 clean 单文件实验保持一致。

---

## 4. 方法说明

23_2 在 Zero-shot logits 的基础上加入 Global Cache logits。

| 组成部分 | 是否使用 |
|---|---:|
| Text prototype / 文本原型 | 是 |
| Point cloud global feature | 是 |
| Zero-shot logits | 是 |
| Global Cache logits | 是 |
| Local Cache logits | 否 |
| Hierarchical Cache | 否 |

Global Cache 的基本作用是：在测试过程中动态缓存高置信度样本的全局点云特征和伪标签，然后对后续样本进行全局特征检索，生成 cache logits，并与 zero-shot logits 融合。

23_2 与 23_1 的主要区别如下：

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 23_1 | 是 | 否 | 否 |
| 23_2 | 是 | 是 | 否 |

因此，23_2 可以用于单独评估 Global Cache 在 OpenShape × ScanObjNN clean hardest 上的影响。

---

## 5. 输出结构

输出目录：

Point-Cache/results/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

23_2_openshape_scanobjnn_clean_hardest_zs_global_clean_YYYYMMDD_HHMMSS.log

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
| 当前复现 accuracy | 41.95 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，23_2 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 7. 当前结果表

| 实验编号 | Dataset | Variant | File | Method | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 23_2_openshape_scanobjnn_clean_hardest_zs_global | sonn_c | hardest | data/sonn_c/hardest/clean.h5 | Zero-shot + Global Cache | 41.95 | done |

该结果表示：OpenShape 在 ScanObjNN clean hardest split 上加入 Global Cache 后，准确率为 41.95。

---

## 8. 与原文结果对比

原文 Point-Cache 的 ScanObjectNN hardest / S-PB T50-RS-C 相关表中，OpenShape / O-Shape 在 Original Data，也就是 ScanObjNN clean hardest 上的 +Global Cache 结果为 42.16。

当前复现结果为 41.95。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache OpenShape / ScanObjNN clean hardest / +Global Cache | 42.16 | 41.95 | -0.21 | 0.21 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与原文差异 | -0.21 |
| 绝对差异 | 0.21 |

分析：

当前复现结果 41.95 比原文 42.16 低 -0.21，差异很小。可以认为 23_2 的 OpenShape × ScanObjNN clean hardest +Global Cache 结果与原文高度接近。

因此，23_2 不只是脚本执行成功，而且数值也基本对齐原文。

---

## 9. 与前序实验 23_1 的对比

23_1 是本实验的直接前序子实验，方法为 Zero-shot，不使用缓存。

| 实验编号 | 方法 | 当前复现值 |
|---|---|---:|
| 23_1_openshape_scanobjnn_clean_hardest_zs | Zero-shot | 41.88 |
| 23_2_openshape_scanobjnn_clean_hardest_zs_global | Zero-shot + Global Cache | 41.95 |

Global Cache 带来的当前复现变化为：

| 比较 | 当前复现变化 |
|---|---:|
| 23_2 - 23_1 | +0.07 |

原文中对应变化为：

| 比较 | 原文变化 |
|---|---:|
| Global Cache - Zero-shot | 42.16 - 41.12 = +1.04 |

对比：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 相对 Zero-shot 的变化 | +1.04 | +0.07 | -0.97 |

分析：

当前复现中，Global Cache 将准确率从 23_1 的 41.88 提升到 23_2 的 41.95，只提升 +0.07。这个提升方向与原文一致，但提升幅度明显弱于原文的 +1.04。

需要分开看两个角度：

| 分析角度 | 结论 |
|---|---|
| 绝对数值对齐 | 23_2 = 41.95，与原文 42.16 差异 -0.21，较好 |
| 相对 23_1 的变化 | 当前为 +0.07，明显弱于原文 +1.04 |
| 是否异常 | 不建议判为异常，因为 23_1 当前复现值比原文高 +0.76，抬高了当前 Zero-shot 基线 |
| 后续重点 | 需要看 23_3 是否能复现完整 Point-Cache 的最终提升 |

当前 Global Cache 增益偏弱的主要原因之一是：23_1 当前复现值 41.88 比原文 Zero-shot 41.12 高 +0.76，而 23_2 当前复现值 41.95 与原文 42.16 接近。因此，以当前 23_1 为基线计算时，Global Cache 的相对提升被压缩。

---

## 10. 与 21_2 ModelNet clean 的关系

21_2 是 OpenShape 在 ModelNet clean 上的 Zero-shot + Global Cache 结果；23_2 是 OpenShape 在 ScanObjNN clean hardest 上的 Zero-shot + Global Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy |
|---|---|---|---:|
| 21_2_openshape_modelnet_clean_zs_global | ModelNet clean | ZS + Global Cache | 84.48 |
| 23_2_openshape_scanobjnn_clean_hardest_zs_global | ScanObjNN clean hardest | ZS + Global Cache | 41.95 |

从 ModelNet clean 到 ScanObjNN clean hardest 的下降为：

| 比较 | 变化 |
|---|---:|
| 23_2 - 21_2 | 41.95 - 84.48 = -42.53 |

分析：

OpenShape + Global Cache 在 ModelNet clean 上达到 84.48，但在 ScanObjNN clean hardest 上只有 41.95，下降 -42.53。

这说明，即使加入 Global Cache，真实扫描 hardest split 仍然是非常困难的数据设置。Global Cache 对 ModelNet-C corrupted setting 有明显帮助，但在 ScanObjNN clean hardest 中单独作用较弱。

---

## 11. 与 22_2 ModelNet-C 的关系

22_2 是 OpenShape 在 ModelNet-C all35 上的 Zero-shot + Global Cache 结果；23_2 是 OpenShape 在 ScanObjNN clean hardest 上的 Zero-shot + Global Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy / Avg |
|---|---|---|---:|
| 22_2_openshape_modelnetc_corruptions_all35_zs_global | ModelNet-C all35 | ZS + Global Cache | 75.14 all35 / 76.46 S2 |
| 23_2_openshape_scanobjnn_clean_hardest_zs_global | ScanObjNN clean hardest | ZS + Global Cache | 41.95 |

对比：

| 比较 | 变化 |
|---|---:|
| 23_2 - 22_2 all35 | 41.95 - 75.14 = -33.19 |
| 23_2 - 22_2 S2 Avg | 41.95 - 76.46 = -34.51 |

分析：

即使与 ModelNet-C corrupted setting 相比，ScanObjNN clean hardest 仍然明显更难。OpenShape + Global Cache 在 ModelNet-C all35 上为 75.14，但在 ScanObjNN clean hardest 上只有 41.95。

这说明 synthetic corruption 与真实扫描域偏移不是同一难度层级。ScanObjNN clean hardest 虽然是 clean 文件，但其真实扫描噪声、背景干扰、遮挡和不完整几何带来的挑战明显更强。

---

## 12. 与 ULIP / ULIP-2 的 ScanObjNN clean hardest 关系

23_2 可以与前面 ULIP、ULIP-2 的 ScanObjNN clean hardest +Global Cache 结果进行横向比较。

| Backbone | 实验编号 | ScanObjNN clean hardest +Global Cache |
|---|---|---:|
| ULIP | 03_2_ulip_scanobjnn_clean_hardest_zs_global | 32.20 |
| ULIP-2 | 13_2_ulip2_scanobjnn_clean_hardest_zs_global | 40.42 |
| OpenShape | 23_2_openshape_scanobjnn_clean_hardest_zs_global | 41.95 |

Backbone 提升：

| 比较 | 提升 |
|---|---:|
| OpenShape - ULIP | 41.95 - 32.20 = +9.75 |
| OpenShape - ULIP-2 | 41.95 - 40.42 = +1.53 |

分析：

加入 Global Cache 后，OpenShape 仍然高于 ULIP 和 ULIP-2。相比 ULIP，OpenShape 高 +9.75；相比 ULIP-2，高 +1.53。

但是，OpenShape 相对 ULIP-2 的优势在 +Global Cache 设置下明显缩小。23_1 Zero-shot 中 OpenShape 比 ULIP-2 高 +7.81，而 23_2 中只高 +1.53。这说明 ULIP-2 在 ScanObjNN clean hardest 上从 Global Cache 中获得了更明显的收益，而 OpenShape 当前 Global Cache 单独收益较弱。

---

## 13. 与后续子实验的关系

23_2 是 23_3 的直接前序实验。

| 后续实验 | 对比方式 |
|---|---|
| 23_3_openshape_scanobjnn_clean_hardest_zs_global_local | 与 23_2 比较，评估 Local Cache 在 Global Cache 基础上的额外影响 |

本文件不展开 23_3 的实际结果。23_3 的数值及 Local Cache 额外影响应记录在 23_3 子实验文档和 23 组 summary 文档中。

需要注意的是，当前 23_2 的 Global Cache 单独增益较弱，因此 23_3 的关键问题是：

| 问题 | 说明 |
|---|---|
| Local Cache 是否能补足 Global Cache 的弱增益？ | 比较 23_3 - 23_2 |
| 完整 Point-Cache 是否接近原文 43.72？ | 比较 23_3 与原文 |
| 当前趋势是否仍是 ZS < Global < Global + Local？ | 判断整体方法趋势 |
| 23 组主要增益来源是否转向 Local Cache？ | 由 23_3 决定 |

---

## 14. 结果含义分析

23_2 的结果说明：OpenShape 在 ScanObjNN clean hardest 上加入 Global Cache 后，绝对结果与原文接近，但相对 23_1 的提升很弱。

| 观察 | 含义 |
|---|---|
| 23_2 = 41.95 | OpenShape + Global Cache 在 ScanObjNN clean hardest 上的结果 |
| 比原文低 -0.21 | 绝对数值与原文高度接近 |
| 比 23_1 高 +0.07 | Global Cache 当前单独增益很弱 |
| 原文 Global 增益为 +1.04 | 当前相对增益明显低于原文 |
| 23_1 当前比原文高 +0.76 | 当前 Zero-shot 基线偏高压缩了 Global 增益 |
| 后续需要看 23_3 | 判断 Local Cache 是否提供主要提升 |

因此，23_2 应记录为一个“数值对齐但相对增益较弱”的实验，而不是简单写成“Global Cache 显著提升”。

---

## 15. 对后续 MCM-PC 的启发

当前 23_2 对后续 MCM-PC 方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| Global Cache 当前只提升 +0.07 | 在真实扫描 clean hardest 上，仅使用全局缓存可能不足 |
| 绝对值仍接近原文 | 不是脚本失败，而是相对基线变化较弱 |
| OpenShape + Global 与 ULIP-2 + Global 差距缩小 | 强 backbone 不一定在 cache 增益上占优 |
| ScanObjNN hardest 仍然困难 | 真实扫描场景需要更细粒度或更可靠的适应机制 |
| 后续 23_3 更关键 | Local Cache 可能在真实扫描数据上发挥更大作用 |

这对 MCM-PC 很重要：如果真实扫描数据中的主要问题来自局部结构缺失、遮挡、背景残留和细节变形，那么单纯的全局缓存可能不足以充分适应。后续方法需要考虑更可靠的局部证据、文本原型和全局缓存之间的协同机制。

---

## 16. 阶段性结论

本实验完成了 OpenShape × ScanObjNN clean hardest 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 23_2 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 41.95。
3. 原文 OpenShape / O-Shape 在 ScanObjNN clean hardest 上的 +Global Cache 结果为 42.16。
4. 当前复现结果比原文低 -0.21，差异很小，可以认为高度接近。
5. 相比 23_1 Zero-shot 的 41.88，23_2 只提升到 41.95，当前 Global Cache 增益为 +0.07。
6. 原文中 Global Cache 相比 Zero-shot 的增益为 +1.04，当前增益明显更弱。
7. 当前 Global Cache 增益偏弱的原因之一是 23_1 当前复现值比原文 Zero-shot 高 +0.76。
8. 与 21_2 ModelNet clean 相比，23_2 低 -42.53，说明真实扫描 hardest split 仍然非常困难。
9. 与 22_2 ModelNet-C all35 相比，23_2 低 -33.19，说明真实扫描域偏移比 synthetic corruption 更难。
10. 23_2 是 23_3 分析 Local Cache 额外贡献的直接对照。
11. 23_2 结果有效，不需要重跑，但文档中应明确记录 Global Cache 单独增益较弱。

---

## 17. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global_single_gpu.sh 1

---

## 18. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global/summary.csv
