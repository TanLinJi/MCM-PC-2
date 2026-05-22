# 31_3_uni3d_modelnet_clean_zs_global_local

## 1. 实验目的

本实验用于复现 Uni3D 在 ModelNet clean 上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 31_3_uni3d_modelnet_clean_zs_global_local |
| Backbone | Uni3D |
| Dataset | ModelNet clean |
| Dataset 参数 | modelnet_c |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 31_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证完整 Point-Cache 对 Uni3D 在 ModelNet clean 上的影响。

本文件只记录 31_3 本身，并与前序子实验 31_1 和 31_2 进行对比。完整 31 组的三方法总表、checkpoint 特殊情况、Global / Local 贡献分解和整体阶段性总结应放在 31 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | Uni3D |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 实际数据文件 | data/modelnet_c/clean.h5 |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 方法脚本 | Point-Cache/scripts/baseline/31_3_uni3d_modelnet_clean_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/31_run_uni3d_modelnet_clean_common.sh |
| cache_type | hierarchical |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |
| 输入点数 | 1024 |
| Uni3D point encoder checkpoint | weights/uni3d/modelnet40/model.pt |
| Uni3D text encoder checkpoint | weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin |
| pc_model | eva_giant_patch14_560 |
| clip_model | EVA02-E-14-plus |
| pc_feat_dim | 1408 |
| num_group | 512 |
| group_size | 64 |
| pc_encoder_dim | 512 |
| embed_dim | 1024 |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 1 |

本实验使用 `modelnet_c` 作为 dataset 参数，并指定：

| 参数 | 值 |
|---|---|
| cor_type | clean |
| npoints | 1024 |
| sonn_variant | obj_only |

实际读取文件为：

data/modelnet_c/clean.h5

---

## 3. 当前实现方式

31 组是 clean 单文件实验，因此不需要 all35 优化 runner。

| 项目 | 说明 |
|---|---|
| 是否需要 35 个 cor_type 循环 | 否 |
| 是否需要 Python 内部循环 | 否 |
| 是否需要 all35 runner | 否 |
| 是否每个子实验只生成 1 行 summary | 是 |
| 是否每个子实验只生成 1 个 log | 是 |

31_3 的执行路径为：

| 层级 | 文件 |
|---|---|
| 单方法脚本 | Point-Cache/scripts/baseline/31_3_uni3d_modelnet_clean_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/31_run_uni3d_modelnet_clean_common.sh |
| 原始推理 runner | Point-Cache/runners/model_with_hierarchical_caches.py |

该结构与 01、11、21、31_1、31_2 等 ModelNet clean 单文件实验保持一致。

---

## 4. 方法说明

31_3 在 Zero-shot logits 的基础上同时加入 Global Cache logits 和 Local Cache logits。

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

31_3 与前两个子实验的关系如下：

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 31_1 | 是 | 否 | 否 |
| 31_2 | 是 | 是 | 否 |
| 31_3 | 是 | 是 | 是 |

因此，31_3 可以用于评估完整 Point-Cache 在 Uni3D × ModelNet clean 上的最终表现，并判断 Local Cache 是否在 Global Cache 基础上带来额外贡献。

---

## 5. Uni3D checkpoint 说明

本实验使用的 Uni3D point encoder checkpoint 为：

weights/uni3d/modelnet40/model.pt

这是本次 31 组复现中非常关键的设置。

此前曾使用服务器原有 checkpoint：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

旧 checkpoint 下 31 组 ModelNet clean 结果整体偏低。尤其 31_3 的结果为 81.60，明显低于原文 83.87。切换到 Uni3D-g 的 ModelNet40 checkpoint 后，31_3 提升到 83.71，与原文 83.87 高度接近。

因此，31_3 正式记录以 `weights/uni3d/modelnet40/model.pt` 为准。

该 checkpoint 的下载脚本已记录在：

Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh

完整的 checkpoint 特殊情况会在 31 组 summary 文档中统一说明。

---

## 6. 输出结构

输出目录：

Point-Cache/results/baseline/31_3_uni3d_modelnet_clean_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

31_3_uni3d_modelnet_clean_zs_global_local_clean_YYYYMMDD_HHMMSS.log

因为本实验只测试 `clean.h5`，所以期望只有 1 行 summary 记录和 1 个对应 log。

---

## 7. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 1 | 1 | clean 实验只包含 1 个测试文件 |
| summary 中唯一 cor_type 数 | 1 | 1 | 只测试 clean |
| summary 中唯一 log_path 数 | 1 | 1 | 每次运行对应 1 个 log |
| logs 目录 .log 文件数 | 1 | 1 | 没有旧日志或重复日志残留 |
| status=done 数 | 1 | 1 | 脚本执行成功 |
| 当前复现 accuracy | 83.71 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，31_3 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 8. 当前结果表

| 实验编号 | Dataset | File | Method | Checkpoint | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 31_3_uni3d_modelnet_clean_zs_global_local | modelnet_c | data/modelnet_c/clean.h5 | Zero-shot + Global Cache + Local Cache | weights/uni3d/modelnet40/model.pt | 83.71 | done |

该结果表示：Uni3D 在 ModelNet clean 上使用完整 Point-Cache 后，准确率为 83.71。

---

## 9. 与原文结果对比

原文 Point-Cache Table 1 中，Uni3D 在 ModelNet clean 上的 +Hierarchical Cache 结果为 83.87。

当前复现结果为 83.71。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Uni3D / ModelNet clean / +Hierarchical Cache | 83.87 | 83.71 | -0.16 | 0.16 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与原文差异 | -0.16 |
| 绝对差异 | 0.16 |

分析：

当前复现结果 83.71 比原文 83.87 低 -0.16，差异很小。因此，31_3 不只是脚本执行成功，而且数值与原文高度接近。

该结果说明，使用 `weights/uni3d/modelnet40/model.pt` 后，Uni3D × ModelNet clean 的完整 Point-Cache baseline 复现成功。

---

## 10. 与前序实验 31_1 和 31_2 的对比

31_1 是无缓存 Zero-shot，31_2 是 Zero-shot + Global Cache，31_3 是完整 Point-Cache。

| 实验编号 | 方法 | 当前复现值 |
|---|---|---:|
| 31_1_uni3d_modelnet_clean_zs | Zero-shot | 81.85 |
| 31_2_uni3d_modelnet_clean_zs_global | Zero-shot + Global Cache | 83.23 |
| 31_3_uni3d_modelnet_clean_zs_global_local | Zero-shot + Global Cache + Local Cache | 83.71 |

当前复现变化：

| 比较 | 当前复现变化 | 含义 |
|---|---:|---|
| 31_2 - 31_1 | +1.38 | Global Cache 相比 Zero-shot 的变化 |
| 31_3 - 31_2 | +0.48 | Local Cache 在 Global Cache 基础上的额外变化 |
| 31_3 - 31_1 | +1.86 | 完整 Point-Cache 相比 Zero-shot 的总体变化 |

原文对应变化：

| 比较 | 原文变化 |
|---|---:|
| Global Cache - Zero-shot | 83.14 - 81.81 = +1.33 |
| Hierarchical Cache - Global Cache | 83.87 - 83.14 = +0.73 |
| Hierarchical Cache - Zero-shot | 83.87 - 81.81 = +2.06 |

变化对齐情况：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 相对 Zero-shot 的变化 | +1.33 | +1.38 | +0.05 |
| Local Cache 额外变化 | +0.73 | +0.48 | -0.25 |
| 完整 Point-Cache 相对 Zero-shot 的变化 | +2.06 | +1.86 | -0.20 |

分析：

当前复现中，完整 Point-Cache 将准确率从 31_1 的 81.85 提升到 31_3 的 83.71，总提升 +1.86。原文总提升为 +2.06，当前略低 -0.20。

Global Cache 的增益与原文几乎完全一致：原文 +1.33，当前 +1.38。Local Cache 的额外增益略弱：原文 +0.73，当前 +0.48，但方向正确，并且最终完整结果与原文只差 -0.16。

因此，31_3 可以认为复现有效。

---

## 11. 与旧 checkpoint 结果对比

本实验曾使用服务器原有 checkpoint 运行过一次：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

旧 checkpoint 下 31_3 的结果为 81.60。切换到 ModelNet40 checkpoint 后，31_3 的结果为 83.71。

| Checkpoint | 31_3 ZS + Global + Local |
|---|---:|
| weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt | 81.60 |
| weights/uni3d/modelnet40/model.pt | 83.71 |
| 原文 Point-Cache Table 1 | 83.87 |

变化：

| 比较 | 变化 |
|---|---:|
| modelnet40 checkpoint - old pc_encoder checkpoint | +2.11 |
| modelnet40 checkpoint - 原文 | -0.16 |
| old pc_encoder checkpoint - 原文 | -2.27 |

分析：

旧 checkpoint 下的 81.60 明显低于原文 83.87，差异为 -2.27。切换到 `weights/uni3d/modelnet40/model.pt` 后，结果提升到 83.71，与原文高度接近。

因此，31 组最初偏低的主要原因可以归结为：Uni3D point encoder checkpoint 与原文 ModelNet40 设置不对应。后续 31 / 32 组应统一使用 `weights/uni3d/modelnet40/model.pt`。

---

## 12. 与 01 / 11 / 21 组 ModelNet clean 的关系

31_3 可以与前面三个 backbone 的 ModelNet clean 完整 Point-Cache 结果进行横向比较。

| Backbone | 实验编号 | ModelNet clean ZS + Global + Local |
|---|---|---:|
| ULIP | 01_3_ulip_modelnet_clean_zs_global_local | 约 64 |
| ULIP-2 | 11_3_ulip2_modelnet_clean_zs_global_local | 74.35 |
| OpenShape | 21_3_openshape_modelnet_clean_zs_global_local | 84.00 |
| Uni3D | 31_3_uni3d_modelnet_clean_zs_global_local | 83.71 |

分析：

完整 Point-Cache 下，Uni3D 在 ModelNet clean 上明显强于 ULIP 和 ULIP-2，略低于 OpenShape。Uni3D 与 OpenShape 非常接近，分别为 83.71 和 84.00。

这说明 Uni3D 是一个很强的 backbone，并且在切换到正确 ModelNet40 checkpoint 后，已经达到与 OpenShape 接近的 clean performance。

---

## 13. 与后续实验的关系

31_3 是 31 组最后一个子实验，因此它本身可以作为后续 31 组 summary 文档的输入。

本文件只记录 31_3 自身及其与前序子实验的关系。完整 31 组总结应在单独 summary 文档中完成，包括：

| 后续 summary 应包含的内容 | 说明 |
|---|---|
| 31_1 / 31_2 / 31_3 总表 | 横向比较三种方法 |
| 与原文三种方法对齐 | 分别比较 81.81、83.14、83.87 |
| checkpoint 特殊情况 | 记录旧 checkpoint 偏低、下载 ModelNet40 checkpoint 后对齐 |
| Global Cache 影响 | 分析 31_2 - 31_1 |
| Local Cache 影响 | 分析 31_3 - 31_2 |
| 与 01 / 11 / 21 组关系 | 分析不同 backbone 的 ModelNet clean 表现 |
| 对 32 组 ModelNet-C 的意义 | 作为 clean reference |

31_3 的结果也为后续 32 组 Uni3D × ModelNet-C all35 提供 clean reference。后续 32 组应继续使用 `weights/uni3d/modelnet40/model.pt`。

---

## 14. 结果含义分析

31_3 的结果说明：完整 Point-Cache 在 Uni3D × ModelNet clean 上有效，并且最终数值高度接近原文。

| 观察 | 含义 |
|---|---|
| 31_3 = 83.71 | 当前完整 Point-Cache 最终结果 |
| 原文为 83.87 | 当前只低 -0.16，复现高度接近 |
| 相比 31_1 提升 +1.86 | 完整 Point-Cache 有明确正增益 |
| 相比 31_2 提升 +0.48 | Local Cache 有额外正贡献 |
| 31_2 相比 31_1 提升 +1.38 | Global Cache 是主要提升来源 |
| 趋势为 ZS < Global < Global + Local | 与原文趋势一致 |

因此，31_3 是 31 组三个子实验中最关键的最终结果。它证明了完整 Point-Cache 的最终效果是可靠的，也说明 Uni3D × ModelNet clean 复现现在已经进入可归档状态。

---

## 15. 对后续 MCM-PC 的启发

当前 31_3 对后续 MCM-PC 方法设计和实验管理有以下启发：

| 观察 | 启发 |
|---|---|
| 完整 Point-Cache 提升 +1.86 | cache 机制在 Uni3D clean 上有效 |
| Global Cache 提供 +1.38 | 全局缓存是主要提升来源 |
| Local Cache 提供 +0.48 | 局部缓存有额外贡献，但弱于 Global |
| 使用 modelnet40 checkpoint 后对齐原文 | checkpoint 与数据集匹配非常关键 |
| 旧 pc_encoder checkpoint 下结果明显偏低 | 不应作为正式 baseline checkpoint |
| 31 / 32 组应使用 modelnet40 checkpoint | ModelNet 系列统一设置 |
| 下载脚本已经归档 | 便于后续复现实验环境 |

这次实验说明：对于 Uni3D，checkpoint 选择本身就是实验设置的一部分，必须在文档中明确记录，不能只写 backbone 名称。

---

## 16. 阶段性结论

本实验完成了 Uni3D × ModelNet clean 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 31_3 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前使用 checkpoint 为 weights/uni3d/modelnet40/model.pt。
3. 当前复现准确率为 83.71。
4. 原文 Point-Cache Table 1 中 Uni3D / ModelNet clean / +Hierarchical Cache 结果为 83.87。
5. 当前复现结果比原文低 -0.16，差异很小，可以认为高度接近。
6. 相比 31_1 Zero-shot 的 81.85，31_3 提升到 83.71，总提升 +1.86。
7. 相比 31_2 Global Cache 的 83.23，31_3 额外提升 +0.48。
8. 当前趋势为 Zero-shot < Global Cache < Global + Local Cache，与原文趋势一致。
9. 旧 checkpoint weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt 下结果为 81.60，比原文低 -2.27。
10. 切换到 ModelNet40 checkpoint 后，结果从 81.60 提升到 83.71，说明此前偏低主要来自 checkpoint 不匹配。
11. 31_3 结果有效，不需要重跑。
12. 该实验可作为 31 组 summary 文档和后续 32 组 ModelNet-C all35 的 clean reference。
13. 31 / 32 组后续应统一使用 weights/uni3d/modelnet40/model.pt。

---

## 17. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/31_3_uni3d_modelnet_clean_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/31_3_uni3d_modelnet_clean_zs_global_local_single_gpu.sh 1

---

## 18. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/31_3_uni3d_modelnet_clean_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/31_3_uni3d_modelnet_clean_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/31_3_uni3d_modelnet_clean_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/31_3_uni3d_modelnet_clean_zs_global_local/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/31_3_uni3d_modelnet_clean_zs_global_local/summary.csv
