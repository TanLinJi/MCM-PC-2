# 33_3_uni3d_scanobjnn_clean_hardest_zs_global_local

## 1. 实验目的

本实验用于复现 Uni3D 在 ScanObjNN clean hardest split 上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 33_3_uni3d_scanobjnn_clean_hardest_zs_global_local |
| Backbone | Uni3D |
| Dataset | ScanObjNN clean hardest |
| Dataset 参数 | sonn_c |
| SONN variant | hardest |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 33_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证完整 Point-Cache 对 Uni3D 在 ScanObjNN clean hardest split 上的影响。

本文件只记录 33_3 本身，并与前序子实验 33_1 和 33_2 进行对比。完整 33 组的三方法总表、checkpoint 特殊情况、Global / Local 贡献分解和整体阶段性总结应放在 33 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | Uni3D |
| 数据集参数 | sonn_c |
| 数据目录 | data/sonn_c |
| 数据集变体 | hardest |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 方法脚本 | Point-Cache/scripts/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/33_run_uni3d_scanobjnn_clean_hardest_common.sh |
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
| cor_type | clean |
| npoints | 1024 |
| sim2real_type | so_obj_only_9 |

实际读取文件为：

data/sonn_c/hardest/clean.h5

---

## 3. 当前实现方式

33 组是 clean 单文件实验，因此不需要 all35 优化 runner。

| 项目 | 说明 |
|---|---|
| 是否需要 35 个 cor_type 循环 | 否 |
| 是否需要 Python 内部循环 | 否 |
| 是否需要 all35 runner | 否 |
| 是否每个子实验只生成 1 行 summary | 是 |
| 是否每个子实验只生成 1 个 log | 是 |

33_3 的执行路径为：

| 层级 | 文件 |
|---|---|
| 单方法脚本 | Point-Cache/scripts/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/33_run_uni3d_scanobjnn_clean_hardest_common.sh |
| 原始推理 runner | Point-Cache/runners/model_with_hierarchical_caches.py |

该结构与 03、13、23、31、33_1、33_2 等 clean 单文件实验保持一致。

---

## 4. 方法说明

33_3 在 Zero-shot logits 的基础上同时加入 Global Cache logits 和 Local Cache logits。

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

33_3 与前两个子实验的关系如下：

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 33_1 | 是 | 否 | 否 |
| 33_2 | 是 | 是 | 否 |
| 33_3 | 是 | 是 | 是 |

因此，33_3 可以用于评估完整 Point-Cache 在 Uni3D × ScanObjNN clean hardest 上的最终表现，并判断 Local Cache 是否在 Global Cache 基础上带来额外贡献。

---

## 5. Uni3D checkpoint 说明

本实验使用的 Uni3D point encoder checkpoint 为：

weights/uni3d/scanobjnn/model.pt

这是本次 33 组复现中非常关键的设置。

此前曾使用服务器原有 checkpoint：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

旧 checkpoint 下 33_3 的结果为 45.49，明显低于原文 51.13。检查 33 组公共脚本后发现，当时脚本的 checkpoint 候选列表没有包含 `weights/uni3d/scanobjnn/model.pt`，因此自动优先选择了已存在的 `weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt`。

修正公共脚本后，将 ScanObjNN checkpoint 加入候选列表第一位：

weights/uni3d/scanobjnn/model.pt

重新运行后，33_3 的结果提升到 51.98，略高于原文 51.13，但整体趋势和增益结构合理。因此，33_3 正式记录以 `weights/uni3d/scanobjnn/model.pt` 为准。

该 checkpoint 的下载脚本已记录在：

Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh

完整的 checkpoint 特殊情况会在 33 组 summary 文档中统一说明。

---

## 6. 输出结构

输出目录：

Point-Cache/results/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

33_3_uni3d_scanobjnn_clean_hardest_zs_global_local_clean_YYYYMMDD_HHMMSS.log

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
| 当前复现 accuracy | 51.98 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |
| 实际 checkpoint | weights/uni3d/scanobjnn/model.pt | scanobjnn checkpoint | 已修正 |

从执行完整性看，33_3 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 8. 当前结果表

| 实验编号 | Dataset | File | Method | Checkpoint | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 33_3_uni3d_scanobjnn_clean_hardest_zs_global_local | sonn_c hardest | data/sonn_c/hardest/clean.h5 | Zero-shot + Global Cache + Local Cache | weights/uni3d/scanobjnn/model.pt | 51.98 | done |

该结果表示：Uni3D 在 ScanObjNN clean hardest 上使用完整 Point-Cache 后，准确率为 51.98。

---

## 9. 与原文结果对比

原文 Point-Cache Table 7 中，Uni3D 在 ScanObjNN hardest clean/original data 上的 +Hierarchical Cache 结果为 51.13。

当前复现结果为 51.98。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Uni3D / ScanObjNN clean hardest / +Hierarchical Cache | 51.13 | 51.98 | +0.85 | 0.85 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与原文差异 | +0.85 |
| 绝对差异 | 0.85 |

分析：

当前复现结果 51.98 比原文 51.13 高 +0.85。相较 33_1 和 33_2，33_3 的正偏差更明显，但仍处于可以接受的复现范围内。

同时，33_3 的方法趋势与原文一致，并且 33_1 / 33_2 已经分别对齐原文。因此，33_3 可以作为有效复现结果保留，但文档中应明确记录：当前完整 Point-Cache 结果略高于原文。

---

## 10. 与前序实验 33_1 和 33_2 的对比

33_1 是无缓存 Zero-shot，33_2 是 Zero-shot + Global Cache，33_3 是完整 Point-Cache。

| 实验编号 | 方法 | 当前复现值 |
|---|---|---:|
| 33_1_uni3d_scanobjnn_clean_hardest_zs | Zero-shot | 45.63 |
| 33_2_uni3d_scanobjnn_clean_hardest_zs_global | Zero-shot + Global Cache | 50.03 |
| 33_3_uni3d_scanobjnn_clean_hardest_zs_global_local | Zero-shot + Global Cache + Local Cache | 51.98 |

当前复现变化：

| 比较 | 当前复现变化 | 含义 |
|---|---:|---|
| 33_2 - 33_1 | +4.40 | Global Cache 相比 Zero-shot 的变化 |
| 33_3 - 33_2 | +1.95 | Local Cache 在 Global Cache 基础上的额外变化 |
| 33_3 - 33_1 | +6.35 | 完整 Point-Cache 相比 Zero-shot 的总体变化 |

原文对应变化：

| 比较 | 原文变化 |
|---|---:|
| Global Cache - Zero-shot | 50.28 - 46.04 = +4.24 |
| Hierarchical Cache - Global Cache | 51.13 - 50.28 = +0.85 |
| Hierarchical Cache - Zero-shot | 51.13 - 46.04 = +5.09 |

变化对齐情况：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 相对 Zero-shot 的变化 | +4.24 | +4.40 | +0.16 |
| Local Cache 额外变化 | +0.85 | +1.95 | +1.10 |
| 完整 Point-Cache 相对 Zero-shot 的变化 | +5.09 | +6.35 | +1.26 |

分析：

当前复现中，完整 Point-Cache 将准确率从 33_1 的 45.63 提升到 33_3 的 51.98，总提升 +6.35。原文总提升为 +5.09，当前高 +1.26。

Global Cache 的增益与原文高度一致：原文 +4.24，当前 +4.40。Local Cache 的额外增益当前更强：原文 +0.85，当前 +1.95。这也是 33_3 比原文高 +0.85 的主要原因。

因此，33_3 可以认为复现有效，但需要在 summary 文档中记录：Local Cache 额外增益高于原文，导致完整 Point-Cache 最终值略高于原文。

---

## 11. 与旧 checkpoint 结果对比

本实验曾使用服务器原有 checkpoint 运行过一次：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

旧 checkpoint 下 33_3 的结果为 45.49。切换到 ScanObjNN checkpoint 后，33_3 的结果为 51.98。

| Checkpoint | 33_3 ZS + Global + Local |
|---|---:|
| weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt | 45.49 |
| weights/uni3d/scanobjnn/model.pt | 51.98 |
| 原文 Point-Cache Table 7 | 51.13 |

变化：

| 比较 | 变化 |
|---|---:|
| scanobjnn checkpoint - old pc_encoder checkpoint | +6.49 |
| scanobjnn checkpoint - 原文 | +0.85 |
| old pc_encoder checkpoint - 原文 | -5.64 |

分析：

旧 checkpoint 下的 45.49 明显低于原文 51.13，差异为 -5.64。切换到 `weights/uni3d/scanobjnn/model.pt` 后，结果提升到 51.98，略高于原文。

因此，33 组最初偏低的主要原因可以归结为：Uni3D point encoder checkpoint 与 ScanObjNN 数据设置不对应。后续 33 / 34 组应统一使用 `weights/uni3d/scanobjnn/model.pt`。

---

## 12. 与 03 / 13 / 23 组 ScanObjNN clean hardest 的关系

33_3 可以与前面几个 backbone 的 ScanObjNN clean hardest 完整 Point-Cache 结果进行横向比较。

| Backbone | 实验编号 | ScanObjNN clean hardest ZS + Global + Local |
|---|---|---:|
| ULIP | 03_3_ulip_scanobjnn_clean_hardest_zs_global_local | 32.48 |
| OpenShape | 23_3_openshape_scanobjnn_clean_hardest_zs_global_local | 43.82 |
| Uni3D | 33_3_uni3d_scanobjnn_clean_hardest_zs_global_local | 51.98 |

分析：

完整 Point-Cache 下，Uni3D 在 ScanObjNN clean hardest 上明显强于 ULIP 和 OpenShape。当前 Uni3D 完整 Point-Cache 为 51.98，比 OpenShape 的 43.82 高 +8.16。

这说明 Uni3D 的 ScanObjNN 对应 checkpoint 对真实扫描 hardest split 有明显优势，也说明在 ScanObjNN clean hardest 上，Uni3D 是当前最强 backbone。

---

## 13. 与后续实验的关系

33_3 是 33 组最后一个子实验，因此它本身可以作为后续 33 组 summary 文档的输入。

本文件只记录 33_3 自身及其与前序子实验的关系。完整 33 组总结应在单独 summary 文档中完成，包括：

| 后续 summary 应包含的内容 | 说明 |
|---|---|
| 33_1 / 33_2 / 33_3 总表 | 横向比较三种方法 |
| 与原文三种方法对齐 | 分别比较 46.04、50.28、51.13 |
| checkpoint 特殊情况 | 记录旧 checkpoint 偏低、切换 scanobjnn checkpoint 后对齐 |
| Global Cache 影响 | 分析 33_2 - 33_1 |
| Local Cache 影响 | 分析 33_3 - 33_2 |
| 与 03 / 13 / 23 组关系 | 分析不同 backbone 的 ScanObjNN clean hardest 表现 |
| 对 34 组 ScanObjNN-C 的意义 | 作为 clean reference |

33_3 的结果也为后续 34 组 Uni3D × ScanObjNN-C hardest all35 提供 clean reference。后续 34 组应继续使用 `weights/uni3d/scanobjnn/model.pt`。

---

## 14. 结果含义分析

33_3 的结果说明：完整 Point-Cache 在 Uni3D × ScanObjNN clean hardest 上有效，并且最终数值略高于原文。

| 观察 | 含义 |
|---|---|
| 33_3 = 51.98 | 当前完整 Point-Cache 最终结果 |
| 原文为 51.13 | 当前高 +0.85，略高但可接受 |
| 相比 33_1 提升 +6.35 | 完整 Point-Cache 有明确正增益 |
| 相比 33_2 提升 +1.95 | Local Cache 有较强额外贡献 |
| 33_2 相比 33_1 提升 +4.40 | Global Cache 是主要提升来源 |
| 趋势为 ZS < Global < Global + Local | 与原文趋势一致 |
| 使用 scanobjnn checkpoint 后对齐 | checkpoint 选择正确 |

因此，33_3 是 33 组三个子实验中最关键的最终结果。它证明了完整 Point-Cache 的最终效果是可靠的，也说明 Uni3D × ScanObjNN clean hardest 复现现在已经进入可归档状态。

---

## 15. 对后续 MCM-PC 的启发

当前 33_3 对后续 MCM-PC 方法设计和实验管理有以下启发：

| 观察 | 启发 |
|---|---|
| 完整 Point-Cache 提升 +6.35 | cache 机制在 Uni3D ScanObjNN clean hardest 上有效 |
| Global Cache 提供 +4.40 | 全局缓存是主要提升来源 |
| Local Cache 提供 +1.95 | 局部缓存有较强额外贡献 |
| 使用 scanobjnn checkpoint 后结果恢复正常 | checkpoint 与数据集匹配非常关键 |
| 旧 pc_encoder checkpoint 下结果明显偏低 | 不应作为正式 baseline checkpoint |
| 33 / 34 组应使用 scanobjnn checkpoint | ScanObjNN 系列统一设置 |
| 33_3 略高于原文 | 文档中需记录该正偏差 |

这次实验说明：对于 Uni3D，checkpoint 选择本身就是实验设置的一部分，必须在文档中明确记录，不能只写 backbone 名称。

---

## 16. 阶段性结论

本实验完成了 Uni3D × ScanObjNN clean hardest 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 33_3 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前使用 checkpoint 为 weights/uni3d/scanobjnn/model.pt。
3. 当前复现准确率为 51.98。
4. 原文 Point-Cache Table 7 中 Uni3D / ScanObjNN clean hardest / +Hierarchical Cache 结果为 51.13。
5. 当前复现结果比原文高 +0.85，略高但可以接受。
6. 相比 33_1 Zero-shot 的 45.63，33_3 提升到 51.98，总提升 +6.35。
7. 相比 33_2 Global Cache 的 50.03，33_3 额外提升 +1.95。
8. 当前趋势为 Zero-shot < Global Cache < Global + Local Cache，与原文趋势一致。
9. 旧 checkpoint weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt 下结果为 45.49，比原文低 -5.64。
10. 切换到 ScanObjNN checkpoint 后，结果从 45.49 提升到 51.98，说明此前偏低主要来自 checkpoint 不匹配。
11. 33_3 结果有效，不需要重跑。
12. 该实验可作为 33 组 summary 文档和后续 34 组 ScanObjNN-C hardest all35 的 clean reference。
13. 33 / 34 组后续应统一使用 weights/uni3d/scanobjnn/model.pt。

---

## 17. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh 1

---

## 18. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local/summary.csv
