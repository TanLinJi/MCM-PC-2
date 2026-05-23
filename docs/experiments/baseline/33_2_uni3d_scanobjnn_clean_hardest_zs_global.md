# 33_2_uni3d_scanobjnn_clean_hardest_zs_global

## 1. 实验目的

本实验用于复现 Uni3D 在 ScanObjNN clean hardest split 上使用 Zero-shot + Global Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 33_2_uni3d_scanobjnn_clean_hardest_zs_global |
| Backbone | Uni3D |
| Dataset | ScanObjNN clean hardest |
| Dataset 参数 | sonn_c |
| SONN variant | hardest |
| Method | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |

本实验是在 33_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存是否能够提升 Uni3D 在 ScanObjNN clean hardest split 上的表现。

本文件只记录 33_2 本身，并与前序子实验 33_1 进行对比。完整 33 组的三方法总表、checkpoint 特殊情况、Global / Local 贡献分解和整体阶段性总结应放在 33 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | Uni3D |
| 数据集参数 | sonn_c |
| 数据目录 | data/sonn_c |
| 数据集变体 | hardest |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 原始核心 runner | runners/model_with_global_cache.py |
| 方法脚本 | Point-Cache/scripts/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/33_run_uni3d_scanobjnn_clean_hardest_common.sh |
| cache_type | global |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |
| 输入点数 | 1024 |
| Uni3D point encoder checkpoint | weights/uni3d/scanobjnn/model.pt |
| Uni3D text encoder checkpoint | weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin |
| pc_model | eva_giant_patch14_560 |
| clip_model | EVA02-E-14-plus |
| pc_feat_dim | 1408 |
| num_group | 512 |
| group_size | 64 |
| pc_encoder_dim | 512 |
| embed_dim | 1024 |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 0 |

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

33_2 的执行路径为：

| 层级 | 文件 |
|---|---|
| 单方法脚本 | Point-Cache/scripts/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/33_run_uni3d_scanobjnn_clean_hardest_common.sh |
| 原始推理 runner | Point-Cache/runners/model_with_global_cache.py |

该结构与 03、13、23、31、33_1 等 clean 单文件实验保持一致。

---

## 4. 方法说明

33_2 在 Zero-shot logits 的基础上加入 Global Cache logits。

| 组成部分 | 是否使用 |
|---|---:|
| Text prototype / 文本原型 | 是 |
| Point cloud global feature | 是 |
| Zero-shot logits | 是 |
| Global Cache logits | 是 |
| Local Cache logits | 否 |
| Hierarchical Cache | 否 |

Global Cache 的基本作用是：在测试过程中动态缓存高置信度样本的全局点云特征和伪标签，然后对后续样本进行全局特征检索，生成 cache logits，并与 zero-shot logits 融合。

33_2 与 33_1 的主要区别如下：

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 33_1 | 是 | 否 | 否 |
| 33_2 | 是 | 是 | 否 |

因此，33_2 可以用于单独评估 Global Cache 在 Uni3D × ScanObjNN clean hardest 上的影响。

---

## 5. Uni3D checkpoint 说明

本实验使用的 Uni3D point encoder checkpoint 为：

weights/uni3d/scanobjnn/model.pt

这是本次 33 组复现中非常关键的设置。

此前曾使用服务器原有 checkpoint：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

旧 checkpoint 下 33_2 的结果为 43.79，明显低于原文 50.28。检查 33 组公共脚本后发现，当时脚本的 checkpoint 候选列表没有包含 `weights/uni3d/scanobjnn/model.pt`，因此自动优先选择了已存在的 `weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt`。

修正公共脚本后，将 ScanObjNN checkpoint 加入候选列表第一位：

weights/uni3d/scanobjnn/model.pt

重新运行后，33_2 的结果提升到 50.03，与原文 50.28 高度接近。因此，33_2 正式记录以 `weights/uni3d/scanobjnn/model.pt` 为准。

该 checkpoint 的下载脚本已记录在：

Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh

完整的 checkpoint 特殊情况会在 33 组 summary 文档中统一说明。

---

## 6. 输出结构

输出目录：

Point-Cache/results/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

33_2_uni3d_scanobjnn_clean_hardest_zs_global_clean_YYYYMMDD_HHMMSS.log

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
| 当前复现 accuracy | 50.03 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |
| 实际 checkpoint | weights/uni3d/scanobjnn/model.pt | scanobjnn checkpoint | 已修正 |

从执行完整性看，33_2 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 8. 当前结果表

| 实验编号 | Dataset | File | Method | Checkpoint | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 33_2_uni3d_scanobjnn_clean_hardest_zs_global | sonn_c hardest | data/sonn_c/hardest/clean.h5 | Zero-shot + Global Cache | weights/uni3d/scanobjnn/model.pt | 50.03 | done |

该结果表示：Uni3D 在 ScanObjNN clean hardest 上加入 Global Cache 后，准确率为 50.03。

---

## 9. 与原文结果对比

原文 Point-Cache Table 7 中，Uni3D 在 ScanObjNN hardest clean/original data 上的 +Global Cache 结果为 50.28。

当前复现结果为 50.03。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Uni3D / ScanObjNN clean hardest / +Global Cache | 50.28 | 50.03 | -0.25 | 0.25 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与原文差异 | -0.25 |
| 绝对差异 | 0.25 |

分析：

当前复现结果 50.03 比原文 50.28 低 -0.25，差异很小。因此，33_2 不只是脚本执行成功，而且数值与原文高度接近。

该结果说明，使用 `weights/uni3d/scanobjnn/model.pt` 后，Uni3D × ScanObjNN clean hardest 的 +Global Cache baseline 复现有效。

---

## 10. 与前序实验 33_1 的对比

33_1 是本实验的直接前序子实验，方法为 Zero-shot，不使用缓存。

| 实验编号 | 方法 | 当前复现值 |
|---|---|---:|
| 33_1_uni3d_scanobjnn_clean_hardest_zs | Zero-shot | 45.63 |
| 33_2_uni3d_scanobjnn_clean_hardest_zs_global | Zero-shot + Global Cache | 50.03 |

Global Cache 带来的当前复现变化为：

| 比较 | 当前复现变化 |
|---|---:|
| 33_2 - 33_1 | +4.40 |

原文中对应变化为：

| 比较 | 原文变化 |
|---|---:|
| Global Cache - Zero-shot | 50.28 - 46.04 = +4.24 |

对比：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 相对 Zero-shot 的变化 | +4.24 | +4.40 | +0.16 |

分析：

当前复现中，Global Cache 将准确率从 33_1 的 45.63 提升到 33_2 的 50.03，提升 +4.40。原文中 Global Cache 提升为 +4.24。二者差异仅 +0.16。

因此，33_2 不仅绝对准确率与原文对齐，Global Cache 的相对增益也与原文高度一致。这说明当前 Uni3D ScanObjNN clean hardest 的 Global Cache 复现是可靠的。

---

## 11. 与旧 checkpoint 结果对比

本实验曾使用服务器原有 checkpoint 运行过一次：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

旧 checkpoint 下 33_2 的结果为 43.79。切换到 ScanObjNN checkpoint 后，33_2 的结果为 50.03。

| Checkpoint | 33_2 ZS + Global |
|---|---:|
| weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt | 43.79 |
| weights/uni3d/scanobjnn/model.pt | 50.03 |
| 原文 Point-Cache Table 7 | 50.28 |

变化：

| 比较 | 变化 |
|---|---:|
| scanobjnn checkpoint - old pc_encoder checkpoint | +6.24 |
| scanobjnn checkpoint - 原文 | -0.25 |
| old pc_encoder checkpoint - 原文 | -6.49 |

分析：

旧 checkpoint 下的 43.79 明显低于原文 50.28，差异为 -6.49。切换到 `weights/uni3d/scanobjnn/model.pt` 后，结果提升到 50.03，与原文高度接近。

因此，33 组最初偏低的主要原因可以归结为：Uni3D point encoder checkpoint 与 ScanObjNN 数据设置不对应。后续 33 / 34 组应统一使用 `weights/uni3d/scanobjnn/model.pt`。

---

## 12. 与 03 / 13 / 23 组 ScanObjNN clean hardest 的关系

33_2 可以与前面三个 backbone 的 ScanObjNN clean hardest +Global Cache 结果进行横向比较。

| Backbone | 实验编号 | ScanObjNN clean hardest +Global Cache |
|---|---|---:|
| ULIP | 03_2_ulip_scanobjnn_clean_hardest_zs_global | 32.20 |
| ULIP-2 | 13_2_ulip2_scanobjnn_clean_hardest_zs_global | 39.38 |
| OpenShape | 23_2_openshape_scanobjnn_clean_hardest_zs_global | 41.95 |
| Uni3D | 33_2_uni3d_scanobjnn_clean_hardest_zs_global | 50.03 |

分析：

加入 Global Cache 后，Uni3D 在 ScanObjNN clean hardest 上明显强于 ULIP、ULIP-2 和 OpenShape。当前 Uni3D +Global Cache 为 50.03，比 OpenShape 的 41.95 高 +8.08。

这说明 Uni3D 的 ScanObjNN 对应 checkpoint 对真实扫描 hardest split 有明显优势，也说明 Global Cache 在该设置下能带来强增益。

---

## 13. 与后续子实验的关系

33_2 是 33_3 的直接前序实验。

| 后续实验 | 对比方式 |
|---|---|
| 33_3_uni3d_scanobjnn_clean_hardest_zs_global_local | 与 33_2 比较，评估 Local Cache 在 Global Cache 基础上的额外影响 |

本文件不展开 33_3 的实际结果。33_3 的数值及 Local Cache 额外影响应记录在 33_3 子实验文档和 33 组 summary 文档中。

需要注意的是，当前 33_2 已经证明 Global Cache 在 Uni3D × ScanObjNN clean hardest 上有效，因此 33_3 的关键问题是：

| 问题 | 说明 |
|---|---|
| Local Cache 是否能在 Global Cache 基础上继续提升？ | 比较 33_3 - 33_2 |
| 完整 Point-Cache 是否接近原文 51.13？ | 比较 33_3 与原文 |
| 当前趋势是否仍是 ZS < Global < Global + Local？ | 判断整体方法趋势 |
| Local Cache 额外贡献是否复现原文趋势？ | 观察 33_3 的额外增益 |

---

## 14. 结果含义分析

33_2 的结果说明：Global Cache 在 Uni3D × ScanObjNN clean hardest 上非常有效，并且当前复现与原文高度一致。

| 观察 | 含义 |
|---|---|
| 33_2 = 50.03 | 当前 Uni3D + Global Cache 结果 |
| 原文为 50.28 | 当前只低 -0.25，复现高度接近 |
| 相比 33_1 提升 +4.40 | Global Cache 有明确正增益 |
| 原文 Global 增益为 +4.24 | 当前增益与原文高度一致 |
| 使用 scanobjnn checkpoint 后对齐 | checkpoint 选择正确 |
| 旧 checkpoint 下明显偏低 | 说明不能使用旧 pc_encoder 权重作为正式复现结果 |

因此，33_2 是 33 组中确认 Global Cache 复现可靠的关键实验。

---

## 15. 对后续 MCM-PC 的启发

当前 33_2 对后续 MCM-PC 方法设计和实验管理有以下启发：

| 观察 | 启发 |
|---|---|
| Global Cache 在 Uni3D ScanObjNN clean hardest 上提升 +4.40 | Uni3D 对 Global Cache 有稳定且显著的正响应 |
| 当前增益与原文高度一致 | 说明脚本和 checkpoint 设置已基本正确 |
| checkpoint 切换后结果明显改善 | 后续 Uni3D 实验必须严格区分数据集对应权重 |
| 33 / 34 组应使用 scanobjnn checkpoint | ScanObjNN 系列统一使用 weights/uni3d/scanobjnn/model.pt |
| 31 / 32 组应使用 modelnet40 checkpoint | ModelNet 系列统一使用 weights/uni3d/modelnet40/model.pt |
| 下载脚本已经归档 | 后续复现实验环境更可控 |

这次结果说明，Uni3D baseline 复现不能只检查 runner、参数和数据路径，还必须把 checkpoint 路径作为实验设置的核心部分记录。

---

## 16. 阶段性结论

本实验完成了 Uni3D × ScanObjNN clean hardest 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 33_2 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前使用 checkpoint 为 weights/uni3d/scanobjnn/model.pt。
3. 当前复现准确率为 50.03。
4. 原文 Point-Cache Table 7 中 Uni3D / ScanObjNN clean hardest / +Global Cache 结果为 50.28。
5. 当前复现结果比原文低 -0.25，差异很小，可以认为高度接近。
6. 相比 33_1 Zero-shot 的 45.63，33_2 提升到 50.03，Global Cache 增益为 +4.40。
7. 原文 Global Cache 增益为 +4.24，当前增益与原文只差 +0.16。
8. 旧 checkpoint weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt 下结果为 43.79，比原文低 -6.49。
9. 切换到 ScanObjNN checkpoint 后，结果从 43.79 提升到 50.03，说明此前偏低主要来自 checkpoint 不匹配。
10. 33_2 结果有效，不需要重跑。
11. 本实验是 33_3 分析 Local Cache 额外贡献的直接对照。
12. 33 / 34 组后续应统一使用 weights/uni3d/scanobjnn/model.pt。

---

## 17. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global_single_gpu.sh 1

---

## 18. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global/summary.csv
