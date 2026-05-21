# 21_3_openshape_modelnet_clean_zs_global_local

## 1. 实验目的

本实验用于复现 OpenShape 在 ModelNet clean 上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 21_3_openshape_modelnet_clean_zs_global_local |
| Backbone | OpenShape |
| Dataset | ModelNet clean |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 21_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证完整 Point-Cache 对 OpenShape 在 ModelNet clean 上的影响。

需要特别注意：OpenShape 在 ModelNet clean 上的 Zero-shot 已经非常强，原文中 +Global Cache 和 +Hierarchical Cache 都没有提升 clean accuracy，而是略低于 Zero-shot。因此，本实验不能简单用 “完整 Point-Cache 是否提升 clean accuracy” 判断是否正常，而应优先比较当前复现值与原文数值是否对齐。

本文件只记录 21_3 本身，并与前序子实验 21_1 和 21_2 进行对比。完整 21 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 21 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | OpenShape |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 实际数据文件 | data/modelnet_c/clean.h5 |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 方法脚本 | Point-Cache/scripts/baseline/21_3_openshape_modelnet_clean_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/21_run_openshape_modelnet_clean_common.sh |
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

本实验使用 `modelnet_c` 作为 dataset 参数，并指定 `cor_type=clean`。实际读取文件为：

data/modelnet_c/clean.h5

---

## 3. 方法说明

21_3 与 21_2 的区别在于：21_2 只使用 Global Cache，而 21_3 进一步加入 Local Cache。

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 21_1 | 是 | 否 | 否 |
| 21_2 | 是 | 是 | 否 |
| 21_3 | 是 | 是 | 是 |

完整 Point-Cache 的预测由三部分组成：

| 组成部分 | 作用 |
|---|---|
| Zero-shot logits | 来自 OpenShape 的原始文本-点云相似度预测 |
| Global Cache logits | 基于全局点云特征的测试时缓存检索结果 |
| Local Cache logits | 基于局部 patch / 局部聚类特征的测试时缓存检索结果 |

在 OpenShape × ModelNet clean 上，由于 OpenShape 的 zero-shot 表征已经很强，完整 Point-Cache 不一定带来正增益。原文中 OpenShape clean 的 +Hierarchical Cache 结果也低于 Zero-shot 和 +Global Cache，因此本实验重点观察：

| 观察点 | 说明 |
|---|---|
| 21_3 是否接近原文 +Hierarchical Cache 值 | 判断复现是否对齐 |
| 21_3 与 21_1 / 21_2 的差值是否符合原文趋势 | 判断 clean 上轻微下降是否正常 |
| Local Cache 是否在 clean setting 上带来额外收益 | 作为后续 ModelNet-C 分析背景 |

---

## 4. 输出结构

输出目录：

Point-Cache/results/baseline/21_3_openshape_modelnet_clean_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

21_3_openshape_modelnet_clean_zs_global_local_clean_YYYYMMDD_HHMMSS.log

因为本实验只测试 `clean.h5`，所以期望只有 1 行 summary 记录和 1 个对应 log。

---

## 5. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 1 | 1 | clean 实验只包含 1 个测试文件 |
| summary 中唯一 cor_type 数 | 1 | 1 | 只测试 clean |
| summary 中唯一 log_path 数 | 1 | 1 | 每次运行对应 1 个 log |
| logs 目录 .log 文件数 | 1 | 1 | 没有旧日志或重复日志残留 |
| status=done 数 | 1 | 1 | 脚本执行成功 |
| 当前复现 accuracy | 84.00 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，21_3 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 6. 当前结果表

| 实验编号 | Dataset | File | Method | Accuracy | Status |
|---|---|---|---|---:|---|
| 21_3_openshape_modelnet_clean_zs_global_local | modelnet_c | data/modelnet_c/clean.h5 | Zero-shot + Global Cache + Local Cache | 84.00 | done |

该结果表示：OpenShape 在 ModelNet clean 数据上使用完整 Point-Cache 后，准确率为 84.00。

---

## 7. 与原文结果对比

原文 Point-Cache Table 1 中，OpenShape / O-Shape 在 ModelNet clean 上的 +Hierarchical Cache 结果为 84.04。

当前复现结果为 84.00。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Table 1 OpenShape / ModelNet clean / +Hierarchical Cache | 84.04 | 84.00 | -0.04 | 0.04 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与原文差异 | -0.04 |
| 绝对差异 | 0.04 |

分析：

当前复现结果 84.00 与原文 84.04 只相差 -0.04，几乎完全一致。可以认为 21_3 的 OpenShape × ModelNet clean +Hierarchical Cache 结果与原文高度对齐。

因此，21_3 不只是脚本执行成功，而且数值也与 Point-Cache Table 1 的 OpenShape clean +Hierarchical Cache 结果基本一致。

---

## 8. 与前序实验 21_1 和 21_2 的对比

21_1 是无缓存 Zero-shot，21_2 是 Zero-shot + Global Cache，21_3 是完整 Point-Cache。

| 实验编号 | 方法 | 当前复现值 |
|---|---|---:|
| 21_1_openshape_modelnet_clean_zs | Zero-shot | 84.72 |
| 21_2_openshape_modelnet_clean_zs_global | Zero-shot + Global Cache | 84.48 |
| 21_3_openshape_modelnet_clean_zs_global_local | Zero-shot + Global Cache + Local Cache | 84.00 |

当前复现变化：

| 比较 | 当前复现变化 | 含义 |
|---|---:|---|
| 21_2 - 21_1 | -0.24 | Global Cache 相比 Zero-shot 的变化 |
| 21_3 - 21_2 | -0.48 | Local Cache 在 Global Cache 基础上的额外变化 |
| 21_3 - 21_1 | -0.72 | 完整 Point-Cache 相比 Zero-shot 的总体变化 |

原文对应变化：

| 比较 | 原文变化 |
|---|---:|
| Global Cache - Zero-shot | 84.52 - 84.56 = -0.04 |
| Hierarchical Cache - Global Cache | 84.04 - 84.52 = -0.48 |
| Hierarchical Cache - Zero-shot | 84.04 - 84.56 = -0.52 |

变化对齐情况：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 相对 Zero-shot 的变化 | -0.04 | -0.24 | -0.20 |
| Local Cache 额外变化 | -0.48 | -0.48 | +0.00 |
| 完整 Point-Cache 相对 Zero-shot 的变化 | -0.52 | -0.72 | -0.20 |

分析：

当前复现中，完整 Point-Cache 将准确率从 21_1 的 84.72 变为 21_3 的 84.00，总变化为 -0.72。这个方向与原文一致，因为原文中 OpenShape clean 上 +Hierarchical Cache 也低于 Zero-shot。

尤其重要的是，当前 Local Cache 的额外变化为 -0.48，与原文中的 -0.48 完全一致。这说明 21_3 不仅绝对数值与原文对齐，而且 Local Cache 在 clean setting 上的相对影响也与原文趋势高度一致。

当前完整 Point-Cache 相比 Zero-shot 的下降幅度略大于原文，主要原因是 21_1 当前复现值 84.72 比原文 Zero-shot 84.56 高 +0.16，而 21_3 当前复现值 84.00 与原文 84.04 几乎一致。

---

## 9. 为什么 OpenShape clean 上完整 Point-Cache 会下降

OpenShape 在 ModelNet clean 上的 Zero-shot 已经达到 84% 以上，说明它的原始文本-点云对齐能力很强。在这种 clean synthetic setting 下，测试时缓存机制的收益空间很小。

完整 Point-Cache 引入 Global Cache 和 Local Cache 后，会使用在线测试样本的伪标签和缓存检索信息修正 zero-shot logits。当基础模型已经非常强时，这些伪标签和缓存检索不一定能进一步提供有效信息，反而可能带来轻微扰动。

因此，OpenShape clean 上出现：

Zero-shot > ZS + Global Cache > ZS + Global Cache + Local Cache

并不意味着完整 Point-Cache 实现错误，而是说明在该 clean setting 下，OpenShape 的 zero-shot 已经足够强，cache 的边际收益不足。

这一点也符合原文现象。原文中 OpenShape clean 的 Zero-shot 为 84.56，+Global Cache 为 84.52，+Hierarchical Cache 为 84.04，同样呈现逐步轻微下降。

---

## 10. 与此前 backbone 的 clean 结果关系

21_3 可以与 ULIP、ULIP-2 的 ModelNet clean 完整 Point-Cache 结果进行横向比较。

| Backbone | 实验编号 | ModelNet clean ZS + Global + Local |
|---|---|---:|
| ULIP | 01_3_ulip_modelnet_clean_zs_global_local | 约 64 |
| ULIP-2 | 11_3_ulip2_modelnet_clean_zs_global_local | 74.35 |
| OpenShape | 21_3_openshape_modelnet_clean_zs_global_local | 84.00 |

观察：

1. OpenShape 完整 Point-Cache 的绝对准确率仍然显著高于 ULIP 和 ULIP-2。
2. OpenShape 的 clean zero-shot 已经很强，因此完整 Point-Cache 在 clean 上没有表现出正向增益。
3. 这说明 backbone 表征能力越强，clean setting 上 cache 的边际收益可能越低。
4. 后续更关键的是观察 corrupted setting，即 22 组 ModelNet-C all35。

---

## 11. 与后续实验的关系

21_3 是 21 组最后一个子实验，因此它本身可以作为后续 21 组 summary 文档的输入。

本文件只记录 21_3 自身及其与前序子实验的关系。完整 21 组总结应在单独 summary 文档中完成，包括：

| 后续 summary 应包含的内容 | 说明 |
|---|---|
| 21_1 / 21_2 / 21_3 总表 | 横向比较三种方法 |
| 与原文三种方法对齐 | 分别比较 84.56、84.52、84.04 |
| Global Cache 影响 | 分析 21_2 - 21_1 |
| Local Cache 影响 | 分析 21_3 - 21_2 |
| 与 ULIP / ULIP-2 clean 对比 | 分析 backbone 差异 |
| 对 22 组 ModelNet-C 的意义 | clean 参考和鲁棒性背景 |

---

## 12. 阶段性结论

本实验完成了 OpenShape × ModelNet clean 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 21_3 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 84.00。
3. 原文 Point-Cache Table 1 中 OpenShape / O-Shape 在 ModelNet clean 上的 +Hierarchical Cache 结果为 84.04。
4. 当前复现结果与原文差异仅 -0.04，可以认为高度对齐。
5. 相比 21_2 Global Cache 的 84.48，21_3 下降 -0.48，与原文中 Hierarchical Cache 相比 Global Cache 的变化 -0.48 完全一致。
6. 相比 21_1 Zero-shot 的 84.72，21_3 下降 -0.72。
7. OpenShape 在 ModelNet clean 上完整 Point-Cache 略低于 Zero-shot 是原文中已有现象，不是实验异常。
8. 当前方法趋势与原文一致：Zero-shot > Global Cache > Global + Local Cache。
9. 该实验说明 clean setting 上 OpenShape 的 zero-shot 已经很强，cache 的主要价值需要在后续 corrupted setting 中观察。
10. 本实验可作为 21 组 summary 文档和 22 组 OpenShape × ModelNet-C all35 实验的 clean 参考。

---

## 13. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/21_3_openshape_modelnet_clean_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/21_3_openshape_modelnet_clean_zs_global_local_single_gpu.sh 1

---

## 14. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/21_3_openshape_modelnet_clean_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/21_3_openshape_modelnet_clean_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/21_3_openshape_modelnet_clean_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/21_3_openshape_modelnet_clean_zs_global_local/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/21_3_openshape_modelnet_clean_zs_global_local/summary.csv
