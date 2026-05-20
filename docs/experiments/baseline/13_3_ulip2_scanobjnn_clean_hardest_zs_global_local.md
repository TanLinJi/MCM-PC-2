# 13_3_ulip2_scanobjnn_clean_hardest_zs_global_local

## 1. 实验目的

本实验用于复现 ULIP-2 在 ScanObjNN clean hardest 上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 13_3_ulip2_scanobjnn_clean_hardest_zs_global_local |
| Backbone | ULIP-2 |
| Dataset | ScanObjNN clean hardest |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 13_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证局部缓存是否能在全局缓存之外继续提升 ULIP-2 在 ScanObjNN clean hardest 上的识别性能。

本文件只记录 13_3 本身，并与前序子实验 13_1 和 13_2 进行对比。完整 13 组三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 13 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP-2 |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 数据目录 | data/sonn_c |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 方法脚本 | Point-Cache/scripts/baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/13_run_ulip2_scanobjnn_clean_hardest_common.sh |
| cache_type | hierarchical |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |
| 输入点数 | 1024 |
| Global Cache shot_capacity | 3 |
| Local Cache shot_capacity | 3 |
| Global / Local alpha | 4.0 |
| Global / Local beta | 3.0 |
| KMeans 聚类数 | 3 |
| Backbone 权重 | weights/ulip/pointbert_ulip2.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| ULIP version | ulip2 |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 0 |

本实验使用 `sonn_c` 作为 dataset 参数，`sonn_variant=hardest`，并指定 `cor_type=clean`。实际读取文件为：

data/sonn_c/hardest/clean.h5

---

## 3. 方法说明

13_3 与 13_2 的区别在于：13_2 只使用 Global Cache，而 13_3 进一步加入 Local Cache。

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 13_1 | 是 | 否 | 否 |
| 13_2 | 是 | 是 | 否 |
| 13_3 | 是 | 是 | 是 |

完整 Point-Cache 的预测由三部分组成：

| 组成部分 | 作用 |
|---|---|
| Zero-shot logits | 来自 ULIP-2 的原始文本-点云相似度预测 |
| Global Cache logits | 基于全局点云特征的测试时缓存检索结果 |
| Local Cache logits | 基于局部 patch / 局部聚类特征的测试时缓存检索结果 |

Global Cache 主要利用点云整体结构信息；Local Cache 进一步利用局部结构信息。理论上，Local Cache 可以补充 Global Cache 对细粒度结构差异和局部几何信息的不足。

因此，本实验重点观察：

| 观察点 | 说明 |
|---|---|
| 13_3 是否高于 13_1 | 判断完整 Point-Cache 是否提升 Zero-shot |
| 13_3 是否高于 13_2 | 判断 Local Cache 是否带来额外收益 |
| 13_3 是否接近原文值 | 判断完整 Point-Cache 复现是否对齐原文 |
| Local Cache 额外增益是否明显 | 判断在真实扫描 clean hardest 上局部缓存的实际贡献 |

---

## 4. 输出结构

输出目录：

Point-Cache/results/baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

13_3_ulip2_scanobjnn_clean_hardest_zs_global_local_clean_YYYYMMDD_HHMMSS.log

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
| 当前复现 accuracy | 42.44 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，13_3 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 6. 当前结果表

| 实验编号 | Dataset | Variant | File | Method | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 13_3_ulip2_scanobjnn_clean_hardest_zs_global_local | sonn_c | hardest | data/sonn_c/hardest/clean.h5 | Zero-shot + Global Cache + Local Cache | 42.44 | done |

该结果表示：ULIP-2 在 ScanObjNN hardest clean 数据上使用完整 Point-Cache 后，准确率为 42.44。

---

## 7. 与原文结果对比

原文 Point-Cache Supplementary Table 7 中，S-PB T50-RS-C 是 ScanObjectNN hardest split；其中 Original Data / SONN 对应 clean hardest。ULIP-2 在该设置下使用 +Hierarchical Cache 的结果为 42.40。

当前复现结果为 42.44。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Supplementary Table 7 ULIP-2 + Hierarchical Cache / SONN Original Data | 42.40 | 42.44 | +0.04 | 0.04 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与 Supplementary Table 7 的差异 | +0.04 |
| 绝对差异 | 0.04 |

分析：

当前复现结果 42.44 与原文 42.40 相差 +0.04，几乎完全一致。可以认为 13_3 的 ULIP-2 + Global Cache + Local Cache 结果与原文高度对齐。

因此，13_3 不只是脚本执行成功，而且数值也与原文 ScanObjNN clean hardest / ULIP-2 + Hierarchical Cache 结果基本一致。

---

## 8. 与前序实验 13_1 和 13_2 的对比

13_1 是无缓存 Zero-shot，13_2 是 Zero-shot + Global Cache，13_3 是完整 Point-Cache。

| 实验编号 | 方法 | 当前复现值 |
|---|---|---:|
| 13_1_ulip2_scanobjnn_clean_hardest_zs | Zero-shot | 34.07 |
| 13_2_ulip2_scanobjnn_clean_hardest_zs_global | Zero-shot + Global Cache | 40.42 |
| 13_3_ulip2_scanobjnn_clean_hardest_zs_global_local | Zero-shot + Global Cache + Local Cache | 42.44 |

当前复现增益：

| 比较 | 当前复现增益 | 含义 |
|---|---:|---|
| 13_2 - 13_1 | +6.35 | Global Cache 相比 Zero-shot 的提升 |
| 13_3 - 13_2 | +2.02 | Local Cache 在 Global Cache 基础上的额外提升 |
| 13_3 - 13_1 | +8.37 | 完整 Point-Cache 相比 Zero-shot 的总体提升 |

原文对应增益：

| 比较 | 原文增益 |
|---|---:|
| Global Cache - Zero-shot | 40.28 - 33.38 = +6.90 |
| Hierarchical Cache - Global Cache | 42.40 - 40.28 = +2.12 |
| Hierarchical Cache - Zero-shot | 42.40 - 33.38 = +9.02 |

增益对齐情况：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 增益 | +6.90 | +6.35 | -0.55 |
| Local Cache 额外增益 | +2.12 | +2.02 | -0.10 |
| 完整 Point-Cache 总增益 | +9.02 | +8.37 | -0.65 |

分析：

当前复现中，完整 Point-Cache 将准确率从 13_1 的 34.07 提升到 13_3 的 42.44，总提升为 +8.37。方法趋势正确：

Zero-shot < ZS + Global Cache < ZS + Global Cache + Local Cache

当前 Local Cache 的额外增益为 +2.02，与原文的 +2.12 高度接近。这说明 13_3 不仅绝对结果与原文对齐，而且 Local Cache 的模块贡献也基本复现了原文趋势。

当前完整 Point-Cache 总增益略低于原文，主要原因是 13_1 Zero-shot 当前复现值 34.07 高于原文 33.38，导致以当前 13_1 为基线计算的总增益略被压缩。

---

## 9. 结果含义分析

ScanObjNN clean hardest 是真实扫描数据，即使没有 synthetic corruption，也存在明显 domain gap。13_1 的 Zero-shot 准确率为 34.07，说明 ULIP-2 原始 zero-shot 在该数据上较困难。

13_2 加入 Global Cache 后提升到 40.42，13_3 进一步加入 Local Cache 后提升到 42.44。说明测试时全局缓存和局部缓存都能缓解真实扫描域偏移。

| 观察 | 含义 |
|---|---|
| 13_3 = 42.44 | 与原文 +Hierarchical Cache 结果高度一致 |
| 13_3 高于 13_2 | Local Cache 有额外正增益 |
| 13_3 高于 13_1 | 完整 Point-Cache 有明显作用 |
| Local Cache 额外提升 +2.02 | 局部缓存对真实扫描 clean hardest 有明显帮助 |
| 完整 Point-Cache 总提升 +8.37 | 测试时缓存机制能明显缓解真实扫描域偏移 |

这说明在 ULIP-2 × ScanObjNN clean hardest 上，完整 Point-Cache 的作用明显强于在 ModelNet clean 上的作用。真实扫描 hardest split 存在更强 domain gap，因此 cache 机制有更大的发挥空间。

---

## 10. Local Cache 贡献分析

本实验中，Local Cache 的额外增益为：

| 比较 | 数值 |
|---|---:|
| 13_3 - 13_2 | +2.02 |

原文中的 Local Cache 额外增益为：

| 比较 | 数值 |
|---|---:|
| Hierarchical Cache - Global Cache | 42.40 - 40.28 = +2.12 |

当前 Local Cache 额外增益比原文低 0.10，差异很小，说明局部缓存贡献复现良好。

可能原因包括：

| 原因 | 解释 |
|---|---|
| ScanObjNN hardest 是真实扫描数据 | 局部结构更不完整、更复杂 |
| Global Cache 只能捕获整体结构 | 对局部细节和缺失的补偿有限 |
| Local Cache 补充局部几何信息 | 能在 Global Cache 基础上继续提升 |
| hardest split domain gap 明显 | 测试时缓存机制有较大发挥空间 |

因此，Local Cache 在 13_3 中不是微弱补充，而是提供了比较明显的额外收益。与 11 组 ModelNet clean 中 Local Cache 只提升 +0.36 不同，13_3 中 Local Cache 额外提升 +2.02，说明局部缓存对真实扫描数据更有价值。

---

## 11. 与 11_3 ModelNet clean 的关系

11_3 是 ULIP-2 在 ModelNet clean 上的完整 Point-Cache 结果；13_3 是 ULIP-2 在 ScanObjNN clean hardest 上的完整 Point-Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy |
|---|---|---|---:|
| 11_3_ulip2_modelnet_clean_zs_global_local | ModelNet clean | Zero-shot + Global + Local | 74.35 |
| 13_3_ulip2_scanobjnn_clean_hardest_zs_global_local | ScanObjNN clean hardest | Zero-shot + Global + Local | 42.44 |

对比：

| 比较 | 变化 |
|---|---:|
| 13_3 - 11_3 | -31.91 |

分析：

即使使用完整 Point-Cache，ScanObjNN clean hardest 仍然比 ModelNet clean 低 31.91 个百分点。这说明真实扫描 hardest split 对 ULIP-2 是非常强的 domain shift。

不过，相比 Zero-shot，完整 Point-Cache 明显缩小了 ModelNet clean 到 ScanObjNN clean hardest 的差距：

| 方法 | ModelNet clean | ScanObjNN clean hardest | 下降 |
|---|---:|---:|---:|
| Zero-shot | 72.20 | 34.07 | -38.13 |
| ZS + Global | 73.99 | 40.42 | -33.57 |
| ZS + Global + Local | 74.35 | 42.44 | -31.91 |

完整 Point-Cache 使跨数据设置下降从 Zero-shot 的 -38.13 缩小到 -31.91，说明 Global + Local Cache 对真实扫描域偏移有实际缓解作用。

---

## 12. 与 03_3 ULIP ScanObjNN clean hardest 的关系

03_3 是 ULIP 在 ScanObjNN clean hardest 上的完整 Point-Cache 结果；13_3 是 ULIP-2 在同一数据设置下的完整 Point-Cache 结果。

| Backbone | 实验编号 | Method | Accuracy |
|---|---|---|---:|
| ULIP | 03_3_ulip_scanobjnn_clean_hardest_zs_global_local | Zero-shot + Global + Local | 32.48 |
| ULIP-2 | 13_3_ulip2_scanobjnn_clean_hardest_zs_global_local | Zero-shot + Global + Local | 42.44 |

对比：

| 比较 | 变化 |
|---|---:|
| ULIP-2 - ULIP | +9.96 |

分析：

在 ScanObjNN clean hardest 上，ULIP-2 的完整 Point-Cache 结果比 ULIP 高 +9.96。说明更强 backbone 对真实扫描数据也带来明显优势。

同时，ULIP-2 的 cache 增益也非常明显：

| Backbone | Zero-shot | ZS + Global | ZS + Global + Local | Full over ZS |
|---|---:|---:|---:|---:|
| ULIP | 29.08 | 32.20 | 32.48 | +3.40 |
| ULIP-2 | 34.07 | 40.42 | 42.44 | +8.37 |

这说明在 ScanObjNN clean hardest 上，ULIP-2 不仅基础 zero-shot 更强，而且 cache 模块带来的总提升也更大。可能原因是 ULIP-2 的特征空间更适合测试时缓存检索，因此 Global / Local Cache 能更有效地利用测试流分布。

---

## 13. 与后续实验的关系

13_3 是 13 组的最后一个子实验，因此它本身可以作为后续 13 组 summary 文档的输入。

本文件只记录 13_3 自身及其与前序子实验的关系。完整 13 组总结应在单独 summary 文档中完成，包括：

| 后续 summary 应包含的内容 | 说明 |
|---|---|
| 13_1 / 13_2 / 13_3 总表 | 横向比较三种方法 |
| 与原文三种方法对齐 | 分别比较 33.38、40.28、42.40 |
| Global Cache 贡献 | 分析 13_2 - 13_1 |
| Local Cache 贡献 | 分析 13_3 - 13_2 |
| 与 11 组 ModelNet clean 对比 | 分析真实扫描域偏移 |
| 与 03 组 ULIP 对比 | 分析 ULIP-2 backbone 提升 |
| 13 组阶段性结论 | 总结 ULIP-2 在 ScanObjNN clean hardest 上的复现状态 |

---

## 14. 阶段性结论

本实验完成了 ULIP-2 × ScanObjNN clean hardest 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 13_3 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 42.44。
3. 原文 Point-Cache Supplementary Table 7 中 ULIP-2 在 S-PB T50-RS-C Original Data / SONN 上的 +Hierarchical Cache 结果为 42.40。
4. 当前复现结果与原文差异仅 +0.04，可以认为 13_3 高度对齐原文。
5. 相比 13_1 Zero-shot 的 34.07，13_3 提升 +8.37。
6. 相比 13_2 Global Cache 的 40.42，13_3 额外提升 +2.02。
7. 当前方法趋势正确：Zero-shot < Global Cache < Global + Local Cache。
8. Local Cache 在 ScanObjNN clean hardest 上有明显正增益，且与原文 Local Cache 增益高度接近。
9. 完整 Point-Cache 明显缓解真实扫描 clean hardest 上的 domain gap。
10. 本实验可作为 13 组 summary 文档和后续 14 组 ULIP-2 × ScanObjNN-C all35 实验的 clean 参考结果。

---

## 15. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh 1

---

## 16. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

cat results/baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local/summary.csv
