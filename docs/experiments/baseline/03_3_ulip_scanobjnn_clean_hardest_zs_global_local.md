# 03_3_ulip_scanobjnn_clean_hardest_zs_global_local

## 1. 实验目的

本实验用于复现 ULIP 在 ScanObjNN clean hardest 上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 03_3_ulip_scanobjnn_clean_hardest_zs_global_local |
| Backbone | ULIP |
| Dataset | ScanObjNN clean hardest |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 03_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证局部缓存是否能在全局缓存之外继续带来额外增益。

本文件只记录 03_3 本身，并与前序子实验 03_1 和 03_2 进行对比。完整 03 组三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 03 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 数据目录 | data/sonn_c |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 方法脚本 | Point-Cache/scripts/baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/03_run_ulip_scanobjnn_clean_hardest_common.sh |
| cache_type | hierarchical |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |
| 输入点数 | 1024 |
| Global Cache shot_capacity | 3 |
| Local Cache shot_capacity | 3 |
| Global / Local alpha | 4.0 |
| Global / Local beta | 3.0 |
| KMeans 聚类数 | 3 |
| 权重 | weights/ulip/pointbert_ulip1.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| GPU | 单张 Tesla T4 |

本实验使用 `sonn_c` 作为 dataset 参数，`sonn_variant=hardest`，并指定 `cor_type=clean`。实际读取文件为：

data/sonn_c/hardest/clean.h5

---

## 3. 方法说明

03_3 与 03_2 的区别在于：03_2 只使用 Global Cache，而 03_3 进一步加入 Local Cache。

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 03_1 | 是 | 否 | 否 |
| 03_2 | 是 | 是 | 否 |
| 03_3 | 是 | 是 | 是 |

完整 Point-Cache 的预测由三部分组成：

| 组成部分 | 作用 |
|---|---|
| Zero-shot logits | 来自 ULIP 的原始文本-点云相似度预测 |
| Global Cache logits | 基于全局点云特征的测试时缓存检索结果 |
| Local Cache logits | 基于局部 patch / 局部聚类特征的测试时缓存检索结果 |

Global Cache 主要利用点云整体结构信息；Local Cache 进一步利用局部结构信息。理论上，Local Cache 可以补充 Global Cache 对细粒度结构差异的不足。

因此，本实验重点观察：

| 观察点 | 说明 |
|---|---|
| 03_3 是否高于 03_1 | 判断完整 Point-Cache 是否提升 Zero-shot |
| 03_3 是否高于 03_2 | 判断 Local Cache 是否带来额外收益 |
| 03_3 是否接近原文值 | 判断完整 Point-Cache 复现是否对齐原文 |
| Local Cache 额外增益是否明显 | 判断在 clean hardest 上局部缓存的实际贡献 |

---

## 4. 输出结构

输出目录：

Point-Cache/results/baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

03_3_ulip_scanobjnn_clean_hardest_zs_global_local_clean_YYYYMMDD_HHMMSS.log

因为本实验只测试 `clean.h5`，所以期望只有 1 行 summary 记录和 1 个对应 log。

---

## 5. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 1 | 1 | clean 实验只包含 1 个测试文件 |
| summary 中唯一 cor_type 数 | 1 | 1 | 只测试 clean |
| summary 中唯一 log_path 数 | 1 | 1 | 每次运行对应 1 个 log |
| status=done 数 | 1 | 1 | 脚本执行成功 |
| 当前复现 accuracy | 32.48 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，03_3 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 6. 当前结果表

| 实验编号 | Dataset | Variant | File | Method | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 03_3_ulip_scanobjnn_clean_hardest_zs_global_local | sonn_c | hardest | data/sonn_c/hardest/clean.h5 | Zero-shot + Global Cache + Local Cache | 32.48 | done |

该结果表示：ULIP 在 ScanObjNN hardest clean 数据上使用完整 Point-Cache 后，准确率为 32.48。

---

## 7. 与原文结果对比

原文中与本实验直接相关的参考值为：

| 来源 | 原文值 | 说明 |
|---|---:|---|
| Supplementary Table 7 | 32.48 | S-PB T50-RS-C hardest split，ULIP + Hierarchical Cache |

当前复现结果为 32.48。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Table 7 ULIP + Hierarchical Cache | 32.48 | 32.48 | +0.00 | 0.00 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与 Table 7 的差异 | +0.00 |
| 绝对差异 | 0.00 |

分析：

当前复现结果 32.48 与原文 Supplementary Table 7 的 32.48 完全一致。因此，03_3 的 Zero-shot + Global Cache + Local Cache 结果不仅脚本执行成功，而且数值与原文完全对齐。

这说明当前 ScanObjNN clean hardest 设置下，完整 Point-Cache 的核心参数、数据路径、模型权重、缓存构建与推理流程基本正确。

---

## 8. 与前序实验 03_1 和 03_2 的对比

03_1 是无缓存 Zero-shot，03_2 是 Zero-shot + Global Cache，03_3 是完整 Point-Cache。

| 实验编号 | 方法 | 当前复现值 |
|---|---|---:|
| 03_1_ulip_scanobjnn_clean_hardest_zs | Zero-shot | 29.08 |
| 03_2_ulip_scanobjnn_clean_hardest_zs_global | Zero-shot + Global Cache | 32.20 |
| 03_3_ulip_scanobjnn_clean_hardest_zs_global_local | Zero-shot + Global Cache + Local Cache | 32.48 |

当前复现增益：

| 比较 | 当前复现增益 | 含义 |
|---|---:|---|
| 03_2 - 03_1 | +3.12 | Global Cache 相比 Zero-shot 的提升 |
| 03_3 - 03_2 | +0.28 | Local Cache 在 Global Cache 基础上的额外提升 |
| 03_3 - 03_1 | +3.40 | 完整 Point-Cache 相比 Zero-shot 的总体提升 |

原文对应增益：

| 比较 | 原文增益 |
|---|---:|
| Global Cache - Original Data | 32.37 - 29.29 = +3.08 |
| Hierarchical Cache - Global Cache | 32.48 - 32.37 = +0.11 |
| Hierarchical Cache - Original Data | 32.48 - 29.29 = +3.19 |

增益对齐情况：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 增益 | +3.08 | +3.12 | +0.04 |
| Local Cache 额外增益 | +0.11 | +0.28 | +0.17 |
| 完整 Point-Cache 总增益 | +3.19 | +3.40 | +0.21 |

分析：

当前复现中，完整 Point-Cache 将准确率从 03_1 的 29.08 提升到 03_3 的 32.48，总提升为 +3.40。这个趋势与原文一致。

Global Cache 是主要提升来源：03_2 相比 03_1 提升 +3.12。Local Cache 在 Global Cache 基础上继续提升 +0.28，但幅度较小。这说明在 ScanObjNN clean hardest 上，完整 Point-Cache 的主要收益仍然来自 Global Cache，Local Cache 起到补充作用。

---

## 9. 结果含义分析

ScanObjNN clean hardest 是真实扫描数据，即使没有 synthetic corruption，也存在明显 domain gap。03_1 的 Zero-shot 仅为 29.08，说明 ULIP 原始 zero-shot 在该数据上较困难。

03_2 加入 Global Cache 后提升到 32.20，说明测试流中的全局特征缓存可以利用在线测试数据分布，缓解真实扫描域偏移。

03_3 进一步加入 Local Cache 后提升到 32.48，说明局部结构信息也能带来额外收益，但收益幅度较小。

| 观察 | 含义 |
|---|---|
| 03_3 高于 03_1 | 完整 Point-Cache 明显提升 Zero-shot |
| 03_3 高于 03_2 | Local Cache 有额外正增益 |
| 03_3 与原文完全一致 | 复现结果非常可靠 |
| Local Cache 额外提升只有 +0.28 | clean hardest 上局部缓存贡献较小 |
| 总提升 +3.40 | Point-Cache 能缓解真实扫描 clean 数据的 domain gap |

---

## 10. Local Cache 贡献分析

本实验中，Local Cache 的额外增益为：

| 比较 | 数值 |
|---|---:|
| 03_3 - 03_2 | +0.28 |

该值说明 Local Cache 在 ScanObjNN clean hardest 上是有效的，但不是主要提升来源。

可能原因包括：

| 原因 | 解释 |
|---|---|
| clean 数据没有 synthetic corruption | 局部结构没有像 corruption 场景中那样严重破坏 |
| Global Cache 已经捕获主要分布信息 | 全局特征缓存能够覆盖大部分可用增益 |
| hardest split 真实扫描噪声复杂 | 局部 patch 可能存在缺失和扫描噪声，限制 Local Cache 效果 |
| clean setting 只有一种数据状态 | 不像 all35 corruption 那样能充分观察局部缓存对不同损坏类型的差异贡献 |

因此，Local Cache 的作用在 03_3 中表现为“小幅但正向的补充增益”。这个结论与 02 组 ModelNet-C all35 中的趋势一致：Global Cache 是主贡献，Local Cache 提供额外但相对较小的提升。

---

## 11. 与后续实验的关系

03_3 是 03 组的最后一个子实验，因此它本身可以作为后续 03 组 summary 文档的输入。

本文件只记录 03_3 自身及其与前序子实验的关系。完整 03 组总结应在单独 summary 文档中完成，包括：

| 后续 summary 应包含的内容 | 说明 |
|---|---|
| 03_1 / 03_2 / 03_3 总表 | 横向比较三种方法 |
| 与原文三种方法对齐 | 分别比较 29.29、32.37、32.48 |
| Global Cache 贡献 | 分析 03_2 - 03_1 |
| Local Cache 贡献 | 分析 03_3 - 03_2 |
| 03 组阶段性结论 | 总结 ULIP 在 ScanObjNN clean hardest 上的复现状态 |

---

## 12. 阶段性结论

本实验完成了 ULIP × ScanObjNN clean hardest 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 03_3 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 32.48。
3. 原文 Supplementary Table 7 中 ULIP 在 S-PB T50-RS-C hardest split 上的 +Hierarchical Cache 结果为 32.48。
4. 当前复现结果与原文完全一致，差异为 +0.00。
5. 相比 03_1 Zero-shot 的 29.08，03_3 提升 +3.40。
6. 相比 03_2 Global Cache 的 32.20，03_3 额外提升 +0.28。
7. 当前结果说明完整 Point-Cache 在 ScanObjNN clean hardest 上复现成功。
8. Global Cache 是主要增益来源，Local Cache 提供小幅额外增益。
9. 本实验可作为后续 03 组 summary 文档和 04 组 ScanObjNN-C all35 实验的 clean 参考结果。

---

## 13. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh 1

---

## 14. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

cat results/baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local/summary.csv
