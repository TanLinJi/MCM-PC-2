# 11_3_ulip2_modelnet_clean_zs_global_local

## 1. 实验目的

本实验用于复现 ULIP-2 在 ModelNet clean 上使用 Zero-shot + Global Cache + Local Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 11_3_ulip2_modelnet_clean_zs_global_local |
| Backbone | ULIP-2 |
| Dataset | ModelNet clean |
| Method | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |

本实验是在 11_2 Zero-shot + Global Cache 的基础上进一步加入 Local Cache，用于验证局部缓存是否能在全局缓存之外继续提升 ULIP-2 在 ModelNet clean 上的识别性能。

本文件只记录 11_3 本身，并与前序子实验 11_1 和 11_2 进行对比。完整 11 组三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 11 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP-2 |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 实际数据文件 | data/modelnet_c/clean.h5 |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 方法简写 | zs_global_local |
| 原始核心 runner | runners/model_with_hierarchical_caches.py |
| 方法脚本 | Point-Cache/scripts/baseline/11_3_ulip2_modelnet_clean_zs_global_local_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/11_run_ulip2_modelnet_clean_common.sh |
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

本实验使用 `modelnet_c` 作为 dataset 参数，并指定 `cor_type=clean`。实际读取文件为：

data/modelnet_c/clean.h5

虽然数据集参数仍然写作 `modelnet_c`，但 `clean.h5` 对应的是 ModelNet clean 测试设置，不是 corrupted setting。

---

## 3. 方法说明

11_3 与 11_2 的区别在于：11_2 只使用 Global Cache，而 11_3 进一步加入 Local Cache。

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 11_1 | 是 | 否 | 否 |
| 11_2 | 是 | 是 | 否 |
| 11_3 | 是 | 是 | 是 |

完整 Point-Cache 的预测由三部分组成：

| 组成部分 | 作用 |
|---|---|
| Zero-shot logits | 来自 ULIP-2 的原始文本-点云相似度预测 |
| Global Cache logits | 基于全局点云特征的测试时缓存检索结果 |
| Local Cache logits | 基于局部 patch / 局部聚类特征的测试时缓存检索结果 |

因此，本实验重点观察：

| 观察点 | 说明 |
|---|---|
| 11_3 是否高于 11_1 | 判断完整 Point-Cache 是否提升 Zero-shot |
| 11_3 是否高于 11_2 | 判断 Local Cache 是否带来额外收益 |
| 11_3 是否接近原文值 | 判断完整 Point-Cache 复现是否对齐原文 |
| Local Cache 额外增益是否明显 | 判断在 ModelNet clean 上局部缓存的实际贡献 |

---

## 4. 输出结构

输出目录：

Point-Cache/results/baseline/11_3_ulip2_modelnet_clean_zs_global_local/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

11_3_ulip2_modelnet_clean_zs_global_local_clean_YYYYMMDD_HHMMSS.log

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
| 当前复现 accuracy | 74.35 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，11_3 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 6. 当前结果表

| 实验编号 | Dataset | File | Method | Accuracy | Status |
|---|---|---|---|---:|---|
| 11_3_ulip2_modelnet_clean_zs_global_local | modelnet_c | data/modelnet_c/clean.h5 | Zero-shot + Global Cache + Local Cache | 74.35 | done |

该结果表示：ULIP-2 在 ModelNet clean 数据上使用完整 Point-Cache 后，准确率为 74.35。

---

## 7. 与原文结果对比

原文 Point-Cache Table 1 中，ULIP-2 在 ModelNet clean 上使用 +Hierarchical Cache 的结果为 74.53。

当前复现结果为 74.35。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Table 1 ULIP-2 + Hierarchical Cache / ModelNet clean | 74.53 | 74.35 | -0.18 | 0.18 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与 Table 1 的差异 | -0.18 |
| 绝对差异 | 0.18 |

分析：

当前复现结果 74.35 与原文 74.53 相差 -0.18，误差很小，可以认为 11_3 的 ULIP-2 + Global Cache + Local Cache 结果与原文基本对齐。

因此，11_3 不只是脚本执行成功，而且数值也与原文 ModelNet clean / ULIP-2 + Hierarchical Cache 结果基本一致。

---

## 8. 与前序实验 11_1 和 11_2 的对比

11_1 是无缓存 Zero-shot，11_2 是 Zero-shot + Global Cache，11_3 是完整 Point-Cache。

| 实验编号 | 方法 | 当前复现值 |
|---|---|---:|
| 11_1_ulip2_modelnet_clean_zs | Zero-shot | 72.20 |
| 11_2_ulip2_modelnet_clean_zs_global | Zero-shot + Global Cache | 73.99 |
| 11_3_ulip2_modelnet_clean_zs_global_local | Zero-shot + Global Cache + Local Cache | 74.35 |

当前复现增益：

| 比较 | 当前复现增益 | 含义 |
|---|---:|---|
| 11_2 - 11_1 | +1.79 | Global Cache 相比 Zero-shot 的提升 |
| 11_3 - 11_2 | +0.36 | Local Cache 在 Global Cache 基础上的额外提升 |
| 11_3 - 11_1 | +2.15 | 完整 Point-Cache 相比 Zero-shot 的总体提升 |

原文对应增益：

| 比较 | 原文增益 |
|---|---:|
| Global Cache - Zero-shot | 73.95 - 71.23 = +2.72 |
| Hierarchical Cache - Global Cache | 74.53 - 73.95 = +0.58 |
| Hierarchical Cache - Zero-shot | 74.53 - 71.23 = +3.30 |

增益对齐情况：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 增益 | +2.72 | +1.79 | -0.93 |
| Local Cache 额外增益 | +0.58 | +0.36 | -0.22 |
| 完整 Point-Cache 总增益 | +3.30 | +2.15 | -1.15 |

分析：

当前复现中，完整 Point-Cache 将准确率从 11_1 的 72.20 提升到 11_3 的 74.35，总提升为 +2.15。方法趋势正确：

Zero-shot < ZS + Global Cache < ZS + Global Cache + Local Cache

但是，当前复现的总增益小于原文。主要原因是 11_1 Zero-shot 当前复现值 72.20 高于原文 71.23，导致以当前 11_1 为基线计算的 cache 增益被压缩。

从绝对值看，11_2 和 11_3 都与原文非常接近，因此当前 11 组的主要问题不是 cache 方法结果偏低，而是 11_1 Zero-shot 略高。

---

## 9. 结果含义分析

ULIP-2 在 ModelNet clean 上的基础性能较强，11_1 Zero-shot 已经达到 72.20。因此，Global Cache 和 Local Cache 的提升空间相对有限。

11_2 加入 Global Cache 后提升到 73.99，11_3 进一步加入 Local Cache 后提升到 74.35。这说明在 ModelNet clean 上，完整 Point-Cache 仍然有效，但增益幅度不大。

| 观察 | 含义 |
|---|---|
| 11_3 = 74.35 | 与原文 +Hierarchical Cache 结果基本一致 |
| 11_3 高于 11_2 | Local Cache 有额外正增益 |
| 11_3 高于 11_1 | 完整 Point-Cache 有效 |
| Local Cache 额外提升 +0.36 | clean synthetic setting 上局部缓存贡献较小 |
| 当前总增益低于原文 | 主要由 11_1 当前复现值偏高导致 |

这说明在 ULIP-2 × ModelNet clean 上，Point-Cache 的作用是正向的，但由于 clean 数据和强 backbone 已经带来较高起点，cache 模块的边际收益较小。

---

## 10. Local Cache 贡献分析

本实验中，Local Cache 的额外增益为：

| 比较 | 数值 |
|---|---:|
| 11_3 - 11_2 | +0.36 |

原文中的 Local Cache 额外增益为：

| 比较 | 数值 |
|---|---:|
| Hierarchical Cache - Global Cache | 74.53 - 73.95 = +0.58 |

当前 Local Cache 额外增益比原文低 0.22，但仍然是正向提升。

可能原因包括：

| 原因 | 解释 |
|---|---|
| ModelNet clean 较规则 | 局部结构本身比较完整，Global Cache 已经能捕获大部分信息 |
| ULIP-2 backbone 较强 | Zero-shot 和 Global Cache 基础性能较高，进一步提升空间有限 |
| clean setting 无 corruption | 局部缓存对局部破坏的补偿作用不如 corrupted setting 明显 |
| 当前 11_1 偏高 | 绝对 baseline 较高会压缩后续整体增益解释空间 |

因此，Local Cache 在 11_3 中表现为“小幅但正向的补充增益”。

---

## 11. 与后续实验的关系

11_3 是 11 组的最后一个子实验，因此它本身可以作为后续 11 组 summary 文档的输入。

本文件只记录 11_3 自身及其与前序子实验的关系。完整 11 组总结应在单独 summary 文档中完成，包括：

| 后续 summary 应包含的内容 | 说明 |
|---|---|
| 11_1 / 11_2 / 11_3 总表 | 横向比较三种方法 |
| 与原文三种方法对齐 | 分别比较 71.23、73.95、74.53 |
| Global Cache 贡献 | 分析 11_2 - 11_1 |
| Local Cache 贡献 | 分析 11_3 - 11_2 |
| 11 组阶段性结论 | 总结 ULIP-2 在 ModelNet clean 上的复现状态 |

---

## 12. 阶段性结论

本实验完成了 ULIP-2 × ModelNet clean 的 Zero-shot + Global Cache + Local Cache baseline 复现。

主要结论如下：

1. 11_3 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 74.35。
3. 原文 Point-Cache Table 1 中 ULIP-2 在 ModelNet clean 上的 +Hierarchical Cache 结果为 74.53。
4. 当前复现结果与原文差异为 -0.18，可以认为 11_3 基本对齐原文。
5. 相比 11_1 Zero-shot 的 72.20，11_3 提升 +2.15。
6. 相比 11_2 Global Cache 的 73.99，11_3 额外提升 +0.36。
7. 当前方法趋势正确：Zero-shot < Global Cache < Global + Local Cache。
8. Local Cache 在 ModelNet clean 上有小幅正增益，但不是主要提升来源。
9. 本实验可作为 11 组 summary 文档和后续 12 组 ULIP-2 × ModelNet-C all35 实验的 clean 参考结果。

---

## 13. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/11_3_ulip2_modelnet_clean_zs_global_local_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/11_3_ulip2_modelnet_clean_zs_global_local_single_gpu.sh 1

---

## 14. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/11_3_ulip2_modelnet_clean_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/11_3_ulip2_modelnet_clean_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/11_3_ulip2_modelnet_clean_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l

cat results/baseline/11_3_ulip2_modelnet_clean_zs_global_local/summary.csv
