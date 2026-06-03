# 11_2_ulip2_modelnet_clean_zs_global

## 1. 实验目的

本实验用于复现 ULIP-2 在 ModelNet clean 上使用 Zero-shot + Global Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 11_2_ulip2_modelnet_clean_zs_global |
| Backbone | ULIP-2 |
| Dataset | ModelNet clean |
| Method | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |

本实验是在 11_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存是否能够提升 ULIP-2 在 ModelNet clean 上的识别性能。

本文件只记录 11_2 本身，并与前序子实验 11_1 进行对比。完整 11 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 11 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP-2 |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 实际数据文件 | data/modelnet_c/clean.h5 |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 原始核心 runner | runners/model_with_global_cache.py |
| 方法脚本 | Point-Cache/scripts/baseline/11_2_ulip2_modelnet_clean_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/11_run_ulip2_modelnet_clean_common.sh |
| cache_type | global |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |
| 输入点数 | 1024 |
| Global Cache shot_capacity | 3 |
| Global Cache alpha | 4.0 |
| Global Cache beta | 3.0 |
| Backbone 权重 | weights/ulip/pointbert_ulip2.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| ULIP version | ulip2 |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 1 |

本实验使用 `modelnet_c` 作为 dataset 参数，并指定 `cor_type=clean`。实际读取文件为：

data/modelnet_c/clean.h5

虽然数据集参数仍然写作 `modelnet_c`，但 `clean.h5` 对应的是 ModelNet clean 测试设置，不是 corrupted setting。

---

## 3. 方法说明

11_2 与 11_1 的区别在于：11_1 只使用 zero-shot logits，而 11_2 在此基础上加入 Global Cache。

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 11_1 | 是 | 否 | 否 |
| 11_2 | 是 | 是 | 否 |

Global Cache 的基本作用是：在测试时动态缓存高置信度样本的全局点云特征及其伪标签，再用后续测试样本与缓存特征之间的相似度生成 cache logits，并与 zero-shot logits 融合。

因此，本实验重点观察：

| 观察点 | 说明 |
|---|---|
| 11_2 是否高于 11_1 | 判断 Global Cache 是否带来有效增益 |
| 11_2 是否接近原文值 | 判断复现是否与 Point-Cache 原始结果对齐 |
| 增益幅度是否合理 | 判断 Global Cache 在当前复现环境下的实际作用 |

---

## 4. 输出结构

输出目录：

Point-Cache/results/baseline/11_2_ulip2_modelnet_clean_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

11_2_ulip2_modelnet_clean_zs_global_clean_YYYYMMDD_HHMMSS.log

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
| 当前复现 accuracy | 73.99 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，11_2 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 6. 当前结果表

| 实验编号 | Dataset | File | Method | Accuracy | Status |
|---|---|---|---|---:|---|
| 11_2_ulip2_modelnet_clean_zs_global | modelnet_c | data/modelnet_c/clean.h5 | Zero-shot + Global Cache | 73.99 | done |

该结果表示：ULIP-2 在 ModelNet clean 数据上加入 Global Cache 后，准确率为 73.99。

---

## 7. 与原文结果对比

原文 Point-Cache Table 1 中，ULIP-2 在 ModelNet clean 上使用 +Global Cache 的结果为 73.95。

当前复现结果为 73.99。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Table 1 ULIP-2 + Global Cache / ModelNet clean | 73.95 | 73.99 | +0.04 | 0.04 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与 Table 1 的差异 | +0.04 |
| 绝对差异 | 0.04 |

分析：

当前复现结果 73.99 与原文 73.95 几乎完全一致，差异仅 +0.04。可以认为 11_2 的 ULIP-2 + Global Cache 结果与原文高度对齐。

因此，11_2 不只是脚本执行成功，而且数值也与原文 ModelNet clean / ULIP-2 + Global Cache 结果高度一致。

---

## 8. 与前序实验 11_1 的对比

11_1 是本实验的直接前序子实验，方法为 Zero-shot，不使用缓存。

| 实验编号 | 方法 | 当前复现值 |
|---|---|---:|
| 11_1_ulip2_modelnet_clean_zs | Zero-shot | 72.20 |
| 11_2_ulip2_modelnet_clean_zs_global | Zero-shot + Global Cache | 73.99 |

Global Cache 带来的当前复现增益为：

| 比较 | 当前复现增益 |
|---|---:|
| 11_2 - 11_1 | +1.79 |

原文中对应的增益为：

| 比较 | 原文增益 |
|---|---:|
| Global Cache - Zero-shot | 73.95 - 71.23 = +2.72 |

对比：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 增益 | +2.72 | +1.79 | -0.93 |

分析：

当前复现中，Global Cache 将准确率从 72.20 提升到 73.99，提升 +1.79，说明 Global Cache 在当前复现环境下仍然有效。

不过，当前 Global Cache 增益小于原文的 +2.72。主要原因不是 11_2 偏低，而是 11_1 Zero-shot 当前复现值 72.20 比原文 71.23 高 +0.97，导致以当前 11_1 为基线计算的 cache 增益被压缩。

因此，11_2 的分析需要分开看：

| 分析角度 | 结论 |
|---|---|
| 绝对数值对齐 | 11_2 = 73.99，与原文 73.95 高度一致 |
| 相对 11_1 的增益 | 当前提升 +1.79，低于原文增益 +2.72 |
| 原因解释 | 主要是当前 11_1 Zero-shot 略高于原文 |

---

## 9. 结果含义分析

ULIP-2 在 ModelNet clean 上的 Zero-shot 基础性能已经较高，11_1 达到 72.20。因此，在 clean synthetic setting 上，Global Cache 的可提升空间相对有限。

11_2 加入 Global Cache 后达到 73.99，说明测试时全局缓存仍然能够带来额外提升。

| 观察 | 含义 |
|---|---|
| 11_2 = 73.99 | 与原文 +Global Cache 结果高度一致 |
| 11_2 高于 11_1 | Global Cache 在当前实验中有效 |
| 当前增益 +1.79 | 增益为正，但小于原文 |
| 11_1 高于原文 | 当前 cache 增益被高 baseline 压缩 |

这说明在 ULIP-2 × ModelNet clean 上，Global Cache 的效果是正向的，但由于 Zero-shot 已经较强，绝对提升不像 corrupted setting 中那么大。

---

## 10. 与后续子实验的关系

11_2 是 11_3 的直接前序实验。

| 后续实验 | 对比方式 |
|---|---|
| 11_3_ulip2_modelnet_clean_zs_global_local | 与 11_2 比较，评估 Local Cache 在 Global Cache 基础上的额外增益 |

本文件不展开 11_3 的实际结果。11_3 的数值及 Local Cache 额外贡献应记录在 11_3 子实验文档和 11 组 summary 文档中。

---

## 11. 阶段性结论

本实验完成了 ULIP-2 × ModelNet clean 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 11_2 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 73.99。
3. 原文 Point-Cache Table 1 中 ULIP-2 在 ModelNet clean 上的 +Global Cache 结果为 73.95。
4. 当前复现结果与原文差异仅 +0.04，可以认为 11_2 高度对齐原文。
5. 相比 11_1 Zero-shot 的 72.20，11_2 提升到 73.99，当前增益为 +1.79。
6. 当前 Global Cache 增益低于原文 +2.72，主要原因是 11_1 当前复现值高于原文。
7. 因此，11_2 的绝对结果可靠，且 Global Cache 在当前复现环境下仍然有效。
8. 本实验是 11_3 分析 Local Cache 额外贡献的直接对照。

---

## 12. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/11_2_ulip2_modelnet_clean_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/11_2_ulip2_modelnet_clean_zs_global_single_gpu.sh 1

---

## 13. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/11_2_ulip2_modelnet_clean_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/11_2_ulip2_modelnet_clean_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/11_2_ulip2_modelnet_clean_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

cat results/baseline/11_2_ulip2_modelnet_clean_zs_global/summary.csv
