# 13_2_ulip2_scanobjnn_clean_hardest_zs_global

## 1. 实验目的

本实验用于复现 ULIP-2 在 ScanObjNN clean hardest 上使用 Zero-shot + Global Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 13_2_ulip2_scanobjnn_clean_hardest_zs_global |
| Backbone | ULIP-2 |
| Dataset | ScanObjNN clean hardest |
| Method | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |

本实验是在 13_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存是否能够提升 ULIP-2 在 ScanObjNN clean hardest 上的识别性能。

本文件只记录 13_2 本身，并与前序子实验 13_1 进行对比。完整 13 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 13 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP-2 |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 数据目录 | data/sonn_c |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 原始核心 runner | runners/model_with_global_cache.py |
| 方法脚本 | Point-Cache/scripts/baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/13_run_ulip2_scanobjnn_clean_hardest_common.sh |
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

本实验使用 `sonn_c` 作为 dataset 参数，`sonn_variant=hardest`，并指定 `cor_type=clean`。实际读取文件为：

data/sonn_c/hardest/clean.h5

---

## 3. 方法说明

13_2 与 13_1 的区别在于：13_1 只使用 zero-shot logits，而 13_2 在此基础上加入 Global Cache。

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 13_1 | 是 | 否 | 否 |
| 13_2 | 是 | 是 | 否 |

Global Cache 的基本作用是：在测试时动态缓存高置信度样本的全局点云特征及其伪标签，再用后续测试样本与缓存特征之间的相似度生成 cache logits，并与 zero-shot logits 融合。

因此，本实验重点观察：

| 观察点 | 说明 |
|---|---|
| 13_2 是否高于 13_1 | 判断 Global Cache 是否带来有效增益 |
| 13_2 是否接近原文值 | 判断复现是否与 Point-Cache 原始结果对齐 |
| 增益幅度是否合理 | 判断 Global Cache 在真实扫描 clean hardest 上的实际作用 |

---

## 4. 输出结构

输出目录：

Point-Cache/results/baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

13_2_ulip2_scanobjnn_clean_hardest_zs_global_clean_YYYYMMDD_HHMMSS.log

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
| 当前复现 accuracy | 40.42 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，13_2 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 6. 当前结果表

| 实验编号 | Dataset | Variant | File | Method | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 13_2_ulip2_scanobjnn_clean_hardest_zs_global | sonn_c | hardest | data/sonn_c/hardest/clean.h5 | Zero-shot + Global Cache | 40.42 | done |

该结果表示：ULIP-2 在 ScanObjNN hardest clean 数据上加入 Global Cache 后，准确率为 40.42。

---

## 7. 与原文结果对比

原文 Point-Cache Supplementary Table 7 中，S-PB T50-RS-C 是 ScanObjectNN hardest split；其中 Original Data / SONN 对应 clean hardest。ULIP-2 在该设置下使用 +Global Cache 的结果为 40.28。

当前复现结果为 40.42。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Supplementary Table 7 ULIP-2 + Global Cache / SONN Original Data | 40.28 | 40.42 | +0.14 | 0.14 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与 Supplementary Table 7 的差异 | +0.14 |
| 绝对差异 | 0.14 |

分析：

当前复现结果 40.42 与原文 40.28 相差 +0.14，误差很小。可以认为 13_2 的 ULIP-2 + Global Cache 结果与原文高度对齐。

因此，13_2 不只是脚本执行成功，而且数值也与原文 ScanObjNN clean hardest / ULIP-2 + Global Cache 结果基本一致。

---

## 8. 与前序实验 13_1 的对比

13_1 是本实验的直接前序子实验，方法为 Zero-shot，不使用缓存。

| 实验编号 | 方法 | 当前复现值 |
|---|---|---:|
| 13_1_ulip2_scanobjnn_clean_hardest_zs | Zero-shot | 34.07 |
| 13_2_ulip2_scanobjnn_clean_hardest_zs_global | Zero-shot + Global Cache | 40.42 |

Global Cache 带来的当前复现增益为：

| 比较 | 当前复现增益 |
|---|---:|
| 13_2 - 13_1 | +6.35 |

原文中对应的增益为：

| 比较 | 原文增益 |
|---|---:|
| Global Cache - Zero-shot | 40.28 - 33.38 = +6.90 |

对比：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 增益 | +6.90 | +6.35 | -0.55 |

分析：

当前复现中，Global Cache 将准确率从 34.07 提升到 40.42，提升 +6.35。该增益非常明显，说明 Global Cache 在 ScanObjNN clean hardest 上有效。

当前 Global Cache 增益略低于原文的 +6.90，主要原因是 13_1 Zero-shot 当前复现值 34.07 比原文 33.38 高 +0.69，导致以当前 13_1 为基线计算的 cache 增益略被压缩。

因此，13_2 的分析需要分开看：

| 分析角度 | 结论 |
|---|---|
| 绝对数值对齐 | 13_2 = 40.42，与原文 40.28 高度一致 |
| 相对 13_1 的增益 | 当前提升 +6.35，略低于原文增益 +6.90 |
| 原因解释 | 主要是当前 13_1 Zero-shot 略高于原文 |
| 方法有效性 | Global Cache 在当前复现环境下仍然带来明显提升 |

---

## 9. 结果含义分析

ScanObjNN clean hardest 是真实扫描数据，即使没有 synthetic corruption，也存在明显 domain gap。13_1 的 Zero-shot 准确率为 34.07，说明 ULIP-2 原始 zero-shot 在该数据上较困难。

13_2 加入 Global Cache 后提升到 40.42，说明测试流中的全局特征缓存可以利用在线测试数据分布，缓解真实扫描域偏移。

| 观察 | 含义 |
|---|---|
| 13_2 = 40.42 | 与原文 +Global Cache 结果高度一致 |
| 13_2 高于 13_1 | Global Cache 在当前实验中有效 |
| 当前增益 +6.35 | 真实扫描 clean hardest 上 Global Cache 贡献很明显 |
| 13_1 略高于原文 | 当前增益略低于原文，但不影响 13_2 绝对值对齐 |
| 13_2 仍远低于 ModelNet clean | 真实扫描 hardest split 仍然非常困难 |

这说明在 ULIP-2 × ScanObjNN clean hardest 上，Global Cache 的效果比在 ModelNet clean 上更明显。原因是 ScanObjNN clean hardest 存在更强的真实域偏移，测试时缓存有更大补偿空间。

---

## 10. 与 11_2 ModelNet clean 的关系

11_2 是 ULIP-2 在 ModelNet clean 上的 Zero-shot + Global Cache 结果；13_2 是 ULIP-2 在 ScanObjNN clean hardest 上的 Zero-shot + Global Cache 结果。

| 实验编号 | Dataset setting | Method | Accuracy |
|---|---|---|---:|
| 11_2_ulip2_modelnet_clean_zs_global | ModelNet clean | Zero-shot + Global Cache | 73.99 |
| 13_2_ulip2_scanobjnn_clean_hardest_zs_global | ScanObjNN clean hardest | Zero-shot + Global Cache | 40.42 |

对比：

| 比较 | 变化 |
|---|---:|
| 13_2 - 11_2 | -33.57 |

分析：

即使加入 Global Cache，ScanObjNN clean hardest 仍然比 ModelNet clean 低 33.57 个百分点。这说明真实扫描 hardest split 对 ULIP-2 是非常强的 domain shift。

不过，与 Zero-shot 的跨数据下降相比，Global Cache 缓解了一部分 gap：

| 方法 | ModelNet clean | ScanObjNN clean hardest | 下降 |
|---|---:|---:|---:|
| Zero-shot | 72.20 | 34.07 | -38.13 |
| ZS + Global | 73.99 | 40.42 | -33.57 |

Global Cache 使 clean ModelNet 到 ScanObjNN hardest 的性能差距从 -38.13 缩小到 -33.57，说明缓存机制对真实扫描域偏移有缓解作用。

---

## 11. 与后续子实验的关系

13_2 是 13_3 的直接前序实验。

| 后续实验 | 对比方式 |
|---|---|
| 13_3_ulip2_scanobjnn_clean_hardest_zs_global_local | 与 13_2 比较，评估 Local Cache 在 Global Cache 基础上的额外增益 |

本文件不展开 13_3 的实际结果。13_3 的数值及 Local Cache 额外贡献应记录在 13_3 子实验文档和 13 组 summary 文档中。

---

## 12. 阶段性结论

本实验完成了 ULIP-2 × ScanObjNN clean hardest 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 13_2 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 40.42。
3. 原文 Point-Cache Supplementary Table 7 中 ULIP-2 在 S-PB T50-RS-C Original Data / SONN 上的 +Global Cache 结果为 40.28。
4. 当前复现结果与原文差异仅 +0.14，可以认为 13_2 高度对齐原文。
5. 相比 13_1 Zero-shot 的 34.07，13_2 提升到 40.42，当前增益为 +6.35。
6. 当前 Global Cache 增益略低于原文 +6.90，主要原因是 13_1 当前复现值高于原文。
7. Global Cache 在 ScanObjNN clean hardest 上带来明显提升，说明测试时全局缓存能有效缓解真实扫描域偏移。
8. 本实验是 13_3 分析 Local Cache 额外贡献的直接对照。

---

## 13. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global_single_gpu.sh 1

---

## 14. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

cat results/baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global/summary.csv
