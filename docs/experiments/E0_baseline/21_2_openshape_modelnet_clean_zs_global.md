# 21_2_openshape_modelnet_clean_zs_global

## 1. 实验目的

本实验用于复现 OpenShape 在 ModelNet clean 上使用 Zero-shot + Global Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 21_2_openshape_modelnet_clean_zs_global |
| Backbone | OpenShape |
| Dataset | ModelNet clean |
| Method | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |

本实验是在 21_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存对 OpenShape 在 ModelNet clean 上的影响。

需要特别注意：OpenShape 在 ModelNet clean 上的 Zero-shot 已经非常强，原文中 Global Cache 并没有提升 clean accuracy，而是略低于 Zero-shot。因此，本实验不能简单用 “Global Cache 是否提升 clean accuracy” 判断是否正常，而应优先比较当前复现值与原文数值是否对齐。

本文件只记录 21_2 本身，并与前序子实验 21_1 进行对比。完整 21 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 21 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | OpenShape |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 实际数据文件 | data/modelnet_c/clean.h5 |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 原始核心 runner | runners/model_with_global_cache.py |
| 方法脚本 | Point-Cache/scripts/baseline/21_2_openshape_modelnet_clean_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/21_run_openshape_modelnet_clean_common.sh |
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

本实验使用 `modelnet_c` 作为 dataset 参数，并指定 `cor_type=clean`。实际读取文件为：

data/modelnet_c/clean.h5

---

## 3. 方法说明

21_2 与 21_1 的区别在于：21_1 只使用 zero-shot logits，而 21_2 在此基础上加入 Global Cache。

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 21_1 | 是 | 否 | 否 |
| 21_2 | 是 | 是 | 否 |

Global Cache 的基本作用是：在测试时动态缓存高置信度样本的全局点云特征及其伪标签，再用后续测试样本与缓存特征之间的相似度生成 cache logits，并与 zero-shot logits 融合。

在 OpenShape × ModelNet clean 上，由于 OpenShape 的 zero-shot 表征已经很强，Global Cache 不一定带来正增益。原文中 OpenShape clean 的 Global Cache 结果也略低于 Zero-shot，因此本实验重点观察：

| 观察点 | 说明 |
|---|---|
| 21_2 是否接近原文 +Global Cache 值 | 判断复现是否对齐 |
| 21_2 与 21_1 的差值是否符合原文趋势 | 判断 clean 上轻微下降是否正常 |
| OpenShape clean 上 cache 是否存在边际收益不足 | 为后续 ModelNet-C 分析提供背景 |

---

## 4. 输出结构

输出目录：

Point-Cache/results/baseline/21_2_openshape_modelnet_clean_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

21_2_openshape_modelnet_clean_zs_global_clean_YYYYMMDD_HHMMSS.log

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
| 当前复现 accuracy | 84.48 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，21_2 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 6. 当前结果表

| 实验编号 | Dataset | File | Method | Accuracy | Status |
|---|---|---|---|---:|---|
| 21_2_openshape_modelnet_clean_zs_global | modelnet_c | data/modelnet_c/clean.h5 | Zero-shot + Global Cache | 84.48 | done |

该结果表示：OpenShape 在 ModelNet clean 数据上加入 Global Cache 后，准确率为 84.48。

---

## 7. 与原文结果对比

原文 Point-Cache Table 1 中，OpenShape / O-Shape 在 ModelNet clean 上的 +Global Cache 结果为 84.52。

当前复现结果为 84.48。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Table 1 OpenShape / ModelNet clean / +Global Cache | 84.52 | 84.48 | -0.04 | 0.04 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与原文差异 | -0.04 |
| 绝对差异 | 0.04 |

分析：

当前复现结果 84.48 与原文 84.52 只相差 -0.04，几乎完全一致。可以认为 21_2 的 OpenShape × ModelNet clean +Global Cache 结果与原文高度对齐。

因此，21_2 不只是脚本执行成功，而且数值也与 Point-Cache Table 1 的 OpenShape clean +Global Cache 结果基本一致。

---

## 8. 与前序实验 21_1 的对比

21_1 是本实验的直接前序子实验，方法为 Zero-shot，不使用缓存。

| 实验编号 | 方法 | 当前复现值 |
|---|---|---:|
| 21_1_openshape_modelnet_clean_zs | Zero-shot | 84.72 |
| 21_2_openshape_modelnet_clean_zs_global | Zero-shot + Global Cache | 84.48 |

Global Cache 带来的当前复现变化为：

| 比较 | 当前复现变化 |
|---|---:|
| 21_2 - 21_1 | -0.24 |

原文中对应变化为：

| 比较 | 原文变化 |
|---|---:|
| Global Cache - Zero-shot | 84.52 - 84.56 = -0.04 |

对比：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 相对 Zero-shot 的变化 | -0.04 | -0.24 | -0.20 |

分析：

当前复现中，Global Cache 将准确率从 21_1 的 84.72 变为 84.48，下降 -0.24。这个现象不应被判断为异常，因为原文中 OpenShape 在 ModelNet clean 上也是 +Global Cache 略低于 Zero-shot。

当前下降幅度比原文略大，主要原因是 21_1 当前复现值 84.72 比原文 84.56 高 +0.16，而 21_2 当前复现值 84.48 与原文 84.52 几乎一致。因此，以当前 21_1 作为基线计算时，Global Cache 的相对下降会显得更明显。

需要分开看两个角度：

| 分析角度 | 结论 |
|---|---|
| 绝对数值对齐 | 21_2 = 84.48，与原文 84.52 高度一致 |
| 相对 21_1 的变化 | 当前为 -0.24，方向与原文一致 |
| 是否异常 | 不异常，OpenShape clean 上 cache 略降是原文已有现象 |
| 后续重点 | 需要看 22 组 ModelNet-C all35 上 Global Cache 是否提升鲁棒性 |

---

## 9. 为什么 OpenShape clean 上 Global Cache 会略降

OpenShape 在 ModelNet clean 上的 Zero-shot 已经达到 84% 以上，说明其原始文本-点云对齐能力很强。在这种 clean synthetic setting 下，测试时缓存机制的收益空间很小。

Global Cache 依赖在线测试样本的伪标签和全局特征检索。当基础 zero-shot 已经很强时，cache 引入的伪标签不一定能提供额外信息，反而可能带来轻微扰动。

因此，OpenShape clean 上出现：

Zero-shot > ZS + Global Cache

并不意味着 Global Cache 实现错误，而是说明在该 clean setting 下，OpenShape 已经接近较高性能区间，cache 的边际收益不足。

这一点也符合原文现象。原文中 OpenShape clean 的 Zero-shot 为 84.56，而 +Global Cache 为 84.52，同样略低。

---

## 10. 与此前 backbone 的 clean 结果关系

21_2 可以与 ULIP、ULIP-2 的 ModelNet clean +Global Cache 结果进行横向比较。

| Backbone | 实验编号 | ModelNet clean +Global Cache |
|---|---|---:|
| ULIP | 01_2_ulip_modelnet_clean_zs_global | 约 62 |
| ULIP-2 | 11_2_ulip2_modelnet_clean_zs_global | 73.99 |
| OpenShape | 21_2_openshape_modelnet_clean_zs_global | 84.48 |

观察：

1. OpenShape +Global Cache 的绝对准确率显著高于 ULIP 和 ULIP-2。
2. OpenShape 的 clean zero-shot 已经很强，因此 Global Cache 在 clean 上没有表现出正向增益。
3. 这说明 backbone 表征能力越强，clean setting 上 cache 的边际收益可能越低。
4. 后续更关键的是观察 corrupted setting，即 22 组 ModelNet-C all35。

---

## 11. 与后续子实验的关系

21_2 是 21_3 的直接前序实验。

| 后续实验 | 对比方式 |
|---|---|
| 21_3_openshape_modelnet_clean_zs_global_local | 与 21_2 比较，评估 Local Cache 在 Global Cache 基础上的额外影响 |

本文件不展开 21_3 的实际结果。21_3 的数值及 Local Cache 额外影响应记录在 21_3 子实验文档和 21 组 summary 文档中。

需要注意的是，原文中 OpenShape clean 上 +Hierarchical Cache 也低于 +Global Cache。因此，如果 21_3 继续低于 21_2，也不应直接判断为异常，而应继续和原文数值及趋势对齐。

---

## 12. 阶段性结论

本实验完成了 OpenShape × ModelNet clean 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 21_2 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 84.48。
3. 原文 Point-Cache Table 1 中 OpenShape / O-Shape 在 ModelNet clean 上的 +Global Cache 结果为 84.52。
4. 当前复现结果与原文差异仅 -0.04，可以认为高度对齐。
5. 相比 21_1 Zero-shot 的 84.72，21_2 下降到 84.48，当前变化为 -0.24。
6. OpenShape 在 ModelNet clean 上 Global Cache 略低于 Zero-shot 是原文中已有现象，不是实验异常。
7. 当前下降幅度略大于原文，主要是因为 21_1 当前复现值略高于原文。
8. 该实验是 21_3 分析 Local Cache 额外影响的直接对照，也是后续 22 组 OpenShape × ModelNet-C all35 的 clean 参考之一。

---

## 13. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/21_2_openshape_modelnet_clean_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/21_2_openshape_modelnet_clean_zs_global_single_gpu.sh 1

---

## 14. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/21_2_openshape_modelnet_clean_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/21_2_openshape_modelnet_clean_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/21_2_openshape_modelnet_clean_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/21_2_openshape_modelnet_clean_zs_global/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/21_2_openshape_modelnet_clean_zs_global/summary.csv
