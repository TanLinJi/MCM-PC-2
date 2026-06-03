# 21_1_openshape_modelnet_clean_zs

## 1. 实验目的

本实验用于复现 OpenShape 在 ModelNet clean 上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 21_1_openshape_modelnet_clean_zs |
| Backbone | OpenShape |
| Dataset | ModelNet clean |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 OpenShape 在 ModelNet clean 上的无缓存基础性能。该结果后续会作为 21_2 和 21_3 的对照基线，但本文件只记录 21_1 本身，不展开整个 21 组的综合分析。

需要特别注意：虽然公共脚本日志中会显示 `Cache type: global`，但 21_1 实际调用的 runner 是 `runners/zs_infer.py`，因此该实验仍然是 Zero-shot，无 Global Cache，也无 Local Cache。这里的 `cache_type=global` 只是为了统一公共脚本参数格式而传入的占位参数，不参与 Zero-shot 推理。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | OpenShape |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 实际数据文件 | data/modelnet_c/clean.h5 |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 方法脚本 | Point-Cache/scripts/baseline/21_1_openshape_modelnet_clean_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/21_run_openshape_modelnet_clean_common.sh |
| 输入点数 | 1024 |
| OpenShape version | vitg14 |
| OpenShape 权重 | weights/openshape/openshape-pointbert-vitg14-rgb/model.pt |
| OpenCLIP 权重 | weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 0 |

本实验使用 `modelnet_c` 作为 dataset 参数，并指定 `cor_type=clean`。实际读取文件为：

data/modelnet_c/clean.h5

---

## 3. 输出结构

输出目录：

Point-Cache/results/baseline/21_1_openshape_modelnet_clean_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

21_1_openshape_modelnet_clean_zs_clean_YYYYMMDD_HHMMSS.log

因为本实验只测试 `clean.h5`，所以期望只有 1 行 summary 记录和 1 个对应 log。

---

## 4. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 1 | 1 | clean 实验只包含 1 个测试文件 |
| summary 中唯一 cor_type 数 | 1 | 1 | 只测试 clean |
| summary 中唯一 log_path 数 | 1 | 1 | 每次运行对应 1 个 log |
| logs 目录 .log 文件数 | 1 | 1 | 没有旧日志或重复日志残留 |
| status=done 数 | 1 | 1 | 脚本执行成功 |
| 当前复现 accuracy | 84.72 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，21_1 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 5. 当前结果表

| 实验编号 | Dataset | File | Method | Accuracy | Status |
|---|---|---|---|---:|---|
| 21_1_openshape_modelnet_clean_zs | modelnet_c | data/modelnet_c/clean.h5 | Zero-shot | 84.72 | done |

该结果表示：OpenShape 在 ModelNet clean 数据上的 Zero-shot 准确率为 84.72。

---

## 6. 与原文结果对比

原文 Point-Cache Table 1 中，OpenShape / O-Shape 在 ModelNet clean 上的 Zero-shot 结果为 84.56。

当前复现结果为 84.72。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache Table 1 OpenShape / ModelNet clean / Zero-shot | 84.56 | 84.72 | +0.16 | 0.16 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与原文差异 | +0.16 |
| 绝对差异 | 0.16 |

分析：

当前复现结果 84.72 比原文 84.56 高 +0.16，差异很小。可以认为 21_1 的 OpenShape × ModelNet clean Zero-shot 结果与原文高度对齐。

因此，21_1 不只是脚本执行成功，而且数值也与 Point-Cache Table 1 的 OpenShape clean 结果基本一致。

---

## 7. 关于 Cache type 显示的说明

在运行日志中，公共脚本会打印：

Cache type: global

这并不表示 21_1 使用了 Global Cache。

21_1 实际调用关系如下：

| 项目 | 实际值 |
|---|---|
| Method | Zero-shot |
| Runner | runners/zs_infer.py |
| METHOD_FULL | Zero-shot |
| CACHE_TYPE | global，占位参数 |

关键判断依据是 runner。21_1 调用的是 `runners/zs_infer.py`，该 runner 执行 zero-shot inference，不会构建 positive cache、global cache 或 local cache，也不会进行 cache logits 融合。

因此，本实验仍然应记录为：

| 是否使用 Global Cache | 是否使用 Local Cache |
|---:|---:|
| 否 | 否 |

后续如果希望避免误解，可以把公共脚本中的显示文字从 `Cache type` 改为 `Cache type arg`，并额外提示 `ignored for Zero-shot runner`。但这不是影响实验结果的问题。

---

## 8. 结果含义分析

OpenShape 在 ModelNet clean 上的 Zero-shot 准确率达到 84.72，明显高于此前 ULIP 和 ULIP-2 在 ModelNet clean 上的 Zero-shot 结果。

| Backbone | 实验编号 | ModelNet clean Zero-shot |
|---|---|---:|
| ULIP | 01_1_ulip_modelnet_clean_zs | 56.16 左右复现水平 |
| ULIP-2 | 11_1_ulip2_modelnet_clean_zs | 72.20 |
| OpenShape | 21_1_openshape_modelnet_clean_zs | 84.72 |

分析：

OpenShape 在 ModelNet clean 上的基础表示能力很强，Zero-shot 本身已经达到较高水平。这意味着在 clean synthetic setting 上，cache 模块的提升空间可能较小，甚至可能因为伪标签缓存扰动导致轻微下降。

这一点与原文现象一致：Point-Cache Table 1 中 OpenShape 在 ModelNet clean 上也是 Zero-shot 略高于 cache 版本。

因此，后续分析 OpenShape 时，不能简单用 “clean 上 cache 是否提升” 来判断实验是否正常。对 OpenShape 更重要的是观察 ModelNet-C、ScanObjNN 和 ScanObjNN-C 上的鲁棒性表现。

---

## 9. 与前序 backbone 的关系

21_1 是 C 组 OpenShape 的第一个实验。它与前面 A 组 ULIP、B 组 ULIP-2 的 clean 结果形成 backbone 对比。

| 组别 | Backbone | 数据设置 | Zero-shot |
|---|---|---|---:|
| A 组 | ULIP | ModelNet clean | 约 56 |
| B 组 | ULIP-2 | ModelNet clean | 72.20 |
| C 组 | OpenShape | ModelNet clean | 84.72 |

观察：

1. OpenShape 在 ModelNet clean 上明显强于 ULIP 和 ULIP-2。
2. OpenShape 的 clean zero-shot 已经很高，因此 cache 边际收益可能较低。
3. 21_1 是后续 22 组 OpenShape × ModelNet-C all35 的 clean 参考。

---

## 10. 与后续子实验的关系

在 21 组内部，21_1 是第一个子实验，因此没有前序 21 组子实验可供对比。

它后续将作为以下实验的基线：

| 后续实验 | 对比方式 |
|---|---|
| 21_2_openshape_modelnet_clean_zs_global | 与 21_1 比较，评估 Global Cache 在 OpenShape clean 上的影响 |
| 21_3_openshape_modelnet_clean_zs_global_local | 与 21_1 和 21_2 比较，评估完整 Point-Cache 及 Local Cache 额外影响 |

需要注意：原文中 OpenShape clean 上 cache 版本略低于 Zero-shot，因此如果后续 21_2、21_3 低于 21_1，不应直接判断为异常，而应优先与原文趋势和数值对齐情况比较。

---

## 11. 阶段性结论

本实验完成了 OpenShape × ModelNet clean 的 Zero-shot baseline 复现。

主要结论如下：

1. 21_1 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 84.72。
3. 原文 Point-Cache Table 1 中 OpenShape / O-Shape 在 ModelNet clean 上的 Zero-shot 结果为 84.56。
4. 当前复现结果比原文高 +0.16，差异很小，可以认为高度对齐。
5. 日志中 `Cache type: global` 是公共脚本占位参数，不表示 21_1 使用了 Global Cache。
6. 21_1 实际调用 `runners/zs_infer.py`，因此应记录为无 cache 的 Zero-shot。
7. OpenShape 在 ModelNet clean 上 Zero-shot 已经很强，后续 cache 在 clean setting 上不一定带来提升。
8. 该实验是 21_2、21_3 以及后续 22 组 ModelNet-C all35 的基础对照。

---

## 12. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/21_1_openshape_modelnet_clean_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/21_1_openshape_modelnet_clean_zs_single_gpu.sh 1

---

## 13. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/21_1_openshape_modelnet_clean_zs/summary.csv | wc -l

tail -n +2 results/baseline/21_1_openshape_modelnet_clean_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/21_1_openshape_modelnet_clean_zs/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/21_1_openshape_modelnet_clean_zs/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/21_1_openshape_modelnet_clean_zs/summary.csv
