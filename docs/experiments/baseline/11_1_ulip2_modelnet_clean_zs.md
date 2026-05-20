# 11_1_ulip2_modelnet_clean_zs

## 1. 实验目的

本实验用于复现 ULIP-2 在 ModelNet clean 上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 11_1_ulip2_modelnet_clean_zs |
| Backbone | ULIP-2 |
| Dataset | ModelNet clean |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 ULIP-2 在 ModelNet clean 数据上的无缓存基础性能。该结果后续会作为 11_2 和 11_3 的对照基线，但本文件只记录 11_1 本身，不展开整个 11 组的综合分析。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP-2 |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 实际数据文件 | data/modelnet_c/clean.h5 |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 方法脚本 | Point-Cache/scripts/baseline/11_1_ulip2_modelnet_clean_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/11_run_ulip2_modelnet_clean_common.sh |
| 输入点数 | 1024 |
| Backbone 权重 | weights/ulip/pointbert_ulip2.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| ULIP version | ulip2 |
| GPU | 单张 Tesla T4 |

本实验使用 `modelnet_c` 作为 dataset 参数，并指定 `cor_type=clean`。实际读取文件为：

data/modelnet_c/clean.h5

虽然数据集参数仍然写作 `modelnet_c`，但 `clean.h5` 对应的是 ModelNet clean 测试设置，不是 corrupted setting。

---

## 3. 输出结构

输出目录：

Point-Cache/results/baseline/11_1_ulip2_modelnet_clean_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

11_1_ulip2_modelnet_clean_zs_clean_YYYYMMDD_HHMMSS.log

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
| 当前复现 accuracy | 72.20 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，11_1 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 5. 当前结果表

| 实验编号 | Dataset | File | Method | Accuracy | Status |
|---|---|---|---|---:|---|
| 11_1_ulip2_modelnet_clean_zs | modelnet_c | data/modelnet_c/clean.h5 | Zero-shot | 72.20 | done |

该结果表示：ULIP-2 在 ModelNet clean 数据上的 Zero-shot 准确率为 72.20。

---

## 6. 与原文结果对比

原文 Point-Cache Table 1 中，ULIP-2 在 ModelNet clean 上的 Zero-shot 结果为 71.23。

当前复现结果为 72.20。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Table 1 ULIP-2 Zero-shot / ModelNet clean | 71.23 | 72.20 | +0.97 | 0.97 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与 Table 1 的差异 | +0.97 |
| 绝对差异 | 0.97 |

分析：

当前复现结果 72.20 比原文 71.23 高 +0.97。该差异略高于 01 组、03 组中常见的 0.1–0.3 左右误差，但仍在可接受范围内。

更严谨地说：11_1 脚本执行成功；从数值上看，当前 ULIP-2 Zero-shot 结果略高于原文，但没有出现明显异常，可以作为 11_2 和 11_3 的基础对照。

需要注意的是，因为 11_1 的复现值高于原文接近 1 个百分点，后续用当前结果计算 cache 增益时，得到的增益可能会小于原文增益。因此后续分析应同时看两个角度：

| 分析角度 | 说明 |
|---|---|
| 绝对值是否接近原文 | 判断每个方法是否复现成功 |
| 当前方法间增益是否合理 | 判断 Global / Local Cache 在当前复现环境下是否有效 |

---

## 7. 结果含义分析

ULIP-2 在 ModelNet clean 上的 Zero-shot 准确率为 72.20，明显高于此前 ULIP 在同一设置下的结果。这符合预期，因为 ULIP-2 是更强的 backbone。

ModelNet clean 是 CAD / synthetic object 数据，相比 ScanObjNN clean hardest 更规则、更干净，因此 ULIP-2 在该数据上可以取得较高的 Zero-shot 准确率。

| 观察 | 含义 |
|---|---|
| Zero-shot = 72.20 | ULIP-2 在 ModelNet clean 上已有较强基础性能 |
| 当前值高于原文 +0.97 | 结果略高，但仍可接受 |
| clean setting 只含 1 个测试文件 | 不涉及 corruption × severity 分析 |
| 后续 cache 增益可能较小 | 因为 Zero-shot 基础性能已经较高 |

---

## 8. 与前序子实验的关系

在 11 组内部，11_1 是第一个子实验，因此没有前序 11 组子实验可供对比。

它后续将作为以下实验的基线：

| 后续实验 | 对比方式 |
|---|---|
| 11_2_ulip2_modelnet_clean_zs_global | 与 11_1 比较，评估 Global Cache 增益 |
| 11_3_ulip2_modelnet_clean_zs_global_local | 与 11_1 和 11_2 比较，评估完整 Point-Cache 及 Local Cache 额外增益 |

具体方法间增益不在本文件展开，应放在 11_2、11_3 子实验文档及 11 组 summary 文档中。

---

## 9. 阶段性结论

本实验完成了 ULIP-2 × ModelNet clean 的 Zero-shot baseline 复现。

主要结论如下：

1. 11_1 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 72.20。
3. 原文 Point-Cache Table 1 中 ULIP-2 在 ModelNet clean 上的 Zero-shot 结果为 71.23。
4. 当前复现结果比原文高 +0.97，略高但仍可接受。
5. 该实验是 11_2 Global Cache 和 11_3 Hierarchical Cache 的基础对照。
6. 因为当前 Zero-shot 结果略高于原文，后续分析 cache 增益时需要同时关注绝对值对齐和方法间相对提升。

---

## 10. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/11_1_ulip2_modelnet_clean_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/11_1_ulip2_modelnet_clean_zs_single_gpu.sh 1

---

## 11. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/11_1_ulip2_modelnet_clean_zs/summary.csv | wc -l

tail -n +2 results/baseline/11_1_ulip2_modelnet_clean_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/11_1_ulip2_modelnet_clean_zs/logs -maxdepth 1 -name '*.log' | wc -l

cat results/baseline/11_1_ulip2_modelnet_clean_zs/summary.csv
