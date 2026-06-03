# 13_1_ulip2_scanobjnn_clean_hardest_zs

## 1. 实验目的

本实验用于复现 ULIP-2 在 ScanObjNN clean hardest 上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 13_1_ulip2_scanobjnn_clean_hardest_zs |
| Backbone | ULIP-2 |
| Dataset | ScanObjNN clean hardest |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 ULIP-2 在真实扫描点云 clean hardest split 上的无缓存基础性能。该结果后续会作为 13_2 和 13_3 的对照基线，但本文件只记录 13_1 本身，不展开整个 13 组的综合分析。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP-2 |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 数据目录 | data/sonn_c |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 方法脚本 | Point-Cache/scripts/baseline/13_1_ulip2_scanobjnn_clean_hardest_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/13_run_ulip2_scanobjnn_clean_hardest_common.sh |
| 输入点数 | 1024 |
| Backbone 权重 | weights/ulip/pointbert_ulip2.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| ULIP version | ulip2 |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 0 |

本实验使用 `sonn_c` 作为 dataset 参数，`sonn_variant=hardest`，并指定 `cor_type=clean`。实际读取文件为：

data/sonn_c/hardest/clean.h5

---

## 3. 输出结构

输出目录：

Point-Cache/results/baseline/13_1_ulip2_scanobjnn_clean_hardest_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

13_1_ulip2_scanobjnn_clean_hardest_zs_clean_YYYYMMDD_HHMMSS.log

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
| 当前复现 accuracy | 34.07 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，13_1 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 5. 当前结果表

| 实验编号 | Dataset | Variant | File | Method | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 13_1_ulip2_scanobjnn_clean_hardest_zs | sonn_c | hardest | data/sonn_c/hardest/clean.h5 | Zero-shot | 34.07 | done |

该结果表示：ULIP-2 在 ScanObjNN hardest clean 数据上的 Zero-shot 准确率为 34.07。

---

## 6. 与原文结果对比

原文 Point-Cache Supplementary Table 7 中，S-PB T50-RS-C 是 ScanObjectNN hardest split；其中 Original Data / SONN 对应 clean hardest。ULIP-2 在该设置下的 Zero-shot 结果为 33.38。

当前复现结果为 34.07。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Supplementary Table 7 ULIP-2 Zero-shot / SONN Original Data | 33.38 | 34.07 | +0.69 | 0.69 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与 Supplementary Table 7 的差异 | +0.69 |
| 绝对差异 | 0.69 |

分析：

当前复现结果 34.07 比原文 33.38 高 +0.69。该差异略高，但仍处于可接受范围内。

因此，严谨表述应为：

13_1 脚本执行成功；从数值上看，当前 ULIP-2 Zero-shot 结果略高于原文 ScanObjNN clean hardest / Original Data 参考值，但没有出现明显异常，可以认为复现结果基本对齐。

需要注意的是，因为 13_1 的当前复现值高于原文，后续用当前结果计算 cache 增益时，得到的增益可能会略低于原文增益。因此后续分析应同时看两个角度：

| 分析角度 | 说明 |
|---|---|
| 绝对值是否接近原文 | 判断每个方法是否复现成功 |
| 当前方法间增益是否合理 | 判断 Global / Local Cache 在当前复现环境下是否有效 |

---

## 7. 结果含义分析

ULIP-2 在 ScanObjNN clean hardest 上的 Zero-shot 准确率为 34.07。相比 ModelNet clean，ScanObjNN clean hardest 明显更困难。

可能原因包括：

| 原因 | 解释 |
|---|---|
| 真实扫描数据 | ScanObjNN 来自真实扫描，不是规则 CAD 模型 |
| 几何缺失 | 对象可能存在遮挡、缺失和扫描不完整 |
| 背景与噪声 | hardest split 中样本复杂度更高 |
| 类别混淆 | 真实物体局部结构更容易相似 |
| 无测试时自适应 | Zero-shot 不利用测试流分布信息 |

因此，本实验的意义不是追求高数值，而是建立 ULIP-2 在真实扫描 clean hardest 设置下的无缓存基线。

---

## 8. 与 11_1 ModelNet clean 的关系

11_1 是 ULIP-2 在 ModelNet clean 上的 Zero-shot 结果；13_1 是 ULIP-2 在 ScanObjNN clean hardest 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | Accuracy |
|---|---|---|---:|
| 11_1_ulip2_modelnet_clean_zs | ModelNet clean | Zero-shot | 72.20 |
| 13_1_ulip2_scanobjnn_clean_hardest_zs | ScanObjNN clean hardest | Zero-shot | 34.07 |

对比：

| 比较 | 变化 |
|---|---:|
| 13_1 - 11_1 | -38.13 |

分析：

ScanObjNN clean hardest 相比 ModelNet clean 下降 38.13 个百分点。这说明真实扫描 hardest split 对 ULIP-2 是一个非常强的 domain shift。

虽然两者都是 clean 数据，但数据性质完全不同：

| 数据设置 | 数据性质 | 难点 |
|---|---|---|
| ModelNet clean | CAD / synthetic object 数据 | 形状规则、背景干扰少 |
| ScanObjNN clean hardest | real-world scanned object 数据 | 遮挡、缺失、背景、扫描噪声、真实域偏移 |

因此，13_1 不能与 11_1 直接判断“高低是否异常”，而应优先与原文 ScanObjNN hardest clean / Original Data 的 ULIP-2 数值对齐。

---

## 9. 与前序子实验的关系

在 13 组内部，13_1 是第一个子实验，因此没有前序 13 组子实验可供对比。

它后续将作为以下实验的基线：

| 后续实验 | 对比方式 |
|---|---|
| 13_2_ulip2_scanobjnn_clean_hardest_zs_global | 与 13_1 比较，评估 Global Cache 增益 |
| 13_3_ulip2_scanobjnn_clean_hardest_zs_global_local | 与 13_1 和 13_2 比较，评估完整 Point-Cache 及 Local Cache 额外增益 |

具体方法间增益不在本文件展开，应放在 13_2、13_3 子实验文档及 13 组 summary 文档中。

---

## 10. 阶段性结论

本实验完成了 ULIP-2 × ScanObjNN clean hardest 的 Zero-shot baseline 复现。

主要结论如下：

1. 13_1 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 34.07。
3. 原文 Point-Cache Supplementary Table 7 中 ULIP-2 在 S-PB T50-RS-C Original Data / SONN 上的 Zero-shot 结果为 33.38。
4. 当前复现结果比原文高 +0.69，略高但仍可接受。
5. 相比 11_1 ModelNet clean 的 72.20，13_1 下降到 34.07，说明真实扫描 hardest split 是明显更困难的数据设置。
6. 该实验是 13_2 Global Cache 和 13_3 Hierarchical Cache 的基础对照。
7. 因为当前 Zero-shot 结果略高于原文，后续分析 cache 增益时需要同时关注绝对值对齐和方法间相对提升。

---

## 11. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/13_1_ulip2_scanobjnn_clean_hardest_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/13_1_ulip2_scanobjnn_clean_hardest_zs_single_gpu.sh 1

---

## 12. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/13_1_ulip2_scanobjnn_clean_hardest_zs/summary.csv | wc -l

tail -n +2 results/baseline/13_1_ulip2_scanobjnn_clean_hardest_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/13_1_ulip2_scanobjnn_clean_hardest_zs/logs -maxdepth 1 -name '*.log' | wc -l

cat results/baseline/13_1_ulip2_scanobjnn_clean_hardest_zs/summary.csv
