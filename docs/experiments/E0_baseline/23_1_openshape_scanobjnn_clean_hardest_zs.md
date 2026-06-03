# 23_1_openshape_scanobjnn_clean_hardest_zs

## 1. 实验目的

本实验用于复现 OpenShape 在 ScanObjNN clean hardest split 上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 23_1_openshape_scanobjnn_clean_hardest_zs |
| Backbone | OpenShape |
| Dataset | ScanObjNN clean hardest |
| Dataset 参数 | sonn_c |
| SONN variant | hardest |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 OpenShape 在 ScanObjNN clean hardest 上的无缓存基础性能。该结果后续会作为 23_2 和 23_3 的对照基线，但本文件只记录 23_1 本身，不展开完整 23 组的方法间对比。

ScanObjNN hardest split 是比 ModelNet clean 更接近真实扫描场景的数据设置。相比 ModelNet clean，ScanObjNN hardest 包含更复杂的真实扫描噪声、遮挡、背景残留和形状不完整问题，因此它更适合观察 OpenShape 在真实扫描域偏移下的基础泛化能力。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | OpenShape |
| 数据集参数 | sonn_c |
| 数据目录 | data/sonn_c |
| 数据集变体 | hardest |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 方法脚本 | Point-Cache/scripts/baseline/23_1_openshape_scanobjnn_clean_hardest_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/23_run_openshape_scanobjnn_clean_hardest_common.sh |
| 输入点数 | 1024 |
| OpenShape version | vitg14 |
| OpenShape 权重 | weights/openshape/openshape-pointbert-vitg14-rgb/model.pt |
| OpenCLIP 权重 | weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 1 |

本实验使用 `sonn_c` 作为 dataset 参数，并指定：

| 参数 | 值 |
|---|---|
| sonn_variant | hardest |
| cor_type | clean |

实际读取文件为：

data/sonn_c/hardest/clean.h5

---

## 3. 当前实现方式

23 组是 clean 单文件实验，因此不需要 all35 优化 runner。

| 项目 | 说明 |
|---|---|
| 是否需要 35 个 cor_type 循环 | 否 |
| 是否需要 Python 内部循环 | 否 |
| 是否需要 all35 runner | 否 |
| 是否每个子实验只生成 1 行 summary | 是 |
| 是否每个子实验只生成 1 个 log | 是 |

23_1 的执行路径为：

| 层级 | 文件 |
|---|---|
| 单方法脚本 | Point-Cache/scripts/baseline/23_1_openshape_scanobjnn_clean_hardest_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/23_run_openshape_scanobjnn_clean_hardest_common.sh |
| 原始推理 runner | Point-Cache/runners/zs_infer.py |

该结构与 03、13、21 等 clean 单文件实验保持一致。

---

## 4. 方法说明

23_1 是纯 Zero-shot 推理，不使用任何 cache。

| 组成部分 | 是否使用 |
|---|---:|
| Text prototype / 文本原型 | 是 |
| Point cloud global feature | 是 |
| Zero-shot logits | 是 |
| Global Cache logits | 否 |
| Local Cache logits | 否 |
| Hierarchical Cache | 否 |

需要注意：公共脚本中仍然会传入 `cache_type` 参数，这是为了统一脚本接口；但是 23_1 实际调用的是 `runners/zs_infer.py`，该 runner 不会构建 Global Cache 或 Local Cache。因此，23_1 应明确记录为无缓存 Zero-shot baseline。

---

## 5. 输出结构

输出目录：

Point-Cache/results/baseline/23_1_openshape_scanobjnn_clean_hardest_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

23_1_openshape_scanobjnn_clean_hardest_zs_clean_YYYYMMDD_HHMMSS.log

因为本实验只测试 `clean.h5`，所以期望只有 1 行 summary 记录和 1 个对应 log。

---

## 6. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 1 | 1 | clean 实验只包含 1 个测试文件 |
| summary 中唯一 cor_type 数 | 1 | 1 | 只测试 clean |
| summary 中唯一 log_path 数 | 1 | 1 | 每次运行对应 1 个 log |
| logs 目录 .log 文件数 | 1 | 1 | 没有旧日志或重复日志残留 |
| status=done 数 | 1 | 1 | 脚本执行成功 |
| 当前复现 accuracy | 41.88 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，23_1 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 7. 当前结果表

| 实验编号 | Dataset | Variant | File | Method | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 23_1_openshape_scanobjnn_clean_hardest_zs | sonn_c | hardest | data/sonn_c/hardest/clean.h5 | Zero-shot | 41.88 | done |

该结果表示：OpenShape 在 ScanObjNN clean hardest split 上的 Zero-shot 准确率为 41.88。

---

## 8. 与原文结果对比

原文 Point-Cache 的 ScanObjectNN hardest / S-PB T50-RS-C 相关表中，OpenShape / O-Shape 在 Original Data，也就是 ScanObjNN clean hardest 上的 Zero-shot 结果为 41.12。

当前复现结果为 41.88。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Point-Cache OpenShape / ScanObjNN clean hardest / Zero-shot | 41.12 | 41.88 | +0.76 | 0.76 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与原文差异 | +0.76 |
| 绝对差异 | 0.76 |

分析：

当前复现结果 41.88 比原文 41.12 高 +0.76。该差异略大于部分 clean 单文件实验中的 0.1 到 0.2 级别差异，但仍在可接受范围内。

因此，23_1 不只是脚本执行成功，而且数值也基本对齐原文。后续判断 23 组是否成功时，不应只看 23_1 单点差异，还要结合 23_2 和 23_3 的最终方法趋势与原文是否一致。

---

## 9. 与 21_1 ModelNet clean 的关系

21_1 是 OpenShape 在 ModelNet clean 上的 Zero-shot 结果；23_1 是 OpenShape 在 ScanObjNN clean hardest 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | Accuracy |
|---|---|---|---:|
| 21_1_openshape_modelnet_clean_zs | ModelNet clean | Zero-shot | 84.72 |
| 23_1_openshape_scanobjnn_clean_hardest_zs | ScanObjNN clean hardest | Zero-shot | 41.88 |

从 ModelNet clean 到 ScanObjNN clean hardest 的下降为：

| 比较 | 变化 |
|---|---:|
| 23_1 - 21_1 | 41.88 - 84.72 = -42.84 |

分析：

OpenShape 在 ModelNet clean 上的 Zero-shot 准确率为 84.72，但在 ScanObjNN clean hardest 上下降到 41.88，下降幅度达到 -42.84。

这说明 ScanObjNN clean hardest 与 ModelNet clean 的难度完全不同。ModelNet clean 更接近标准 CAD / synthetic object classification，而 ScanObjNN hardest 是真实扫描数据，包含背景、遮挡、缺失和扫描噪声，因此对 zero-shot 点云-文本对齐提出更高要求。

这个结果也说明，不能仅凭 ModelNet clean 上的高准确率判断 OpenShape 已经足够鲁棒。真实扫描域偏移仍然会导致显著性能下降。

---

## 10. 与 22_1 ModelNet-C 的关系

22_1 是 OpenShape 在 ModelNet-C all35 上的 Zero-shot 结果；23_1 是 OpenShape 在 ScanObjNN clean hardest 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | Accuracy / Avg |
|---|---|---|---:|
| 22_1_openshape_modelnetc_corruptions_all35_zs | ModelNet-C all35 | Zero-shot | 72.57 all35 / 73.57 S2 |
| 23_1_openshape_scanobjnn_clean_hardest_zs | ScanObjNN clean hardest | Zero-shot | 41.88 |

对比：

| 比较 | 变化 |
|---|---:|
| 23_1 - 22_1 all35 | 41.88 - 72.57 = -30.69 |
| 23_1 - 22_1 S2 Avg | 41.88 - 73.57 = -31.69 |

分析：

即使与 corrupted ModelNet-C 相比，ScanObjNN clean hardest 仍然显著更难。OpenShape 在 ModelNet-C all35 的 Zero-shot 平均为 72.57，但在 ScanObjNN clean hardest 上只有 41.88。

这说明真实扫描域偏移比 synthetic corruption 更具挑战。ModelNet-C 的 corruption 是在 ModelNet 点云上施加的受控扰动，而 ScanObjNN hardest 的数据本身来自真实扫描，几何不完整性、背景干扰和类别外观差异更复杂。

因此，23 组是评估 OpenShape 在真实扫描数据上泛化能力的重要节点。

---

## 11. 与 ULIP / ULIP-2 的 ScanObjNN clean hardest 关系

23_1 可以与前面 ULIP、ULIP-2 的 ScanObjNN clean hardest Zero-shot 结果进行横向比较。

| Backbone | 实验编号 | ScanObjNN clean hardest Zero-shot |
|---|---|---:|
| ULIP | 03_1_ulip_scanobjnn_clean_hardest_zs | 29.08 |
| ULIP-2 | 13_1_ulip2_scanobjnn_clean_hardest_zs | 34.07 |
| OpenShape | 23_1_openshape_scanobjnn_clean_hardest_zs | 41.88 |

Backbone 提升：

| 比较 | 提升 |
|---|---:|
| OpenShape - ULIP | 41.88 - 29.08 = +12.80 |
| OpenShape - ULIP-2 | 41.88 - 34.07 = +7.81 |

分析：

OpenShape 在 ScanObjNN clean hardest 上明显强于 ULIP 和 ULIP-2。相比 ULIP，OpenShape 提升 +12.80；相比 ULIP-2，提升 +7.81。

这说明 OpenShape 的基础点云-文本对齐能力在真实扫描数据上也更强。但需要注意，即使 OpenShape 明显优于 ULIP / ULIP-2，它在 ScanObjNN clean hardest 上仍只有 41.88，远低于 ModelNet clean 的 84.72。这说明真实扫描域偏移仍然是关键挑战。

---

## 12. 与后续子实验的关系

23_1 是 23 组第一个子实验，因此没有前序 23 组子实验可供对比。

它后续将作为以下实验的基线：

| 后续实验 | 对比方式 |
|---|---|
| 23_2_openshape_scanobjnn_clean_hardest_zs_global | 与 23_1 比较，评估 Global Cache 在 OpenShape × ScanObjNN clean hardest 上的影响 |
| 23_3_openshape_scanobjnn_clean_hardest_zs_global_local | 与 23_1 和 23_2 比较，评估完整 Point-Cache 及 Local Cache 额外影响 |

原文中 OpenShape 在 ScanObjNN clean hardest 上的趋势为：

| 方法 | 原文值 |
|---|---:|
| Zero-shot | 41.12 |
| + Global Cache | 42.16 |
| + Hierarchical Cache | 43.72 |

因此，后续 23_2 和 23_3 的重点是观察：

1. Global Cache 是否提升；
2. Local Cache 是否在 Global Cache 基础上继续提升；
3. 最终完整 Point-Cache 是否接近原文 43.72；
4. 当前方法趋势是否保持 Zero-shot < Global < Global + Local。

---

## 13. 结果含义分析

23_1 的意义不只是给出一个 clean accuracy，而是说明 OpenShape 在真实扫描 hardest split 上仍然存在明显域偏移问题。

| 观察 | 含义 |
|---|---|
| 23_1 = 41.88 | OpenShape 在 ScanObjNN hardest 上的基础性能 |
| 比原文高 +0.76 | 数值略高但可接受 |
| 比 21_1 ModelNet clean 低 -42.84 | 真实扫描域偏移非常明显 |
| 比 22_1 ModelNet-C all35 低 -30.69 | 真实扫描 clean hardest 甚至比 synthetic corruption 更难 |
| 比 ULIP / ULIP-2 更高 | OpenShape backbone 更强 |
| 仍然只有 41.88 | 强 backbone 也无法完全解决真实扫描域偏移 |

因此，23_1 是一个非常关键的 baseline：它说明后续 cache 模块在真实扫描数据上仍然有必要发挥作用。

---

## 14. 对后续 MCM-PC 的启发

当前 23_1 对后续 MCM-PC 方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| OpenShape 在 ScanObjNN hardest 上远低于 ModelNet clean | 真实扫描域偏移是必须重点验证的场景 |
| OpenShape 仍明显强于 ULIP / ULIP-2 | 强 backbone 可以提高基础性能，但不能消除域偏移 |
| ScanObjNN clean hardest 比 ModelNet-C all35 更难 | 方法不能只在 synthetic corruption 上验证 |
| Zero-shot 只有 41.88 | 测试时缓存仍有明显提升空间 |
| 后续 23_2 / 23_3 将判断 cache 是否有效 | 需要重点观察 Global 与 Local 的贡献分解 |

这对 MCM-PC 的意义是：如果后续方法只在 ModelNet / ModelNet-C 上有效，但在 ScanObjNN hardest 上无效，那么顶会说服力会不足。真实扫描 hardest split 是后续方法必须重视的核心实验设置之一。

---

## 15. 阶段性结论

本实验完成了 OpenShape × ScanObjNN clean hardest 的 Zero-shot baseline 复现。

主要结论如下：

1. 23_1 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 41.88。
3. 原文 OpenShape / O-Shape 在 ScanObjNN clean hardest 上的 Zero-shot 结果为 41.12。
4. 当前复现结果比原文高 +0.76，差异可接受。
5. 相比 21_1 ModelNet clean 的 84.72，23_1 下降到 41.88，下降 -42.84。
6. 相比 22_1 ModelNet-C all35 的 72.57，23_1 低 -30.69。
7. OpenShape 在 ScanObjNN clean hardest 上明显强于 ULIP 和 ULIP-2，分别高 +12.80 和 +7.81。
8. 23_1 说明真实扫描 hardest split 是比 ModelNet clean 和 ModelNet-C 更具挑战的数据设置。
9. 本实验是 23_2 Global Cache 和 23_3 Global + Local Cache 的基础对照。
10. 23_1 结果有效，不需要重跑。

---

## 16. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/23_1_openshape_scanobjnn_clean_hardest_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/23_1_openshape_scanobjnn_clean_hardest_zs_single_gpu.sh 1

---

## 17. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/23_1_openshape_scanobjnn_clean_hardest_zs/summary.csv | wc -l

tail -n +2 results/baseline/23_1_openshape_scanobjnn_clean_hardest_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/23_1_openshape_scanobjnn_clean_hardest_zs/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/23_1_openshape_scanobjnn_clean_hardest_zs/summary.csv | cut -d',' -f13 | sort | uniq -c

cat results/baseline/23_1_openshape_scanobjnn_clean_hardest_zs/summary.csv
