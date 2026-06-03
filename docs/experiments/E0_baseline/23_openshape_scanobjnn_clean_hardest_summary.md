# 23_openshape_scanobjnn_clean_hardest_summary

## 1. 实验组目的

本总文档汇总 OpenShape 在 ScanObjNN clean hardest split 上的三组 baseline 复现实验。

23 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | OpenShape |
| Dataset | ScanObjNN clean hardest |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 输入点数 | 1024 |
| 测试设置数 | 1 个 clean hardest setting |

三组实验分别为：

| 实验编号 | 方法 | 作用 |
|---|---|---|
| 23_1_openshape_scanobjnn_clean_hardest_zs | Zero-shot | 无缓存基础对照 |
| 23_2_openshape_scanobjnn_clean_hardest_zs_global | Zero-shot + Global Cache | 验证全局缓存在真实扫描 clean hardest 上的影响 |
| 23_3_openshape_scanobjnn_clean_hardest_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 及 Local Cache 额外影响 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| OpenShape 在 ScanObjNN clean hardest 上的 Zero-shot 基础性能是多少？ | 由 23_1 给出 |
| Global Cache 在真实扫描 clean hardest 上是否提升？ | 比较 23_2 - 23_1 |
| Local Cache 在 Global Cache 基础上是否继续提升？ | 比较 23_3 - 23_2 |
| 三个结果是否与原文对齐？ | 分别与原文 Table 7 的 Original Data 结果对比 |
| ScanObjNN clean hardest 与 ModelNet clean / ModelNet-C 有何差异？ | 与 21 组、22 组对比 |
| 后续 MCM-PC 应重点关注哪些问题？ | 从真实扫描域偏移和全局/局部缓存贡献中寻找方法改进方向 |

需要特别注意：23 组是 clean 单文件实验，不是 all35 corruption 实验。因此本文档不包含 corruption × severity 矩阵，而是围绕 ScanObjNN clean hardest 的单点 accuracy、原文对齐、方法间变化、真实扫描域偏移和 backbone 对比展开分析。

---

## 2. 当前实现方式

23 组使用普通 bash 脚本执行，不使用 all35 优化 runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/23_run_openshape_scanobjnn_clean_hardest_common.sh |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |
| 数据文件 | Point-Cache/data/sonn_c/hardest/clean.h5 |
| OpenShape 权重 | Point-Cache/weights/openshape/openshape-pointbert-vitg14-rgb/model.pt |
| OpenCLIP 权重 | Point-Cache/weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin |

之所以不使用 all35 优化 runner，是因为 23 组只运行一个 clean hardest 文件，不存在 35 次重复加载模型的问题。保持普通脚本结构更简单，也与 03 组、13 组、21 组 clean 单文件实验组织方式一致。

---

## 3. 三个实验的输出完整性检查

| 实验编号 | 方法 | summary 行数 | cor_type 唯一数 | log_path 唯一数 | logs 文件数 | status | 状态 |
|---|---|---:|---:|---:|---:|---|---|
| 23_1_openshape_scanobjnn_clean_hardest_zs | Zero-shot | 1 | 1 | 1 | 1 | done | 完成 |
| 23_2_openshape_scanobjnn_clean_hardest_zs_global | ZS + Global | 1 | 1 | 1 | 1 | done | 完成 |
| 23_3_openshape_scanobjnn_clean_hardest_zs_global_local | ZS + Global + Local | 1 | 1 | 1 | 1 | done | 完成 |

说明：

1. 23 组每个子实验都只对应 `data/sonn_c/hardest/clean.h5` 一个测试文件，因此 summary 行数应为 1。
2. 三个子实验均为 `status=done`。
3. 每个子实验都有唯一 log_path，说明结果和日志可以一一对应。
4. 每个 logs 目录均只有 1 个 log 文件，没有旧日志或重复日志残留。
5. 执行完整性正常并不等于结果正常；结果是否正常还需要与原文参考值对比。

---

## 4. 核心结果总表

| 实验编号 | 方法 | 当前复现值 | 原文参考值 | Diff = 当前 - 原文 | 是否对齐 |
|---|---|---:|---:|---:|---|
| 23_1_openshape_scanobjnn_clean_hardest_zs | Zero-shot | 41.88 | 41.12 | +0.76 | 可接受，略高 |
| 23_2_openshape_scanobjnn_clean_hardest_zs_global | ZS + Global Cache | 41.95 | 42.16 | -0.21 | 高度接近 |
| 23_3_openshape_scanobjnn_clean_hardest_zs_global_local | ZS + Global + Local Cache | 43.82 | 43.72 | +0.10 | 高度对齐 |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.22 |
| MAE | 0.36 |
| Max Abs Diff | 0.76 |

分析：

23 组三个子实验均完成，并且三种方法的绝对结果整体与原文 Table 7 对齐。

其中 23_3 最关键，当前完整 Point-Cache 结果为 43.82，与原文 43.72 仅相差 +0.10，说明最终方法复现非常稳定。23_2 与原文也很接近，差异为 -0.21。23_1 Zero-shot 比原文高 +0.76，略高但仍可接受。

因此，23 组可以认为是有效复现结果，不需要重跑。

---

## 5. 方法间变化分析

### 5.1 当前复现变化

| 比较 | 当前复现值 | 变化 |
|---|---:|---:|
| 23_1 Zero-shot | 41.88 | — |
| 23_2 ZS + Global | 41.95 | +0.07 over 23_1 |
| 23_3 ZS + Global + Local | 43.82 | +1.87 over 23_2 |
| 23_3 ZS + Global + Local | 43.82 | +1.94 over 23_1 |

### 5.2 原文变化

| 比较 | 原文值 | 变化 |
|---|---:|---:|
| Zero-shot | 41.12 | — |
| + Global Cache | 42.16 | +1.04 over Zero-shot |
| + Hierarchical Cache | 43.72 | +1.56 over Global Cache |
| + Hierarchical Cache | 43.72 | +2.60 over Zero-shot |

### 5.3 当前变化与原文变化对比

| 变化来源 | 原文变化 | 当前复现变化 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | +1.04 | +0.07 | -0.97 |
| Local Cache extra over Global | +1.56 | +1.87 | +0.31 |
| Full Point-Cache over Zero-shot | +2.60 | +1.94 | -0.66 |

分析：

当前复现中的方法趋势为：

Zero-shot < ZS + Global Cache < ZS + Global Cache + Local Cache

这个趋势与原文一致。原文中 OpenShape 在 ScanObjNN clean hardest 上也是完整 Point-Cache 最高，Zero-shot 最低。

但当前的贡献分解与原文有所不同。原文中 Global Cache 和 Local Cache 都有较明显贡献；当前复现中 Global Cache 单独增益非常弱，只提升 +0.07，而 Local Cache 在 Global Cache 基础上额外提升 +1.87，是主要提升来源。

这说明 23 组不能写成 “Global Cache 是主要提升来源”。更准确的结论是：当前复现中，完整 Point-Cache 的最终效果高度对齐原文，但主要增益来自 Local Cache。

---

## 6. 方法贡献分解

以当前复现结果为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

43.82 - 41.88 = +1.94

其中：

| 贡献来源 | Accuracy 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +0.07 | 3.61% |
| Local Cache | +1.87 | 96.39% |
| 完整 Point-Cache | +1.94 | 100.00% |

以原文结果为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

43.72 - 41.12 = +2.60

其中：

| 贡献来源 | Accuracy 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +1.04 | 40.00% |
| Local Cache | +1.56 | 60.00% |
| 完整 Point-Cache | +2.60 | 100.00% |

分析：

当前复现与原文的最终完整结果非常接近，但贡献来源不同。当前 Global Cache 贡献被压缩，Local Cache 贡献更突出。

这种差异的一个原因是当前 23_1 Zero-shot 复现值 41.88 比原文 41.12 高 +0.76，抬高了当前基线；而 23_2 Global Cache 复现值 41.95 与原文 42.16 接近。因此，以当前 23_1 为基线计算时，Global Cache 的相对增益显得很弱。

---

## 7. 为什么 23 组中 Local Cache 更重要

ScanObjNN clean hardest 虽然是 clean 文件，但它不是 ModelNet clean 那种标准 CAD / synthetic setting，而是真实扫描 hardest split。它常见的问题包括：

| 真实扫描挑战 | 对点云识别的影响 |
|---|---|
| 局部遮挡 | 物体局部结构不完整 |
| 背景残留 | 全局特征可能混入非目标区域 |
| 扫描噪声 | 点分布不规则 |
| 形状缺失 | 全局轮廓不稳定 |
| 姿态与采样差异 | 同类样本内部变化更大 |

Global Cache 主要依赖整 object 的全局特征。当真实扫描数据中全局形状受到遮挡、背景、缺失和噪声影响时，仅使用全局特征可能不足以稳定补偿 zero-shot logits。

Local Cache 进一步利用局部 patch / 局部聚类特征。对于真实扫描 hardest split，局部结构可能更能捕捉可辨别部件或残留有效几何信息。因此，在 23 组中，Local Cache 在 Global Cache 基础上带来 +1.87 的额外提升，是符合数据特点的。

---

## 8. 与 21 组 ModelNet clean 的关系

21 组是 OpenShape 在 ModelNet clean 上的结果；23 组是 OpenShape 在 ScanObjNN clean hardest 上的结果。

| 方法 | 21 组 ModelNet clean | 23 组 ScanObjNN clean hardest | 变化 |
|---|---:|---:|---:|
| Zero-shot | 84.72 | 41.88 | -42.84 |
| ZS + Global | 84.48 | 41.95 | -42.53 |
| ZS + Global + Local | 84.00 | 43.82 | -40.18 |

分析：

从 ModelNet clean 到 ScanObjNN clean hardest，OpenShape 性能大幅下降。即使使用完整 Point-Cache，准确率也从 84.00 降到 43.82，下降 -40.18。

这说明 ScanObjNN clean hardest 与 ModelNet clean 的难度完全不同。ModelNet clean 更接近标准 CAD / synthetic object classification，而 ScanObjNN hardest 是真实扫描数据，包含背景、遮挡、缺失和扫描噪声，因此对 zero-shot 点云-文本对齐和测试时适应提出更高要求。

不过，完整 Point-Cache 的下降幅度最小，说明 Local Cache 在真实扫描 hardest split 中确实起到一定补偿作用。

---

## 9. 与 22 组 ModelNet-C 的关系

22 组是 OpenShape 在 ModelNet-C all35 上的结果；23 组是 OpenShape 在 ScanObjNN clean hardest 上的结果。

| 方法 | 22 组 ModelNet-C all35 Avg | 23 组 ScanObjNN clean hardest | 变化 |
|---|---:|---:|---:|
| Zero-shot | 72.57 | 41.88 | -30.69 |
| ZS + Global | 75.14 | 41.95 | -33.19 |
| ZS + Global + Local | 75.14 | 43.82 | -31.32 |

如果用 22 组 severity=2 Average 对比：

| 方法 | 22 组 ModelNet-C S2 Avg | 23 组 ScanObjNN clean hardest | 变化 |
|---|---:|---:|---:|
| Zero-shot | 73.57 | 41.88 | -31.69 |
| ZS + Global | 76.46 | 41.95 | -34.51 |
| ZS + Global + Local | 76.33 | 43.82 | -32.51 |

分析：

即使与 corrupted ModelNet-C 相比，ScanObjNN clean hardest 仍然明显更难。OpenShape 在 ModelNet-C all35 的完整 Point-Cache 平均为 75.14，但在 ScanObjNN clean hardest 上只有 43.82。

这说明 synthetic corruption 与真实扫描域偏移不是同一难度层级。ModelNet-C 的 corruption 是在标准 ModelNet 点云上施加的受控扰动，而 ScanObjNN hardest 本身来自真实扫描，背景、遮挡、缺失和采样不规则更复杂。

因此，23 组是比 21 组、22 组更能体现真实扫描域泛化难度的实验组。

---

## 10. 与 ULIP / ULIP-2 ScanObjNN clean hardest 的关系

23 组可以与前面 ULIP、ULIP-2 的 ScanObjNN clean hardest 结果进行横向比较。

| Backbone | Zero-shot | ZS + Global | ZS + Global + Local |
|---|---:|---:|---:|
| ULIP | 29.08 | 32.20 | 32.48 |
| ULIP-2 | 34.07 | 40.42 | 42.44 |
| OpenShape | 41.88 | 41.95 | 43.82 |

OpenShape 相比 ULIP 的提升：

| 方法 | 提升 |
|---|---:|
| Zero-shot | 41.88 - 29.08 = +12.80 |
| ZS + Global | 41.95 - 32.20 = +9.75 |
| ZS + Global + Local | 43.82 - 32.48 = +11.34 |

OpenShape 相比 ULIP-2 的提升：

| 方法 | 提升 |
|---|---:|
| Zero-shot | 41.88 - 34.07 = +7.81 |
| ZS + Global | 41.95 - 40.42 = +1.53 |
| ZS + Global + Local | 43.82 - 42.44 = +1.38 |

分析：

OpenShape 在 ScanObjNN clean hardest 上整体高于 ULIP 和 ULIP-2。尤其是 Zero-shot 阶段，OpenShape 比 ULIP-2 高 +7.81，说明 OpenShape 的基础表征能力更强。

但加入 cache 后，OpenShape 相对 ULIP-2 的优势明显缩小。完整 Point-Cache 下，OpenShape 只比 ULIP-2 高 +1.38。这说明 ULIP-2 在 ScanObjNN clean hardest 上从 cache 中获得了更大相对收益，而 OpenShape 的优势主要来自更强的 Zero-shot 起点。

---

## 11. 与 13 组 ULIP-2 的贡献结构对比

13 组 ULIP-2 × ScanObjNN clean hardest 的结果为：

| 方法 | 13 组 ULIP-2 | 方法变化 |
|---|---:|---:|
| Zero-shot | 34.07 | — |
| ZS + Global | 40.42 | +6.35 |
| ZS + Global + Local | 42.44 | +2.02 over Global |

23 组 OpenShape 的结果为：

| 方法 | 23 组 OpenShape | 方法变化 |
|---|---:|---:|
| Zero-shot | 41.88 | — |
| ZS + Global | 41.95 | +0.07 |
| ZS + Global + Local | 43.82 | +1.87 over Global |

分析：

ULIP-2 的主要提升来自 Global Cache，Global Cache 单独提升 +6.35；OpenShape 的 Global Cache 单独提升只有 +0.07。

但两者的 Local Cache 额外收益都比较明显：ULIP-2 为 +2.02，OpenShape 为 +1.87。这说明在 ScanObjNN clean hardest 上，Local Cache 对真实扫描数据具有较稳定价值。

---

## 12. 与 22 组 Local Cache 现象的差异

22 组 OpenShape × ModelNet-C all35 中，Local Cache 在 Global Cache 基础上的额外贡献几乎为零：

| 组别 | 数据设置 | Local Cache extra |
|---|---|---:|
| 22 组 | OpenShape × ModelNet-C all35 | +0.01 all35 / -0.13 S2 |
| 23 组 | OpenShape × ScanObjNN clean hardest | +1.87 |

分析：

这说明 Local Cache 的价值具有数据依赖性。

在 ModelNet-C 上，OpenShape 的全局特征本身已经比较强，Global Cache 已经提供了主要鲁棒性补偿，Local Cache 边际收益很弱。

在 ScanObjNN clean hardest 上，真实扫描数据存在局部遮挡、背景残留、缺失和不完整结构，局部 patch 信息可能能补充全局特征的不足，因此 Local Cache 贡献更明显。

这对后续 MCM-PC 很重要：不能简单地假设 Local Cache 在所有数据集上都有同样作用，应考虑样本类型、域偏移类型和全局-局部一致性。

---

## 13. 当前结果意义分析

23 组结果说明：

| 观察 | 解释 |
|---|---|
| Zero-shot = 41.88 | OpenShape 在 ScanObjNN hardest 上的基础性能 |
| ZS + Global = 41.95 | Global Cache 单独几乎没有提升 |
| ZS + Global + Local = 43.82 | 完整 Point-Cache 明显优于 Zero-shot |
| 23_3 与原文只差 +0.10 | 完整方法复现高度可靠 |
| ScanObjNN clean hardest 远低于 ModelNet clean | 真实扫描域偏移非常强 |
| ScanObjNN clean hardest 远低于 ModelNet-C | 真实扫描域偏移比 synthetic corruption 更难 |
| Local Cache 是当前主要增益来源 | 局部结构信息对真实扫描 hardest split 很重要 |

23 组是一个非常关键的实验组。它说明 OpenShape 在真实扫描 clean hardest 上仍然存在明显域偏移，而完整 Point-Cache 可以提供有效补偿，尤其是 Local Cache。

---

## 14. 对后续 24 组的意义

23 组是 OpenShape × ScanObjNN clean hardest，后续 24 组将进入：

OpenShape × ScanObjNN-C hardest all35

也就是在 ScanObjNN hardest split 上进一步施加 7 种 corruption × 5 个 severity。

23 组将作为 24 组的 clean hardest 参考：

| 后续比较 | 目的 |
|---|---|
| 24_1 vs 23_1 | 观察 corruption 相比 clean hardest 对 OpenShape Zero-shot 的影响 |
| 24_2 vs 23_2 | 观察 corruption 相比 clean hardest 对 Global Cache 的影响 |
| 24_3 vs 23_3 | 观察 corruption 相比 clean hardest 对完整 Point-Cache 的影响 |
| 24_2 - 24_1 | 评估 Global Cache 在 ScanObjNN-C hardest 上是否提升鲁棒性 |
| 24_3 - 24_2 | 评估 Local Cache 在 ScanObjNN-C hardest 上是否继续有额外贡献 |

由于 23 组已经显示 Local Cache 在真实扫描 clean hardest 上有明显贡献，因此 24 组需要重点关注：当真实扫描 hardest 进一步叠加 corruption 后，Local Cache 是否仍然有效，或者是否会像 22 组那样在部分 corrupted setting 上变弱。

---

## 15. 对后续 MCM-PC 的启发

当前 23 组实验对后续方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| ScanObjNN clean hardest 远难于 ModelNet clean | 真实扫描域偏移必须作为核心实验场景 |
| OpenShape Zero-shot 明显强于 ULIP / ULIP-2 | 强 backbone 提供更高起点 |
| Global Cache 单独增益很弱 | 仅依赖全局特征可能不足 |
| Local Cache 额外贡献明显 | 真实扫描数据上局部证据很重要 |
| 完整 Point-Cache 与原文高度对齐 | 当前复现可靠，可作为后续方法对照 |
| 22 组 Local 弱、23 组 Local 强 | Local Cache 价值依赖数据类型 |
| OpenShape 相比 ULIP-2 的完整方法优势很小 | 强 backbone 上提升空间有限，需要更精细机制 |

这对 MCM-PC 很重要。后续方法不应简单固定 Global / Local 的融合权重，而应根据样本可靠性、全局-局部一致性、伪标签置信度和域偏移类型动态决定缓存贡献。

对于真实扫描 hardest split，局部结构信号可能比 synthetic corruption 场景更重要。后续 MCM-PC 的改进可以围绕如何筛选可靠局部证据、如何抑制错误局部伪标签、如何平衡文本原型、全局缓存和局部缓存展开。

---

## 16. 阶段性结论

23 组 OpenShape × ScanObjNN clean hardest baseline 已完成。

主要结论如下：

1. 三个子实验均完成，summary.csv 行数均为 1，status 均为 done。
2. 23_1 Zero-shot 当前复现值为 41.88，原文参考值为 41.12，差异 +0.76。
3. 23_2 ZS + Global 当前复现值为 41.95，原文参考值为 42.16，差异 -0.21。
4. 23_3 ZS + Global + Local 当前复现值为 43.82，原文参考值为 43.72，差异 +0.10。
5. 三个结果整体与原文对齐，完整 Point-Cache 结果尤其稳定。
6. 当前方法趋势为 Zero-shot < Global Cache < Global + Local Cache，与原文一致。
7. Global Cache 相比 Zero-shot 当前只提升 +0.07，明显弱于原文 +1.04。
8. Local Cache 在 Global Cache 基础上当前额外提升 +1.87，高于原文 +1.56。
9. 当前 23 组主要提升来源是 Local Cache，而不是 Global Cache。
10. 相比 21 组 ModelNet clean，ScanObjNN clean hardest 性能大幅下降，说明真实扫描域偏移非常强。
11. 相比 22 组 ModelNet-C all35，ScanObjNN clean hardest 仍然低很多，说明真实扫描域偏移比 synthetic corruption 更难。
12. OpenShape 在 ScanObjNN clean hardest 上整体强于 ULIP 和 ULIP-2，但完整 Point-Cache 下相对 ULIP-2 的优势只有 +1.38。
13. 23 组完成了 OpenShape 的第三个数据设置 baseline 复现，可作为后续 24 组 ScanObjNN-C hardest all35 的 clean reference。

---

## 17. 运行命令汇总

23_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/23_1_openshape_scanobjnn_clean_hardest_zs_single_gpu.sh 1

23_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global_single_gpu.sh 1

23_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh 0

---

## 18. 检查命令汇总

23_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/23_1_openshape_scanobjnn_clean_hardest_zs/summary.csv | wc -l

tail -n +2 results/baseline/23_1_openshape_scanobjnn_clean_hardest_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/23_1_openshape_scanobjnn_clean_hardest_zs/logs -maxdepth 1 -name '*.log' | wc -l

23_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

23_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
