# 02_1_ulip_modelnetc_corruptions_all35_zs

## 1. 实验目的

复现 ULIP 在 ModelNet-C 全部 35 个损坏设置上的 Zero-shot 结果。

本实验属于 baseline 复现阶段的 02 组实验。02 组实验固定使用 ULIP backbone，并在 ModelNet-C 的 35 个 corrupted setting 上评估一种方法。

本实验是 02 组的基础对照实验，不使用 Global Cache，也不使用 Local Cache。后续 02_2 和 02_3 的所有增益都需要以本实验作为基准进行比较。

具体而言，本实验回答三个问题：

| 问题 | 说明 |
|---|---|
| Zero-shot 基础性能是多少？ | 得到 ULIP 在 ModelNet-C 全 35 个 corrupted setting 上的原始鲁棒性水平 |
| 复现结果是否与原论文对齐？ | 使用 severity=2 的结果与原论文 Table 1 对比 |
| 新版优化 runner 是否可靠？ | 与旧版 bash 循环 runner 的结果进行对齐，确认单进程 all35 runner 没有改变实验结论 |

需要特别注意：原论文 Table 1 只报告 corruption severity level = 2 下的结果，而本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原论文 Table 1 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

---

## 2. 当前实现方式

本实验的外部命名规则保持不变：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 02_1_ulip_modelnetc_corruptions_all35_zs |
| 方法脚本 | Point-Cache/scripts/baseline/02_1_ulip_modelnetc_corruptions_all35_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/02_run_ulip_modelnetc_corruptions_all35_common.sh |
| 新增 Python runner | Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/02_1_ulip_modelnetc_corruptions_all35_zs/ |

重要变更：

| 旧实现 | 新实现 |
|---|---|
| bash 外层循环 35 次 |
| 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 |
| 模型只加载一次，每个 cor_type 重新创建 DataLoader |
| bash 通过 tee 生成单个 cor_type 的 log |
| Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv |
| summary.csv 的列结构保持不变 |

本实验虽然改用了优化 runner，但实验定义没有改变。优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、预测逻辑和输出结构。

---

## 3. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 当前优化 runner | runners/baseline/run_ulip_modelnetc_corruptions_all35.py |
| cache_type 参数 | global |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |
| Severity 编号 | 0, 1, 2, 3, 4 |
| 损坏组合数 | 7 × 5 = 35 |
| 输入点数 | 1024 |
| 权重 | weights/ulip/pointbert_ulip1.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| GPU | 单张 Tesla T4 |

其中 `cache_type=global` 只是为了保持参数接口一致。对于 Zero-shot 实验，实际不会使用 Global Cache 或 Local Cache。

---

## 4. 损坏类型

| corruption | severity 编号 |
|---|---|
| add_global | 0, 1, 2, 3, 4 |
| add_local | 0, 1, 2, 3, 4 |
| dropout_global | 0, 1, 2, 3, 4 |
| dropout_local | 0, 1, 2, 3, 4 |
| rotate | 0, 1, 2, 3, 4 |
| scale | 0, 1, 2, 3, 4 |
| jitter | 0, 1, 2, 3, 4 |

注意：severity 文件编号从 0 开始，到 4 结束。不要写成 1 到 5。

文件命名形式为：

| 示例 | 含义 |
|---|---|
| add_global_0.h5 | add_global corruption，severity=0 |
| add_global_2.h5 | add_global corruption，severity=2 |
| jitter_4.h5 | jitter corruption，severity=4 |

---

## 5. 输出结构

输出目录：

Point-Cache/results/baseline/02_1_ulip_modelnetc_corruptions_all35_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则保持不变：

02_1_ulip_modelnetc_corruptions_all35_zs_add_global_0_YYYYMMDD_HHMMSS.log

也就是说，新版 runner 虽然只启动一次 Python，但仍然会为 35 个 cor_type 生成 35 个独立 log。当前已经清理旧版 20260517 日志，只保留新版优化 runner 产生并被 summary.csv 引用的 35 个 log。

---

## 6. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 35 | 35 | 说明 35 个 cor_type 全部完成 |
| summary 中唯一 cor_type 数 | 35 | 35 | 说明没有漏跑或重复写入 |
| summary 中唯一 log_path 数 | 35 | 35 | 说明每个 cor_type 都有独立日志路径 |
| logs 目录当前 .log 文件数 | 35 | 35 | 说明日志目录已清理干净 |
| status=done 数 | 35 | 35 | 说明没有失败项 |
| severity=2 Average | 47.68 | 用于论文对齐 | 与原论文 Table 1 对比 |
| all35 Average | 46.85 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

结论：02_1 实验完整，summary.csv、log_path 和 logs 文件数量完全一致，可以作为后续 02_2 和 02_3 的基础对照。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 45.38 | 38.57 | 34.00 | 29.98 | 26.50 | 34.89 |
| add_local | 51.05 | 47.08 | 43.92 | 41.69 | 39.10 | 44.57 |
| dropout_global | 56.12 | 55.83 | 54.70 | 52.55 | 45.42 | 52.92 |
| dropout_local | 56.77 | 54.78 | 50.57 | 47.89 | 41.09 | 50.22 |
| rotate | 56.40 | 56.20 | 55.19 | 51.34 | 44.33 | 52.69 |
| scale | 53.08 | 52.51 | 50.89 | 50.12 | 48.42 | 51.00 |
| jitter | 55.02 | 51.58 | 44.49 | 33.35 | 23.87 | 41.66 |
| **Average** | **53.40** | **50.94** | **47.68** | **43.85** | **38.39** | **46.85** |

整体观察：

1. 随着 severity 从 S0 增加到 S4，平均准确率从 53.40 下降到 38.39。
2. all35 Average 为 46.85，表示 ULIP 在 ModelNet-C 全 35 个 corrupted setting 上的整体 Zero-shot 水平。
3. add_global 的平均准确率最低，为 34.89，是本实验中最困难的 corruption。
4. dropout_global 和 rotate 的平均准确率最高，分别为 52.92 和 52.69，说明这两类 corruption 对 ULIP Zero-shot 的破坏相对较弱。
5. jitter 在 S4 下降到 23.87，是所有 35 个 setting 中最低的结果，说明强坐标扰动会严重破坏 ULIP 的几何判别能力。

---

## 8. Severity 维度分析

### 8.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 53.40 | — | 0.00 |
| S1 | 50.94 | -2.47 | -2.47 |
| S2 | 47.68 | -3.26 | -5.72 |
| S3 | 43.85 | -3.83 | -9.56 |
| S4 | 38.39 | -5.46 | -15.01 |

分析：

随着 severity 增大，ULIP Zero-shot 的平均准确率单调下降。S0 到 S4 总下降 15.01 个百分点，说明 severity 编号确实反映了损坏强度。

下降幅度并不是完全线性的。S3 到 S4 的下降最大，为 -5.46，说明当 corruption 达到较高强度时，模型性能会出现更明显的退化。

这也说明后续 cache 方法不应该只看 severity=2，而应该观察 S0-S4 全范围下是否都能带来稳定增益。特别是 S4 高损坏场景，更能体现方法的鲁棒性上限。

### 8.2 severity 趋势结论

| 观察 | 结论 |
|---|---|
| S0 到 S4 平均准确率单调下降 | 数据读取和 severity 编号逻辑合理 |
| S3 到 S4 下降最大 | 高强度 corruption 对 ULIP 表征破坏更明显 |
| S2 平均为 47.68 | 与原论文 Table 1 的 severity=2 设置对应 |
| S4 平均仅 38.39 | 强 corruption 下 Zero-shot 仍有较大提升空间 |

---

## 9. Corruption 难度分析

### 9.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | add_global | 34.89 | 全局异常点最困难，所有 severity 下都偏低 |
| 2 | jitter | 41.66 | 高 severity 下急剧退化，S4 仅 23.87 |
| 3 | add_local | 44.57 | 局部异常点有明显影响，但弱于全局异常点 |
| 4 | dropout_local | 50.22 | 局部缺失会降低性能，但仍保留部分整体结构 |
| 5 | scale | 51.00 | 对尺度扰动相对稳定 |
| 6 | rotate | 52.69 | 整体表现较高，但高 severity 仍下降明显 |
| 7 | dropout_global | 52.92 | 当前 Zero-shot 下平均最高，破坏相对较弱 |

分析：

add_global 和 jitter 是 Zero-shot 下最困难的两类 corruption。二者都不是简单的信息缺失，而是会引入错误几何扰动。相比之下，dropout_global 和 dropout_local 主要是删除点或结构，模型仍可能从剩余结构中保留类别线索。

因此，从 Zero-shot 结果看，错误结构扰动比单纯结构缺失更容易导致模型失败。这对后续 MCM-PC 的设计有启发：如果伪标签错误主要集中在异常点或几何扰动场景中，那么后续方法应重点增强对错误结构扰动的可靠性诊断，而不是只依赖全局置信度。

### 9.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 53.08 | 48.42 | 4.66 | 8.78% | 51.00 |
| dropout_global | 56.12 | 45.42 | 10.70 | 19.07% | 52.92 |
| add_local | 51.05 | 39.10 | 11.95 | 23.41% | 44.57 |
| rotate | 56.40 | 44.33 | 12.07 | 21.40% | 52.69 |
| dropout_local | 56.77 | 41.09 | 15.68 | 27.62% | 50.22 |
| add_global | 45.38 | 26.50 | 18.88 | 41.60% | 34.89 |
| jitter | 55.02 | 23.87 | 31.15 | 56.62% | 41.66 |

分析：

scale 的退化最小，从 S0 到 S4 只下降 4.66，说明 ULIP 对尺度变化相对鲁棒。

jitter 的退化最大，从 S0 的 55.02 下降到 S4 的 23.87，绝对下降 31.15，相对下降 56.62%。这说明强 jitter 会严重破坏点云几何结构，使 ULIP 的 zero-shot 表征大幅失效。

add_global 的平均准确率最低，同时 S0 到 S4 下降 18.88，说明全局异常点不仅在高 severity 下困难，在低 severity 下也已经显著影响模型。

---

## 10. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | add_global | 45.38 | dropout_local | 56.77 | 11.39 |
| S1 | add_global | 38.57 | rotate | 56.20 | 17.63 |
| S2 | add_global | 34.00 | rotate | 55.19 | 21.19 |
| S3 | add_global | 29.98 | dropout_global | 52.55 | 22.57 |
| S4 | jitter | 23.87 | scale | 48.42 | 24.55 |

分析：

随着 severity 增大，不同 corruption 之间的性能差距逐步扩大。S0 的 best-worst gap 为 11.39，而 S4 扩大到 24.55。这说明高强度 corruption 不只是整体降低准确率，还会放大不同 corruption 类型之间的难度差异。

在 S0 到 S3 中，add_global 一直是最难 corruption。到 S4 时，jitter 成为最难 corruption，说明极强的坐标扰动对模型破坏更严重。

---

## 11. 低准确率区域分析

| 条件 | 数量 | 占比 | 主要涉及 corruption |
|---|---:|---:|---|
| Acc < 50 | 17 / 35 | 48.57% | add_global, add_local, jitter, dropout_local, rotate, scale |
| Acc < 45 | 12 / 35 | 34.29% | add_global, add_local, jitter, dropout_local, rotate |
| Acc < 40 | 7 / 35 | 20.00% | add_global, add_local, jitter |
| Acc < 35 | 5 / 35 | 14.29% | add_global, jitter |
| Acc < 30 | 3 / 35 | 8.57% | add_global, jitter |
| Acc < 25 | 1 / 35 | 2.86% | jitter |

分析：

低准确率区域高度集中在 add_global 和 jitter 上。35 个 setting 中低于 40 的有 7 个，其中 add_global 占 4 个，jitter 占 2 个，add_local 占 1 个。

这说明 ULIP Zero-shot 的失败不是均匀分布在所有 corruption 上，而是集中发生在“异常点引入”和“坐标扰动”相关设置中。后续分析 02_2 和 02_3 时，需要重点观察 Global Cache 和 Local Cache 是否能缓解这些低性能区域。

---

## 12. 与原论文 severity=2 结果对比

原论文 Table 1 报告的是 severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原论文 S2 | Diff | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 34.00 | 33.55 | +0.45 | 0.45 |
| add_local | 43.92 | 43.92 | +0.00 | 0.00 |
| dropout_global | 54.70 | 54.70 | +0.00 | 0.00 |
| dropout_local | 50.57 | 50.89 | -0.32 | 0.32 |
| rotate | 55.19 | 55.27 | -0.08 | 0.08 |
| scale | 50.89 | 50.20 | +0.69 | 0.69 |
| jitter | 44.49 | 44.08 | +0.41 | 0.41 |
| **Average** | **47.68** | **47.52** | **+0.16** | **0.28 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.16 |
| MAE | 0.28 |
| RMSE | 0.37 |
| Max Abs Diff | 0.69 |

分析：

当前复现与原论文的 severity=2 结果高度一致。平均差异仅 +0.16，MAE 为 0.28，最大单项差异为 scale 的 +0.69，均处于可接受范围。

这说明当前数据路径、模型权重、文本原型构建、推理流程和原始 Point-Cache baseline 基本对齐。因此，02_1 可以作为可信的 Zero-shot 复现结果。

---

## 13. 新版优化 runner 与旧版 bash runner 对齐

本实验曾先使用旧版 bash 外层循环方式跑过一次，随后改为单进程 Python 内部循环 35 个 cor_type 的优化 runner。为了确认优化 runner 没有改变实验结论，这里比较两版结果。

| 指标 | 旧版 bash 循环 | 新版优化 runner | Diff |
|---|---:|---:|---:|
| severity=2 Average | 47.64 | 47.68 | +0.04 |
| all35 Average | 46.86 | 46.85 | -0.01 |
| summary 行数 | 35 | 35 | 0 |
| 唯一 log_path 数 | 35 | 35 | 0 |
| logs 文件数 | 35 | 35 | 0 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 35 个 setting 的平均绝对差异 | 0.22 |
| 35 个 setting 的 RMSE | 0.29 |
| 最大单项差异 | 0.69 |
| all35 平均差异 | -0.01 |
| severity=2 平均差异 | +0.04 |

分析：

新版优化 runner 与旧版 bash runner 的结果高度一致。个别 setting 存在 0.1 到 0.7 的小幅浮动，但 severity=2 Average 和 all35 Average 基本不变。

这说明把 35 次 Python 启动改为单个 Python 进程内部循环，并没有改变 Zero-shot 实验结论。后续 02_2 和 02_3 可以继续使用该优化 runner。

---

## 14. 对后续 02_2 和 02_3 的意义

本实验给出了 ULIP 在 ModelNet-C all35 下的基础 Zero-shot 水平：

| 指标 | 数值 |
|---|---:|
| severity=2 Average | 47.68 |
| all35 Average | 46.85 |

后续比较重点如下：

| 比较 | 目的 |
|---|---|
| 02_2 - 02_1 | 评估 Global Cache 相比 Zero-shot 的提升 |
| 02_3 - 02_2 | 评估 Local Cache 在 Global Cache 基础上的额外提升 |
| 02_3 - 02_1 | 评估完整 Point-Cache 相比 Zero-shot 的总体提升 |

特别需要关注的 corruption：

| Corruption | 原因 |
|---|---|
| add_global | Zero-shot 平均最低，是最困难 corruption |
| jitter | 高 severity 下退化最严重 |
| add_local | 局部异常点也造成明显性能下降 |
| scale | Zero-shot 已相对稳定，cache 增益可能较小 |
| dropout_global | Zero-shot 表现较好，cache 增益空间可能有限 |

如果 Global Cache 和 Local Cache 有效，理想情况应该是：

1. severity=2 Average 高于 47.68；
2. all35 Average 高于 46.85；
3. add_global 和 jitter 等困难 corruption 获得明显提升；
4. S4 高 severity 场景下退化得到缓解；
5. Global + Local 的结果应高于只使用 Global Cache 的结果。

---

## 15. 阶段性结论

本实验完成了 ULIP × ModelNet-C 全 35 corrupted setting 的 Zero-shot baseline 复现。

主要结论如下：

1. 实验完整性正常：summary.csv 有 35 行，cor_type 唯一数为 35，log_path 唯一数为 35，logs 文件数为 35。
2. severity=2 Average 为 47.68，与原论文 47.52 基本一致，说明复现结果可靠。
3. all35 Average 为 46.85，这是本复现实验额外统计的完整 35 setting 平均值。
4. 准确率随 severity 增大而单调下降，说明数据读取和 severity 编号逻辑正常。
5. add_global 是平均意义下最困难的 corruption，jitter 是高 severity 下退化最严重的 corruption。
6. 低准确率区域主要集中在 add_global、jitter 和 add_local，说明异常点和坐标扰动是 ULIP Zero-shot 的主要弱点。
7. 新版优化 runner 与旧版 bash runner 高度对齐，说明工程优化没有改变实验结论。
8. 该实验可作为 02_2 Global Cache 和 02_3 Global + Local Cache 的基础对照。

---

## 16. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/02_1_ulip_modelnetc_corruptions_all35_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/02_1_ulip_modelnetc_corruptions_all35_zs_single_gpu.sh 1

---

## 17. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv | wc -l

tail -n +2 results/baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/02_1_ulip_modelnetc_corruptions_all35_zs/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv | cut -d',' -f13 | sort | uniq -c
