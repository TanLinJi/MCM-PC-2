# 12_2_ulip2_modelnetc_corruptions_all35_zs_global

## 1. 实验目的

本实验用于复现 ULIP-2 在 ModelNet-C 全部 35 个损坏设置上使用 Zero-shot + Global Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 12_2_ulip2_modelnetc_corruptions_all35_zs_global |
| Backbone | ULIP-2 |
| Dataset | ModelNet-C |
| Method | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |

本实验是在 12_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存是否能够提升 ULIP-2 在 ModelNet-C corrupted setting 上的鲁棒性。

本文件只记录 12_2 本身，并与前序子实验 12_1 进行对比。完整 12 组的三方法总表、Global / Local 贡献分解和整体阶段性总结应放在 12 组 summary 文档中。

需要特别注意：原文 Point-Cache Table 1 只报告 corruption severity level = 2 下 7 种 corruption 的结果；本实验额外跑了 severity 0 到 4 的全部 35 个设置。因此本文档同时记录两个指标：

| 指标 | 含义 |
|---|---|
| severity=2 Average | 与原文 Point-Cache Table 1 对齐的指标 |
| all35 Average | 7 种 corruption × 5 个 severity 的整体平均，属于本复现实验的扩展统计 |

---

## 2. 当前实现方式

本实验的外部命名规则如下：

| 项目 | 路径或名称 |
|---|---|
| 实验编号 | 12_2_ulip2_modelnetc_corruptions_all35_zs_global |
| 方法脚本 | Point-Cache/scripts/baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/12_run_ulip2_modelnetc_corruptions_all35_common.sh |
| 优化 Python runner | Point-Cache/runners/baseline/run_ulip2_modelnetc_corruptions_all35.py |
| 结果目录 | Point-Cache/results/baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global/ |

本实验是 all35 实验，因此使用优化 runner：

| 旧实现风险 | 当前实现 |
|---|---|
| bash 外层循环 35 次 |
| 一个 Python 进程内部循环 35 个 cor_type |
| 每个 cor_type 都重新启动 Python、重新加载模型 |
| 模型只加载一次，每个 cor_type 重新创建 DataLoader、重新初始化 Global Cache |
| bash 通过 tee 生成单个 cor_type 的 log |
| Python 内部使用 Tee 为每个 cor_type 生成独立 log |
| 每个 cor_type 独立写入 summary.csv |
| summary.csv 的列结构保持不变 |

该优化只减少重复加载模型、重复初始化 CUDA 和重复构建文本原型的开销，不改变 cor_type、DataLoader、Global Cache 初始化、预测逻辑和输出结构。

---

## 3. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP-2 |
| 数据集参数 | modelnet_c |
| 数据目录 | data/modelnet_c |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 原始核心 runner | runners/model_with_global_cache.py |
| 优化 runner | runners/baseline/run_ulip2_modelnetc_corruptions_all35.py |
| cache_type | global |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |
| Severity 编号 | 0, 1, 2, 3, 4 |
| 损坏组合数 | 7 × 5 = 35 |
| 输入点数 | 1024 |
| Global Cache shot_capacity | 3 |
| Global Cache alpha | 4.0 |
| Global Cache beta | 3.0 |
| Backbone 权重 | weights/ulip/pointbert_ulip2.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| ULIP version | ulip2 |
| GPU | 单张 Tesla T4，本次运行记录为 GPU 1 |

本实验使用 `modelnet_c` 作为 dataset 参数，并在 Python runner 内部循环 35 个 `cor_type`。实际读取文件形式为：

data/modelnet_c/{corruption}_{severity}.h5

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

Point-Cache/results/baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | 35 个 cor_type 的准确率汇总，每个 cor_type 一行 |
| logs/ | 每个 cor_type 一个独立 log，共 35 个 |
| wandb/ | wandb offline 日志 |

log 命名规则：

12_2_ulip2_modelnetc_corruptions_all35_zs_global_add_global_0_YYYYMMDD_HHMMSS.log

也就是说，优化 runner 虽然只启动一次 Python，但仍然会为 35 个 cor_type 生成 35 个独立 log。

---

## 6. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 35 | 35 | 说明 35 个 cor_type 全部完成 |
| summary 中唯一 cor_type 数 | 35 | 35 | 说明没有漏跑或重复写入 |
| summary 中唯一 log_path 数 | 35 | 35 | 说明每个 cor_type 都有独立日志路径 |
| logs 目录当前 .log 文件数 | 35 | 35 | 说明日志目录状态正常 |
| status=done 数 | 35 | 35 | 说明没有失败项 |
| severity=2 Average | 61.22 | 用于论文对齐 | 与原文 Point-Cache Table 1 对比 |
| all35 Average | 61.09 | 本实验扩展统计 | 全 35 个 corrupted setting 的总体平均 |

从执行完整性看，12_2 脚本执行成功，summary.csv 生成正常，结果记录完整。进一步是否可接受，需要与原文 ModelNet-C severity=2 参考值进行对比。

---

## 7. 当前结果表：corruption × severity

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 70.10 | 68.52 | 67.30 | 66.94 | 65.23 | 67.62 |
| add_local | 67.30 | 62.88 | 59.44 | 56.89 | 52.74 | 59.85 |
| dropout_global | 72.24 | 72.28 | 70.58 | 64.75 | 48.26 | 65.62 |
| dropout_local | 71.47 | 67.10 | 61.55 | 56.24 | 47.45 | 60.76 |
| rotate | 73.01 | 73.38 | 72.00 | 70.02 | 68.60 | 71.40 |
| scale | 71.35 | 71.39 | 68.76 | 66.61 | 65.72 | 68.77 |
| jitter | 59.51 | 42.01 | 28.93 | 18.12 | 15.64 | 32.84 |
| **Average** | **69.28** | **65.37** | **61.22** | **57.08** | **52.52** | **61.09** |

整体观察：

1. all35 Average 为 61.09，表示 ULIP-2 在 ModelNet-C 全 35 个 corrupted setting 上加入 Global Cache 后的整体水平。
2. severity=2 Average 为 61.22，用于和原文 Point-Cache Table 1 对齐。
3. rotate 的平均准确率最高，为 71.40。
4. jitter 的平均准确率最低，为 32.84。
5. jitter_4 为 15.64，是全部 35 个 setting 中最低的结果。
6. 相比 12_1 Zero-shot，Global Cache 在所有 35 个 setting 上均带来正向提升。

---

## 8. 与原文结果对比

原文 Point-Cache Table 1 报告的是 ModelNet-C severity level = 2 下 7 种 corruption 的结果。因此，本节只使用 S2 列进行对比。

| Corruption | 当前复现 S2 | 原文 S2 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| add_global | 67.30 | 67.02 | +0.28 | 0.28 |
| add_local | 59.44 | 59.32 | +0.12 | 0.12 |
| dropout_global | 70.58 | 71.35 | -0.77 | 0.77 |
| dropout_local | 61.55 | 61.59 | -0.04 | 0.04 |
| rotate | 72.00 | 72.37 | -0.37 | 0.37 |
| scale | 68.76 | 68.40 | +0.36 | 0.36 |
| jitter | 28.93 | 28.20 | +0.73 | 0.73 |
| **Average** | **61.22** | **61.18** | **+0.04** | **0.38 MAE** |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.04 |
| MAE | 0.38 |
| RMSE | 0.46 |
| Max Abs Diff | 0.77 |

分析：

当前复现的 severity=2 Average 为 61.22，原文为 61.18，差异仅 +0.04。单项最大绝对差异为 dropout_global 的 0.77，整体误差较小。

因此，12_2 不只是脚本跑通，而且数值也与原文 ULIP-2 + Global Cache 在 ModelNet-C severity=2 上的结果高度对齐。

---

## 9. Severity 维度分析

### 9.1 不同 severity 的平均准确率

| Severity | Average Accuracy | 相比上一档变化 | 相比 S0 变化 |
|---:|---:|---:|---:|
| S0 | 69.28 | — | 0.00 |
| S1 | 65.37 | -3.91 | -3.91 |
| S2 | 61.22 | -4.15 | -8.06 |
| S3 | 57.08 | -4.14 | -12.20 |
| S4 | 52.52 | -4.56 | -16.76 |

分析：

随着 severity 从 S0 增大到 S4，平均准确率从 69.28 下降到 52.52，总下降 16.76 个百分点。整体上，severity 越高，ULIP-2 + Global Cache 的准确率越低，说明 corruption severity 仍然持续影响性能。

与 12_1 Zero-shot 相比，12_2 在所有 severity 上整体上移，说明 Global Cache 不是只改善某个特定 severity，而是在整个 severity 范围内都有效。

### 9.2 与 12_1 Zero-shot 的 severity 维度对比

| Severity | 12_1 Zero-shot Avg | 12_2 ZS + Global Avg | Gain |
|---:|---:|---:|---:|
| S0 | 67.19 | 69.28 | +2.09 |
| S1 | 61.93 | 65.37 | +3.44 |
| S2 | 58.02 | 61.22 | +3.20 |
| S3 | 53.92 | 57.08 | +3.16 |
| S4 | 49.29 | 52.52 | +3.24 |
| **all35** | **58.07** | **61.09** | **+3.02** |

分析：

Global Cache 在所有 severity 上都带来正向提升。当前 all35 平均提升为 +3.02，severity=2 提升为 +3.20。

原文 severity=2 下 Global Cache 增益为：

61.18 - 57.95 = +3.23

当前复现 severity=2 下 Global Cache 增益为：

61.22 - 58.02 = +3.20

二者差异为 -0.03，说明 Global Cache 的增益幅度与原文高度一致。

---

## 10. Corruption 难度分析

### 10.1 按 all35 平均准确率排序

| 难度排名 | Corruption | Avg(S0-S4) | 主要现象 |
|---:|---|---:|---|
| 1 | jitter | 32.84 | 平均最低，高 severity 下仍然非常困难 |
| 2 | add_local | 59.85 | 局部异常点仍然较困难 |
| 3 | dropout_local | 60.76 | 局部缺失仍有明显影响 |
| 4 | dropout_global | 65.62 | 高 severity 下降较大 |
| 5 | add_global | 67.62 | 表现中等偏高 |
| 6 | scale | 68.77 | 较稳定 |
| 7 | rotate | 71.40 | 当前最容易 |

分析：

加入 Global Cache 后，jitter 仍然是绝对最困难的 corruption，但平均准确率从 12_1 的 27.28 提升到 32.84，说明 Global Cache 对 jitter 有明显帮助。

rotate 和 scale 仍然表现较好，说明 ULIP-2 对旋转和尺度变化相对更鲁棒。

### 10.2 每种 corruption 从 S0 到 S4 的退化强度

| Corruption | S0 | S4 | 绝对下降 S0-S4 | 相对下降 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|
| scale | 71.35 | 65.72 | 5.63 | 7.89% | 68.77 |
| rotate | 73.01 | 68.60 | 4.41 | 6.04% | 71.40 |
| add_global | 70.10 | 65.23 | 4.87 | 6.95% | 67.62 |
| add_local | 67.30 | 52.74 | 14.56 | 21.63% | 59.85 |
| dropout_global | 72.24 | 48.26 | 23.98 | 33.20% | 65.62 |
| dropout_local | 71.47 | 47.45 | 24.02 | 33.61% | 60.76 |
| jitter | 59.51 | 15.64 | 43.87 | 73.72% | 32.84 |

分析：

rotate、scale 和 add_global 在 severity 增大时相对稳定；jitter 的退化仍然最严重，从 S0 的 59.51 下降到 S4 的 15.64，绝对下降 43.87，相对下降 73.72%。

dropout_local 和 dropout_global 也有明显退化，说明点云缺失类 corruption 在高 severity 下仍会严重影响 ULIP-2 + Global Cache 的表现。

---

## 11. 每个 severity 下的最难与最易 corruption

| Severity | 最难 corruption | Acc | 最易 corruption | Acc | Gap |
|---:|---|---:|---|---:|---:|
| S0 | jitter | 59.51 | rotate | 73.01 | 13.50 |
| S1 | jitter | 42.01 | rotate | 73.38 | 31.37 |
| S2 | jitter | 28.93 | rotate | 72.00 | 43.07 |
| S3 | jitter | 18.12 | rotate | 70.02 | 51.90 |
| S4 | jitter | 15.64 | rotate | 68.60 | 52.96 |

分析：

在所有 severity 下，jitter 都是最困难 corruption，而 rotate 始终是最容易 corruption。随着 severity 增大，best-worst gap 从 S0 的 13.50 扩大到 S4 的 52.96。

Global Cache 提升了整体准确率，但没有改变 jitter 是最主要失败区域这一事实。

---

## 12. 低准确率区域分析

### 12.1 低准确率 setting 数量

| 条件 | 12_1 Zero-shot 数量 | 12_2 ZS + Global 数量 | 减少数量 |
|---|---:|---:|---:|
| Acc < 65 | 18 / 35 | 12 / 35 | -6 |
| Acc < 60 | 13 / 35 | 7 / 35 | -6 |
| Acc < 55 | 10 / 35 | 5 / 35 | -5 |
| Acc < 50 | 7 / 35 | 4 / 35 | -3 |
| Acc < 45 | 4 / 35 | 4 / 35 | 0 |
| Acc < 40 | 4 / 35 | 3 / 35 | -1 |
| Acc < 35 | 4 / 35 | 3 / 35 | -1 |
| Acc < 30 | 3 / 35 | 3 / 35 | 0 |
| Acc < 25 | 3 / 35 | 2 / 35 | -1 |

分析：

加入 Global Cache 后，低准确率区域明显减少。例如 Acc < 60 的 setting 从 13 个减少到 7 个，Acc < 55 的 setting 从 10 个减少到 5 个。

但是，严重低准确率区域仍然主要由 jitter 主导。Acc < 30 的 setting 数量仍为 3 个，说明 Global Cache 对高 severity jitter 的修复有限。

### 12.2 当前仍然困难的低准确率区域

| cor_type | Accuracy | 说明 |
|---|---:|---|
| jitter_4 | 15.64 | 最高 severity 坐标扰动仍然最难 |
| jitter_3 | 18.12 | 中高 severity 坐标扰动仍然偏低 |
| jitter_2 | 28.93 | severity=2 jitter 仍然明显低于其他 corruption |
| jitter_1 | 42.01 | 虽有提升，但仍低于大多数 corruption |
| dropout_local_4 | 47.45 | 高 severity 局部缺失仍有明显影响 |
| dropout_global_4 | 48.26 | 高 severity 全局缺失仍有明显影响 |

分析：

12_2 中最困难区域仍然集中在 jitter，尤其是 jitter_3 和 jitter_4。说明 Global Cache 不能完全解决强坐标扰动问题。

---

## 13. Global Cache 相比 12_1 Zero-shot 的逐项提升

| Corruption | S0 Gain | S1 Gain | S2 Gain | S3 Gain | S4 Gain | Avg Gain |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +1.91 | +2.15 | +1.78 | +3.20 | +3.16 | +2.44 |
| add_local | +4.62 | +5.30 | +4.78 | +5.51 | +4.44 | +4.93 |
| dropout_global | +0.81 | +1.90 | +1.66 | +2.88 | -0.65 | +1.32 |
| dropout_local | +1.66 | +3.20 | +3.24 | +4.90 | +2.31 | +3.06 |
| rotate | +1.09 | +1.38 | +0.97 | +1.75 | +2.51 | +1.54 |
| scale | +2.31 | +2.71 | +2.03 | -0.08 | +0.53 | +1.50 |
| jitter | +2.22 | +7.41 | +7.94 | +3.94 | +6.32 | +5.56 |
| **Average** | **+2.09** | **+3.44** | **+3.20** | **+3.16** | **+3.24** | **+3.02** |

分析：

Global Cache 在 33 / 35 个 setting 上带来正向提升，只有 dropout_global_4 和 scale_3 出现轻微负增益。

平均提升最大的 corruption 是 jitter，Avg Gain 为 +5.56。说明 Global Cache 对坐标扰动有较明显补偿作用，但由于 jitter 的原始准确率太低，即使提升后仍然是最困难 corruption。

add_local 的提升也很明显，Avg Gain 为 +4.93，说明 Global Cache 对局部异常点有较强修正作用。

---

## 14. 与前序实验的关系

12_2 的直接前序子实验是 12_1，即 ULIP-2 在 ModelNet-C all35 上的 Zero-shot 结果。

| 实验编号 | Dataset setting | Method | S2 Avg | all35 Avg |
|---|---|---|---:|---:|
| 12_1_ulip2_modelnetc_corruptions_all35_zs | ModelNet-C all35 | Zero-shot | 58.02 | 58.07 |
| 12_2_ulip2_modelnetc_corruptions_all35_zs_global | ModelNet-C all35 | ZS + Global Cache | 61.22 | 61.09 |

当前结果说明：在 ModelNet-C all35 上，Global Cache 能稳定提升 ULIP-2 Zero-shot。

| 比较 | 变化 |
|---|---:|
| 12_2 S2 Avg - 12_1 S2 Avg | +3.20 |
| 12_2 all35 Avg - 12_1 all35 Avg | +3.02 |

分析：

12_2 相比 12_1 的提升与原文 Global Cache 增益高度一致。原文 severity=2 下 Global Cache 增益为 +3.23，当前复现 severity=2 增益为 +3.20，差异仅 -0.03。

因此，12_2 不仅绝对数值对齐原文，而且相对 12_1 的模块增益也对齐原文。

---

## 15. 阶段性结论

本实验完成了 ULIP-2 × ModelNet-C all35 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 12_2 脚本执行成功，summary.csv 有 35 行，status=done 数为 35。
2. 当前 severity=2 Average 为 61.22，原文 Point-Cache Table 1 中 ULIP-2 + Global Cache Avg 为 61.18，差异仅 +0.04。
3. 当前 all35 Average 为 61.09，是本实验额外统计的 35 个 corrupted setting 总平均。
4. 当前结果与原文 severity=2 数值高度对齐，可以认为 12_2 复现成功。
5. 相比 12_1 Zero-shot，12_2 的 severity=2 Average 提升 +3.20，all35 Average 提升 +3.02。
6. 当前 Global Cache 增益与原文 severity=2 增益 +3.23 高度一致。
7. Global Cache 对 jitter 的平均提升最大，为 +5.56，但 jitter 仍然是最困难 corruption。
8. Global Cache 对 add_local 的提升也明显，Avg Gain 为 +4.93。
9. 低准确率区域明显减少，但高 severity jitter 仍然是主要失败点。
10. 本实验是 12_3 分析 Local Cache 额外贡献的直接对照，不在本文件中展开完整 12 组方法间对比。

---

## 16. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global_single_gpu.sh 1

---

## 17. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

tail -n +2 results/baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global/summary.csv | cut -d',' -f13 | sort | uniq -c
