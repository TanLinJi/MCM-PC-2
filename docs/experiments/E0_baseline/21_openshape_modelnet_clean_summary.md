# 21_openshape_modelnet_clean_summary

## 1. 实验组目的

本总文档汇总 OpenShape 在 ModelNet clean 上的三组 baseline 复现实验。

21 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | OpenShape |
| Dataset | ModelNet clean |
| 数据集参数 | modelnet_c |
| 实际数据文件 | data/modelnet_c/clean.h5 |
| 输入点数 | 1024 |
| 测试设置数 | 1 个 clean setting |

三组实验分别为：

| 实验编号 | 方法 | 作用 |
|---|---|---|
| 21_1_openshape_modelnet_clean_zs | Zero-shot | 无缓存基础对照 |
| 21_2_openshape_modelnet_clean_zs_global | Zero-shot + Global Cache | 验证全局缓存在 clean setting 上的影响 |
| 21_3_openshape_modelnet_clean_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 在 clean setting 上的影响 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| OpenShape 在 ModelNet clean 上的 Zero-shot 基础性能是多少？ | 由 21_1 给出 |
| Global Cache 在 OpenShape clean 上是否提升？ | 比较 21_2 - 21_1 |
| Local Cache 在 Global Cache 基础上是否继续提升？ | 比较 21_3 - 21_2 |
| 三个结果是否与原文对齐？ | 分别与原文 Point-Cache Table 1 对比 |
| OpenShape clean 上 cache 略降是否异常？ | 需要结合原文趋势判断 |
| 后续 22 组 ModelNet-C 应如何分析？ | 21 组作为 clean 参考 |

需要特别注意：21 组是 clean 单文件实验，不是 all35 corruption 实验。因此本文档不包含 corruption × severity 矩阵，而是围绕 ModelNet clean 的单点 accuracy、原文对齐、方法间变化和 backbone 对比展开分析。

---

## 2. 当前实现方式

21 组使用普通 bash 脚本执行，不使用 all35 优化 runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/21_run_openshape_modelnet_clean_common.sh |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |
| 数据文件 | Point-Cache/data/modelnet_c/clean.h5 |
| OpenShape 权重 | Point-Cache/weights/openshape/openshape-pointbert-vitg14-rgb/model.pt |
| OpenCLIP 权重 | Point-Cache/weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin |

之所以不使用 all35 优化 runner，是因为 21 组只运行一个 clean 文件，不存在 35 次重复加载模型的问题。保持普通脚本结构更简单，也与 01 组、11 组 ModelNet clean 的实验组织方式一致。

---

## 3. 三个实验的输出完整性检查

| 实验编号 | 方法 | summary 行数 | cor_type 唯一数 | log_path 唯一数 | logs 文件数 | status | 状态 |
|---|---|---:|---:|---:|---:|---|---|
| 21_1_openshape_modelnet_clean_zs | Zero-shot | 1 | 1 | 1 | 1 | done | 完成 |
| 21_2_openshape_modelnet_clean_zs_global | ZS + Global | 1 | 1 | 1 | 1 | done | 完成 |
| 21_3_openshape_modelnet_clean_zs_global_local | ZS + Global + Local | 1 | 1 | 1 | 1 | done | 完成 |

说明：

1. 21 组每个子实验都只对应 `clean.h5` 一个测试文件，因此 summary 行数应为 1。
2. 三个子实验均为 `status=done`。
3. 每个子实验都有唯一 log_path，说明结果和日志可以一一对应。
4. 每个 logs 目录均只有 1 个 log 文件，没有旧日志或重复日志残留。
5. 执行完整性正常并不等于结果正常；结果是否正常还需要与原文参考值对比。

---

## 4. 核心结果总表

| 实验编号 | 方法 | 当前复现值 | 原文参考值 | Diff = 当前 - 原文 | 是否对齐 |
|---|---|---:|---:|---:|---|
| 21_1_openshape_modelnet_clean_zs | Zero-shot | 84.72 | 84.56 | +0.16 | 高度对齐 |
| 21_2_openshape_modelnet_clean_zs_global | ZS + Global Cache | 84.48 | 84.52 | -0.04 | 高度对齐 |
| 21_3_openshape_modelnet_clean_zs_global_local | ZS + Global + Local Cache | 84.00 | 84.04 | -0.04 | 高度对齐 |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.03 |
| MAE | 0.08 |
| Max Abs Diff | 0.16 |

分析：

21 组三个子实验均完成，并且三种方法的绝对结果与原文 Point-Cache Table 1 高度对齐。

其中 21_2 和 21_3 与原文几乎完全一致，差异均只有 -0.04。21_1 Zero-shot 比原文高 +0.16，差异也很小。整体看，21 组 OpenShape × ModelNet clean 的复现非常稳定。

---

## 5. 方法间变化分析

### 5.1 当前复现变化

| 比较 | 当前复现值 | 变化 |
|---|---:|---:|
| 21_1 Zero-shot | 84.72 | — |
| 21_2 ZS + Global | 84.48 | -0.24 over 21_1 |
| 21_3 ZS + Global + Local | 84.00 | -0.48 over 21_2 |
| 21_3 ZS + Global + Local | 84.00 | -0.72 over 21_1 |

### 5.2 原文变化

| 比较 | 原文值 | 变化 |
|---|---:|---:|
| Zero-shot | 84.56 | — |
| + Global Cache | 84.52 | -0.04 over Zero-shot |
| + Hierarchical Cache | 84.04 | -0.48 over Global Cache |
| + Hierarchical Cache | 84.04 | -0.52 over Zero-shot |

### 5.3 当前变化与原文变化对比

| 变化来源 | 原文变化 | 当前复现变化 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | -0.04 | -0.24 | -0.20 |
| Local Cache extra over Global | -0.48 | -0.48 | +0.00 |
| Full Point-Cache over Zero-shot | -0.52 | -0.72 | -0.20 |

分析：

当前复现中的方法趋势为：

Zero-shot > ZS + Global Cache > ZS + Global Cache + Local Cache

这个趋势与原文完全一致。原文中 OpenShape 在 ModelNet clean 上也是 Zero-shot 最高，Global Cache 略低，Hierarchical Cache 进一步略低。

当前 Global Cache 相对 Zero-shot 的下降幅度比原文大 0.20，主要原因是 21_1 当前复现值 84.72 比原文 84.56 高 +0.16，而 21_2 当前复现值 84.48 与原文 84.52 几乎一致。

Local Cache 的额外变化当前为 -0.48，与原文 -0.48 完全一致，说明 21_3 的模块相对变化也复现良好。

---

## 6. 为什么 OpenShape clean 上 cache 会下降

OpenShape 在 ModelNet clean 上的 Zero-shot 已经达到 84% 以上，说明其原始文本-点云对齐能力很强。在这种 clean synthetic setting 下，测试时缓存机制的收益空间很小。

Point-Cache 依赖在线测试样本的伪标签和缓存检索信息。当基础 zero-shot 已经很强时，cache 产生的额外 logits 不一定提供新的有效信息，反而可能对原本正确的高质量 zero-shot logits 造成轻微扰动。

因此，OpenShape clean 上出现以下趋势并不异常：

Zero-shot > ZS + Global Cache > ZS + Global Cache + Local Cache

该趋势在原文 Point-Cache Table 1 中也存在：

| 方法 | 原文 ModelNet clean |
|---|---:|
| Zero-shot | 84.56 |
| + Global Cache | 84.52 |
| + Hierarchical Cache | 84.04 |

当前复现也呈现同样趋势：

| 方法 | 当前 ModelNet clean |
|---|---:|
| Zero-shot | 84.72 |
| + Global Cache | 84.48 |
| + Global + Local Cache | 84.00 |

因此，21 组的 clean 下降不应解释为实验失败，而应解释为 OpenShape clean setting 下 cache 边际收益不足。

---

## 7. 与 ULIP / ULIP-2 ModelNet clean 的关系

21 组是 OpenShape 在 ModelNet clean 上的结果；前面 01 组和 11 组分别是 ULIP 和 ULIP-2 在同一数据设置上的结果。

| Backbone | Zero-shot | ZS + Global | ZS + Global + Local |
|---|---:|---:|---:|
| ULIP | 约 56 | 约 62 | 约 64 |
| ULIP-2 | 72.20 | 73.99 | 74.35 |
| OpenShape | 84.72 | 84.48 | 84.00 |

分析：

OpenShape 在 ModelNet clean 上的绝对准确率显著高于 ULIP 和 ULIP-2。尤其是 Zero-shot 阶段，OpenShape 已经达到 84.72，明显高于 ULIP-2 的 72.20。

这说明 OpenShape 的基础表征能力和文本-点云对齐能力在 ModelNet clean 上更强。也正因为 OpenShape 的 Zero-shot 起点很高，cache 在 clean setting 上的边际收益更小，甚至可能略微下降。

从三个 backbone 的 clean setting 可以看出：

| Backbone | clean 上 cache 现象 |
|---|---|
| ULIP | cache 明显提升 |
| ULIP-2 | cache 小幅提升 |
| OpenShape | cache 略微下降 |

这说明 cache 的 clean 收益与 backbone 的基础 zero-shot 强度有关：基础模型越弱，cache 越容易提供补偿；基础模型越强，cache 越可能只带来很小收益甚至轻微扰动。

---

## 8. 与后续 22 组 ModelNet-C 的关系

21 组是 OpenShape 在 ModelNet clean 上的 baseline。后续 22 组将进入：

OpenShape × ModelNet-C all35

即 7 种 corruption × 5 个 severity 的完整 corrupted setting。

21 组将作为 22 组的 clean 参考：

| 后续比较 | 目的 |
|---|---|
| 22_1 vs 21_1 | 观察 corruption 相比 clean 对 OpenShape Zero-shot 的影响 |
| 22_2 vs 21_2 | 观察 corruption 相比 clean 对 Global Cache 的影响 |
| 22_3 vs 21_3 | 观察 corruption 相比 clean 对完整 Point-Cache 的影响 |
| 22_2 - 22_1 | 评估 Global Cache 在 ModelNet-C 上是否提升鲁棒性 |
| 22_3 - 22_2 | 评估 Local Cache 在 ModelNet-C 上是否有额外贡献 |

需要特别注意：OpenShape clean 上 cache 下降，不代表 corrupted setting 上也会下降。原文 Table 1 中，OpenShape 在 ModelNet-C severity=2 的 corruption 平均值从 Zero-shot 的 73.49 提升到 Global Cache 的 76.43，再提升到 Hierarchical Cache 的 76.59。也就是说，OpenShape 在 clean 上 cache 略降，但在 corrupted setting 上 cache 有明显鲁棒性收益。

因此，22 组才是判断 OpenShape cache 价值的关键实验组。

---

## 9. 当前结果意义分析

21 组结果说明：

| 观察 | 解释 |
|---|---|
| Zero-shot = 84.72 | OpenShape 在 ModelNet clean 上基础性能很强 |
| ZS + Global = 84.48 | Global Cache 在 clean 上没有带来提升 |
| ZS + Global + Local = 84.00 | Local Cache 在 clean 上进一步轻微降低 |
| 三个结果均高度对齐原文 | 当前 OpenShape clean 复现可靠 |
| clean 上 cache 略降是原文现象 | 不应误判为脚本或实现错误 |
| 后续重点应转向 ModelNet-C | OpenShape cache 价值主要体现在 corrupted setting |

OpenShape 与 ULIP / ULIP-2 的差异很重要。ULIP 和 ULIP-2 在 clean 上仍有 cache 提升空间，而 OpenShape 在 clean 上已经很强。因此，OpenShape clean 结果主要用于确认权重、数据路径、推理流程和 baseline 数值正确，而不用于证明 cache 的鲁棒性贡献。

---

## 10. 对后续 MCM-PC 的启发

当前 21 组实验对后续方法设计有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| OpenShape clean Zero-shot 很强 | 强 backbone 上不能默认 cache 一定提升 clean accuracy |
| cache clean 上略降 | 需要设计可靠性机制，避免在高置信正确样本上过度扰动 |
| Local Cache clean 上进一步下降 | 局部缓存需要更谨慎地参与融合 |
| 原文也存在该现象 | 该问题不是复现错误，而是方法性质 |
| corrupted setting 原文仍有收益 | cache 的主要价值在 domain shift / corruption 场景 |
| 22 组将更关键 | 需要重点观察 ModelNet-C 上 cache 是否恢复正增益 |

这对 MCM-PC 的启发是：后续方法不能只追求“所有场景 cache 都增强”，而应考虑条件化或可靠性感知的融合策略。对于 OpenShape clean 这种高质量 zero-shot 场景，cache 应该尽量避免引入不必要扰动；对于 corrupted setting，cache 才应发挥更强补偿作用。

---

## 11. 阶段性结论

21 组 OpenShape × ModelNet clean baseline 已完成。

主要结论如下：

1. 三个子实验均完成，summary.csv 行数均为 1，status 均为 done。
2. 21_1 Zero-shot 当前复现值为 84.72，原文参考值为 84.56，差异 +0.16。
3. 21_2 ZS + Global 当前复现值为 84.48，原文参考值为 84.52，差异 -0.04。
4. 21_3 ZS + Global + Local 当前复现值为 84.00，原文参考值为 84.04，差异 -0.04。
5. 三个结果整体与原文高度对齐，最大差异只有 0.16。
6. 当前方法趋势为 Zero-shot > Global Cache > Global + Local Cache，与原文一致。
7. Global Cache 相比 Zero-shot 当前变化为 -0.24，原文变化为 -0.04。
8. Local Cache 在 Global Cache 基础上的当前变化为 -0.48，与原文 -0.48 完全一致。
9. OpenShape 在 ModelNet clean 上 cache 轻微下降是原文已有现象，不是实验异常。
10. 21 组说明 OpenShape clean zero-shot 非常强，cache 的主要价值需要在后续 22 组 ModelNet-C all35 中观察。
11. 21 组完成了 C 组 OpenShape 的第一个数据设置 baseline 复现。

---

## 12. 运行命令汇总

21_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/21_1_openshape_modelnet_clean_zs_single_gpu.sh 0

21_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/21_2_openshape_modelnet_clean_zs_global_single_gpu.sh 0

21_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/21_3_openshape_modelnet_clean_zs_global_local_single_gpu.sh 0

---

## 13. 检查命令汇总

21_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/21_1_openshape_modelnet_clean_zs/summary.csv | wc -l

tail -n +2 results/baseline/21_1_openshape_modelnet_clean_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/21_1_openshape_modelnet_clean_zs/logs -maxdepth 1 -name '*.log' | wc -l

21_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/21_2_openshape_modelnet_clean_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/21_2_openshape_modelnet_clean_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/21_2_openshape_modelnet_clean_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

21_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/21_3_openshape_modelnet_clean_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/21_3_openshape_modelnet_clean_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/21_3_openshape_modelnet_clean_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
