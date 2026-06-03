# 11_ulip2_modelnet_clean_summary

## 1. 实验组目的

本总文档汇总 ULIP-2 在 ModelNet clean 上的三组 baseline 复现实验。

11 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | ULIP-2 |
| Dataset | ModelNet clean |
| 数据集参数 | modelnet_c |
| 实际数据文件 | data/modelnet_c/clean.h5 |
| 输入点数 | 1024 |
| 测试设置数 | 1 个 clean setting |

三组实验分别为：

| 实验编号 | 方法 | 作用 |
|---|---|---|
| 11_1_ulip2_modelnet_clean_zs | Zero-shot | 无缓存基础对照 |
| 11_2_ulip2_modelnet_clean_zs_global | Zero-shot + Global Cache | 验证全局缓存增益 |
| 11_3_ulip2_modelnet_clean_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 增益 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| ULIP-2 在 ModelNet clean 上的 Zero-shot 基础性能是多少？ | 由 11_1 给出 |
| Global Cache 是否有效？ | 比较 11_2 - 11_1 |
| Local Cache 是否有额外贡献？ | 比较 11_3 - 11_2 |
| 三个结果是否与原文对齐？ | 分别与原文 Point-Cache Table 1 对比 |
| ULIP-2 相比 ULIP 的 clean 表现如何？ | 可作为后续 backbone 横向比较的基础 |

需要特别注意：11 组是 clean 单文件实验，不是 all35 corruption 实验。因此本文档不包含 corruption × severity 矩阵，而是围绕 ModelNet clean 的单点 accuracy、原文对齐和方法间增益展开分析。

---

## 2. 当前实现方式

11 组使用普通 bash 脚本执行，不使用 all35 优化 runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/11_run_ulip2_modelnet_clean_common.sh |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |
| 数据文件 | Point-Cache/data/modelnet_c/clean.h5 |
| Backbone 权重 | Point-Cache/weights/ulip/pointbert_ulip2.pt |
| 文本编码器权重 | Point-Cache/weights/ulip/slip_base_100ep.pt |

之所以不使用 all35 优化 runner，是因为 11 组只运行一个 clean 文件，不存在 35 次重复加载模型的问题。保持普通脚本结构更简单，也与 01 组 ModelNet clean 的实验组织方式一致。

---

## 3. 三个实验的输出完整性检查

| 实验编号 | 方法 | summary 行数 | cor_type 唯一数 | log_path 唯一数 | logs 文件数 | status | 状态 |
|---|---|---:|---:|---:|---:|---|---|
| 11_1_ulip2_modelnet_clean_zs | Zero-shot | 1 | 1 | 1 | 1 | done | 完成 |
| 11_2_ulip2_modelnet_clean_zs_global | ZS + Global | 1 | 1 | 1 | 1 | done | 完成 |
| 11_3_ulip2_modelnet_clean_zs_global_local | ZS + Global + Local | 1 | 1 | 1 | 1 | done | 完成 |

说明：

1. 11 组每个子实验都只对应 `clean.h5` 一个测试文件，因此 summary 行数应为 1。
2. 三个子实验均为 `status=done`。
3. 每个子实验都有唯一 log_path，说明结果和日志可以一一对应。
4. 执行完整性正常并不等于结果正常；结果是否正常还需要与原文参考值对比。

---

## 4. 核心结果总表

| 实验编号 | 方法 | 当前复现值 | 原文参考值 | Diff = 当前 - 原文 | 是否对齐 |
|---|---|---:|---:|---:|---|
| 11_1_ulip2_modelnet_clean_zs | Zero-shot | 72.20 | 71.23 | +0.97 | 基本对齐，略高 |
| 11_2_ulip2_modelnet_clean_zs_global | ZS + Global Cache | 73.99 | 73.95 | +0.04 | 高度对齐 |
| 11_3_ulip2_modelnet_clean_zs_global_local | ZS + Global + Local Cache | 74.35 | 74.53 | -0.18 | 高度对齐 |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.28 |
| MAE | 0.40 |
| Max Abs Diff | 0.97 |

分析：

11 组三个子实验均完成，并且三种方法的绝对结果整体与原文 Point-Cache Table 1 对齐。

其中 11_2 和 11_3 与原文非常接近，差异分别只有 +0.04 和 -0.18。11_1 Zero-shot 比原文高 +0.97，略高但仍可接受。由于 11_1 偏高，后续基于当前复现值计算的 cache 增益会小于原文增益。

---

## 5. 方法间增益分析

### 5.1 当前复现增益

| 比较 | 当前复现值 | 增益 |
|---|---:|---:|
| 11_1 Zero-shot | 72.20 | — |
| 11_2 ZS + Global | 73.99 | +1.79 over 11_1 |
| 11_3 ZS + Global + Local | 74.35 | +0.36 over 11_2 |
| 11_3 ZS + Global + Local | 74.35 | +2.15 over 11_1 |

### 5.2 原文增益

| 比较 | 原文值 | 增益 |
|---|---:|---:|
| Zero-shot | 71.23 | — |
| + Global Cache | 73.95 | +2.72 over Zero-shot |
| + Hierarchical Cache | 74.53 | +0.58 over Global Cache |
| + Hierarchical Cache | 74.53 | +3.30 over Zero-shot |

### 5.3 当前增益与原文增益对比

| 增益来源 | 原文增益 | 当前复现增益 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | +2.72 | +1.79 | -0.93 |
| Local Cache extra over Global | +0.58 | +0.36 | -0.22 |
| Full Point-Cache over Zero-shot | +3.30 | +2.15 | -1.15 |

分析：

当前复现中的方法趋势正确：

Zero-shot < ZS + Global Cache < ZS + Global Cache + Local Cache

Global Cache 和 Local Cache 都带来正向提升。但是当前方法间增益低于原文，主要原因是当前 11_1 Zero-shot 复现值 72.20 高于原文 71.23，导致以当前 11_1 为基线计算的 cache 增益被压缩。

因此，11 组需要同时从两个角度理解：

| 角度 | 结论 |
|---|---|
| 绝对值对齐 | 11_2 和 11_3 高度对齐原文，11_1 略高但可接受 |
| 方法趋势 | 三种方法递进关系正确 |
| 增益幅度 | 当前增益小于原文，主要受 11_1 偏高影响 |

---

## 6. 方法贡献分解

以当前复现结果为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

74.35 - 72.20 = +2.15

其中：

| 贡献来源 | 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +1.79 | 83.26% |
| Local Cache | +0.36 | 16.74% |
| 完整 Point-Cache | +2.15 | 100.00% |

分析：

在 ULIP-2 × ModelNet clean 上，Global Cache 是主要提升来源，占完整提升的约 83.26%。Local Cache 有正向贡献，但占比约 16.74%，属于小幅补充增益。

这一趋势和 ULIP 的 01 组、03 组、04 组方向一致：Global Cache 通常贡献主要提升，Local Cache 更多是额外补充。

---

## 7. 与 01 组 ULIP ModelNet clean 的关系

01 组是 ULIP 在 ModelNet clean 上的结果；11 组是 ULIP-2 在 ModelNet clean 上的结果。

| 组号 | Backbone | Dataset | 分析重点 |
|---|---|---|---|
| 01 | ULIP | ModelNet clean | ULIP clean baseline |
| 11 | ULIP-2 | ModelNet clean | ULIP-2 clean baseline |

11 组结果整体高于 01 组是预期现象，因为 ULIP-2 是更强的 backbone。ModelNet clean 是较规则的 CAD / synthetic object 数据，ULIP-2 的 zero-shot 起点已经较高，因此 cache 模块的边际提升空间相对有限。

---

## 8. 结果意义分析

11 组结果说明：

| 观察 | 解释 |
|---|---|
| Zero-shot = 72.20 | ULIP-2 在 ModelNet clean 上已有较强基础性能 |
| ZS + Global = 73.99 | Global Cache 仍然能提供额外提升 |
| ZS + Global + Local = 74.35 | Local Cache 在 Global Cache 上继续带来小幅正增益 |
| 11_2 与原文差 +0.04 | Global Cache 绝对结果高度复现 |
| 11_3 与原文差 -0.18 | 完整 Point-Cache 绝对结果高度复现 |
| 当前增益低于原文 | 主要因为 11_1 当前 Zero-shot 偏高 |

ModelNet clean 本身不是 corrupted setting，也不是真实扫描 hardest split，因此该实验主要用于确认 backbone 权重、文本编码器、clean 数据读取和三种方法流程是否正常。真正的鲁棒性分析需要进入后续 12 组 ModelNet-C all35。

---

## 9. 对后续 12 组的意义

11 组完成后，ULIP-2 在 ModelNet clean 上的 baseline 已经明确：

| 方法 | ModelNet clean Accuracy |
|---|---:|
| Zero-shot | 72.20 |
| ZS + Global | 73.99 |
| ZS + Global + Local | 74.35 |

下一步 12 组将进入：

ULIP-2 × ModelNet-C all35

也就是 7 种 corruption × 5 个 severity 的完整 corrupted setting。

11 组将作为 12 组的 clean 参考：

| 后续比较 | 目的 |
|---|---|
| 12_1 vs 11_1 | 观察 corruption 相比 clean 对 ULIP-2 Zero-shot 的影响 |
| 12_2 vs 11_2 | 观察 corruption 相比 clean 对 Global Cache 的影响 |
| 12_3 vs 11_3 | 观察 corruption 相比 clean 对完整 Point-Cache 的影响 |
| 12_2 - 12_1 | 评估 Global Cache 在 ModelNet-C 上的增益 |
| 12_3 - 12_2 | 评估 Local Cache 在 ModelNet-C 上的额外增益 |

特别需要关注：ULIP-2 clean 起点较高，但在 corrupted setting 下是否仍然保持优势，以及 Global / Local Cache 是否在 ModelNet-C all35 上带来比 clean setting 更明显的增益。

---

## 10. 阶段性结论

11 组 ULIP-2 × ModelNet clean baseline 已完成。

主要结论如下：

1. 三个子实验均完成，summary.csv 行数均为 1，status 均为 done。
2. 11_1 Zero-shot 当前复现值为 72.20，原文参考值为 71.23，差异 +0.97。
3. 11_2 ZS + Global 当前复现值为 73.99，原文参考值为 73.95，差异 +0.04。
4. 11_3 ZS + Global + Local 当前复现值为 74.35，原文参考值为 74.53，差异 -0.18。
5. 三个结果整体与原文基本对齐，其中 11_2 和 11_3 高度对齐。
6. 当前复现中 Global Cache 带来 +1.79 提升，是主要增益来源。
7. Local Cache 在 Global Cache 基础上额外提升 +0.36，贡献较小但为正。
8. 完整 Point-Cache 相比 Zero-shot 提升 +2.15。
9. 当前方法趋势正确：Zero-shot < Global Cache < Global + Local Cache。
10. 当前 cache 增益小于原文，主要是因为 11_1 Zero-shot 复现值略高于原文。
11. 11 组结果将作为后续 12 组 ULIP-2 × ModelNet-C all35 的 clean 参考。

---

## 11. 运行命令汇总

11_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/11_1_ulip2_modelnet_clean_zs_single_gpu.sh 0

11_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/11_2_ulip2_modelnet_clean_zs_global_single_gpu.sh 0

11_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/11_3_ulip2_modelnet_clean_zs_global_local_single_gpu.sh 0

---

## 12. 检查命令汇总

11_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/11_1_ulip2_modelnet_clean_zs/summary.csv | wc -l

tail -n +2 results/baseline/11_1_ulip2_modelnet_clean_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/11_1_ulip2_modelnet_clean_zs/logs -maxdepth 1 -name '*.log' | wc -l

11_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/11_2_ulip2_modelnet_clean_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/11_2_ulip2_modelnet_clean_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/11_2_ulip2_modelnet_clean_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

11_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/11_3_ulip2_modelnet_clean_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/11_3_ulip2_modelnet_clean_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/11_3_ulip2_modelnet_clean_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
