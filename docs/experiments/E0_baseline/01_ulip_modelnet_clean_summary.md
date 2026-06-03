# 01_ulip_modelnet_clean_summary

## 1. 实验名称

ULIP × ModelNet clean baseline 复现总结。

本实验组用于复现 Point-Cache 在 **ULIP backbone + ModelNet clean** 上的 baseline 结果。

本组包含三个子实验：

| 实验编号 | 方法 | 说明 |
|---|---|---|
| 01_1_ulip_modelnet_clean_zs | Zero-shot | 不使用缓存，仅使用 ULIP 原始 zero-shot 推理 |
| 01_2_ulip_modelnet_clean_zs_global | ZS + Global Cache | 在 zero-shot 基础上加入全局缓存 |
| 01_3_ulip_modelnet_clean_zs_global_local | ZS + Global + Local Cache | 在 Global Cache 基础上进一步加入 Local Cache，即 Point-Cache 的 hierarchical cache |

---

## 2. 实验目的

本实验用于复现 ULIP 在 clean ModelNet 数据上的基础结果。

与 `02_ulip_modelnetc_corruptions_all35_summary` 不同，本实验不涉及 corruption，也不涉及 severity。

本实验只回答一个问题：

| 问题 | 说明 |
|---|---|
| 在 clean ModelNet 上，Point-Cache 是否能复现原论文的 ULIP baseline？ | 对比 Zero-shot、Global Cache、Hierarchical Cache 三种方法的准确率 |

---

## 3. 实验环境与路径

| 项目 | 内容 |
|---|---|
| 项目根目录 | /root/autodl-tmp/MCM-PC-2 |
| Point-Cache 根目录 | /root/autodl-tmp/MCM-PC-2/Point-Cache |
| 数据集参数 | modelnet_c |
| 实际数据文件 | Point-Cache/data/modelnet_c/clean.h5 |
| Backbone | ULIP |
| 点数 | 1024 |
| corruption 类型 | clean |
| severity | 不适用 |
| 方法数 | 3 |
| 总运行数 | 3 runs |

注意：这里虽然数据集参数写作 `modelnet_c`，但实际读取的是 `data/modelnet_c/clean.h5`，也就是 clean 数据，不是 corrupted 数据。

---

## 4. 脚本与结果目录

| 实验编号 | 脚本 | 结果目录 |
|---|---|---|
| 01_1 | Point-Cache/scripts/baseline/01_1_ulip_modelnet_clean_zs_single_gpu.sh | Point-Cache/results/baseline/01_1_ulip_modelnet_clean_zs |
| 01_2 | Point-Cache/scripts/baseline/01_2_ulip_modelnet_clean_zs_global_single_gpu.sh | Point-Cache/results/baseline/01_2_ulip_modelnet_clean_zs_global |
| 01_3 | Point-Cache/scripts/baseline/01_3_ulip_modelnet_clean_zs_global_local_single_gpu.sh | Point-Cache/results/baseline/01_3_ulip_modelnet_clean_zs_global_local |
| 汇总 | 无单独运行脚本 | Point-Cache/results/baseline/01_ulip_modelnet_clean_summary |

---

## 5. 实验完成情况

| 实验编号 | 方法 | 状态 |
|---|---|---|
| 01_1 | Zero-shot | 完成 |
| 01_2 | ZS + Global Cache | 完成 |
| 01_3 | ZS + Global + Local Cache | 完成 |

三个实验均成功运行，并且均成功写入各自的 `summary.csv`。

---

## 6. 与原论文 Table 1 对齐结果

| 方法 | 当前复现 Acc | 原论文 Acc | Diff |
|---|---:|---:|---:|
| Zero-shot | 56.77 | 56.16 | +0.61 |
| ZS + Global Cache | 62.12 | 62.12 | +0.00 |
| ZS + Global + Local Cache | 64.18 | 64.22 | -0.04 |

分析：

1. Zero-shot 当前复现结果为 56.77，原论文为 56.16，差值为 +0.61。
2. ZS + Global Cache 当前复现结果为 62.12，与原论文 62.12 完全一致。
3. ZS + Global + Local Cache 当前复现结果为 64.18，原论文为 64.22，差值仅为 -0.04。
4. 三个结果整体与原论文高度一致，可以认为 clean ModelNet 上的 ULIP baseline 复现成功。

---

## 7. 模块增益分析

| 比较 | 计算 | 提升 |
|---|---:|---:|
| Global - ZS | 62.12 - 56.77 | +5.35 |
| Global + Local - ZS | 64.18 - 56.77 | +7.41 |
| Global + Local - Global | 64.18 - 62.12 | +2.06 |

分析：

1. Global Cache 带来 +5.35 的提升，是主要提升来源。
2. Local Cache 在 Global Cache 基础上进一步带来 +2.06 的提升。
3. 完整 Point-Cache 相比原始 Zero-shot 总提升为 +7.41。
4. 结果趋势为 Zero-shot < Global Cache < Global + Local Cache。

---

## 8. 横向结果表

| Backbone | Dataset | Zero-shot | ZS + Global | ZS + Global + Local | Global - ZS | Hier - ZS | Hier - Global |
|---|---|---:|---:|---:|---:|---:|---:|
| ULIP | ModelNet clean | 56.77 | 62.12 | 64.18 | +5.35 | +7.41 | +2.06 |

---

## 9. 与 02 组实验的关系

`01` 组和 `02` 组的关系如下：

| 实验组 | 数据 | 是否 corrupted | 作用 |
|---|---|---|---|
| 01_ulip_modelnet_clean | data/modelnet_c/clean.h5 | 否 | 验证 clean ModelNet 上的基础复现 |
| 02_ulip_modelnetc_corruptions_all35 | data/modelnet_c/*.h5 | 是 | 验证 ModelNet-C corruption 下的鲁棒性复现 |

可以这样理解：

1. `01` 组验证方法在 clean 数据上的正常表现。
2. `02` 组验证方法在 corrupted 数据上的鲁棒性表现。
3. `01` 组结果主要和原论文 Table 1 的 clean ModelNet 列对齐。
4. `02` 组结果主要和原论文 Table 1 的 ModelNet-C severity=2 七类 corruption 平均值对齐。

---

## 10. 关键观察

| 观察 | 说明 |
|---|---|
| clean 上 Point-Cache 仍然有效 | 即使没有 corruption，Global Cache 和 Local Cache 仍然能提升准确率 |
| Global Cache 是主要提升来源 | 从 56.77 提升到 62.12 |
| Local Cache 继续带来增益 | 从 62.12 提升到 64.18 |
| 复现结果与原论文高度一致 | Global 完全一致，Hier 仅差 -0.04 |
| clean 实验可作为后续 corruption 实验的 sanity check | 说明权重、数据、脚本、runner 基本正确 |

---

## 11. 阶段性结论

本实验组可以正式记录为：

ULIP × ModelNet clean baseline 复现成功。

主要依据如下：

1. 三个子实验全部成功运行。
2. Zero-shot、Global Cache、Hierarchical Cache 均成功得到准确率。
3. 当前复现结果与原论文 Table 1 中 clean ModelNet 的 ULIP 结果高度一致。
4. Global Cache 和 Local Cache 的增益趋势正确。
5. 该组实验为后续 ModelNet-C all35 和其他 backbone 复现提供了基础验证。

---

## 12. 后续记录

`01` 组后续还需要补充三个单实验文档：

| 顺序 | 文档 |
|---:|---|
| 1 | 01_1_ulip_modelnet_clean_zs.md |
| 2 | 01_2_ulip_modelnet_clean_zs_global.md |
| 3 | 01_3_ulip_modelnet_clean_zs_global_local.md |

下一步建议补充：

01_1_ulip_modelnet_clean_zs.md

