# E1 Smoke Test 00_1：manual_full 结果分析

更新日期：2026-06-03

## 1. 实验目的

本实验用于验证 E1 新增的文本原型接口是否会破坏 Point-Cache 原始 zero-shot 文本路径。

方法名称：

- 中文名称：原始完整手工模板
- 方法名：manual_full
- prompt source：manual_full

该方法使用 Point-Cache 原始完整手工模板集合，也就是 E0 baseline 默认使用的文本模板。

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 损坏强度 | severity=2 |
| 损坏类型数量 | 7 |
| 任务设置 | zero-shot |
| 脚本 | Point-Cache/scripts/E1_text_prototype_enhancement/00_1_ulip_modelnetc_s2_zs_manual_full_smoke.sh |
| 结果目录 | Point-Cache/results/E1_text_prototype_enhancement/00_1_ulip_modelnetc_s2_zs_manual_full_smoke/ |

## 3. 实验结果

| 损坏类型 | 准确率 |
|---|---:|
| add_global | 34.00 |
| add_local | 43.92 |
| dropout_global | 54.70 |
| dropout_local | 50.57 |
| rotate | 55.19 |
| scale | 50.89 |
| jitter | 44.49 |
| 平均 | 47.68 |

## 4. 结果分析

manual_full 的平均准确率为 47.68，与 E0 baseline 中对应的 ULIP × ModelNet-C severity=2 zero-shot 结果一致。

该结果说明：

- E1 新增的 prompt-source 接口没有破坏原始文本模板路径；
- 原始完整手工模板仍然是当前最稳定的文本原型基准；
- 后续所有 E1 文本增强方法都应与该结果进行比较。

## 5. 结论

manual_full 是 E1 smoke test 中的 baseline 方法，不是新方法。

该实验的作用是确认代码正确性，并提供后续 manual_3d、llm_only 和 manual_full_llm_fusion 的比较基准。
