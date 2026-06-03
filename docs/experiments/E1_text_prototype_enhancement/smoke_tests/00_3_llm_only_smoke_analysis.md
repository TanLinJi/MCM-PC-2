# E1 Smoke Test 00_3：llm_only 结果分析

更新日期：2026-06-03

## 1. 实验目的

本实验用于验证 LLM 生成的类别级多视角描述是否可以独立替代 Point-Cache 原始完整手工模板。

方法名称：

- 中文名称：只使用 LLM 生成的类别级多视角描述
- 方法名：llm_only
- prompt source：llm_dynamic_init

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 损坏强度 | severity=2 |
| 损坏类型数量 | 7 |
| 任务设置 | zero-shot |
| 每类 LLM 描述数量 | 10 |
| LLM 描述策略 | 部分 2D 视觉语义 + 部分 3D 点云几何 |
| 脚本 | Point-Cache/scripts/E1_text_prototype_enhancement/00_3_ulip_modelnetc_s2_zs_llm_only_smoke.sh |
| 结果目录 | Point-Cache/results/E1_text_prototype_enhancement/00_3_ulip_modelnetc_s2_zs_llm_only_smoke/ |

## 3. 实验结果

| 损坏类型 | manual_full | llm_only | 差值 |
|---|---:|---:|---:|
| add_global | 34.00 | 26.66 | -7.34 |
| add_local | 43.92 | 34.44 | -9.48 |
| dropout_global | 54.70 | 50.69 | -4.01 |
| dropout_local | 50.57 | 43.92 | -6.65 |
| rotate | 55.19 | 46.23 | -8.96 |
| scale | 50.89 | 44.21 | -6.68 |
| jitter | 44.49 | 28.97 | -15.52 |
| 平均 | 47.68 | 39.30 | -8.38 |

## 4. 与 manual_3d 的比较

| 方法 | 平均准确率 |
|---|---:|
| manual_3d | 35.63 |
| llm_only | 39.30 |

llm_only 比 manual_3d 高 3.67 个百分点，说明 LLM 生成的多视角描述确实比简单过滤后的 3D-only 手工模板更有表达能力。

## 5. 结果分析

llm_only 的结果高于 manual_3d，但仍明显低于 manual_full。

这说明：

- LLM 生成描述能够提供一定的类别级语义和几何信息；
- 但单独使用 LLM 描述无法充分对齐 ULIP 的预训练文本空间；
- LLM 生成文本的表达方式、语义粒度和稳定性与原始手工模板存在差异；
- LLM 描述不适合作为 manual_full 的直接替代。

## 6. 结论

LLM 生成描述具有补充价值，但不能单独替代原始手工模板。

该实验为下一步 manual_full_llm_fusion 提供了依据：LLM 描述应作为补充语义分支，而不是替代原始模板。
