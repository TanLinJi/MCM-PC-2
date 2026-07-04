# E1 Smoke Test 00_4：manual_full_llm_fusion 结果分析

更新日期：2026-06-03

## 1. 实验目的

本实验用于验证 E1 当前主方法：

原始完整手工模板文本原型与 LLM 多视角描述文本原型进行加权融合，是否能够提升 zero-shot 分类性能。

方法名称：

- 中文名称：原始完整手工模板与 LLM 描述融合
- 方法名：manual_full_llm_fusion
- prompt source：manualfull_llm_dynamic_init

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 损坏强度 | severity=2 |
| 损坏类型数量 | 7 |
| 任务设置 | zero-shot |
| 每类 LLM 描述数量 | 10 |
| 默认融合权重 | manual_full:LLM = 0.75:0.25 |
| 脚本 | Point-Cache/scripts/E1_text_prototype_enhancement/00_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_smoke.sh |
| 结果目录 | Point-Cache/results/E1_text_prototype_enhancement/00_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_smoke/ |

## 3. 实验结果

| 损坏类型 | manual_full | manual_full_llm_fusion | 差值 |
|---|---:|---:|---:|
| add_global | 34.00 | 33.55 | -0.45 |
| add_local | 43.92 | 44.61 | +0.69 |
| dropout_global | 54.70 | 57.01 | +2.31 |
| dropout_local | 50.57 | 53.44 | +2.87 |
| rotate | 55.19 | 56.36 | +1.17 |
| scale | 50.89 | 52.76 | +1.87 |
| jitter | 44.49 | 44.45 | -0.04 |
| 平均 | 47.68 | 48.88 | +1.20 |

## 4. 结果分析

manual_full_llm_fusion 的平均准确率为 48.88，比 manual_full 提升 1.20 个百分点。

在 7 个损坏类型中，该方法有 5 个损坏类型超过 manual_full：

- add_local
- dropout_global
- dropout_local
- rotate
- scale

仅在 add_global 和 jitter 上略低于 manual_full，且下降幅度很小。

这说明：

- manual_full 提供稳定的 CLIP-style 视觉语义锚点；
- LLM 描述提供额外的类别级视觉语义和 3D 点云几何信息；
- 二者不是替代关系，而是互补关系；
- 加权融合能够利用 manual_full 的稳定性，同时吸收 LLM 描述的补充语义。

## 5. 结论

manual_full_llm_fusion 是当前 E1 smoke test 中唯一超过 manual_full baseline 的方法。

该结果是 E1 当前阶段的核心正结果，支持后续继续开展：

- 融合权重消融；
- ModelNet-C all35 zero-shot 完整验证；
- global cache 和 local cache 设置扩展；
- 跨数据集和多 backbone 验证。
