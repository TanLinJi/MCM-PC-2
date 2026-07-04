# E1 Restart Ablation Plan

更新日期：2026-06-16

## 1. 当前已完成基线

ULIP + ModelNet-C severity=2 + zero-shot：

| 设置 | 平均准确率 |
|---|---:|
| `manual_full` | 47.68 |
| `manual_full_llm_fusion`, 10 prompts, 0.90:0.10 | 48.41 |
| `manual_full_llm_fusion`, 10 prompts, 0.85:0.15 | 48.62 |
| `manual_full_llm_fusion`, 10 prompts, 0.80:0.20 | 48.84 |
| `manual_full_llm_fusion`, 10 prompts, 0.75:0.25 | 48.88 |
| `manual_full_llm_fusion`, 10 prompts, 0.50:0.50 | 48.37 |

## 2. 新增实验

### E1-S1b：0.80:0.20 权重补点

目的：补齐 `0.85:0.15` 与 `0.75:0.25` 之间的中间权重，判断最佳区间是否更接近 0.80。

状态：已完成。S2 平均准确率为 `48.84`，低于 `0.75:0.25` 的 `48.88` 约 `0.04`，但高于 `0.85:0.15` 的 `48.62`。

脚本：

```text
Point-Cache/scripts/E1_text_prototype_enhancement/01_5_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020.sh
```

输出：

```text
Point-Cache/results/E1_text_prototype_enhancement/01_5_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020/
```

### E1-S2：15 prompts 数量与 2D/3D 比例消融

固定融合权重：

```text
manual_full:LLM = 0.80:0.20
```

原因：0.80:0.20 位于当前两个强候选 0.85:0.15 与 0.75:0.25 中间，适合作为 prompt 数量/比例消融的默认权重。

| 编号 | prompt 数量 | LLM prompt mode | 2D:3D | 生成脚本 | 评估脚本 |
|---|---:|---|---|---|---|
| E1-S2a | 15 | `multiview_2d3d_2to1` | 10:5 | `02_1_generate_modelnetc_llm_prompts_p15_2d3d_2to1.sh` | `02_3_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020_p15_2d3d_2to1.sh` |
| E1-S2b | 15 | `multiview_2d3d_1to2` | 5:10 | `02_2_generate_modelnetc_llm_prompts_p15_2d3d_1to2.sh` | `02_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020_p15_2d3d_1to2.sh` |

对照项：

```text
10 prompts, multiview_2d3d, manual_full:LLM = 0.80:0.20
```

即 E1-S1b。

## 3. 判定标准

新增实验完成后，按以下顺序判断：

1. `0.80:0.20` 是否超过或接近 `0.75:0.25 = 48.88`。
2. 15 prompts 是否超过 10 prompts 的 `0.80:0.20`。
3. 2D:3D = 2:1 是否比 1:2 更稳定。
4. 每个 corruption 是否存在明显负迁移，尤其关注 `add_global` 和 `jitter`。

## 4. 进入 all35 的条件

如果以下任一设置在 S2 上成立，即进入 all35：

1. 平均准确率超过 `48.88`；
2. 平均准确率接近 `48.88`，但 7 个 corruption 更稳定；
3. 明显改善 `add_global` 或 `jitter`，且总体不低于 `0.85:0.15 = 48.62`。
