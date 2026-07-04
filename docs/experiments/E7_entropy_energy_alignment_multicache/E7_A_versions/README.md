# E7-A Versions Index

日期：2026-06-12

本文档索引 E7-A 系列实验。每个实验版本单独维护一个文档。

| 版本 | 文档 | 状态 |
|---|---|---|
| E7-A0 | `A0_manual_multicache.md` | 已完成，已记录结果分析 |
| E7-A1 | `A1_reduced_weights_logits_norm.md` | 已完成，已记录结果分析 |
| E7-A2 | `A2_gated_cache_fusion.md` | 已完成，已记录结果分析 |
| E7-A3 | `A3_alignment_zs_correctness_diag.md` | 已完成，已记录结果分析 |
| E7-A4 | `A4_candidate_pool_alignment_core.md` | 已完成修复后结果分析，候选池-对齐核心缓存可靠准入 |
| E7-A4-B1 | `A4_B1_cache_norm_clip.md` | 已完成，结果显示固定范数裁剪不是主线 |
| E7-A4-B2 | `A4_B2_candidate_pool_top_promotion.md` | 已完成，结果显示 top1 晋升不是主线 |
| E7-B3-Diag-A4 | `B3_Diag_cache_voting_diagnostics.md` | 已完成单扰动诊断，证明 A4 fixed 载体偏弱 |
| E7-B3-Diag-02_9_2 | `B3_Diag_0292_cache_branch_diagnostics.md` | 已实现，待运行与分析，基于当前最好方案诊断各缓存分支 |
| E7-A4 old | `A4_forward_alignment_distribution_plan.md` | 早期临时计划，已被 A3 结果和正式 A4 方案取代 |

共同代码目录：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache
```

共同脚本目录：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache
```

共同结果目录：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache
```
