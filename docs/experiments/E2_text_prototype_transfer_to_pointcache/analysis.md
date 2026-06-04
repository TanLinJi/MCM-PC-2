# E2 实验分析：文本原型增强向 Point-Cache 缓存流程的传递验证

本文档用于记录 E2 的实验结果和分析。

当前 E2 尚未开始运行。

E2 需要重点回答：

1. `manual_full_llm_fusion` 是否能在 `zs_global` 下继续优于 `manual_full`；
2. `manual_full_llm_fusion` 是否能在 `zs_global_local` 下继续优于 `manual_full`；
3. Point-Cache 的 cache 分支是否会保留、放大或抵消 E1 的文本收益；
4. E2 是否足以支持后续进入 ModelNet-C all35 完整验证。

## 2026-06-04：E2 global cache smoke test 结果

详细分析文档：

    docs/experiments/E2_text_prototype_transfer_to_pointcache/smoke_tests/00_global_cache_transfer_analysis.md

核心结果：

| 设置 | 文本方法 | 平均准确率 |
|---|---|---:|
| E1 zero-shot baseline | manual_full | 47.68 |
| E1 zero-shot fusion | manual_full_llm_fusion | 48.88 |
| E2 global cache baseline | manual_full + zs_global | 52.66 |
| E2 global cache fusion | manual_full_llm_fusion + zs_global | 53.18 |

结论：

- `manual_full_llm_fusion + zs_global` 相比 `manual_full + zs_global` 提升 +0.52；
- E1 的文本原型融合收益可以部分传递到 Point-Cache global cache；
- 7 个损坏类型中有 6 个提升；
- `add_global` 出现明显下降，说明 global cache 可能会放大部分初始伪标签偏差；
- 后续需要继续验证 full Point-Cache，即 `zs_global_local` 设置。

## 2026-06-04：E2 smoke test 完整分析

完整分析文档：

    docs/experiments/E2_text_prototype_transfer_to_pointcache/smoke_tests/00_e2_smoke_test_full_analysis.md

结果汇总文档：

    docs/experiments/E2_text_prototype_transfer_to_pointcache/results_summary.md

E2 当前四组 smoke test 已完成：

| 编号 | 设置 | 文本方法 | 平均准确率 |
|---|---|---|---:|
| 00_1 | zs_global | manual_full | 52.66 |
| 00_2 | zs_global | manual_full_llm_fusion | 53.18 |
| 00_3 | zs_global_local | manual_full | 54.00 |
| 00_4 | zs_global_local | manual_full_llm_fusion | 54.21 |

结合 E1 zero-shot 结果：

| 阶段 | manual_full | manual_full_llm_fusion | 提升 |
|---|---:|---:|---:|
| zero-shot | 47.68 | 48.88 | +1.20 |
| global cache | 52.66 | 53.18 | +0.52 |
| full Point-Cache | 54.00 | 54.21 | +0.21 |

结论：

- E1 文本原型融合收益可以传递到 Point-Cache 缓存流程；
- cache 分支越强，文本融合的边际贡献越小；
- 当前最优为 `manual_full_llm_fusion + zs_global_local`，平均准确率 54.21；
- `add_global` 是特殊现象，当前只记录，不作为 E2 主结论重点；
- 后续建议进入 ModelNet-C all35 的 full Point-Cache 横向完整验证。

## 术语修正：纵向实验与横向实验

为避免后续实验管理混乱，当前项目中统一采用以下定义：

- 纵向实验：只围绕所提方法做最小验证，目的是把方法链路跑通，证明方向成立；
- 横向实验：完整验证或大规模验证，包括 all35、多个数据集、多个 cache 设置、多个 backbone、多个方法对比等。

因此，E2 当前 smoke test 属于纵向最小验证；后续 ModelNet-C all35、跨数据集、跨 backbone 和完整方法矩阵属于横向完整验证。

