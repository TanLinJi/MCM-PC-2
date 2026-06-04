# E2 结果汇总

更新日期：2026-06-04

## 1. 总体结果

| 阶段 | 设置 | 文本方法 | 平均准确率 |
|---|---|---|---:|
| E1 | zero-shot | manual_full | 47.68 |
| E1 | zero-shot | manual_full_llm_fusion | 48.88 |
| E2 | zs_global | manual_full | 52.66 |
| E2 | zs_global | manual_full_llm_fusion | 53.18 |
| E2 | zs_global_local | manual_full | 54.00 |
| E2 | zs_global_local | manual_full_llm_fusion | 54.21 |

## 2. 核心结论

`manual_full_llm_fusion` 在 zero-shot、global cache 和 full Point-Cache 三个阶段均优于 `manual_full`：

| 阶段 | 提升 |
|---|---:|
| zero-shot | +1.20 |
| global cache | +0.52 |
| full Point-Cache | +0.21 |

说明 E1 的文本原型融合收益可以传递到 Point-Cache 缓存流程中。

## 3. 当前最佳结果

当前 E2 smoke test 最佳结果：

    manual_full_llm_fusion + zs_global_local = 54.21

相对于 E1 zero-shot baseline：

    +6.53

相对于原始完整 Point-Cache：

    +0.21
