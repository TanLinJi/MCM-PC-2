# E3 实验分析：全局原型对齐缓存

本文档用于记录 E3 实验结果和分析。

当前 E3 尚未开始运行。

E3 当前重点验证：

    顺序式 Global Prototype-Alignment Cache 是否能够提升完整 Point-Cache 的整体分类准确率。

对比对象来自 E2：

| 文本方法 | E2 原始 full Point-Cache |
|---|---:|
| manual_full | 54.00 |
| manual_full_llm_fusion | 54.21 |

## 2026-06-05：E3-V1 GPA-only center 初版结果

详细分析文档：

    docs/experiments/E3_global_prototype_alignment_cache/smoke_tests/00_1_e3_v1_gpa_only_manual_full_analysis.md

当前结果：

| 损坏类型 | E2 原始 full Point-Cache | E3-V1 GPA-only center | 差值 |
|---|---:|---:|---:|
| add_global | 47.81 | 50.36 | +2.55 |
| add_local | 46.68 | 48.22 | +1.54 |
| dropout_global | 59.20 | 56.12 | -3.08 |
| dropout_local | 56.69 | 56.81 | +0.12 |
| rotate | 62.07 | 60.09 | -1.98 |
| scale | 55.23 | 54.13 | -1.10 |
| jitter | 50.32 | 48.34 | -1.98 |
| 平均 | 54.00 | 53.44 | -0.56 |

结论：

    顺序式 GPA Cache + GPA-only center 当前不是正向结果。
    该方案在 add_global 和 add_local 上提升明显，
    但在 dropout_global、rotate、jitter、scale 上下降，整体平均低于 E2 baseline。

后续优先事项：

1. 修复 GPA 统计保存；
2. 增加替换事件日志，记录新旧样本熵和距离；
3. 在顺序式关系不变的条件下测试 Center-B：Entropy-only center；
4. 在顺序式关系不变的条件下测试 Center-C：Entropy+GPA union center；
5. 后续再实现 MCP-style 并列更新方案。
