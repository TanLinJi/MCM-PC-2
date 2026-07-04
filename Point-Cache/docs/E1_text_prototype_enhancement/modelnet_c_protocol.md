# ModelNet-C Protocol

更新日期：2026-06-16

## 当前作用

ModelNet-C 是 E1 的第一阶段消融数据集。E1 不单独在某个扰动等级上做快速消融；所有 ModelNet-C 实验都直接运行完整 ModelNet-C。

完整 ModelNet-C 指：

```text
7 corruption types x 5 severities = 35 evaluations
```

## 全量消融阶段

| 阶段 | 内容 |
|---|---|
| E1_10-E1_14 | `manual_full:LLM` 融合权重消融 |
| E1_20-E1_21 | 15 prompts 的 image / pointcloud 组成消融 |

## 记录要求

每个 E1 ModelNet-C 实验都必须记录：

1. 35 个 corruption-severity 组合的逐项准确率；
2. 35 项平均准确率；
3. 每个 corruption type 跨 5 个 severity 的平均值；
4. 相对 E0 `manual_full` baseline 的差值；
5. 对应 prompt JSON、融合权重、runner、脚本、结果目录。
