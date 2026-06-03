# E1 Smoke Test 总结

更新日期：2026-06-03

## 1. Smoke Test 定义

当前 E1 smoke test 是一个最小验证实验，用于快速判断文本原型增强方向是否值得继续。

实验设置：

| 项目 | 设置 |
|---|---|
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 损坏强度 | severity=2 |
| 损坏类型数量 | 7 |
| 任务设置 | zero-shot |

## 2. 四种文本方法

| 编号 | 方法名 | 中文名称 | 作用 |
|---|---|---|---|
| 00_1 | manual_full | 原始完整手工模板 | baseline |
| 00_2 | manual_3d | 删除 2D 模板后的 3D 手工模板 | 失败消融 |
| 00_3 | llm_only | 只使用 LLM 类别描述 | 对照实验 |
| 00_4 | manual_full_llm_fusion | 原始模板与 LLM 描述融合 | 当前主方法 |

## 3. 总体结果

| 方法 | 平均准确率 | 相对 manual_full |
|---|---:|---:|
| manual_full | 47.68 | 0.00 |
| manual_3d | 35.63 | -12.05 |
| llm_only | 39.30 | -8.38 |
| manual_full_llm_fusion | 48.88 | +1.20 |

## 4. 分损坏类型结果

| 损坏类型 | manual_full | manual_3d | llm_only | manual_full_llm_fusion |
|---|---:|---:|---:|---:|
| add_global | 34.00 | 19.49 | 26.66 | 33.55 |
| add_local | 43.92 | 31.16 | 34.44 | 44.61 |
| dropout_global | 54.70 | 45.91 | 50.69 | 57.01 |
| dropout_local | 50.57 | 38.94 | 43.92 | 53.44 |
| rotate | 55.19 | 41.82 | 46.23 | 56.36 |
| scale | 50.89 | 41.37 | 44.21 | 52.76 |
| jitter | 44.49 | 30.75 | 28.97 | 44.45 |
| 平均 | 47.68 | 35.63 | 39.30 | 48.88 |

## 5. 关键结论

### 5.1 manual_3d 失败

manual_3d 相比 manual_full 下降 12.05 个百分点。

这说明：

- 不能简单删除 2D 图像风格模板；
- 2D/image-style prompts 对 ULIP 的 CLIP-style 文本空间具有重要语义锚定作用；
- 纯 3D 几何模板会导致文本原型偏移。

### 5.2 llm_only 不能替代 manual_full

llm_only 比 manual_3d 高 3.67 个百分点，但比 manual_full 低 8.38 个百分点。

这说明：

- LLM 生成描述有补充价值；
- LLM 描述不能单独替代原始手工模板；
- 直接替代会损失预训练文本空间中的稳定语义先验。

### 5.3 manual_full_llm_fusion 是当前有效方向

manual_full_llm_fusion 比 manual_full 高 1.20 个百分点，是当前唯一超过 baseline 的方法。

这说明：

- manual_full 应作为稳定语义锚点保留；
- LLM 描述应作为补充语义分支引入；
- 二者的加权融合比单独使用任一分支更合理。

## 6. 后续计划

下一步优先进行 manual_full_llm_fusion 的权重消融。

候选权重：

| manual_full 权重 | LLM 权重 |
|---:|---:|
| 0.90 | 0.10 |
| 0.85 | 0.15 |
| 0.75 | 0.25 |
| 0.50 | 0.50 |

权重消融完成后，再选择最佳权重运行 ModelNet-C all35 zero-shot 完整验证。
