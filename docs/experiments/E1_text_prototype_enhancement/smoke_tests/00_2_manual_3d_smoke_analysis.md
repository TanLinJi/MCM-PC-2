# E1 Smoke Test 00_2：manual_3d 结果分析

更新日期：2026-06-03

## 1. 实验目的

本实验用于验证一个早期假设：

如果点云任务更依赖 3D 几何结构，那么从原始手工模板中删除明显的 2D 图像风格模板，只保留 3D 几何相关模板，是否会提升文本原型质量。

方法名称：

- 中文名称：删除 2D 图像风格模板后的 3D 手工模板
- 方法名：manual_3d
- prompt source：manual_3d

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 损坏强度 | severity=2 |
| 损坏类型数量 | 7 |
| 任务设置 | zero-shot |
| 脚本 | Point-Cache/scripts/E1_text_prototype_enhancement/00_2_ulip_modelnetc_s2_zs_manual_3d_smoke.sh |
| 结果目录 | Point-Cache/results/E1_text_prototype_enhancement/00_2_ulip_modelnetc_s2_zs_manual_3d_smoke/ |

## 3. 实验结果

| 损坏类型 | manual_full | manual_3d | 差值 |
|---|---:|---:|---:|
| add_global | 34.00 | 19.49 | -14.51 |
| add_local | 43.92 | 31.16 | -12.76 |
| dropout_global | 54.70 | 45.91 | -8.79 |
| dropout_local | 50.57 | 38.94 | -11.63 |
| rotate | 55.19 | 41.82 | -13.37 |
| scale | 50.89 | 41.37 | -9.52 |
| jitter | 44.49 | 30.75 | -13.74 |
| 平均 | 47.68 | 35.63 | -12.05 |

## 4. 结果分析

manual_3d 的平均准确率为 35.63，相比 manual_full 下降 12.05 个百分点。

该结果说明：

- 简单删除 2D 图像风格模板不可行；
- 原始模板中的 2D/image-style prompts 虽然不直接描述点云几何，但对 ULIP 的 CLIP-style 文本空间具有重要语义锚定作用；
- 纯 3D 几何风格模板会使文本原型偏离预训练文本空间，从而造成明显性能下降。

## 5. 结论

manual_3d 是一个失败但有价值的消融实验。

该实验验证了 E1 的关键判断：

对于由二维视觉-语言模型迁移而来的 3D 视觉-语言模型，文本原型不能完全由三维几何描述构成，仍然需要保留一定比例的二维视觉语义描述。
