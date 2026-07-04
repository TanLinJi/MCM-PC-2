# E7-A3：对齐缓存样本 Zero-shot 伪标签正确率诊断

日期：2026-06-12  
状态：已完成

---

## 0. 简要总结

E7-A3 的核心目标是做离线诊断：统计历史上实际进入或替换过对齐缓存
（alignment cache）的样本，其零样本伪标签（zero-shot pseudo-label）是否
足够可靠。结果显示，该假设不成立。

最关键的累计指标是：

```text
all_alignment_entered_zs_acc = 451 / 916 = 49.24
```

这个正确率只比整体零样本（zero-shot）准确率 `47.68` 高 `+1.56`，远不能
称为高可信。同时，实际进入对齐缓存样本的正确率 `49.24` 低于同时进入
熵缓存和能量缓存的有资格样本（eligible sample）正确率 `51.60`，说明当前
对齐缓存自身的分布替换规则没有进一步净化样本。

最终准确率方面，E7-A3 的 ModelNet-C S2 平均准确率为 `52.72`，相比 A2
恢复明显，但仍低于 A0 的 `53.31`、原始 Global + Local 的 `54.00`，以及
当前主线锚点 `02_9_2` 的 `54.71`。因此，A3 的主要结论是：

```text
当前“同时进入熵缓存和能量缓存 -> 再进入对齐缓存”的弱筛选条件，
不能保证对齐缓存样本足够干净。
```

这意味着不应直接把对齐缓存前置为高可信分布来源；如果后续要做前置
对齐分布，需要先加入更强的进入条件，例如更大的分类间隔（large margin）
或更严格的冷启动保护。

---

## 1. 实验目的

E7-A3 的核心目的不是验证最终准确率是否提升，而是验证一个关键前提：

```text
历史上实际进入或替换过对齐缓存的样本，
其 zero-shot 伪标签是否具有很高正确率。
```

这里的 zero-shot 伪标签指：

```text
argmax(S_zs)
```

离线诊断时，将它与真实标签（ground-truth label）比较，得到累计正确率。真实标签只用于实验统计，不参与测试时适应（Test-Time Adaptation, TTA）的缓存更新、替换、门控或最终预测。

这个诊断用于判断后续是否可以把对齐缓存前置：如果进入对齐缓存的样本本身已经高度正确，则可以考虑先建立高可信对齐分布；如果正确率不高，则前置对齐分布会有较大污染风险。

---

## 2. 载体说明

A3 使用 A0 风格作为实验载体：

```text
熵缓存 || 能量缓存
        -> 同一样本同时进入二者后，才有资格进入对齐缓存
```

注意：A0 结构只是载体。A3 的核心观察对象是“对齐缓存进入样本的累计 zero-shot 伪标签正确率”，不是最终融合准确率。

---

## 3. 运行脚本

脚本位置：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_4_ulip_modelnetc_s2_zs_global_e7_a3_alignment_zs_diag_c4_h1p8_e1p8_a2p5_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh
```

运行命令：

```bash
cd Point-Cache
bash scripts/E7_entropy_energy_alignment_multicache/00_4_ulip_modelnetc_s2_zs_global_e7_a3_alignment_zs_diag_c4_h1p8_e1p8_a2p5_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh 0
```

预期结果目录：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_4_ulip_modelnetc_s2_zs_global_e7_a3_alignment_zs_diag_c4_h1p8_e1p8_a2p5_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

---

## 4. 参数设置

| 参数 | 数值 | 含义 |
|---|---:|---|
| `E7_ENTROPY_CAPACITY` | 4 | 每类熵缓存容量 |
| `E7_ENERGY_CAPACITY` | 4 | 每类能量缓存容量 |
| `E7_ALIGNMENT_CAPACITY` | 3 | 每类对齐缓存容量，保持不变 |
| `alpha_ZS` | 1.0 | 零样本得分（zero-shot logits）权重 |
| `alpha_H` | 1.8 | 熵缓存得分（entropy cache logits）权重 |
| `alpha_E` | 1.8 | 能量缓存得分（energy cache logits）权重 |
| `alpha_A` | 2.5 | 对齐缓存得分（alignment cache logits）权重 |
| `beta_H` | 3.0 | 熵缓存相似度温度 |
| `beta_E` | 3.0 | 能量缓存相似度温度 |
| `beta_A` | 3.0 | 对齐缓存相似度温度 |
| `E7_GATED_FUSION` | 0 | 不使用 A2 门控，保持 A0 风格融合 |

最终得分仍为：

```text
S_final = S_zs + 1.8*S_H + 1.8*S_E + 2.5*S_A
```

---

## 5. 新增诊断指标

A3 新增两个累计正确率口径。这里的“累计”指历史累计，包括预构建阶段（build phase）和正式测试阶段（test phase）中所有曾经进入过相关事件的样本，而不是最终缓存中仍然保留的样本快照。

| 指标 | 含义 |
|---|---|
| `build_alignment_eligible_zs_acc` | 预构建阶段，同时进入熵缓存和能量缓存样本的 zero-shot 伪标签正确率 |
| `build_alignment_entered_zs_acc` | 预构建阶段，实际加入或替换对齐缓存样本的 zero-shot 伪标签正确率 |
| `test_alignment_eligible_zs_total` | 测试阶段，历史上同时进入熵缓存和能量缓存的样本数 |
| `test_alignment_eligible_zs_correct` | 上述样本中 `argmax(S_zs) == target` 的数量 |
| `test_alignment_eligible_zs_acc` | 上述样本的累计 zero-shot 伪标签正确率 |
| `test_alignment_entered_zs_total` | 测试阶段，历史上实际加入或替换对齐缓存的样本数 |
| `test_alignment_entered_zs_correct` | 上述样本中 `argmax(S_zs) == target` 的数量 |
| `test_alignment_entered_zs_acc` | 上述样本的累计 zero-shot 伪标签正确率 |
| `all_alignment_eligible_zs_acc` | build + test 历史累计 eligible 样本的 zero-shot 伪标签正确率 |
| `all_alignment_entered_zs_acc` | build + test 历史累计实际进入/替换对齐缓存样本的 zero-shot 伪标签正确率 |

其中最重要的是：

```text
all_alignment_entered_zs_acc
```

因为它对应真正参与对齐缓存历史分布更新的样本质量。

---

## 6. 判断标准

结果分析时重点比较：

| 对比 | 目的 |
|---|---|
| `all_alignment_entered_zs_acc` vs 整体 Zero-shot 准确率 | 判断对齐缓存进入规则是否筛出了更可靠样本 |
| `all_alignment_entered_zs_acc` vs `all_alignment_eligible_zs_acc` | 判断对齐缓存自身分布替换规则是否进一步净化样本 |
| 每个 corruption 的 entered 正确率 | 判断高可信条件是否只在某些 corruption 上成立 |
| entered 样本数量 | 防止正确率高但样本太少，导致统计不稳定 |

如果 `all_alignment_entered_zs_acc` 明显高于整体 zero-shot 准确率，并且样本数量足够，则支持后续做“前置高可信对齐分布”。如果 entered 正确率不高或样本太少，则不应直接前置对齐分布。

---

## 7. 结果

整体准确率对比：

| 方法 | ModelNet-C S2 平均准确率 |
|---|---:|
| Zero-shot | 47.68 |
| 原始 Global Cache | 52.66 |
| 原始 Global + Local | 54.00 |
| `02_9_2` | 54.71 |
| E7-A0 | 53.31 |
| E7-A1 | 50.97 |
| E7-A2 | 49.98 |
| E7-A3 | 52.72 |

逐项对比 A0/A1/A2：

| corruption | A0 | A1 | A2 | A3 |
|---|---:|---:|---:|---:|
| add_global_2 | 48.50 | 42.38 | 40.15 | 45.91 |
| add_local_2 | 47.57 | 46.64 | 45.58 | 48.14 |
| dropout_global_2 | 57.66 | 56.44 | 56.24 | 56.97 |
| dropout_local_2 | 56.16 | 53.53 | 52.11 | 55.71 |
| rotate_2 | 61.14 | 58.71 | 58.02 | 59.44 |
| scale_2 | 53.04 | 52.27 | 51.74 | 53.53 |
| jitter_2 | 49.11 | 46.84 | 46.03 | 49.31 |
| **Average** | **53.31** | **50.97** | **49.98** | **52.72** |

A3 相比 A2 有明显恢复，平均提升 `+2.73`；相比 A1 提升 `+1.74`；但仍低于 A0 `-0.60`，低于完整 Point-Cache `-1.28`，低于 `02_9_2` `-1.99`。

---

## 8. 核心诊断结果

A3 的核心诊断是进入对齐缓存（alignment cache）的样本 zero-shot 伪标签累计正确率。

整体统计：

| 统计口径 | correct / total | 正确率 |
|---|---:|---:|
| `all_alignment_eligible_zs` | 679 / 1316 | 51.60 |
| `all_alignment_entered_zs` | 451 / 916 | 49.24 |
| `build_alignment_entered_zs` | 389 / 806 | 48.26 |
| `test_alignment_entered_zs` | 62 / 110 | 56.36 |

其中最关键的是：

```text
all_alignment_entered_zs_acc = 49.24
```

这个值只比整体 Zero-shot 的 `47.68` 高 `+1.56`，远不能称为高可信。

逐 corruption 统计：

| corruption | all entered total | all entered acc | all eligible acc | build entered acc | test entered acc |
|---|---:|---:|---:|---:|---:|
| add_global_2 | 113 | 57.52 | 56.60 | 55.67 | 68.75 |
| add_local_2 | 120 | 36.67 | 41.95 | 36.45 | 38.46 |
| dropout_global_2 | 138 | 57.25 | 57.43 | 57.02 | 58.82 |
| dropout_local_2 | 137 | 44.53 | 51.78 | 45.00 | 41.18 |
| rotate_2 | 146 | 57.53 | 57.97 | 54.76 | 75.00 |
| scale_2 | 139 | 46.76 | 48.22 | 46.83 | 46.15 |
| jitter_2 | 123 | 43.09 | 46.11 | 41.28 | 57.14 |

---

## 9. 假设判断

原假设：

```text
同时进入熵缓存和能量缓存，并最终进入对齐缓存的样本，
其 zero-shot 伪标签正确率应该很高。
```

A3 结果不支持这个假设。

主要原因：

1. `all_alignment_entered_zs_acc` 只有 `49.24`，不明显高于整体 Zero-shot 的 `47.68`。
2. `all_alignment_entered_zs_acc` 甚至低于 `all_alignment_eligible_zs_acc` 的 `51.60`，说明当前对齐缓存自身分布替换规则没有进一步净化样本，反而略降。
3. 结果在不同 corruption 上非常不稳定：`add_global_2/dropout_global_2/rotate_2` 在 `57%` 左右，但 `add_local_2` 只有 `36.67`，`jitter_2` 只有 `43.09`，`scale_2` 也只有 `46.76`。
4. 预构建阶段（build phase）占 entered 样本大多数：`806/916`。其正确率只有 `48.26`，说明对齐缓存历史分布从早期就不够干净。

因此，当前“低熵 + 低能量 + 同时进入双缓存”不能作为强可靠样本筛选规则。直接把对齐缓存前置，并用这些样本建立高可信对齐分布，风险较大。

---

## 10. 最终准确率分析

A3 使用 A0 风格载体，但调整为：

```text
熵缓存容量 = 4
能量缓存容量 = 4
对齐缓存容量 = 3
alpha_H = 1.8
alpha_E = 1.8
alpha_A = 2.5
```

这个设置让 A3 从 A2 的 `49.98` 恢复到 `52.72`，说明 A2 的门控融合确实损害较大；同时，A3 的权重和容量设置比 A1 更接近有效缓存干预，因此比 A1 更好。

但 A3 仍低于 A0 的 `53.31` 和 `02_9_2` 的 `54.71`。这说明：

```text
只靠容量/权重调整可以恢复一部分性能，
但不能解决缓存样本质量不足的问题。
```

logits norm 诊断也支持这一点：A3 的正缓存总得分范数平均明显大于 A1/A2，预测改变率也更高，因此它恢复的是“缓存干预强度”，不是“缓存更干净”。

按 corruption 看，A3 相比 A0：

| corruption | A3 - A0 |
|---|---:|
| add_global_2 | -2.59 |
| add_local_2 | +0.57 |
| dropout_global_2 | -0.69 |
| dropout_local_2 | -0.45 |
| rotate_2 | -1.70 |
| scale_2 | +0.49 |
| jitter_2 | +0.20 |

A3 在 `add_local_2/scale_2/jitter_2` 上略高于 A0，但在 `add_global_2/rotate_2` 上下降明显，整体仍低于 A0。

---

## 11. 当前结论

A3 的主要价值是验证并否定了一个关键假设：

```text
当前规则下，进入对齐缓存的样本并不是高可信样本。
```

因此，不建议直接进入原 A4 设想中的“前置高可信对齐分布”。如果要继续做前置对齐分布，需要先设计更强的进入条件，而不能只依赖“同时进入熵缓存和能量缓存”。

可能的后续方向包括：

1. 对齐缓存前置之前，加入更严格的 zero-shot 置信度条件，例如更低熵、更低能量、更大 margin。
2. 先区分 build 阶段和 test 阶段，因为 test entered 正确率 `56.36` 高于 build entered 的 `48.26`，污染主要来自早期预构建。
3. 重新考虑是否应该预构建对齐缓存，或者延迟对齐分布启用，等历史统计稳定后再更新。
4. 对不同 corruption 的行为做拆分，避免一个统一规则在 `add_local_2` 这类场景中严重污染。

---

## 12. 下一步计划状态

结果分析已完成。下一步计划需要和用户确认后再写入本文档。
