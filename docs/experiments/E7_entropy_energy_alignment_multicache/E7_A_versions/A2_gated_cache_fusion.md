# E7-A2：样本级门控缓存融合

日期：2026-06-12  
状态：已完成

---

## 1. 实验目的

E7-A2 在 E7-A1 的基础上加入样本级门控（sample-wise gating）。

E7-A1 表明，单纯降低缓存权重会削弱缓存的正向纠错能力。因此 A2 不继续只调固定权重，而是让每个测试样本根据零样本得分和缓存得分的关系，自适应决定缓存参与强度。

验证问题：

```text
在不改变缓存更新规则的情况下，
通过样本级门控是否能保留缓存纠错收益，同时减少错误缓存证据对最终预测的干扰。
```

---

## 2. 运行脚本

脚本位置：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_3_ulip_modelnetc_s2_zs_global_e7_a2_gated_cache_h0p6_e0p6_a0p9_gagree1p0_gcorrect1p2_gfallback0p2_zsm5p0_cm1p0_sim0p60_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh
```

运行命令：

```bash
cd Point-Cache
bash scripts/E7_entropy_energy_alignment_multicache/00_3_ulip_modelnetc_s2_zs_global_e7_a2_gated_cache_h0p6_e0p6_a0p9_gagree1p0_gcorrect1p2_gfallback0p2_zsm5p0_cm1p0_sim0p60_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh 0
```

预期结果目录：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_3_ulip_modelnetc_s2_zs_global_e7_a2_gated_cache_h0p6_e0p6_a0p9_gagree1p0_gcorrect1p2_gfallback0p2_zsm5p0_cm1p0_sim0p60_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

---

## 3. 参数设置

| 参数 | 数值 | 含义 |
|---|---:|---|
| `alpha_ZS` | 1.0 | 零样本得分（zero-shot logits）权重 |
| `alpha_H` | 0.6 | 熵缓存得分（entropy cache logits）权重 |
| `alpha_E` | 0.6 | 能量缓存得分（energy cache logits）权重 |
| `alpha_A` | 0.9 | 对齐缓存得分（alignment cache logits）权重 |
| `E7_GATED_FUSION` | 1 | 开启样本级门控融合 |
| `E7_GATE_AGREE` | 1.0 | 缓存预测与零样本预测一致时的门控系数 |
| `E7_GATE_CORRECT` | 1.2 | 缓存满足纠错条件时的门控系数 |
| `E7_GATE_FALLBACK` | 0.2 | 缓存不可信时的保守门控系数 |
| `E7_GATE_ZS_MARGIN_MAX` | 5.0 | 判断零样本预测“不够自信”的最大 margin |
| `E7_GATE_CACHE_MARGIN_MIN` | 1.0 | 判断缓存预测“较自信”的最小 margin |
| `E7_GATE_SIM_MIN` | 0.60 | 判断缓存相似度足够高的最小阈值 |

---

## 4. 得分计算

先分别计算三个缓存得分：

```text
S_H = alpha_H * 熵缓存相似度投票
S_E = alpha_E * 能量缓存相似度投票
S_A = alpha_A * 对齐缓存相似度投票
```

然后求和：

```text
S_cache = S_H + S_E + S_A
```

所以 `argmax(S_cache)` 是：

```text
先把三个缓存的类别得分向量逐类别相加；
再取相加后得分最高的类别。
```

例如有 3 个类别：

```text
S_H = [0.2, 0.7, 0.1]
S_E = [0.1, 0.5, 0.4]
S_A = [0.0, 0.3, 0.2]

S_cache = [0.3, 1.5, 0.7]
argmax(S_cache) = 1
```

A2 最终得分：

```text
S_final = S_zs + g(x) * S_cache
```

---

## 5. 门控规则

先计算：

```text
zs_pred = argmax(S_zs)
cache_pred = argmax(S_cache)
zs_margin = top1(S_zs) - top2(S_zs)
cache_margin = top1(S_cache) - top2(S_cache)
cache_similarity = 当前样本与三个缓存中所有样本的最大余弦相似度
```

其中 `cache_margin` 的计算方式是：

```text
先得到 S_cache；
找到 S_cache 里最高和第二高的类别得分；
cache_margin = 最高得分 - 第二高得分。
```

`cache_similarity` 的计算方式是：

```text
收集熵缓存、能量缓存、对齐缓存中的所有 pc_feats；
计算当前样本 pc_feats 与每个缓存样本 pc_feats 的点积相似度；
取最大值作为 cache_similarity。
```

因为特征已经归一化，这里的点积就是余弦相似度（cosine similarity）。

门控规则：

```text
如果 cache_pred == zs_pred:
    g = 1.0
否则如果 zs_margin <= 5.0 且 cache_margin >= 1.0 且 cache_similarity >= 0.60:
    g = 1.2
否则:
    g = 0.2
```

直觉：

1. 缓存和零样本预测一致时，缓存是在增强已有判断，比较安全。
2. 缓存和零样本预测不一致时，只有在零样本不够自信、缓存较自信、且当前样本与缓存很相似时，才允许缓存较强地纠错。
3. 其他冲突场景只保留很小缓存影响，避免错误缓存证据强行改写预测。

---

## 6. 诊断指标

A2 继续保留 A1 的 logits norm（得分向量范数）诊断，并新增：

| 指标 | 含义 |
|---|---|
| `test_gate_agree_count` | 使用 `g=1.0` 的次数 |
| `test_gate_correct_count` | 使用 `g=1.2` 的次数 |
| `test_gate_fallback_count` | 使用 `g=0.2` 的次数 |
| `test_gate_value_mean` | 平均门控系数 |
| `test_cache_similarity_mean` | 平均最大缓存相似度 |
| `test_zs_margin_mean` | 平均零样本 margin |
| `test_cache_margin_mean` | 平均缓存 margin |

---

## 7. 注意事项

当前实现沿用 Point-Cache/E4 的 update-then-logits 约定：当前样本先参与缓存更新，再计算最终 logits。因此 `cache_similarity` 可能受到当前样本刚进入缓存的影响，出现偏高现象。

这不会改变 A2 的运行定义，但分析结果时需要重点检查：

```text
test_gate_correct_count
test_cache_similarity_mean
test_gate_value_mean
```

如果 `cache_similarity` 几乎总是很高，说明相似度阈值筛选力不足，后续需要改成“更新缓存前计算相似度”或“只在旧缓存上计算相似度”。

---

## 8. 结果

| 方法 | ModelNet-C S2 平均准确率 |
|---|---:|
| Zero-shot | 47.68 |
| 原始 Global Cache | 52.66 |
| 原始 Global + Local | 54.00 |
| `02_9_2` | 54.71 |
| E7-A0 | 53.31 |
| E7-A1 | 50.97 |
| E7-A2 | 49.98 |

逐项对比 A0/A1：

| corruption | A0 | A1 | A2 | A2 - A1 | A2 - A0 |
|---|---:|---:|---:|---:|---:|
| add_global_2 | 48.50 | 42.38 | 40.15 | -2.23 | -8.35 |
| add_local_2 | 47.57 | 46.64 | 45.58 | -1.06 | -1.99 |
| dropout_global_2 | 57.66 | 56.44 | 56.24 | -0.20 | -1.42 |
| dropout_local_2 | 56.16 | 53.53 | 52.11 | -1.42 | -4.05 |
| rotate_2 | 61.14 | 58.71 | 58.02 | -0.69 | -3.12 |
| scale_2 | 53.04 | 52.27 | 51.74 | -0.53 | -1.30 |
| jitter_2 | 49.11 | 46.84 | 46.03 | -0.81 | -3.08 |

---

## 9. 诊断结果

门控分支占比：

| 指标 | 数值 | 含义 |
|---|---:|---|
| `g=1.0` agree | 51.86% | 缓存预测与零样本预测一致 |
| `g=1.2` corrective | 6.74% | 缓存与零样本不一致，且满足纠错分支条件 |
| `g=0.2` fallback | 41.40% | 缓存证据被弱化 |
| 平均门控系数 | 0.682 | 样本级平均 `g(x)` |

注意：`g=1.2` 的 corrective 只表示“满足设计中的纠错条件”，不表示 ground-truth 上一定预测正确。

门控相关诊断：

| 指标 | 数值 | 解释 |
|---|---:|---|
| 平均 `cache_similarity` | 0.762 | 当前样本与缓存中样本的最大余弦相似度 |
| 平均 `zs_margin` | 3.254 | 零样本 top1-top2 间隔 |
| 平均 `cache_margin` | 0.891 | 缓存 top1-top2 间隔 |
| 熵缓存/能量缓存预测一致率 | 81.41% | `argmax(S_H) == argmax(S_E)` 的比例 |
| 平均预测改变率 | 6.01% | 最终预测相对 zero-shot 发生变化的比例 |

logits norm（得分向量范数）：

| 项目 | 平均范数 |
|---|---:|
| `S_zs` | 33.48 |
| `S_cache = S_H + S_E + S_A` | 7.04 |
| `S_final` | 35.73 |

按 corruption 的门控诊断：

| corruption | agree | corrective | fallback | sim | cache margin | pred change |
|---|---:|---:|---:|---:|---:|---:|
| add_global_2 | 38.45% | 12.84% | 48.70% | 0.873 | 0.997 | 11.02% |
| add_local_2 | 52.15% | 4.50% | 43.35% | 0.764 | 0.755 | 4.78% |
| dropout_global_2 | 59.16% | 4.62% | 36.22% | 0.727 | 0.975 | 3.97% |
| dropout_local_2 | 51.18% | 3.61% | 45.22% | 0.719 | 0.695 | 3.65% |
| rotate_2 | 56.56% | 6.89% | 36.55% | 0.717 | 1.020 | 5.11% |
| scale_2 | 52.92% | 5.96% | 41.13% | 0.707 | 0.851 | 4.82% |
| jitter_2 | 52.59% | 8.75% | 38.65% | 0.827 | 0.947 | 8.71% |

对齐缓存测试阶段触发：

| 指标 | 数值 |
|---|---:|
| `test_alignment_eligible` | 196 |
| `test_alignment_add_not_full` | 8 |
| `test_alignment_replace_joint` | 78 |

---

## 10. 结果分析

A2 没有达到预期。它的平均准确率为 `49.98`，低于 A1 的 `50.97`，更明显低于 A0 的 `53.31`、完整 Point-Cache 的 `54.00` 和当前 anchor `02_9_2` 的 `54.71`。

这说明当前样本级门控（sample-wise gating）没有恢复缓存纠错收益。A2 虽然把预测改变率压低到约 `6.01%`，但被保留下来的缓存干预并没有带来足够正收益。换句话说，A2 主要是在“少改预测”，而不是“更准确地改预测”。

门控诊断显示，`g=1.2` corrective 分支只占 `6.74%`，触发较少；同时 `cache_margin` 平均只有 `0.891`，说明缓存自己的类别区分并不强。缓存证据不是很有决断力，门控规则自然也难以可靠判断什么时候应该相信缓存。

`cache_similarity` 平均达到 `0.762`，其中 `add_global_2` 达到 `0.873`。这需要谨慎解读：当前实现沿用 Point-Cache/E4 的 update-then-logits 约定，先更新缓存，再计算缓存相似度和缓存 logits。因此如果当前样本刚刚进入缓存，它与自身的相似度会接近 1，导致 `cache_similarity` 偏高。这个问题在原始 Point-Cache 中也存在于“先更新缓存、再用缓存打分”的路径里，但 A2 额外把相似度作为门控证据，因此这种自相似会直接削弱门控筛选力。

从 logits norm 看，`S_cache` 平均范数约 `7.04`，明显小于 `S_zs` 的 `33.48`，所以 A2 的失败不应主要归因为缓存得分尺度过大。更合理的解释仍然是：

```text
缓存证据方向质量不足，且当前门控特征无法可靠地区分有益缓存干预和有害缓存干预。
```

熵缓存和能量缓存预测一致率达到 `81.41%`，说明二者高度同向，但这种同向并没有转化为更高准确率。这进一步支持一个判断：仅把熵缓存和能量缓存并列相加，不足以形成真正互补的证据。

---

## 11. 核心假设验证状态

本实验还有一个更核心的验证目标：

```text
同时进入熵缓存和能量缓存、因此有资格进入对齐缓存的样本，
其 zero-shot 伪标签是否高度正确。
```

当前 A2 日志没有记录这个量。具体来说，代码只记录了：

| 已记录指标 | 能说明什么 | 不能说明什么 |
|---|---|---|
| `test_alignment_eligible` | 有多少样本同时被熵缓存和能量缓存接受 | 这些样本的 zero-shot 伪标签是否正确 |
| `test_alignment_add_not_full` / `test_alignment_replace_joint` | 对齐缓存实际添加或替换次数 | 被添加/替换样本是否属于 ground-truth 类别 |
| `test_cache_agreement_HE` | 熵缓存和能量缓存投票是否一致 | 一致的预测是否正确 |
| 最终 `final_acc` | 融合后最终准确率 | 进入对齐缓存之前的伪标签质量 |

因此，当前 A2 不能回答“进入对齐缓存的样本是不是基本都是正确样本”。也不能据此直接支持“把对齐缓存放到前面、先建立高可信对齐分布”的设想。

要验证这个核心假设，需要新增只用于诊断的统计：

| 诊断项 | 定义 |
|---|---|
| `alignment_eligible_zs_correct` | 同一样本同时进入熵缓存和能量缓存时，`argmax(S_zs) == target` 的次数 |
| `alignment_eligible_total` | 同一样本同时进入熵缓存和能量缓存的总次数 |
| `alignment_eligible_zs_acc` | `alignment_eligible_zs_correct / alignment_eligible_total` |
| `alignment_entered_zs_correct` | 样本实际进入或替换对齐缓存时，`argmax(S_zs) == target` 的次数 |
| `alignment_entered_total` | 样本实际进入或替换对齐缓存的总次数 |
| `alignment_entered_zs_acc` | `alignment_entered_zs_correct / alignment_entered_total` |

其中更关键的是 `alignment_entered_zs_acc`，因为它对应真正参与对齐缓存分布更新的样本质量。

---

## 12. 当前结论

A2 的门控融合不适合作为主线方案。它提供的价值主要是诊断性的：

1. 只在最终融合阶段加门控，不能从根本上清理缓存。
2. update-then-logits 会让当前样本自相似影响 `cache_similarity`，使相似度门控偏乐观。
3. 缓存证据的主要问题不是权重尺度，而是进入缓存/进入分布之前的可靠性筛选。

但是，A2 当前结果尚不能证明或否定“对齐缓存样本本身是否高可信”。在决定是否把对齐缓存前置之前，必须先补充 `alignment_entered_zs_acc` 这类诊断。

---

## 13. 下一步计划状态

结果分析已完成。下一步计划需要和用户确认后再写入本文档。
