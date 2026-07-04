# E7-A4-B1：缓存总得分范数裁剪

日期：2026-06-14
状态：已完成，结果已分析

---

## 0. 简要总结

E7-A4-B1 是在修复后 A4 上继续做的第一步改进实验。它不改变候选池
（candidate pool）、对齐核心缓存（alignment core cache）、熵缓存
（entropy cache）、能量缓存（energy cache）的准入和替换规则，只控制缓存得分
对最终预测的影响强度。

核心规则：

```text
S_cache = S_A + S_H + S_E
如果 ||S_cache||_2 > 20:
    S_cache = S_cache * 20 / (||S_cache||_2 + eps)

S_final = S_zs + S_cache
```

这里：

| 符号 | 含义 |
|---|---|
| `S_zs` | 零样本得分（zero-shot logits） |
| `S_A` | 对齐核心缓存得分（alignment core cache logits） |
| `S_H` | 熵缓存得分（entropy cache logits） |
| `S_E` | 能量缓存得分（energy cache logits） |
| `S_cache` | 三个正缓存的总得分 |
| `||S_cache||_2` | 缓存总得分向量的 L2 范数 |

---

## 1. 实验动机

修复后 A4 的主要结果：

| 方法 | ModelNet-C S2 平均准确率 |
|---|---:|
| A4 修复前 | 52.40 |
| A4 修复后 | 53.18 |
| E7-A0 | 53.31 |
| 原始 Global + Local | 54.00 |
| `02_9_2` 当前主线 | 54.71 |

修复后 A4 证明了候选池 + 多指标可靠性准入能提高进入对齐核心缓存样本的可靠性：

| 统计口径 | 正确率 |
|---|---:|
| 候选池累计 zero-shot 伪标签正确率 | 61.45 |
| 对齐核心缓存累计 zero-shot 伪标签正确率 | 62.98 |
| 对齐核心缓存 test 阶段正确率 | 62.63 |

但最终准确率没有超过 A0 和 `02_9_2`。主要异常来自 `add_global_2`：

| corruption | A4 修复后 | A0 | `02_9_2` |
|---|---:|---:|---:|
| add_global_2 | 43.52 | 48.50 | 47.89 |

诊断显示，`add_global_2` 的缓存总得分范数明显偏高：

| corruption | 修复后 `||S_cache||_2` 均值 |
|---|---:|
| add_global_2 | 27.36 |
| add_local_2 | 15.22 |
| dropout_global_2 | 10.86 |
| dropout_local_2 | 13.59 |
| jitter_2 | 18.48 |
| rotate_2 | 10.96 |
| scale_2 | 10.61 |

因此，A4-B1 的目标是先控制缓存得分强度，避免缓存得分在某些扰动下压过
zero-shot 得分。

---

## 2. 严格 TTA 边界

A4-B1 仍然遵守免训练测试时适应（training-free Test-Time Adaptation,
training-free TTA）设定：

1. 不更新点云编码器（point encoder）参数。
2. 不更新文本编码器（text encoder）参数。
3. 不更新文本原型（text prototypes）或分类头。
4. 不使用真实标签参与任何测试时适应决策。
5. 只允许更新候选池、缓存、历史分布、统计量、替换规则和得分融合规则。

缓存总得分范数裁剪只使用当前样本的在线 logits 和缓存得分，不使用真实标签，因此
符合 training-free TTA。

---

## 3. A4-B1 得分规则

修复后 A4 的得分为：

```text
S_final = S_zs + S_A + S_H + S_E
```

A4-B1 改为：

```text
S_cache = S_A + S_H + S_E
cache_norm = ||S_cache||_2
cache_norm_cap = 20.0

如果 cache_norm > cache_norm_cap:
    S_cache = S_cache * cache_norm_cap / (cache_norm + eps)

S_final = S_zs + S_cache
```

对于当前样本进入或替换了对齐核心缓存、熵缓存、能量缓存中任意一个的情况，仍然沿用
A4 的旧/新得分平均机制：

```text
S_old = 用更新前缓存计算并裁剪后的最终得分
S_new = 用更新后缓存计算并裁剪后的最终得分
S_final = 0.5 * S_old + 0.5 * S_new
```

如果当前样本只进入候选池，或者完全被拒绝：

```text
S_final = S_old
```

候选池不参与最终得分。

---

## 4. 为什么第一版用固定阈值 20

固定阈值 `20.0` 的好处是简单、可解释、容易诊断。

选择依据：

1. `add_global_2` 的 `||S_cache||_2` 均值为 `27.36`，明显异常。
2. 其他 corruption 的均值大多低于 `20.0`。
3. 因此 `20.0` 主要压制异常强缓存得分，不会大面积削弱所有缓存贡献。

这个阈值不是最终论文方法的固定结论，只是第一轮验证用的诊断阈值。

---

## 5. 新增诊断

A4-B1 必须新增更完整的诊断，尤其记录历史上进入过候选池的样本分布。

### 5.1 缓存得分范数诊断

| 指标 | 含义 |
|---|---|
| `test_cache_norm_before_mean` | 每次打分调用中，裁剪前 `||S_cache||_2` 均值 |
| `test_cache_norm_before_max` | 每次打分调用中，裁剪前 `||S_cache||_2` 最大值 |
| `test_cache_norm_after_mean` | 每次打分调用中，裁剪后 `||S_cache||_2` 均值 |
| `test_cache_norm_after_max` | 每次打分调用中，裁剪后 `||S_cache||_2` 最大值 |
| `test_cache_norm_clip_count` | 打分调用中触发裁剪的次数 |
| `test_cache_norm_clip_rate` | 打分调用中触发裁剪的比例 |
| `test_positive_cache_total_before_clip_logits_norm_mean` | 当前测试样本最终使用的正缓存总得分在裁剪前的 L2 范数均值 |
| `test_positive_cache_total_after_clip_logits_norm_mean` | 当前测试样本最终使用的正缓存总得分在裁剪后的 L2 范数均值 |
| `zs_vs_final_pred_change` | 最终预测相对 zero-shot 预测改变比例 |

这里“正缓存总得分”明确指：

```text
对齐核心缓存得分（alignment core cache logits）
+ 熵缓存得分（entropy cache logits）
+ 能量缓存得分（energy cache logits）
```

### 5.2 候选池历史样本分布诊断

必须记录历史上进入过候选池的样本，而不是只记录最终候选池快照。

每个进入候选池的样本记录以下统计量：

| 指标 | 含义 |
|---|---|
| `candidate_history_B` | 瓶颈可靠性 `B(x)` |
| `candidate_history_C` | 理想点接近度 `C(x)` |
| `candidate_history_entropy` | 熵（entropy） |
| `candidate_history_energy` | 能量（energy） |
| `candidate_history_margin` | 分类间隔（margin） |
| `candidate_history_zs_correct` | zero-shot 伪标签是否正确，仅离线诊断 |

每个指标至少输出：

```text
count
mean
std
min
p25
p50
p75
p90
p95
max
```

这些诊断用于回答：

1. 进入候选池的样本到底是什么分布；
2. 高 B/C 样本是否真的更可靠；
3. 候选池通过但后续被对齐核心缓存拒绝的样本是否仍然可靠；
4. 是否存在可以提高准入率的阈值区间。

### 5.3 候选池通过但后续未进入三个缓存的样本

需要单独记录：

```text
candidate_only = 进入候选池，但当前样本没有进入或替换：
1. 对齐核心缓存（alignment core cache）
2. 熵缓存（entropy cache）
3. 能量缓存（energy cache）
中的任意一个。
```

记录这些样本的：

```text
B(x), C(x), entropy, energy, margin, zero-shot correctness
```

并与以下两类样本对比：

1. 被候选池拒绝的样本；
2. 成功进入或替换对齐核心缓存、熵缓存、能量缓存中任意一个的样本。

### 5.4 分布一致性拒绝样本诊断

修复后 A4 的测试阶段，对齐核心缓存拒绝中最多的是：

```text
alignment_core_reject_distribution = 1131
```

因此 A4-B1 需要记录被分布一致性拒绝样本的：

```text
B(x), C(x), entropy, energy, margin, current_joint_score, worst_joint_score
```

这样才能判断：这些样本是否其实很可靠，只是分布门控太保守。

---

## 6. 判断标准

A4-B1 成功的最低标准：

1. `add_global_2` 明显恢复；
2. `dropout_global_2`、`scale_2`、`jitter_2` 等修复后 A4 的正向场景不明显下降；
3. 平均准确率超过修复后 A4 的 `53.18`；
4. 最好超过 A0 的 `53.31`；
5. 候选池和对齐核心缓存样本正确率不能明显下降。

如果 A4-B1 提升主要来自 `add_global_2`，且其他 corruption 基本保持，则说明缓存得分强度控制是必要模块。

如果 A4-B1 平均准确率下降，说明固定 `20.0` 裁剪不是有效主线；此时不再继续推进
相对范数裁剪，而应转向“提高可靠样本进入正式缓存的比例”。

---

## 7. 为什么不继续做相对范数裁剪

相对范数裁剪（relative cache norm clipping）原本是 B1 之后的备选思路：

```text
||S_cache||_2 <= rho * ||S_zs||_2
```

但 A4-B1 的结果显示，固定范数裁剪已经让 `add_global_2` 从 `43.52` 下降到
`41.25`。这说明问题不能简单理解为“正缓存总得分太大”。如果继续做相对范数裁剪，
尤其是原计划 `rho=0.6`，很可能进一步压低 `add_global_2` 中仍然有用的缓存信号。

因此该思路已归入 B1 的结果分析，不再作为独立 B2 实验保留。

B2 改为：

```text
候选池 top 样本晋升（candidate pool top-sample promotion）
```

对应文档：

```text
A4_B2_candidate_pool_top_promotion.md
```

---

## 8. B2 逻辑：候选池 top 样本晋升

### 8.1 动机

修复后 A4 测试阶段：

```text
进入候选池：1423
只进入候选池、未进入后续三个缓存：1269
进入/替换对齐核心缓存、熵缓存、能量缓存中任意一个：154
```

这说明候选池里有大量已经通过多指标筛选的样本，没有被后续正式缓存吸收。

这里的正式缓存明确指：

1. 对齐核心缓存（alignment core cache）
2. 熵缓存（entropy cache）
3. 能量缓存（energy cache）

当前 A4 的逻辑更接近：

```text
当前样本进入候选池后，
只让当前样本尝试进入对齐核心缓存。
```

问题是：候选池里可能已经有比当前样本更可靠的历史样本，但它们没有被重新拿出来尝试晋升。

### 8.2 B2 基本思路

B2 的核心是：

```text
候选池不是直接参与最终得分；
候选池 top 样本可以作为晋升候选，尝试进入对齐核心缓存。
```

流程：

```text
1. 当前样本先按 A4 规则尝试进入候选池。
2. 如果候选池发生新增或替换，则重新计算该类别候选池内所有样本的 B(x), C(x)。
3. 从该类别候选池中选出 top1 候选样本。
4. 这个 top1 样本尝试进入对齐核心缓存。
5. 只有成功进入或替换对齐核心缓存的样本，才继续尝试更新熵缓存和能量缓存。
6. 候选池本身仍不直接参与最终得分。
```

第一版建议：

```text
top_k = 1
```

也就是每次候选池更新后，只拿当前类别候选池中最可靠的 top1 样本尝试晋升。

第一版中，top1 晋升成功后从候选池删除。这样候选池作为可靠样本缓冲池，可以持续释放容量，
让后续高质量样本有机会进入。

### 8.3 晋升条件

候选池 top 样本晋升到对齐核心缓存时，仍应满足对齐核心缓存的原始替换逻辑：

```text
如果对齐核心缓存未满：
    加入；

如果对齐核心缓存已满：
    top 样本在 B/C 上优于对齐核心缓存中最差样本；
    且 top 样本的分布一致性得分不能明显差于最差样本；
    才替换。
```

也就是说，B2 主要扩大“谁有资格尝试晋升”的来源。分布一致性不应继续作为过硬的一票否决，
因为 A4-B1 诊断显示，被分布一致性拒绝样本的 zero-shot 伪标签正确率仍有 `67.96%`。

### 8.4 与候选池直接参与得分的区别

B2 不等于候选池直接参与最终得分。

| 方案 | 是否让候选池直接得分 | 风险 |
|---|---|---|
| 候选池直接参与得分 | 是 | 可能把候选池噪声直接加到最终预测 |
| B2 候选池 top 晋升 | 否 | 仍需通过对齐核心缓存筛选，风险较低 |

B2 更稳妥，因为它保留了对齐核心缓存作为二次筛选，而不是把候选池整体变成预测模块。

### 8.5 当前样本与历史样本的得分处理

B2 需要区分晋升样本来源：

| 晋升来源 | 含义 | 当前样本最终得分处理 |
|---|---|---|
| 当前样本 | 当前测试样本本身是候选池 top 样本，并成功晋升 | 沿用 A4 旧/新得分平均 |
| 历史样本 | 候选池中以前进入的样本被选为 top 样本，并成功晋升 | 只更新缓存，当前样本仍使用旧缓存得分 |

这样设计的原因是：

1. 如果晋升的是当前样本，它确实改变了当前样本自己的缓存状态，使用旧/新平均可以削弱
   自我强化。
2. 如果晋升的是历史样本，它并不是当前输入样本；让这个历史晋升立即影响当前预测，会把
   “当前样本预测”和“历史样本晋升”混在一起，难以分析收益来源。
3. 因此历史样本晋升只影响后续测试样本，不直接改变当前样本预测。

第一版 B2 应采用这个保守设定。

### 8.6 需要额外诊断

B2 必须记录：

| 指标 | 含义 |
|---|---|
| `promotion_attempt_count` | 候选池 top 样本尝试晋升次数 |
| `promotion_success_count` | 晋升成功次数 |
| `promotion_success_rate` | 晋升成功率 |
| `promotion_source_current_count` | 晋升样本就是当前样本的次数 |
| `promotion_source_history_count` | 晋升样本来自候选池历史样本的次数 |
| `promotion_removed_from_candidate_count` | 晋升成功后从候选池删除的次数 |
| `promotion_zs_acc` | 晋升样本 zero-shot 伪标签正确率 |

这些诊断可以回答：

1. 候选池里是否有被当前 A4 浪费掉的可靠样本；
2. 从候选池 top 样本晋升是否提高后续三个缓存的准入率；
3. 准入率提高是否以牺牲缓存正确率为代价。

---

## 9. 运行文件规划

已新建独立文件，不直接覆盖 A4 fixed 文件。

模型文件：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache/model_e7_a4_b1_cache_norm_clip.py
```

runner 文件：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache/run_e7_a4_b1_ulip_modelnetc_s2_cache_norm_clip.py
```

common 脚本：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_run_e7_a4_b1_ulip_modelnetc_s2_common.sh
```

入口脚本：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_6_ulip_modelnetc_s2_zs_global_e7_a4_b1_cache_norm_clip_cap20_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh
```

建议相对运行命令：

```bash
cd Point-Cache
bash scripts/E7_entropy_energy_alignment_multicache/00_6_ulip_modelnetc_s2_zs_global_e7_a4_b1_cache_norm_clip_cap20_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh 0
```

结果目录：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_6_ulip_modelnetc_s2_zs_global_e7_a4_b1_cache_norm_clip_cap20_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

---

## 10. 实验结果

结果文件：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_6_ulip_modelnetc_s2_zs_global_e7_a4_b1_cache_norm_clip_cap20_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist/summary.csv
```

### 10.1 逐扰动对比

| 扰动 | PointCache baseline | `02_9_2` | E7-A0 | E7-A4 fixed | E7-A4-B1 | B1-A4 | B1-`02_9_2` |
|---|---:|---:|---:|---:|---:|---:|---:|
| add_global_2 | 47.81 | 47.89 | 48.50 | 43.52 | 41.25 | -2.27 | -6.64 |
| add_local_2 | 46.68 | 50.85 | 47.57 | 48.82 | 48.82 | +0.00 | -2.03 |
| dropout_global_2 | 59.20 | 59.12 | 57.66 | 59.64 | 59.85 | +0.21 | +0.73 |
| dropout_local_2 | 56.69 | 57.21 | 56.16 | 55.43 | 55.51 | +0.08 | -1.70 |
| rotate_2 | 62.07 | 61.30 | 61.14 | 59.36 | 59.28 | -0.08 | -2.02 |
| scale_2 | 55.23 | 55.92 | 53.04 | 55.59 | 55.59 | +0.00 | -0.33 |
| jitter_2 | 50.32 | 50.65 | 49.11 | 49.92 | 49.76 | -0.16 | -0.89 |
| **平均** | **54.00** | **54.71** | **53.31** | **53.18** | **52.87** | **-0.32** | **-1.84** |

### 10.2 主要结论

A4-B1 没有达到预期。固定缓存总得分范数裁剪（fixed cache total logits norm
clipping）把平均准确率从 A4 fixed 的 `53.18` 降到 `52.87`。

最大问题仍然来自 `add_global_2`：

```text
A4 fixed: 43.52
A4-B1:   41.25
变化:    -2.27
```

这说明 `add_global_2` 的问题不能简单解释为“正缓存总得分太大”。虽然 A4-B1
确实把该扰动的正缓存总得分范数从约 `27.36` 压到约 `19.91`，但最终准确率更低。
因此，固定 `cap=20` 裁剪削弱了部分有用缓存信号。

### 10.3 范数裁剪诊断

| 扰动 | 裁剪前 `||S_cache||_2` | 裁剪后 `||S_cache||_2` | 裁剪触发率 | B1-A4 |
|---|---:|---:|---:|---:|
| add_global_2 | 27.36 | 19.91 | 96.17% | -2.27 |
| add_local_2 | 15.21 | 15.21 | 0.56% | +0.00 |
| dropout_global_2 | 10.83 | 10.82 | 0.52% | +0.21 |
| dropout_local_2 | 13.56 | 13.56 | 1.52% | +0.08 |
| rotate_2 | 10.95 | 10.94 | 0.80% | -0.08 |
| scale_2 | 10.59 | 10.59 | 0.20% | +0.00 |
| jitter_2 | 18.48 | 18.14 | 24.48% | -0.16 |

整体裁剪触发率为 `17.71%`，但主要集中在 `add_global_2` 和 `jitter_2`。
`add_global_2` 几乎每个样本都被裁剪，但准确率下降最大；`jitter_2` 有约四分之一
样本被裁剪，准确率小幅下降。这说明固定范数裁剪不是当前最合适的主改进方向。

### 10.4 候选池与缓存可靠性

跨 7 个 severity=2 扰动的累计统计：

| 样本组 | 数量 | zero-shot 伪标签正确率 |
|---|---:|---:|
| 进入候选池的历史样本 | 6886 | 61.36% |
| 只进入候选池、未进入对齐核心/熵/能量缓存 | 6536 | 60.79% |
| 进入或替换对齐核心缓存、熵缓存、能量缓存中任意一个 | 350 | 72.00% |
| 被候选池拒绝 | 27666 | 44.19% |
| 被对齐核心缓存分布一致性拒绝 | 4138 | 67.96% |

这些结果说明：

1. 候选池的多指标可靠性筛选是有效的：被候选池拒绝样本正确率只有 `44.19%`，
   而进入候选池样本正确率为 `61.36%`。
2. 真正进入对齐核心缓存、熵缓存、能量缓存的样本更可靠，正确率达到 `72.00%`。
3. 但分布一致性门控过于保守：被对齐核心缓存分布一致性拒绝的样本正确率仍有
   `67.96%`。这部分样本可能包含大量本应被吸收的可靠样本。

### 10.5 对下一步的影响

A4-B1 的结果推翻了一个简单假设：

```text
只要限制正缓存总得分范数，就能修复 add_global_2。
```

实际结果表明，`add_global_2` 中较强的缓存得分并不一定全是噪声；固定裁剪会把有用
信号一起压掉。因此不建议继续直接沿着 `cap=20` 固定裁剪做主线。

相对范数裁剪思路原计划使用：

```text
||S_cache||_2 <= rho * ||S_zs||_2
```

但如果第一版 `rho=0.6`，对于 `add_global_2` 来说阈值大约会低于或接近 `20`，
很可能比 B1 更强地削弱缓存信号。因此该思路已归入 B1 结果分析，不再作为独立实验保留。

更值得推进的方向是：

1. 提高可靠样本进入对齐核心缓存、熵缓存、能量缓存的比例；
2. 重新设计分布一致性门控，让它从硬拒绝改为可靠性主导、分布辅助；
3. 将“候选池 top 样本晋升”作为 A4-B2 实验。

A4-B2 方向是：

```text
候选池 top 样本晋升 + 分布一致性软门控
```

其中候选池仍不直接参与最终得分，只作为可靠样本来源；对齐核心缓存仍是最终进入
熵缓存、能量缓存之前的核心筛选层。
