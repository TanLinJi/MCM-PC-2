# E7-A4-B2：候选池 Top1 晋升

日期：2026-06-15
状态：已实现，待运行与分析

---

## 0. 简要总结

B2 是在 A4 fixed 和 B1 结果基础上提出的后续实验。

B1 的固定范数裁剪（fixed cache norm clipping）没有解决问题，尤其让
`add_global_2` 从 `43.52` 下降到 `41.25`。因此，B2 不再沿着“调节缓存得分强度”
继续做，而是改为解决另一个瓶颈：

```text
可靠样本从候选池进入正式缓存的比例太低。
```

B2 的核心规则：

```text
候选池 top1 晋升（candidate pool top1 promotion）
```

每次候选池更新后，只选择当前预测类别候选池中最可靠的 top1 样本尝试进入对齐核心缓存
（alignment core cache）。如果晋升成功，就从候选池删除该样本。

这里的正式缓存明确指：

1. 对齐核心缓存（alignment core cache）
2. 熵缓存（entropy cache）
3. 能量缓存（energy cache）

候选池（candidate pool）不直接参与最终得分。

---

## 1. 严格 TTA 边界

B2 仍然是免训练测试时适应（training-free Test-Time Adaptation, training-free TTA）：

1. 不更新点云编码器（point encoder）。
2. 不更新文本编码器（text encoder）。
3. 不更新文本原型（text prototypes）或分类头。
4. 不使用真实标签参与任何测试时适应决策。
5. 真实标签只用于离线诊断，例如晋升样本的 zero-shot 伪标签正确率。
6. 不反向传播（backpropagation-free）。

B2 仍保留预构建缓存阶段（build cache in advance），以便和 A4 fixed、B1 保持可比。

---

## 2. 实验动机

B1 结果说明，固定裁剪正缓存总得分不是有效方向：

| 方法 | ModelNet-C S2 平均准确率 |
|---|---:|
| A4 fixed | 53.18 |
| B1 | 52.87 |

关键诊断：

| 样本组 | 数量 | zero-shot 伪标签正确率 |
|---|---:|---:|
| 进入候选池的历史样本 | 6886 | 61.36% |
| 被候选池拒绝 | 27666 | 44.19% |
| 进入或替换对齐核心缓存、熵缓存、能量缓存中任意一个 | 350 | 72.00% |
| 被对齐核心缓存分布一致性拒绝 | 4138 | 67.96% |

这说明：

1. 候选池筛选有效，被候选池拒绝的样本明显更不可靠。
2. 真正进入正式缓存的样本更可靠，正确率达到 `72.00%`。
3. 但从候选池到正式缓存的流动太弱，大量候选样本没有被利用。

B2 的目标是提高可靠样本进入正式缓存的比例，而不是继续增强或裁剪缓存得分。

---

## 3. B2 与 A4/B1 的区别

### 3.1 A4 原规则

A4 的主要逻辑：

```text
当前样本进入候选池后；
只有当前样本自己尝试进入对齐核心缓存。
```

对齐核心缓存第一次初始化时，A4 会从候选池 top4 直接填满对齐核心缓存。之后，
对齐核心缓存已满，当前样本要替换缓存样本时必须同时满足：

```text
B/C 优于对齐核心缓存中的最差样本；
分布一致性得分高于被替换样本。
```

这个规则的问题是：候选池中的历史 top 样本后续不会主动尝试晋升。

### 3.2 B1 规则

B1 不改变缓存准入路径，只改变最终得分强度：

```text
S_cache = S_A + S_H + S_E
如果 ||S_cache||_2 > 20:
    S_cache = S_cache * 20 / ||S_cache||_2
S_final = S_zs + S_cache
```

B1 失败说明，主要问题不是简单的缓存得分范数过大。

### 3.3 B2 规则

B2 不做范数裁剪，最终得分仍沿用 A4：

```text
S_final = S_zs + S_A + S_H + S_E
```

B2 改变的是候选池到对齐核心缓存的准入路径：

```text
当前样本更新候选池；
候选池更新后，选择该类别候选池 top1；
top1 尝试晋升到对齐核心缓存；
晋升成功后，从候选池删除 top1。
```

一句话：

```text
B1 改得分强度；
B2 改可靠样本流动路径。
```

---

## 4. 缓存结构与容量

B2 继续使用四类结构：

| 结构 | 中文含义 | 容量 |
|---|---|---:|
| candidate pool | 候选池 | 每类 8 |
| alignment core cache | 对齐核心缓存 | 每类 4 |
| entropy cache | 熵缓存 | 每类 3 |
| energy cache | 能量缓存 | 每类 3 |

层级关系：

```text
候选池 -> 对齐核心缓存 -> 熵缓存 / 能量缓存
```

候选池不是得分模块，只是可靠样本缓冲池。

---

## 5. 启动与预构建缓存阶段

B2 保留当前 A4/B1 的预构建缓存阶段：

```text
阶段 1：build cache in advance
阶段 2：test-time evaluation
```

每个 corruption 开始时，缓存和分布都为空：

```text
候选池为空；
对齐核心缓存为空；
熵缓存为空；
能量缓存为空；
对齐核心历史分布为空；
熵历史分布为空；
能量历史分布为空。
```

预构建阶段和正式测试阶段使用同一套 B2 更新规则。这样 B2 和 A4/B1 的差异只来自
“候选池 top1 晋升路径”，而不是来自是否预构建缓存。

---

## 6. 当前样本处理顺序

对每个测试样本，严格按以下顺序：

```text
1. 用旧缓存计算当前样本最终得分；
2. 得到当前样本的 zero-shot logits、预测类别、熵、能量、margin；
3. 当前样本尝试更新候选池；
4. 如果候选池发生新增或替换，则触发 top1 晋升；
5. 如果 top1 晋升到对齐核心缓存，则继续尝试更新熵缓存和能量缓存；
6. 如果晋升样本就是当前样本，则用旧/新缓存得分平均；
7. 如果晋升样本是历史候选样本，则当前样本仍使用旧缓存得分。
```

这样可以避免当前样本先改变缓存，再立刻过度受益于自己造成的缓存变化。

---

## 7. 候选池规则

### 7.1 候选池未满

对于预测类别 `c`：

```text
如果候选池 c 的样本数 < 8：
    当前样本直接加入候选池；
```

如果加入后候选池仍未满 8，则不触发晋升。

如果加入后候选池达到 8，则触发一次 top1 晋升。

### 7.2 候选池已满

候选池已满时：

```text
把当前样本和已有 8 个候选样本放在一起；
重新计算 B(x), C(x)；
按 B(x) 从大到小排序；
B(x) 相同再按 C(x) 从大到小排序；
保留前 8 个。
```

如果当前样本没有进入前 8：

```text
候选池不变；
不触发晋升。
```

如果当前样本进入前 8 并替换掉旧候选样本：

```text
候选池发生更新；
触发 top1 晋升。
```

### 7.3 B(x) 和 C(x)

| 符号 | 中文含义 |
|---|---|
| `B(x)` | 瓶颈可靠性（bottleneck reliability），取熵、能量、分类间隔三项归一化可靠性的最小值 |
| `C(x)` | 理想点接近度（closeness to ideal point），衡量样本整体接近理想可靠样本的程度 |

候选池更新时，先构造临时集合：

```text
P_tmp = 当前类别候选池已有样本 + 当前新样本
```

如果候选池未满，`P_tmp` 就是候选池加入新样本后的集合；如果候选池已满，
`P_tmp` 最多包含 9 个样本。

在 `P_tmp` 内部做 min-max 归一化，并统一成“越大越可靠”：

```text
q_H(x) = (H_max - H(x)) / (H_max - H_min + eps)
q_E(x) = (E_max - E(x)) / (E_max - E_min + eps)
q_M(x) = (M(x) - M_min) / (M_max - M_min + eps)
```

其中：

```text
H(x) = 熵（entropy），越低越可靠
E(x) = 能量（energy），越低越可靠
M(x) = 分类间隔（margin），越高越可靠
```

归一化后：

```text
q_H(x) 越大，说明熵越可靠；
q_E(x) 越大，说明能量越可靠；
q_M(x) 越大，说明分类间隔越可靠。
```

如果某个指标在 `P_tmp` 中所有样本都相同，则该指标没有区分能力。实现时将该指标
所有样本的归一化可靠性设为 `1.0`，让排序由其他指标决定。

瓶颈可靠性（bottleneck reliability）定义为：

```text
B(x) = min(q_H(x), q_E(x), q_M(x))
```

`B(x)` 表示该样本三个可靠性维度中最弱的一项。只要熵、能量、分类间隔中任意一项很差，
`B(x)` 就会很低。

理想点接近度（closeness to ideal point）定义为：

```text
ideal = (1, 1, 1)
nadir = (0, 0, 0)
```

计算样本 `x` 到理想点和负理想点的距离：

```text
D_pos(x) = sqrt((1 - q_H(x))^2 + (1 - q_E(x))^2 + (1 - q_M(x))^2)
D_neg(x) = sqrt(q_H(x)^2 + q_E(x)^2 + q_M(x)^2)
```

然后：

```text
C(x) = D_neg(x) / (D_pos(x) + D_neg(x) + eps)
```

`C(x)` 越大，表示样本越接近理想可靠样本，同时越远离负理想样本。

B2 的排序规则：

```text
先按 B(x) 从大到小排序；
如果 B(x) 相同或几乎相同，再按 C(x) 从大到小排序。
```

B2 中，B/C 只用于候选池更新和 top1 选择，不再要求 top1 的 B/C 必须优于对齐核心缓存
中的最差样本。

---

## 8. 对齐核心缓存规则

### 8.1 对齐核心缓存未满

B2 不再使用 A4 的“候选池第一次满 8 后 top4 一次性初始化”规则。

B2 改为逐步初始化：

```text
候选池达到 8 或发生更新后；
选择该类别候选池 top1；
如果该类别对齐核心缓存未满 4：
    top1 直接进入对齐核心缓存；
    从候选池删除 top1；
    更新对齐核心历史分布。
```

这样对齐核心缓存由多个 top1 晋升逐步填充，而不是一次性 top4 填满。

### 8.2 对齐核心缓存已满

当该类别对齐核心缓存已经满 4：

```text
选择候选池 top1；
计算 top1 的对齐分布得分；
计算对齐核心缓存中每个样本的对齐分布得分；
找到分布得分最低的缓存样本 worst；
如果 top1 的分布得分 > worst 的分布得分：
    top1 替换 worst；
    从候选池删除 top1；
    更新对齐核心历史分布；
否则：
    top1 留在候选池；
    本轮不晋升。
```

B2 第一版只保留分布一致性替换条件，不再额外要求：

```text
top1 的 B/C 必须优于对齐核心缓存中最差样本。
```

理由是：top1 已经由候选池 B/C 排序筛过，再加 B/C 替换条件会过于保守。

---

## 9. 熵缓存和能量缓存规则

熵缓存和能量缓存不直接接收候选池样本。

只有样本成功进入或替换对齐核心缓存后，才有资格继续更新熵缓存和能量缓存。

### 9.1 熵缓存

```text
如果熵缓存未满：
    直接加入；

如果熵缓存已满：
    新样本熵必须低于当前熵缓存中最高熵样本；
    且新样本分布得分高于被替换样本；
    才替换。
```

### 9.2 能量缓存

```text
如果能量缓存未满：
    直接加入；

如果能量缓存已满：
    新样本能量必须低于当前能量缓存中最高能量样本；
    且新样本分布得分高于被替换样本；
    才替换。
```

---

## 10. 最终得分

B2 暂时不参考 BayesMM 改最终得分。

B2 仍沿用 A4 的缓存投票得分：

```text
S_final = S_zs + S_A + S_H + S_E
```

其中：

| 符号 | 含义 |
|---|---|
| `S_zs` | 零样本得分（zero-shot logits） |
| `S_A` | 对齐核心缓存得分（alignment core cache logits） |
| `S_H` | 熵缓存得分（entropy cache logits） |
| `S_E` | 能量缓存得分（energy cache logits） |

如果当前样本本身晋升成功：

```text
S_final = 0.5 * S_old + 0.5 * S_new
```

如果晋升成功的是历史候选样本：

```text
当前样本仍使用 S_old；
缓存更新只影响后续样本。
```

如果没有晋升：

```text
当前样本使用 S_old。
```

后续 B3 可以考虑参考 BayesMM，将最终得分从缓存投票改为分布似然得分。

---

## 11. B2 暂不做的内容

B2 第一版只验证 top1 晋升路径，因此暂不做：

1. 不做软分布更新（soft distribution update）。
2. 不做多峰分布（multi-modal distribution）。
3. 不把最终得分改成分布得分。
4. 不做缓存总得分范数裁剪。
5. 不让候选池直接参与最终得分。

这些方向可以作为后续 B3/B4。

---

## 12. 需要记录的诊断

B2 至少记录：

| 指标 | 含义 |
|---|---|
| `promotion_attempt_count` | 候选池 top1 尝试晋升次数 |
| `promotion_success_count` | 晋升成功次数 |
| `promotion_success_rate` | 晋升成功率 |
| `promotion_source_current_count` | 晋升样本是当前样本的次数 |
| `promotion_source_history_count` | 晋升样本来自历史候选池的次数 |
| `promotion_removed_from_candidate_count` | 晋升成功后从候选池删除的次数 |
| `promotion_zs_acc` | 晋升样本 zero-shot 伪标签正确率，仅离线诊断 |
| `promotion_reject_distribution_count` | top1 因分布得分不足而未晋升次数 |
| `candidate_not_full_count` | 候选池未满导致不能晋升次数 |
| `candidate_to_alignment_gap` | 进入候选池但未进入正式缓存的样本数量 |
| `entered_true_cache_rate` | 当前样本或历史样本进入对齐核心/熵/能量缓存的比例 |

需要逐扰动分析：

1. `add_global_2` 是否恢复；
2. `dropout_global_2`、`scale_2` 等 A4 正向扰动是否保持；
3. 晋升样本正确率是否高于候选池整体；
4. 正式缓存准入率是否高于 A4/B1；
5. 是否因为晋升后删除导致候选池长期不满。

---

## 13. 判断标准

B2 成功的最低标准：

1. 平均准确率超过 A4 fixed 的 `53.18`。
2. 最好超过 E7-A0 的 `53.31`。
3. `add_global_2` 不能继续低于 B1 的 `41.25`。
4. 晋升样本 zero-shot 伪标签正确率应高于候选池整体的 `61.36%`。
5. 正式缓存准入率应高于 A4/B1 的约 `0.89%`。

B2 如果只提高准入率但准确率下降，说明晋升条件太宽，需要恢复部分可靠性约束或进入 B3 的软分布更新/分布得分路线。

---

## 14. 代码与运行入口

### 14.1 代码文件

模型实现：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache/model_e7_a4_b2_candidate_pool_top1_promotion.py
```

扰动数据集 runner：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache/run_e7_a4_b2_ulip_modelnetc_s2_candidate_pool_top1_promotion.py
```

干净数据集 runner：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache/run_e7_a4_b2_ulip_modelnetc_clean_candidate_pool_top1_promotion.py
```

### 14.2 扰动数据集运行命令

从 `Point-Cache` 目录运行：

```bash
bash scripts/E7_entropy_energy_alignment_multicache/00_7_ulip_modelnetc_s2_zs_global_e7_a4_b2_candidate_pool_top1_promotion_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh 0
```

该脚本运行 ModelNet-C severity=2 的 7 个扰动类型。

### 14.3 干净数据集运行命令

从 `Point-Cache` 目录运行：

```bash
bash scripts/E7_entropy_energy_alignment_multicache/00_8_clean_ulip_modelnetc_clean_zs_global_e7_a4_b2_candidate_pool_top1_promotion_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh 0
```

该脚本运行：

```text
data/modelnet_c/clean.h5
```

### 14.4 当前固定参数

```text
候选池（candidate pool）容量 = 8
对齐核心缓存（alignment core cache）容量 = 4
熵缓存（entropy cache）容量 = 3
能量缓存（energy cache）容量 = 3

alpha_ZS = 1.0
alpha_A = 2.0
alpha_H = 2.0
alpha_E = 2.0

beta_A = 3.0
beta_H = 3.0
beta_E = 3.0

text distribution weight = 0.15
score norm mode = running_zscore
old/new score average = 0.5 / 0.5
```

---

## 15. 结果分析

日期：2026-06-15

结果目录：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_7_ulip_modelnetc_s2_zs_global_e7_a4_b2_candidate_pool_top1_promotion_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist/
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_8_clean_ulip_modelnetc_clean_zs_global_e7_a4_b2_candidate_pool_top1_promotion_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist/
```

### 15.1 准确率对比

| Corruption | 02_9_2 | A4 fixed | B1 | B2 | B2 - A4 | B2 - 02_9_2 |
|---|---:|---:|---:|---:|---:|---:|
| add_global_2 | 47.89 | 43.52 | 41.25 | 42.14 | -1.38 | -5.75 |
| add_local_2 | 50.85 | 48.82 | 48.82 | 47.73 | -1.09 | -3.12 |
| dropout_global_2 | 59.12 | 59.64 | 59.85 | 58.83 | -0.81 | -0.29 |
| dropout_local_2 | 57.21 | 55.43 | 55.51 | 56.04 | +0.61 | -1.17 |
| rotate_2 | 61.30 | 59.36 | 59.28 | 58.67 | -0.69 | -2.63 |
| scale_2 | 55.92 | 55.59 | 55.59 | 54.98 | -0.61 | -0.94 |
| jitter_2 | 50.65 | 49.92 | 49.76 | 49.15 | -0.77 | -1.50 |
| **Average** | **54.71** | **53.18** | **52.87** | **52.51** | **-0.68** | **-2.20** |

clean 结果：

| Method | Clean Accuracy |
|---|---:|
| Zero-shot clean baseline | 56.77 |
| PointCache global clean baseline | 62.12 |
| PointCache global+local clean baseline | 64.18 |
| 02_9_2 clean | 63.86 |
| B2 clean | 59.52 |

### 15.2 诊断结论

| Method | 候选池进入数 | 候选池 zero-shot 伪标签正确率 | 对齐核心缓存进入数 | 对齐核心缓存 zero-shot 伪标签正确率 | 测试阶段当前样本进入真正缓存比例 |
|---|---:|---:|---:|---:|---:|
| A4 fixed | 6888 | 61.45% | 1221 | 62.98% | 0.89% |
| B1 | 6886 | 61.36% | 1223 | 63.12% | 0.89% |
| B2 | 8571 | 59.69% | 1140 | 65.88% | 0.60% |

B2 的机制没有达到预期：

1. 候选池进入数从 `6888` 增加到 `8571`，说明删除 top1 后确实释放了候选池位置。
2. 但候选池正确率从 `61.45%` 降到 `59.69%`，说明释放位置后进入候选池的样本整体更杂。
3. 对齐核心缓存进入样本正确率提高到 `65.88%`，说明 top1 晋升样本更可靠。
4. 但对齐核心缓存进入数从 `1221` 降到 `1140`，没有提高正式缓存准入率。
5. 测试阶段当前样本进入真正缓存比例从 `0.89%` 降到 `0.60%`，导致旧/新得分平均几乎不起作用。
6. 最终准确率下降，说明“提高晋升样本纯度”不能弥补“准入率更低”和“缓存投票得分仍可能误导”的问题。

### 15.3 当前判断

B2 第一版不成立。

它验证了一个有用现象：

```text
候选池 top1 样本比候选池整体更可靠。
```

但它没有解决核心问题：

```text
可靠样本进入最终有效预测路径的比例仍然太低；
并且当前的最终得分仍由缓存投票决定，不是由分布似然直接决定。
```

因此，继续只改候选池晋升顺序，收益空间有限。

### 15.4 下一步方向

下一步不建议继续微调 B2 的 top1 晋升规则。

更值得做的是进入 BayesMM 启发的方向：

```text
候选池 / 对齐核心缓存用于维护干净可靠的分布；
最终得分逐步从缓存投票转向分布得分；
缓存样本保留为分布支撑点和近邻证据，而不是直接用伪标签投票主导预测。
```

一个合理的后续实验可以是：

```text
E7-B3:
保留 B2 的候选池 top1 可靠样本发现机制；
但最终分数增加或替换为对齐分布得分；
同时加入与 zero-shot 得分一致性相关的门控，避免低质量分布过早主导预测。
```
