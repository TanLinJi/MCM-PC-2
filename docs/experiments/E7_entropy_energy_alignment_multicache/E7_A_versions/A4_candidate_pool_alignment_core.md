# E7-A4：候选池-对齐核心缓存可靠准入

日期：2026-06-13  
状态：代码已实现，待运行  

---

## 0. 简要总结

E7-A4 是 E7-A 系列的正式第四版实验。它替代早期临时计划
`A4_forward_alignment_distribution_plan.md`，原因是 E7-A3 已经证明：

```text
当前“同时进入熵缓存和能量缓存 -> 再进入对齐缓存”的弱筛选条件，
不能保证对齐缓存样本足够干净。
```

A4 的核心改进是：

```text
候选池（candidate pool）
    -> 对齐核心缓存（alignment core cache）
        -> 熵缓存（entropy cache）与能量缓存（energy cache）
```

也就是说，A4 不再让熵缓存和能量缓存独立决定可信样本；先由候选池
收集候选样本，再由对齐核心缓存维护真正高可信样本，最后熵缓存和
能量缓存只从已经进入对齐核心缓存的样本中更新。

A4 第一版仍使用手动权重（manual weighting），不引入自适应权重
（adaptive weighting）。自适应权重作为后续版本单独设计。

---

## 1. 实验目标

A4 的主要目标不是先追求最终准确率，而是验证：

```text
通过候选池 + 多指标可靠性准入 + 对齐核心缓存，
是否能显著提高真正进入缓存/分布样本的 zero-shot 伪标签正确率。
```

重点对比 A3：

```text
A3 all_alignment_entered_zs_acc = 451 / 916 = 49.24
```

A4 希望验证候选池和更严格准入是否能让进入对齐核心缓存的样本明显更干净。

---

## 2. 严格 TTA 边界

A4 必须遵守免训练测试时适应（training-free Test-Time Adaptation,
training-free TTA）设定：

1. 不更新点云编码器（point encoder）参数。
2. 不更新文本编码器（text encoder）参数。
3. 不更新文本原型（text prototypes）或分类头参数。
4. 不使用真实标签（ground-truth labels）参与任何测试时适应决策。
5. 允许更新的只有候选池、缓存、缓存统计、分布统计、替换规则和诊断计数。

真实标签只用于离线诊断，例如统计进入对齐核心缓存样本的 zero-shot
伪标签正确率。该统计不参与样本进入、替换、最终得分或任何在线更新。

诊断逻辑必须明确标记。等完整实验线跑通后，这类仅用于内部验证的统计逻辑
需要删除或关闭，只保留在归档诊断版本中。

---

## 3. 与早期 A4 临时计划的关系

早期 `A4_forward_alignment_distribution_plan.md` 的想法是：

```text
如果当前对齐缓存进入样本已经高度正确，
则把对齐分布前置。
```

但 A3 结果显示，旧规则下进入对齐缓存样本并不高可信。因此，新 A4 不再
直接沿用“同时进入熵缓存和能量缓存”的样本，而是重新设计入口：

```text
先候选池筛选，再对齐核心缓存接受，最后熵/能量缓存派生更新。
```

---

## 4. 总体结构

A4 的结构是顺序关系，不是三缓存并行关系：

```text
当前样本 x
  -> zero-shot 推理，得到 S_zs、伪标签、熵、能量、分类间隔
  -> 先用旧缓存计算旧得分 S_old
  -> 更新候选池
  -> 若候选池接受样本，则尝试更新对齐核心缓存
  -> 若样本进入/替换对齐核心缓存，则尝试更新熵缓存和能量缓存
  -> 若样本进入了任何真正参与得分的缓存，则用更新后缓存再算 S_new
  -> 最终得分按是否进入缓存决定
```

这里有一个关键区分：

```text
进入候选池 != 进入缓存
```

候选池只负责临时收集和筛选，不参与最终得分。真正参与最终得分的是：

1. 对齐核心缓存（alignment core cache）
2. 熵缓存（entropy cache）
3. 能量缓存（energy cache）

---

## 5. 组件与容量

| 组件 | 容量 | 是否参与最终得分 | 作用 |
|---|---:|---|---|
| 候选池（candidate pool） | 8 / class | 否 | 收集候选样本，并用多指标准则筛选 |
| 对齐核心缓存（alignment core cache） | 4 / class | 是 | 主可信缓存，建立可信对齐分布 |
| 熵缓存（entropy cache） | 3 / class | 是 | 从对齐核心样本中维护低熵视角 |
| 能量缓存（energy cache） | 3 / class | 是 | 从对齐核心样本中维护低能量视角 |

ModelNet40-C 有 40 类，因此候选池最多保存：

```text
40 * 8 = 320
```

这一规模很小，候选池替换计算可以在 CPU 上完成。特征向量仍按当前缓存逻辑
保存即可。

---

## 6. 每个样本保存内容

### 6.1 候选池样本

候选池样本保存：

| 字段 | 含义 |
|---|---|
| `feat` | 点云全局特征（global point-cloud feature） |
| `label` | zero-shot 伪标签，即 `argmax(S_zs)` |
| `entropy` | 基于 zero-shot logits 计算的熵 |
| `energy` | 基于 zero-shot logits 计算的能量 |
| `margin` | `top1_logit - top2_logit` |

候选池不保存真实标签。真实标签只在诊断统计中离线使用。

### 6.2 对齐核心缓存样本

A4 的对齐核心缓存不同于 A0-A3 的后置对齐缓存。A0-A3 中，对齐缓存不需要
保存熵和能量，因为它只接收已经被熵缓存和能量缓存接受的样本。

A4 中，对齐核心缓存本身承担可靠性替换职责，因此需要保存：

| 字段 | 含义 |
|---|---|
| `feat` | 点云全局特征 |
| `label` | zero-shot 伪标签 |
| `entropy` | 用于候选可靠性比较和后续熵缓存更新 |
| `energy` | 用于候选可靠性比较和后续能量缓存更新 |
| `margin` | 用于候选可靠性比较 |

### 6.3 熵缓存样本

熵缓存样本保存：

| 字段 | 含义 |
|---|---|
| `feat` | 点云全局特征 |
| `label` | zero-shot 伪标签 |
| `ctrl` | 熵（entropy） |

熵缓存只对低熵视角负责，不保存能量作为控制量。

### 6.4 能量缓存样本

能量缓存样本保存：

| 字段 | 含义 |
|---|---|
| `feat` | 点云全局特征 |
| `label` | zero-shot 伪标签 |
| `ctrl` | 能量（energy） |

能量缓存只对低能量视角负责，不保存熵作为控制量。

---

## 7. 候选池多指标替换准则

### 7.1 问题

候选池需要同时考虑三个指标：

| 指标 | 方向 | 含义 |
|---|---|---|
| 熵（entropy） | 越低越好 | 预测越确定 |
| 能量（energy） | 越低越好 | logits 整体解释能力越强 |
| 分类间隔（margin） | 越高越好 | 第一类相对第二类优势越明显 |

这三个量纲不同、方向不同，不能直接相加，也不能直接取最大值或最小值。

### 7.2 方向统一与归一化

当某一类候选池已满，新样本到来时，先构造临时集合：

```text
P_tmp = P_c ∪ {x_new}
```

其中 `P_c` 是类别 `c` 当前 8 个候选样本，`x_new` 是新样本。
因此 `P_tmp` 中最多有 9 个样本。

在 `P_tmp` 内部做 min-max 归一化，并统一成“越大越可靠”：

```text
q_H(x) = (H_max - H(x)) / (H_max - H_min + eps)
q_E(x) = (E_max - E(x)) / (E_max - E_min + eps)
q_M(x) = (M(x) - M_min) / (M_max - M_min + eps)
```

其中：

```text
H = entropy
E = energy
M = margin
```

归一化后：

```text
q_H 越大，熵越可靠
q_E 越大，能量越可靠
q_M 越大，分类间隔越可靠
```

如果某个指标在 `P_tmp` 中所有样本都相同，则该指标没有区分能力。实现时可将
该指标所有样本的归一化可靠性设为 `1.0`，让排序由其他指标决定。

### 7.3 瓶颈可靠性

对每个样本定义瓶颈可靠性（bottleneck reliability）：

```text
B(x) = min(q_H(x), q_E(x), q_M(x))
```

`B(x)` 表示该样本三个可靠性维度中最差的一项。A4 优先保留 `B(x)` 高的样本，
因为这类样本没有明显短板。

### 7.4 理想点接近度

再定义理想点接近度（ideal-point closeness）。理想可靠样本为：

```text
ideal = (1, 1, 1)
```

负理想样本为：

```text
nadir = (0, 0, 0)
```

对样本 `x` 计算：

```text
D_pos(x) = sqrt((1 - q_H)^2 + (1 - q_E)^2 + (1 - q_M)^2)
D_neg(x) = sqrt(q_H^2 + q_E^2 + q_M^2)

C(x) = D_neg(x) / (D_pos(x) + D_neg(x) + eps)
```

`C(x)` 越大，表示样本越接近理想可靠样本，同时越远离负理想样本。

### 7.5 排序规则

A4 使用字典序排序（lexicographic ordering）：

```text
先按 B(x) 从大到小排序；
如果 B(x) 相同或几乎相同，再按 C(x) 从大到小排序。
```

也就是：

```text
x_i 优于 x_j，当且仅当：

1. B(x_i) > B(x_j) + eps_B

或

2. |B(x_i) - B(x_j)| <= eps_B 且 C(x_i) > C(x_j) + eps_C
```

第一版使用：

```text
eps_B = 1e-6
eps_C = 1e-6
```

只处理浮点误差，不人为制造“差不多”的区间。

### 7.6 候选池替换

候选池未满时：

```text
直接加入新样本。
```

候选池已满时：

```text
1. 构造 P_tmp = P_c ∪ {x_new}。
2. 对 P_tmp 中 9 个样本计算 q_H, q_E, q_M。
3. 计算 B(x) 和 C(x)。
4. 按 (B desc, C desc) 排序。
5. 保留前 8 个，删除最后 1 个。
```

如果被删除的是新样本：

```text
新样本被候选池拒绝。
```

如果被删除的是旧样本：

```text
新样本进入候选池，并替换旧样本。
```

如果排序完全并列：

```text
保留旧样本，拒绝新样本。
```

---

## 8. 候选池是否维护分布

候选池不维护正式分布。

原因是：候选池样本只是候选样本，还没有被确认可靠。如果用候选池直接构建
分布，会把污染提前引入分布。

A4 允许记录候选池统计作为诊断，例如：

```text
candidate_pool_zs_acc
candidate_pool_entropy_mean
candidate_pool_energy_mean
candidate_pool_margin_mean
```

但这些诊断统计不能参与测试时适应决策，后续正式方法代码中也应删除或关闭。

正式分布只由历史上进入或替换过对齐核心缓存的样本维护。

---

## 9. 对齐核心缓存更新规则

对齐核心缓存是 A4 的主可信缓存。

### 9.1 初始化

某类别候选池达到 8 个样本后，使用第 7 节的候选池排序规则，选出前 4 个样本
初始化该类对齐核心缓存：

```text
alignment_core_capacity = 4
```

这些样本同时用于初始化该类对齐历史分布（alignment history distribution）。

### 9.2 后续替换

当新样本成功进入候选池后，它才有资格尝试进入对齐核心缓存。

如果该类对齐核心缓存未满：

```text
直接加入对齐核心缓存，并更新对齐历史分布。
```

如果该类对齐核心缓存已满：

```text
1. 在 alignment_core ∪ {x_new} 中重新计算 q_H, q_E, q_M。
2. 计算 B(x), C(x)。
3. 找到当前对齐核心缓存中 (B, C) 最差的已有样本 x_worst。
4. 若 x_new 在 (B, C) 上优于 x_worst，则继续检查分布一致性。
5. 若对齐分布可用，要求 x_new 的分布一致性得分高于 x_worst。
6. 同时满足可靠性更好和分布一致性更好时，替换 x_worst。
```

如果对齐分布暂不可用，第一版采用保守策略：

```text
不做满缓存替换。
```

这样可以避免退回到纯指标替换导致早期污染继续扩散。

### 9.3 对齐历史分布

对齐历史分布由历史上所有进入或替换过对齐核心缓存的样本构成，而不是只由
当前缓存快照构成。

```text
alignment_distribution_history =
    all samples ever accepted by alignment core cache
```

这延续 DPC-Point 当前主线思想：分布由历史接受样本维护，而不是仅由当前小容量
缓存维护。

---

## 10. 熵缓存与能量缓存更新规则

熵缓存和能量缓存不再从原始测试样本直接更新。它们只接收已经成功进入或替换
对齐核心缓存的样本。

也就是说：

```text
如果 x 没有进入/替换对齐核心缓存：
    不更新熵缓存；
    不更新能量缓存。

如果 x 成功进入/替换对齐核心缓存：
    x 才有资格更新熵缓存和能量缓存。
```

### 10.1 熵缓存

容量：

```text
entropy_capacity = 3 / class
```

未满时：

```text
直接加入。
```

已满时，找到当前熵最高的样本：

```text
x_worst = argmax entropy
```

新样本替换条件：

```text
entropy_new < entropy_worst
且
distribution_score_new > distribution_score_worst
```

其中分布一致性得分使用熵缓存自身的历史分布，必要时也可以融合文本分布
（text distribution），保持与 E4/E7 当前分布打分框架一致。

如果分布得分不可用，第一版采用保守策略：

```text
不做满缓存替换。
```

### 10.2 能量缓存

容量：

```text
energy_capacity = 3 / class
```

未满时：

```text
直接加入。
```

已满时，找到当前能量最高的样本：

```text
x_worst = argmax energy
```

新样本替换条件：

```text
energy_new < energy_worst
且
distribution_score_new > distribution_score_worst
```

其中分布一致性得分使用能量缓存自身的历史分布，必要时也可以融合文本分布。

如果分布得分不可用，第一版采用保守策略：

```text
不做满缓存替换。
```

### 10.3 与用户确认的规则

本实验采用如下规则：

```text
熵缓存和能量缓存的更新对象，必须来自已经进入或替换对齐核心缓存的样本。

熵缓存替换：新样本熵低于当前熵缓存中最高熵样本，且更符合当前熵缓存分布。

能量缓存替换：新样本能量低于当前能量缓存中最高能量样本，且更符合当前能量缓存分布。
```

---

## 11. 最终得分计算

A4 使用 A0 手动权重：

| 权重 | 数值 | 含义 |
|---|---:|---|
| `alpha_ZS` | 1.0 | 零样本得分权重 |
| `alpha_A` | 2.0 | 对齐核心缓存得分权重 |
| `alpha_H` | 2.0 | 熵缓存得分权重 |
| `alpha_E` | 2.0 | 能量缓存得分权重 |

为了避免误解，正式公式写成：

```text
S_final =
  1.0 * S_zs
+ 2.0 * S_A_raw
+ 2.0 * S_H_raw
+ 2.0 * S_E_raw
```

其中缓存原始得分均为相似度加权投票：

```text
S_cache_raw(c) =
    sum_i exp[-beta * (1 - f · f_i)] * 1[y_i = c]
```

`beta` 第一版沿用 A0：

```text
beta_A = beta_H = beta_E = 3.0
```

---

## 12. 文本端处理与预测解耦

A4 必须继承 `02_9_2` 的文本端解耦设计。

`02_9_2` 是当前 DPC-Point 主线的最强完整 severity=2 锚点：

```text
方法：E4-C-A0+E1-textdist-only
E4_TEXT_SCORE_WEIGHT=0.15
ULIP + ModelNet-C severity=2
Average accuracy = 54.70595045
```

它的关键不是把 E1 的融合文本原型直接用于最终分类，而是：

```text
最终 zero-shot 分类器：继续使用 manual_full 文本原型。
E1 LLM 描述：只用于构建文本分布（text distribution）先验。
```

A4 因此采用同样原则：

| 模块 | 文本来源 | 说明 |
|---|---|---|
| `S_zs` / zero-shot logits | `manual_full` | 最终文本分类器保持原始 Point-Cache 文本锚点 |
| 伪标签 `argmax(S_zs)` | `manual_full` | 候选池类别归属、熵、能量、margin 都基于该 logits |
| 熵（entropy） | `manual_full` logits | 不使用融合文本原型计算 |
| 能量（energy） | `manual_full` logits | `energy = -logsumexp(S_zs)` |
| 分类间隔（margin） | `manual_full` logits | `top1_logit - top2_logit` |
| 文本分布（text distribution） | E1 `manualfull_llm_dynamic_init` | 只作为分布一致性/替换判断的先验 |

E1 文本分布继续按 E1 的分支权重构建：

```text
manual_full branch weight = 0.75
LLM description branch weight = 0.25
```

这表示 A4 使用 E1 的多视角 LLM 描述作为缓存净化的文本先验，但不让它改变
基础 zero-shot 预测器。这样可以避免 `02_7` 中出现的耦合问题：最终
`clip_weights`、伪标签、缓存初始化、负缓存和最终 logits 同时被 E1 prompt
fusion 改变，导致无法判断收益来自哪里。

一句话规则：

```text
A4 的预测端使用 manual_full；
A4 的分布端使用 E1 text distribution；
二者必须解耦。
```

---

## 13. 先旧缓存打分，再更新，再可选平均

A4 修改 A0-A3 的处理顺序。

旧逻辑：

```text
先更新缓存，再计算当前样本最终得分。
```

A4 新逻辑：

```text
先用旧缓存计算 S_old；
再执行候选池和缓存更新；
如果样本进入了参与最终 logits 计算的缓存，再用新缓存计算 S_new；
最终按是否进入缓存决定。
```

这里的“参与最终 logits 计算的缓存”指最终得分缓存（scoring cache），包括：

1. 对齐核心缓存（alignment core cache）
2. 熵缓存（entropy cache）
3. 能量缓存（energy cache）

候选池（candidate pool）不属于最终得分缓存。候选池只负责临时筛选样本，不直接参与
`S_final` 的计算。

### 13.1 未进入最终得分缓存

如果样本只进入候选池，或者完全被拒绝，没有进入对齐核心缓存、熵缓存、能量缓存
中的任何一个：

```text
S_final = S_old
```

候选池不参与最终得分。

### 13.2 进入最终得分缓存

如果样本进入或替换了对齐核心缓存、熵缓存、能量缓存中的任意一个：

```text
S_new = 1.0*S_zs + 2.0*S_A_raw_new + 2.0*S_H_raw_new + 2.0*S_E_raw_new

S_final = 0.5 * S_old + 0.5 * S_new
```

这样既保留即时适应收益，又削弱当前样本给自己伪标签直接加分造成的自我强化。

---

## 14. 诊断指标

A4 第一版必须记录以下诊断指标。注意：这些统计用于离线分析，后续正式方法代码
需要删除或关闭。

### 14.1 候选池诊断

| 指标 | 含义 |
|---|---|
| `candidate_add_not_full` | 候选池未满直接加入次数 |
| `candidate_replace` | 新样本替换旧候选样本次数 |
| `candidate_reject` | 新样本被候选池拒绝次数 |
| `candidate_B_mean` | 候选池瓶颈可靠性均值 |
| `candidate_C_mean` | 候选池理想点接近度均值 |
| `candidate_zs_acc` | 候选池进入样本 zero-shot 伪标签离线正确率 |

### 14.2 对齐核心缓存诊断

| 指标 | 含义 |
|---|---|
| `alignment_core_add` | 对齐核心缓存加入次数 |
| `alignment_core_replace` | 对齐核心缓存替换次数 |
| `alignment_core_reject_reliability` | 因 `(B, C)` 不优被拒绝次数 |
| `alignment_core_reject_distribution` | 因分布一致性不优被拒绝次数 |
| `alignment_core_entered_zs_acc` | 历史进入/替换对齐核心缓存样本 zero-shot 伪标签正确率 |

### 14.3 熵/能量缓存诊断

| 指标 | 含义 |
|---|---|
| `entropy_add` | 熵缓存加入次数 |
| `entropy_replace` | 熵缓存替换次数 |
| `entropy_reject_ctrl` | 因熵不低于当前最差样本被拒绝次数 |
| `entropy_reject_distribution` | 因分布一致性不优被拒绝次数 |
| `energy_add` | 能量缓存加入次数 |
| `energy_replace` | 能量缓存替换次数 |
| `energy_reject_ctrl` | 因能量不低于当前最差样本被拒绝次数 |
| `energy_reject_distribution` | 因分布一致性不优被拒绝次数 |

### 14.4 得分诊断

| 指标 | 含义 |
|---|---|
| `score_old_acc` | 旧缓存得分预测准确率 |
| `score_new_acc` | 更新后缓存得分预测准确率，仅对进入缓存样本统计 |
| `score_avg_acc` | 平均后最终得分预测准确率 |
| `entered_cache_rate` | 当前样本进入最终得分缓存的比例，即进入对齐核心缓存、熵缓存、能量缓存中任意一个的比例 |
| `candidate_only_rate` | 只进入候选池但未进入最终得分缓存的比例 |

---

## 15. 实现文件、脚本与命令

模型文件：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache/model_e7_a4_candidate_pool_alignment_core.py
```

Runner 文件：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache/run_e7_a4_ulip_modelnetc_s2_candidate_pool_alignment_core.py
```

公共启动脚本：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_run_e7_a4_ulip_modelnetc_s2_common.sh
```

实验入口脚本：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_5_ulip_modelnetc_s2_zs_global_e7_a4_candidate_pool_alignment_core_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh
```

运行命令：

```bash
cd Point-Cache
bash scripts/E7_entropy_energy_alignment_multicache/00_5_ulip_modelnetc_s2_zs_global_e7_a4_candidate_pool_alignment_core_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh 0
```

结果目录：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_5_ulip_modelnetc_s2_zs_global_e7_a4_candidate_pool_alignment_core_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

命名含义：

| 片段 | 含义 |
|---|---|
| `e7_a4` | E7-A4 实验 |
| `candidate_pool_alignment_core` | 候选池 + 对齐核心缓存 |
| `k8` | 候选池容量 8 |
| `a4` | 对齐核心缓存容量 4 |
| `h3` | 熵缓存容量 3 |
| `e3` | 能量缓存容量 3 |
| `a0weights` | 使用 A0 权重 |
| `manualfull_llm_dynamic_init_textdist` | 沿用 E7 当前文本分布设定 |

---

## 16. 预期判断标准

A4 结果分析重点不是只看最终准确率，而是先看样本质量：

| 对比 | 目标 |
|---|---|
| `alignment_core_entered_zs_acc` vs A3 `49.24` | 判断对齐核心缓存是否更干净 |
| A4 final acc vs A0 `53.31` | 判断结构变化是否损害最终预测 |
| A4 final acc vs `02_9_2` `54.71` | 判断是否接近当前主线锚点 |
| `score_old_acc` vs `score_avg_acc` | 判断更新后平均得分是否有帮助 |
| `candidate_reject/replace` | 判断候选池是否真的在筛选 |

如果进入对齐核心缓存样本正确率明显高于 A3，说明 A4 的可靠准入方向成立。

如果进入对齐核心缓存样本仍接近 `50%`，说明熵、能量、分类间隔三类信号仍不足，
需要寻找新的可靠性信号，而不是继续调权重。

---

## 17. 方法依据

A4 的设计参考以下思想，但不直接采用它们的训练式更新：

1. 测试时适应中的可靠样本选择（reliable sample selection）：EATA 指出并非所有
   测试样本都应参与适应，高熵样本可能带来噪声。
2. 在线记忆库维护（online memory bank）：RoTTA 使用记忆库并考虑不确定性和
   类别均衡，说明在线样本库维护是 TTA 中合理的结构。
3. 理想点多指标排序（TOPSIS-style multi-criteria ranking）：对不同方向、不同
   尺度的指标先归一化，再比较其接近理想点和远离负理想点的程度。

A4 与这些工作的关键区别是：A4 保持免训练 TTA，不做反向传播，不更新模型参数，
只更新候选池、缓存和历史分布。

---

## 18. 后续计划状态

本文件记录的是 E7-A4 已确认并已实现的实验方案。下一步应先做 quick/smoke
运行检查，再运行完整 S2 七类 corruption 实验。

实验结果出来后，可以直接补写结果分析；下一步改进计划仍需和用户确认后再写入。

---

## 19. 实验结果分析

日期：2026-06-13

结果目录：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_5_ulip_modelnetc_s2_zs_global_e7_a4_candidate_pool_alignment_core_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

完整性检查：

| 检查项 | 当前值 | 说明 |
|---|---:|---|
| `summary.csv` 数据行 | 7 | S2 七类 corruption 全部完成 |
| `status=done` 行数 | 7 | 无失败项 |
| `e7_stats` 文件数 | 7 | 每个 corruption 都有诊断统计 |

### 19.1 最终准确率

| 方法 | ModelNet-C S2 平均准确率 |
|---|---:|
| Zero-shot | 47.68 |
| 原始 Global Cache | 52.66 |
| 原始 Global + Local | 54.00 |
| `02_9_2` | 54.71 |
| E7-A0 | 53.31 |
| E7-A3 | 52.72 |
| E7-A4 | 52.40 |

E7-A4 相比 Zero-shot 仍有 `+4.72` 的提升，但低于原始 Global Cache `-0.26`，
低于原始 Global + Local `-1.60`，低于当前主线 `02_9_2` `-2.30`，也低于
E7-A0 `-0.91` 和 E7-A3 `-0.31`。

逐 corruption 结果：

| corruption | A0 | A3 | A4 | A4-A0 | A4-A3 |
|---|---:|---:|---:|---:|---:|
| add_global_2 | 48.50 | 45.91 | 46.39 | -2.11 | +0.48 |
| add_local_2 | 47.57 | 48.14 | 47.85 | +0.28 | -0.29 |
| dropout_global_2 | 57.66 | 56.97 | 56.89 | -0.77 | -0.08 |
| dropout_local_2 | 56.16 | 55.71 | 54.94 | -1.22 | -0.77 |
| rotate_2 | 61.14 | 59.44 | 59.12 | -2.02 | -0.32 |
| scale_2 | 53.04 | 53.53 | 53.61 | +0.57 | +0.08 |
| jitter_2 | 49.11 | 49.31 | 48.01 | -1.10 | -1.30 |
| **Average** | **53.31** | **52.72** | **52.40** | **-0.91** | **-0.31** |

从最终准确率看，当前这次 A4 运行不是有效提升版本。

### 19.2 核心诊断：进入样本是否更可靠

A3 的关键累计指标是：

```text
A3 all_alignment_entered_zs_acc = 451 / 916 = 49.24
```

A4 的累计结果为：

| 统计口径 | correct / total | 正确率 |
|---|---:|---:|
| A4 all candidate entered | 4233 / 6888 | 61.45 |
| A4 all alignment core entered | 691 / 1298 | 53.24 |
| A4 test candidate entered | 970 / 1423 | 68.17 |
| A4 test alignment core entered | 94 / 143 | 65.73 |

这里有两个相反信号：

1. 候选池（candidate pool）筛选是有价值的。累计候选池进入样本 zero-shot 伪标签
   正确率达到 `61.45`，明显高于整体 Zero-shot `47.68`。
2. 对齐核心缓存（alignment core cache）没有继续净化样本。累计对齐核心缓存进入
   正确率只有 `53.24`，比候选池低 `8.22`，说明第二阶段筛选反而损失了样本质量。

与 A3 相比，A4 的累计对齐核心缓存正确率从 `49.24` 提升到 `53.24`，提升
`+3.99`。这说明多指标候选池有正向信号，但提升还不够大，不能称为高可信入口。

测试阶段单独看，A4 的对齐核心缓存进入正确率为 `65.73`，明显高于 A3 的测试阶段
`56.36`。这说明测试流里的候选筛选更干净；但累计统计仍被 build 阶段早期样本稀释。

逐 corruption 诊断：

| corruption | A4 all candidate | A4 all alignment core | A4 test alignment core |
|---|---:|---:|---:|
| add_global_2 | 60.59 | 56.13 | 59.09 |
| add_local_2 | 56.75 | 41.32 | 38.89 |
| dropout_global_2 | 67.97 | 61.50 | 79.17 |
| dropout_local_2 | 59.81 | 52.63 | 66.67 |
| jitter_2 | 54.83 | 49.46 | 73.68 |
| rotate_2 | 66.98 | 56.86 | 66.67 |
| scale_2 | 61.95 | 53.03 | 72.22 |

`add_local_2` 是最明显的异常：候选池累计正确率为 `56.75`，但对齐核心缓存只有
`41.32`。这说明对齐核心缓存的进入/替换逻辑在该 corruption 上明显选错样本。

### 19.3 得分诊断

| 指标 | 数值 |
|---|---:|
| 测试样本总数 | 17276 |
| 进入最终得分缓存样本数 | 143 |
| 进入最终得分缓存比例 | 0.83 |
| 只进入候选池比例 | 7.41 |
| 旧缓存得分准确率 | 52.39 |
| 平均后最终得分准确率 | 52.40 |
| 新缓存得分准确率，限进入缓存样本 | 65.03 |
| 最终预测相对 zero-shot 改变比例 | 19.09 |

`S_old/S_new` 平均机制几乎没有改变最终结果：全测试集只多正确 2 个样本。原因是
进入最终得分缓存的测试样本只有 `0.83%`，触发率太低。这里的最终得分缓存指会参与
`S_final` 计算的对齐核心缓存、熵缓存和能量缓存，不包括候选池。进入最终得分缓存的
样本本身较可靠，但数量不足以显著改变整体准确率。

logits 范数诊断：

| 项目 | 平均范数 |
|---|---:|
| zero-shot logits | 33.48 |
| positive cache total logits | 16.11 |
| final logits | 42.54 |

因此本轮主要问题不是缓存 logits 范数过大，而是缓存方向和缓存样本质量不足。

### 19.4 实现偏差：当前运行并不完全等价于设计版 A4

这次结果里发现一个关键实现偏差：

```text
alignment_core_init_from_candidate = 0
```

按照设计，某一类候选池满 8 个样本后，应该从候选池中按 `(B, C)` 选出 top4
初始化对齐核心缓存。但当前代码实际运行中，对齐核心缓存没有通过候选池 top4 初始化。

原因是：当候选池未满 8 时，当前样本仍会进入 `_update_alignment_core`，随后由于该类
对齐核心缓存未满，样本会被直接加入对齐核心缓存。这样对齐核心缓存会被早期样本填满，
导致后续“候选池满 8 后 top4 初始化”的逻辑永远不会触发。

对应统计也支持这一点：

| 事件 | build | test | all |
|---|---:|---:|---:|
| `alignment_core_init_from_candidate` | 0 | 0 | 0 |
| `alignment_core_add_not_full` | 1018 | 14 | 1032 |
| `alignment_core_replace` | 137 | 129 | 266 |
| `alignment_core_reject_distribution` | 4011 | 1232 | 5243 |

这意味着当前 A4 实际上不是“候选池先筛选，再初始化对齐核心缓存”，而是：

```text
候选池记录样本
同时早期样本直接进入对齐核心缓存
之后再用候选池/分布规则做少量替换
```

这个偏差会污染对齐核心缓存的早期历史分布，也解释了为什么候选池正确率明显高于
对齐核心缓存正确率。

### 19.5 当前结论

这次 A4 结果不能直接否定“候选池 + 多指标可靠性准入”方向，因为候选池本身表现出
明显的可靠样本筛选能力：

```text
all candidate entered zs acc = 61.45
test candidate entered zs acc = 68.17
```

但这次运行也不能作为设计版 A4 的有效验证，因为对齐核心缓存没有按设计从候选池 top4
初始化。当前准确率下降和累计对齐核心样本质量不足，很大程度上可能来自这一实现偏差。

因此，本轮最稳妥的判断是：

```text
候选池多指标筛选方向有价值；
当前对齐核心缓存实现存在偏差；
需要修复“候选池满后再初始化/更新对齐核心缓存”的逻辑后，才能判断正式 A4 是否成立。
```

下一步改进计划待与用户确认后再写入本文档。

### 19.6 修复记录

日期：2026-06-14

已修复 A4 代码中的候选池/对齐核心缓存状态边界问题。

修复前：

```text
候选池未满 8 时，当前样本仍可能直接进入对齐核心缓存。
```

修复后：

```text
候选池未满 8 时：
    只更新候选池；
    不更新对齐核心缓存；
    不更新熵缓存；
    不更新能量缓存。

候选池达到 8 后：
    才按 (B, C) 排序选 top4 初始化对齐核心缓存。
```

这次问题属于实现逻辑没有忠实执行设计，不是原始 A4 方案本身必然矛盾。修复后需要重新
运行 A4，新的结果才能代表设计版 A4。

---

## 20. 修复后实验结果分析

日期：2026-06-14

结果目录：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_5_ulip_modelnetc_s2_zs_global_e7_a4_candidate_pool_alignment_core_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

修复前结果备份目录：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_5_ulip_modelnetc_s2_zs_global_e7_a4_candidate_pool_alignment_core_k8_a4_h3_e3_a0weights_manual_full_clip_manualfull_llm_dynamic_init_textdist_before_fix_20260614
```

### 20.1 完整性与修复确认

| 检查项 | 当前值 | 说明 |
|---|---:|---|
| `summary.csv` 数据行 | 7 | S2 七类 corruption 全部完成 |
| `status=done` 行数 | 7 | 无失败项 |
| `e7_stats` 文件数 | 7 | 每个 corruption 都有诊断统计 |
| `alignment_core_init_from_candidate` | 980 | 候选池满后初始化对齐核心缓存已触发 |
| `alignment_core_wait_candidate_full` | 1791 | 候选池未满时等待，不提前写入对齐核心缓存 |
| `alignment_core_add_not_full` | 0 | 已修复“未满直接加入对齐核心缓存”的问题 |

结论：修复后的代码现在符合 A4 设计，即候选池未满时不会提前污染对齐核心缓存。

### 20.2 最终准确率

| 方法 | ModelNet-C S2 平均准确率 |
|---|---:|
| Zero-shot | 47.68 |
| 原始 Global Cache | 52.66 |
| 原始 Global + Local | 54.00 |
| `02_9_2` 当前主线 | 54.71 |
| E7-A0 | 53.31 |
| E7-A3 | 52.72 |
| E7-A4 修复前 | 52.40 |
| E7-A4 修复后 | 53.18 |

修复后 A4 相比修复前提升 `+0.78`，相比 A3 提升 `+0.47`，相比原始 Global Cache
提升 `+0.52`。但它仍低于 A0 `-0.13`，低于原始 Global + Local `-0.82`，低于
`02_9_2` 当前主线 `-1.52`。

逐 corruption 对比：

| corruption | A0 | A3 | A4 修复前 | A4 修复后 | 修复后-A0 | 修复后-`02_9_2` |
|---|---:|---:|---:|---:|---:|---:|
| add_global_2 | 48.50 | 45.91 | 46.39 | 43.52 | -4.98 | -4.37 |
| add_local_2 | 47.57 | 48.14 | 47.85 | 48.82 | +1.25 | -2.03 |
| dropout_global_2 | 57.66 | 56.97 | 56.89 | 59.64 | +1.98 | +0.52 |
| dropout_local_2 | 56.16 | 55.71 | 54.94 | 55.43 | -0.73 | -1.78 |
| rotate_2 | 61.14 | 59.44 | 59.12 | 59.36 | -1.78 | -1.94 |
| scale_2 | 53.04 | 53.53 | 53.61 | 55.59 | +2.55 | -0.33 |
| jitter_2 | 49.11 | 49.31 | 48.01 | 49.92 | +0.81 | -0.73 |
| **Average** | **53.31** | **52.72** | **52.40** | **53.18** | **-0.13** | **-1.52** |

正向场景主要是 `dropout_global_2`、`scale_2`、`jitter_2` 和 `add_local_2`。负向场景
最明显的是 `add_global_2`。

### 20.3 核心诊断：对齐核心缓存是否变干净

修复后，A4 的对齐核心缓存样本质量明显改善：

| 统计口径 | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|
| 候选池累计 zero-shot 伪标签正确率 | 61.45 | 61.45 | +0.00 |
| 对齐核心缓存累计 zero-shot 伪标签正确率 | 53.24 | 62.98 | +9.74 |
| 对齐核心缓存 build 阶段正确率 | 51.69 | 63.05 | +11.36 |
| 对齐核心缓存 test 阶段正确率 | 65.73 | 62.63 | -3.10 |

候选池正确率保持不变，说明修复没有改变候选池本身。对齐核心缓存累计正确率从
`53.24` 提升到 `62.98`，说明“候选池满后再初始化/更新对齐核心缓存”的设计确实
提高了缓存样本质量。

与 A3 的关键诊断相比：

```text
A3 all_alignment_entered_zs_acc = 49.24
A4 fixed all_alignment_core_entered_zs_acc = 62.98
```

这说明 A4 的候选池 + 多指标准入机制，至少在“进入对齐核心缓存的样本是否可靠”这个
目标上是成立的。

逐 corruption 的对齐核心缓存累计正确率：

| corruption | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|
| add_global_2 | 56.13 | 59.68 | +3.55 |
| add_local_2 | 41.32 | 58.67 | +17.35 |
| dropout_global_2 | 61.50 | 70.62 | +9.12 |
| dropout_local_2 | 52.63 | 58.15 | +5.52 |
| jitter_2 | 49.46 | 58.99 | +9.53 |
| rotate_2 | 56.86 | 67.98 | +11.12 |
| scale_2 | 53.03 | 63.83 | +10.80 |

所有 corruption 的对齐核心缓存样本质量都提高，其中 `add_local_2` 提升最大。

### 20.4 得分诊断

| 指标 | 修复后数值 |
|---|---:|
| 测试样本总数 | 17276 |
| 当前测试样本进入/替换对齐核心缓存、熵缓存、能量缓存中任意一个的数量 | 154 |
| 当前测试样本进入/替换对齐核心缓存、熵缓存、能量缓存中任意一个的比例 | 0.89 |
| 只进入候选池比例 | 7.35 |
| 旧缓存得分准确率 | 53.18 |
| 平均后最终得分准确率 | 53.18 |
| 新缓存得分准确率，限进入缓存样本 | 70.13 |
| 最终预测相对 zero-shot 改变比例 | 19.27 |

修复后，当前测试样本进入或替换对齐核心缓存、熵缓存、能量缓存中任意一个时更可靠：
`S_new` 在这些样本上的准确率达到 `70.13`。但这样的测试样本仍只有
`154/17276 = 0.89%`，触发率太低，因此
`S_old/S_new` 平均机制对整体结果几乎没有贡献，只多正确 1 个样本。

也就是说，A4 现在解决了“进入对齐核心缓存的样本是否更干净”的问题，但没有充分解决
“干净样本如何有效影响最终预测”的问题。

### 20.5 add_global_2 异常

`add_global_2` 是修复后最主要的负向来源：

```text
A4 fixed add_global_2 = 43.52
A0 add_global_2 = 48.50
02_9_2 add_global_2 = 47.89
原始 Global + Local add_global_2 = 47.81
```

诊断上，`add_global_2` 的对齐核心缓存累计正确率为 `59.68`，并不是最低；但它的
缓存总得分范数明显偏高：

| corruption | positive cache total logits norm |
|---|---:|
| add_global_2 | 27.36 |
| add_local_2 | 15.22 |
| dropout_global_2 | 10.86 |
| dropout_local_2 | 13.59 |
| jitter_2 | 18.48 |
| rotate_2 | 10.96 |
| scale_2 | 10.61 |

这说明 `add_global_2` 的主要问题不是“对齐核心缓存样本完全不干净”，而是缓存得分
在该 corruption 上过强，导致最终预测相对 zero-shot 改变比例达到 `27.96%`，
高于其他 corruption。换句话说，缓存投票方向即使有一定可靠性，也可能被过大的缓存
得分权重放大，最终压过 zero-shot 的正确判断。

这里的缓存总得分范数（positive cache total logits norm）指：

```text
S_cache = S_A + S_H + S_E
positive_cache_total_logits_norm = ||S_cache||_2
```

其中：

| 符号 | 含义 |
|---|---|
| `S_A` | 对齐核心缓存（alignment core cache）产生的类别得分 |
| `S_H` | 熵缓存（entropy cache）产生的类别得分 |
| `S_E` | 能量缓存（energy cache）产生的类别得分 |
| `||.||_2` | L2 范数，即整条类别得分向量的大小 |

它在每个测试样本计算最终得分时得到，并在记录预测前写入诊断统计。它本身只是诊断量；
真正参与预测的是 `S_cache` 这个向量：

```text
S_final = S_zs + S_cache
```

所以如果 `||S_cache||_2` 过大，缓存得分就会强烈改变最终预测，即使缓存样本质量不低，
也可能把一部分原本正确的 zero-shot 判断拉向错误类别。

### 20.6 当前结论

修复后的 A4 证明了一个重要点：

```text
候选池 + 多指标可靠性准入，可以显著提高进入对齐核心缓存样本的可靠性。
```

但它还没有成为最终性能最优方法：

```text
最终准确率 53.18，仍低于 A0 53.31，也明显低于 02_9_2 的 54.71。
```

因此，A4 的价值更像是证明“可靠样本筛选机制有效”，而不是直接作为当前最强版本。
当前瓶颈从“样本是否干净”转移到了：

1. 干净样本触发率太低；
2. 缓存得分对不同 corruption 的强度不稳定；
3. `add_global_2` 中缓存得分过强，导致明显负迁移；
4. A4 当前只使用全局特征，没有使用原始 Point-Cache 的局部缓存补充。

下一步改进计划见第 21 节。

---

## 21. 下一步实验方向

根据 20 节结果，下一步优先处理的不是“继续提高对齐核心缓存样本正确率”，而是控制
缓存得分对最终预测的影响强度。修复后 A4 已经证明对齐核心缓存样本更干净，但
`add_global_2` 显示缓存得分可能过强，造成负迁移。

### 21.1 A4-B1：缓存总得分范数裁剪

目标：

```text
限制 S_cache = S_A + S_H + S_E 的整体强度，
避免某些 corruption 下缓存得分压过 zero-shot 得分。
```

第一版规则：

```text
cache_norm = ||S_cache||_2
cache_norm_cap = 20.0

如果 cache_norm > cache_norm_cap:
    S_cache = S_cache * cache_norm_cap / (cache_norm + eps)

S_final = S_zs + S_cache
```

这样不会改变缓存得分类别方向，只缩小缓存得分向量的整体长度。这个规则仍符合
免训练测试时适应（training-free Test-Time Adaptation, training-free TTA），因为它
只使用当前样本的在线 logits 和缓存得分，不使用真实标签，也不更新模型参数。

为什么先选 `20.0`：

| corruption | 修复后 `||S_cache||_2` 均值 |
|---|---:|
| add_global_2 | 27.36 |
| add_local_2 | 15.22 |
| dropout_global_2 | 10.86 |
| dropout_local_2 | 13.59 |
| jitter_2 | 18.48 |
| rotate_2 | 10.96 |
| scale_2 | 10.61 |

`20.0` 主要会压制 `add_global_2` 的异常强缓存得分，对其他 corruption 的均值影响较小。
因此它适合作为第一轮诊断实验。

需要记录的新诊断：

| 指标 | 含义 |
|---|---|
| `cache_norm_clip_count` | 触发范数裁剪的样本数 |
| `cache_norm_clip_rate` | 触发范数裁剪比例 |
| `cache_norm_before_mean` | 裁剪前缓存总得分范数均值 |
| `cache_norm_after_mean` | 裁剪后缓存总得分范数均值 |
| `zs_vs_final_pred_change` | 最终预测相对 zero-shot 改变比例 |

判断标准：

1. `add_global_2` 是否明显恢复；
2. `dropout_global_2`、`scale_2`、`jitter_2` 等正向场景是否没有被明显削弱；
3. 平均准确率是否超过 A4 修复后 `53.18`，并尽量超过 A0 `53.31`。

### 21.2 A4-B2：相对范数裁剪，暂不优先

如果 A4-B1 有正向趋势，但固定阈值 `20.0` 不够稳定，可以尝试相对范数裁剪：

```text
cache_norm <= rho * ||S_zs||_2
```

例如第一版可测试：

```text
rho = 0.6
```

该规则比固定阈值更自适应，但也多一个超参数。为了避免过早增加复杂度，先不作为下一轮
首选实验。

### 21.3 候选池直接参与得分：记录为消融实验，暂缓

用户提出的方向：

```text
去掉对齐核心缓存；
候选池 -> 熵缓存 / 能量缓存；
候选池直接参与最终得分计算。
```

当前判断：

1. 这个方向可以做消融实验（ablation），因为候选池累计 zero-shot 伪标签正确率为
   `61.45`，已经明显高于整体 zero-shot `47.68`。
2. 但候选池不是强可信缓存，它只是相对可靠的临时池。如果直接参与最终得分，可能扩大
   样本数量，同时也可能把更多噪声票带入预测。
3. 修复后对齐核心缓存累计正确率为 `62.98`，已经高于候选池的 `61.45`，所以当前
   不能简单认为对齐核心缓存有害。

因此该方向暂时记录为后续消融实验，不作为 A4-B1 的替代方案。更合理的消融设置应当是：

| 版本 | 目的 |
|---|---|
| 保留对齐核心缓存，不让候选池得分 | 当前 A4 fixed 主线 |
| 保留对齐核心缓存，让候选池低权重得分 | 测试候选池是否能提高触发覆盖 |
| 去掉对齐核心缓存，让候选池得分 | 测试对齐核心缓存是否必要 |
| 去掉对齐核心缓存，候选池只派生熵/能量缓存 | 测试候选池作为入口是否足够 |

该消融应在缓存得分强度控制之后再做，否则候选池直接得分可能与 logits 范数过强问题混在一起，
难以判断收益来源。

---

## 22. 候选池准入率与 B/C 排序补充分析

日期：2026-06-14

### 22.1 候选池准入率不是 0.89%

前文的 `154 / 17276 = 0.89%` 指的是：

```text
当前测试样本进入或替换了：
1. 对齐核心缓存（alignment core cache）
2. 熵缓存（entropy cache）
3. 能量缓存（energy cache）
中的任意一个。
```

它不是候选池准入率。

修复后 A4 的测试阶段候选池准入统计如下：

| 统计项 | 数值 |
|---|---:|
| 测试样本总数 | 17276 |
| 进入候选池样本数 | 1423 |
| 进入候选池比例 | 8.24 |
| 进入候选池样本 zero-shot 伪标签正确率 | 68.17 |
| 只进入候选池、未进入后续三个缓存的样本数 | 1269 |
| 只进入候选池、未进入后续三个缓存的比例 | 7.35 |
| 被候选池拒绝样本数 | 15853 |
| 被候选池拒绝比例 | 91.76 |

逐 corruption：

| corruption | 候选池进入比例 | 候选池进入样本正确率 | 进入后续三个缓存中任意一个的比例 |
|---|---:|---:|---:|
| add_global_2 | 6.52 | 65.84 | 0.61 |
| add_local_2 | 7.78 | 64.58 | 0.65 |
| dropout_global_2 | 8.91 | 74.09 | 1.01 |
| dropout_local_2 | 9.00 | 68.92 | 0.97 |
| jitter_2 | 7.90 | 61.54 | 0.81 |
| rotate_2 | 8.87 | 72.15 | 1.26 |
| scale_2 | 8.67 | 68.22 | 0.93 |

结论：

```text
候选池确实很严格，但不是 0.89%。
0.89% 是当前测试样本进入/替换对齐核心缓存、熵缓存、能量缓存中任意一个的比例。
候选池实际准入率是 8.24%，且正确率达到 68.17%。
```

### 22.2 B(x), C(x) 排序逻辑检查

代码中的 B/C 排序逻辑如下：

```text
q_H = (H_max - H) / (H_max - H_min + eps)
q_E = (E_max - E) / (E_max - E_min + eps)
q_M = (M - M_min) / (M_max - M_min + eps)

B(x) = min(q_H, q_E, q_M)
C(x) = D_neg / (D_pos + D_neg + eps)
```

方向检查：

| 指标 | 原始方向 | 归一化后方向 | 当前代码是否正确 |
|---|---|---|---|
| 熵（entropy） | 越低越可靠 | `q_H` 越大越可靠 | 是 |
| 能量（energy） | 越低越可靠 | `q_E` 越大越可靠 | 是 |
| 分类间隔（margin） | 越大越可靠 | `q_M` 越大越可靠 | 是 |

排序检查：

```text
先按 B(x) 从大到小；
B(x) 完全相同或近似相同时，再按 C(x) 从大到小；
如果仍完全相同，保留旧样本，拒绝新样本。
```

小规模 sanity test 已通过：

1. 低熵、低能量、大分类间隔的样本排第一。
2. 高熵、高能量、小分类间隔的样本排最后。
3. 完全并列时旧样本排在新样本前面，符合保守替换设定。

因此，当前没有发现 B/C 方向写反、排序写反这类代码逻辑错误。

但这不表示当前 B/C 规则没有问题。它的设计本身非常保守：

```text
B(x) = min(q_H, q_E, q_M)
```

这意味着只要一个样本在熵、能量、分类间隔三个维度中任意一项很弱，`B(x)` 就会很低。
候选池满后，新样本必须在当前 8 个候选样本加自己组成的 9 个样本里不是最差，才会进入
候选池。因此候选池拒绝率达到 `91.76%` 是这个保守设计的直接结果。

### 22.3 真正的后端瓶颈

候选池虽然严格，但后端瓶颈更明显。测试阶段：

```text
进入候选池：1423
只进入候选池、未进入后续三个缓存：1269
进入/替换对齐核心缓存、熵缓存、能量缓存中任意一个：154
```

也就是说，进入候选池的样本里，大约 `89.18%` 仍没有进入后续三个缓存。

后续对齐核心缓存的主要拒绝原因是分布一致性：

| 事件 | test 次数 |
|---|---:|
| `alignment_core_wait_candidate_full` | 52 |
| `alignment_core_init_from_candidate` | 40 |
| `alignment_core_replace` | 150 |
| `alignment_core_reject_reliability` | 80 |
| `alignment_core_reject_distribution` | 1131 |

这说明当前低触发率不只是候选池严格造成的，更主要是：

```text
候选池通过后，对齐核心缓存的分布一致性门控仍然拒绝了大量样本。
```

### 22.4 如何在保持正确率的情况下提高准入率

当前不能简单放宽所有条件。因为候选池进入样本正确率为 `68.17`，而对齐核心缓存累计
正确率为 `62.98`。盲目放宽可能提高数量，但会降低缓存纯度。

更合理的路线是分阶段做：

1. A4-B1 先做缓存总得分范数裁剪，解决 `add_global_2` 负迁移问题，不改变候选池和
   对齐核心缓存准入规则。
2. A4-B1 同时新增候选池诊断，记录进入/拒绝样本的 `B(x)`、`C(x)` 均值和分位数，
   以及候选池通过但被对齐核心缓存拒绝样本的统计。
3. 如果 A4-B1 后仍需要提高准入率，再设计 A4-B3 准入率实验。候选方案包括：

| 方案 | 作用 | 风险 |
|---|---|---|
| 增大候选池容量，例如 8 -> 12 | 提高候选池覆盖率 | 候选池正确率可能下降 |
| 对齐核心缓存从“只尝试当前样本”改为“从候选池 top 样本中尝试晋升” | 利用已经进入候选池但未被后续缓存吸收的高质量样本 | 旧候选样本可能滞后，需记录晋升来源 |
| 对高 B/C 样本放宽分布一致性门控 | 提高从候选池到对齐核心缓存的转换率 | 可能破坏对齐核心缓存纯度 |
| 候选池低权重参与最终得分 | 提高候选池信息利用率 | 可能把更多噪声票带入最终预测 |

其中最值得优先考虑的是：

```text
从候选池 top 样本中尝试晋升到对齐核心缓存。
```

原因是它不直接降低候选池标准，也不直接让候选池参与最终得分，而是更充分利用已经通过
候选池筛选的样本。这比直接放宽候选池或直接让候选池得分更稳妥。
