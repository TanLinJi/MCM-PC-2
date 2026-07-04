# E7：熵-能量-对齐多缓存（Entropy-Energy-Alignment Multi-Cache）实验设计

日期：2026-06-11  
项目根目录：`/root/autodl-tmp/MCM-PC-2`  
实验目录：`docs/experiments/E7_entropy_energy_alignment_multicache`  
当前对比基线：`02_9_2`，即 `E4-C-A0+E1-textdist-only`，`E4_TEXT_SCORE_WEIGHT=0.15`

---

## 1. 文档目的

本文记录 E7 的完整设计方案。E7 来自一个新的研究思路：不再沿用 Point-Cache 中“全局缓存（global cache）+ 局部缓存（local cache）”的层级结构，而是只使用点云编码器输出的全局特征（global feature）。E7 先并行构造熵缓存（entropy cache）和能量缓存（energy cache），再由二者共同推导位于后置阶段的对齐缓存（alignment cache）。

E7 的核心目标是：

```text
在严格测试时适应（Test-Time Adaptation, TTA）设定下，
用并行的熵缓存、能量缓存，以及二者之后的对齐缓存，共同形成更可靠的在线测试样本知识库，
并在最终分类得分中显式融合这些缓存证据。
```

这里的严格测试时适应（Test-Time Adaptation, TTA）设定指：

| 约束 | E7 是否遵守 |
|---|---|
| 不知道当前样本来自干净数据还是损坏数据 | 遵守 |
| 不使用真实标签（ground-truth labels）参与缓存更新 | 遵守 |
| 不使用源域训练数据 | 遵守 |
| 不反向传播更新模型参数 | E7-A 遵守 |
| 只使用当前测试样本输出和历史在线缓存状态 | 遵守 |

本文是 E7 的设计文档，不是结果报告。后续 E7-A/E7-B/E7-C 的代码、脚本、结果和分析应继续在本目录下补充。

---

## 2. 当前事实基础

### 2.1 当前最好基线

当前主线中，ModelNet-C severity=2 的最好完整七类 corruption 结果是 `02_9_2`：

```text
方法：E4-C-A0+E1-textdist-only
文本分布权重（text distribution weight）：0.15
分数归一化（score normalization）：running_zscore
backbone：ULIP
cache setting：zs_global_local
ModelNet-C severity=2 七类 corruption 平均准确率：54.7057
clean accuracy：63.86
```

`02_9_2` 的主要优点是：E1 的 LLM 文本描述只用于文本分布（text distribution），最终分类器仍使用 `manual_full` 文本原型，因此比直接更换最终文本原型更稳定。

`02_9_2` 的主要问题是：它仍然基于 Point-Cache 的全局缓存（global cache）和局部缓存（local cache）框架，并且最终 logits 的正向缓存证据主要来自原始缓存投票。我们还没有系统检验“熵、能量、一致性”三个维度是否可以构成更干净的缓存结构。

### 2.2 最近负缓存实验的启发

`08_1` 高熵负缓存（high-entropy negative cache）实验表明：

| 实验 | 结果 |
|---|---|
| 将负缓存进入条件从中等熵改为高熵 | S2 平均下降约 `-0.40` |
| 大多数 corruption 中高熵负缓存几乎没有样本进入 | 说明高熵区间不是稳定负缓存来源 |
| `add_global_2` 从 `47.89` 降到 `45.46` | 说明原始中等熵负缓存对部分场景有实际作用 |

因此 E7 不应把“高熵负缓存”作为主贡献。负缓存（negative cache）可以作为 E7-B 的可选模块，但 E7-A 应先验证“两条并行候选缓存 + 一个后置对齐缓存”的结构：熵缓存（entropy cache）和能量缓存（energy cache）并行更新，对齐缓存（alignment cache）由二者共同推导。

---

## 3. 设计边界

### 3.1 不去掉类别标记（CLS token）

最初想法中提到：点云输入点云编码器时，不再像论文中那样添加类别标记（CLS token），而是像经典 PointNet 那样直接输入点云。

经过检查，E7 第一版不采用这个修改。

原因是：ULIP/PointBERT 的点云编码器（point cloud encoder）内部确实使用类别标记（CLS token），并且全局特征依赖它。如果去掉类别标记（CLS token），就等于修改预训练编码器结构，原始权重的语义可能被破坏。这已经不是单纯的测试时缓存改进，而是改模型结构。

因此 E7 第一版遵循：

```text
保留预训练点云编码器（pretrained point cloud encoder）的原始结构；
不删除类别标记（CLS token）；
不修改 ULIP/PointBERT 权重；
只改变测试时缓存结构和最终 logits 融合方式。
```

### 3.2 不使用局部缓存（local cache）

E7 第一版取消 Point-Cache 的局部缓存（local cache），只使用全局点云特征（global point-cloud feature）：

```text
输入点云 -> 点云编码器 -> 全局特征 pc_feats
```

这样做的目的不是否定局部缓存，而是先构造一个更清晰的全局多缓存框架，避免局部特征、局部聚类、全局特征、负缓存同时耦合，导致结果难以解释。

---

## 4. 零样本推理与文本原型解耦

E7 沿用 `02_9_2` 的核心解耦原则：LLM 文本描述和最终文本分类器是两条独立路径。

### 4.1 零样本分类器（最终文本原型，不动）

```text
文本侧（manual_full）：
class prompts (hand-crafted templates) -> text encoder -> text prototype matrix W

点云侧：
point cloud -> point cloud encoder -> point feature z

零样本 logits：
S_zs = scale * z @ W
```

其中：

| 符号 | 中文解释 |
|---|---|
| `z` | 点云全局特征（point-cloud global feature） |
| `W` | 类别文本原型矩阵（text prototype matrix），由 `manual_full` 手工模板构造 |
| `S_zs` | 零样本得分向量（zero-shot logits） |
| `scale` | 当前代码中使用 `100.0` |

`S_zs` 使用稳定的 `manual_full` 文本原型，在整个实验过程中不变。这是 02-7 的教训——把 LLM prompt fusion 直接替换最终文本分类器会导致明显下降。

### 4.2 文本分布（仅用于缓存替换，不进入最终分类器）

E1 的 LLM 生成描述**只用于文本分布（text distribution）**，角色与 `02_9_2` 完全一致：

```text
E1 cached LLM descriptions -> text encoder -> per-prompt text features
-> prompt-level text distribution (per class)
-> 参与缓存替换时的 joint score 计算（text distribution score 部分）
```

注意：LLM 文本信息**不进入**最终 logits 融合公式。最终得分中没有任何独立的"文本项"——文本只通过影响缓存维护来间接影响缓存投票得分 `S_H`、`S_E`、`S_A`，不直接作为分类器证据。

### 4.3 解耦总结

| 组件 | 文本原型来源 | 是否在实验过程中变化 | 作用阶段 |
|---|---|---|---|
| 零样本分类器 `S_zs` | `manual_full` | 不变 | 最终分类 |
| 文本分布打分 | E1 LLM 描述 | 不变（预计算缓存） | 缓存替换判断 |
| 缓存得分 `S_H/S_E/S_A` | 视觉特征 + 伪标签 | 在线更新 | 最终分类 |
| 最终 logits | `S_zs` + 缓存得分 | — | 最终分类 |

文本信息完全不走"最终分类器直接项"这条路径，只通过控制缓存质量来间接起作用。这是 E7 设计中对 02-9-2 经验的直接继承。

---

## 5. 核心思想

E7 将在线测试样本分成两个并行候选视角，并在二者之后构造一个后置对齐视角：

| 缓存 | 关注点 | 直觉 |
|---|---|---|
| 熵缓存（entropy cache） | 低熵样本 | 模型预测更确定 |
| 能量缓存（energy cache） | 低能量样本 | 样本更像已知类别/更可信 |
| 对齐缓存（alignment cache） | 位于熵缓存和能量缓存之后，只接收同时被二者接受的样本。一旦进入对齐缓存，样本不会因为后来在熵缓存或能量缓存中被替换而自动从对齐缓存中删除。对齐缓存有独立的容量和替换逻辑。 | 熵和能量达成一致，可靠性更高 |
| 负缓存（negative cache） | 高熵或高能量样本 | 潜在混淆或不可靠样本，用于抑制 |

E7-A 先只实现两个并行正向缓存和一个后置对齐缓存：

```text
熵缓存（entropy cache） || 能量缓存（energy cache）
                 -> 对齐缓存（alignment cache）
```

其中熵缓存和能量缓存是并行关系；对齐缓存不是第三条并行分支，而是二者之后的汇合缓存。负缓存（negative cache）作为 E7-B 的可选扩展。自动权重（adaptive weighting）作为 E7-C 的可选扩展。

---

## 6. 缓存内容设计

### 6.1 熵缓存

熵缓存（entropy cache）每个样本只保存：

| 字段 | 中文解释 | 用途 |
|---|---|---|
| `pc_feats` | 点云全局特征（point-cloud global feature） | 计算新样本与缓存样本相似度 |
| `pseudo_label` | 伪标签（pseudo label） | 作为缓存投票类别 |
| `entropy` | 熵（entropy） | 控制进入和替换 |

不保存能量（energy）。熵缓存只对熵负责。

### 6.2 能量缓存

能量缓存（energy cache）每个样本只保存：

| 字段 | 中文解释 | 用途 |
|---|---|---|
| `pc_feats` | 点云全局特征（point-cloud global feature） | 计算新样本与缓存样本相似度 |
| `pseudo_label` | 伪标签（pseudo label） | 作为缓存投票类别 |
| `energy` | 能量（energy） | 控制进入和替换 |

不保存熵（entropy）。能量缓存只对能量负责。

### 6.3 对齐缓存

对齐缓存（alignment cache）每个样本只保存：

| 字段 | 中文解释 | 用途 |
|---|---|---|
| `pc_feats` | 点云全局特征（point-cloud global feature） | 计算新样本与缓存样本相似度 |
| `pseudo_label` | 伪标签（pseudo label） | 作为缓存投票类别 |

不保存熵（entropy），不保存能量（energy），也不保存 `from_entropy_cache=True` 或 `from_energy_cache=True`。

原因是：对齐缓存的进入条件由代码逻辑保证。只有同时被熵缓存和能量缓存接受的样本，才可能进入对齐缓存，因此不需要在缓存项中重复记录来源。

**重要：对齐缓存与熵缓存/能量缓存之间没有级联删除。** 一旦样本进入了对齐缓存，即使后来它在熵缓存或能量缓存中被替换出去，对齐缓存中的该样本也不受影响。对齐缓存有独立的容量 K=3 和自己的替换逻辑，它的成员资格由历史接受事件（acceptance event）决定，而不是由熵缓存或能量缓存的当前成员资格（current membership）决定。这样对齐缓存才能自然维护自己的历史原型分布。

### 6.4 负缓存

负缓存（negative cache）先不进入 E7-A。若在 E7-B 中加入，建议每个样本保存：

| 字段 | 中文解释 | 用途 |
|---|---|---|
| `pc_feats` | 点云全局特征（point-cloud global feature） | 计算新样本与负缓存样本相似度 |
| `pseudo_label` | 伪标签（pseudo label） | 记录当前样本预测类别，诊断用 |
| `trigger_type` | 触发类型（trigger type） | 记录样本因高熵或高能量进入负缓存 |
| `trigger_score` | 触发分数（trigger score） | 记录用于替换比较的熵或能量数值 |
| `topk_confusing_labels` | 前 k 个混淆类别（top-k confusing labels） | 负缓存最终抑制的类别 |

这里不建议直接保存完整 `prob_map` 作为主要抑制依据。

`prob_map` 是类别概率分布（class probability distribution），即 softmax 后的概率向量。如果一共有 3 个类别：

```text
logits = [0.1, 0.4, 0.7]
prob_map = softmax(logits) = [0.25, 0.34, 0.41]
```

当前 Point-Cache 代码中的负缓存使用完整概率分布，并通过阈值抑制一批类别。这个方式可能过度抑制。E7-B 更推荐保存 `topk_confusing_labels`，也就是只保存少数最容易混淆的类别。

例如：

```text
prob_map = [0.25, 0.34, 0.41]
pseudo_label = 2
top1_confusing_labels = [1]
top2_confusing_labels = [1, 0]
```

这表示：当前样本预测为类别 2，因此负缓存不抑制类别 2，而只抑制概率次高的类别 1，或者再加上类别 0。

---

## 7. 缓存容量

E7 初始容量设置：

| 缓存 | 每类容量 |
|---|---:|
| 熵缓存（entropy cache） | 5 |
| 能量缓存（energy cache） | 5 |
| 对齐缓存（alignment cache） | 3 |
| 负缓存（negative cache, E7-B 可选） | 待定，建议 2 或 3 |

容量解释：

1. 熵缓存和能量缓存每类容量设为 5，是为了比原始 Point-Cache 的 `K=3` 有更大的历史覆盖。
2. 对齐缓存每类容量设为 3，是为了让它更像“高可信核心原型缓存”，避免过宽。
3. 负缓存容量不应太大，因为负缓存用于抑制，容量过大容易误伤。

---

## 8. 历史原型分布

E7 的关键原则：用于判断“是否更符合原型分布”的统计量，必须来自历史上所有被该缓存接受过的样本，而不是只来自当前缓存内容。

这沿用 `02_9_2` 的 accepted-history 思路。

原因是：

```text
当前缓存内容会被容量限制和替换规则影响；
如果只用当前缓存构建分布，分布会随替换剧烈波动；
用历史接受样本构建分布，可以提供更稳定的类别原型统计。
```

每个正向缓存都维护自己的历史分布：

| 缓存 | 历史分布来源 |
|---|---|
| 熵缓存（entropy cache） | 历史上所有被熵缓存接受过的样本 |
| 能量缓存（energy cache） | 历史上所有被能量缓存接受过的样本 |
| 对齐缓存（alignment cache） | 历史上所有被对齐缓存接受过的样本 |

每个类别 `c` 可维护：

| 统计量 | 中文解释 |
|---|---|
| `count_c` | 类别 `c` 历史接受样本数 |
| `mean_c` | 类别 `c` 特征均值 |
| `var_c` | 类别 `c` 特征方差，可先用对角方差（diagonal variance） |

E7-A 第一版建议沿用 E4-C 中已经验证过的对角高斯式分布打分（diagonal Gaussian-style distribution score），先不引入完整协方差（full covariance）。

---

## 9. 缓存更新规则

### 9.1 基本变量

对每个测试样本，先计算：

| 变量 | 中文解释 |
|---|---|
| `pc_feats` | 点云全局特征（point-cloud global feature） |
| `clip_logits` | 零样本 logits（zero-shot logits） |
| `pseudo_label` | 伪标签，即 `argmax(clip_logits)` |
| `entropy` | 熵（entropy） |
| `energy` | 能量（energy） |

熵（entropy）越低，表示预测越确定。  
能量（energy）可以为负，通常越低、越负，表示模型越自信。

能量建议计算为：

```text
energy = -logsumexp(clip_logits)
```

### 9.2 熵缓存更新

熵缓存为空或当前预测类别未满时：

```text
直接加入熵缓存。
更新熵缓存历史原型分布。
```

熵缓存中当前预测类别已满时：

```text
找到该类别缓存中熵最高的样本 h_max。

新样本必须同时满足：
1. entropy_new < h_max
2. 新样本比被替换样本更符合熵缓存历史原型分布

若满足，则替换 h_max 对应样本，并更新熵缓存历史原型分布。
否则拒绝。
```

注意：历史原型分布只由历史上被熵缓存接受过的样本维护。

### 9.3 能量缓存更新

能量缓存为空或当前预测类别未满时：

```text
直接加入能量缓存。
更新能量缓存历史原型分布。
```

能量缓存中当前预测类别已满时：

```text
找到该类别缓存中能量最高的样本 e_max。

新样本必须同时满足：
1. energy_new < e_max
2. 新样本比被替换样本更符合能量缓存历史原型分布

若满足，则替换 e_max 对应样本，并更新能量缓存历史原型分布。
否则拒绝。
```

注意：因为能量越低越可信，所以要替换的是当前类别缓存中能量最高的样本。

### 9.4 对齐缓存更新

对齐缓存只由熵缓存和能量缓存共同推导。具体流程如下。

#### 9.4.1 进入资格

```text
只有当新样本同时被熵缓存接受、并且被能量缓存接受时，
它才有资格进入对齐缓存。
```

这里的"接受"指 acceptance event——即样本成功进入了熵缓存或能量缓存（包括首次加入和后续替换）。仅被拒绝的样本不算"接受"。

#### 9.4.2 同步判断：替换事件的对齐推导

由于熵缓存和能量缓存是并行更新的，一个样本可能同时触发两者的替换。需要做同步判断：

```text
在处理每个测试样本时，维护两个标记：
- entered_entropy: bool，当前样本是否进入了熵缓存
- entered_energy: bool，当前样本是否进入了能量缓存

如果 entered_entropy=True AND entered_energy=True：
    该样本有资格进入对齐缓存，执行对齐缓存更新逻辑。
否则：
    跳过对齐缓存更新。
```

注意：这个判断只针对当前样本本身。之前已经进入对齐缓存的其他样本不受影响——即使它们的来源样本后来在熵缓存或能量缓存中被替换出去，对齐缓存中的条目仍然保留。

#### 9.4.3 非级联删除规则

```text
对齐缓存与熵缓存/能量缓存之间没有级联删除。
一旦样本进入对齐缓存，它只由对齐缓存自己的替换逻辑管理。
熵缓存或能量缓存中对应样本的后续替换不会自动从对齐缓存中删除该样本。
```

原因：

1. 对齐缓存本身有独立的容量 K=3 和自己的替换逻辑，如果把它定义成"熵缓存和能量缓存当前内容的严格交集"，它就不再是一个独立缓存。
2. 对齐缓存需要维护自己的历史原型分布。如果上游缓存替换就自动删除对齐缓存中的条目，对齐缓存的分布统计会被打乱，难以稳定收敛。
3. 进入对齐缓存代表该样本通过了双重筛选（低熵 AND 低能量），这个"资格"是历史事实，不应因为上游缓存的容量压力而被撤销。

#### 9.4.4 替换逻辑

对齐缓存为空或当前预测类别未满时：

```text
直接加入对齐缓存。
更新对齐缓存历史原型分布。
```

对齐缓存中当前预测类别已满时：

```text
不再检查熵或能量，因为进入对齐缓存之前已经通过双重筛选。

新样本必须满足：
1. 新样本比该类别对齐缓存中某个已有样本更符合对齐缓存历史原型分布

若满足，则替换分布匹配最差的已有样本，并更新对齐缓存历史原型分布。
否则拒绝。
```

对齐缓存最开始为空，因此在未达到最小可用条件前不参与最终 logits 计算。

建议设置：

```text
alignment_min_count_per_class = 1
alignment_min_total = 5 或 10
```

第一版可以简单设为：

```text
只要对齐缓存非空，就参与最终 logits；
但诊断中记录 alignment_cache_total。
```

如果结果不稳定，再加入最小样本数门控。

### 9.5 负缓存更新（E7-B 可选）

负缓存不是 E7-A 的主线。若在 E7-B 中加入，规则如下。

负缓存为空或当前预测类别未满时：

```text
若样本满足高熵 或 高能量，满足一个即可进入负缓存。
```

负缓存中当前预测类别已满时：

```text
新样本需要满足以下任一条件：
1. 高熵 且 更符合负缓存历史原型分布
2. 高能量 且 更符合负缓存历史原型分布

若满足，则替换负缓存中分布匹配最差的样本。
否则拒绝。
```

负缓存历史原型分布同样由历史上所有被负缓存接受过的样本维护。

负缓存最终得分建议用 `topk_confusing_labels` 构造，而不是完整 `prob_map`。

---

## 10. 缓存得分如何计算

每个缓存最终都必须输出一个长度为类别数的得分向量（logits-like vector）。

以熵缓存（entropy cache）为例。

假设新样本特征为 `z_q`，熵缓存中有若干样本：

```text
(z_i, y_i, h_i)
```

其中：

| 符号 | 中文解释 |
|---|---|
| `z_i` | 缓存样本特征 |
| `y_i` | 缓存样本伪标签 |
| `h_i` | 缓存样本熵 |

先计算新样本与每个缓存样本的相似度权重：

```text
a_i = exp[-beta_H * (1 - z_q · z_i)]
```

然后按伪标签投票，得到每个类别的熵缓存得分：

```text
S_H(c) = sum_i a_i * 1[y_i = c]
```

这里的 `1[y_i = c]` 是指示函数：如果缓存样本的伪标签等于类别 `c`，取 1，否则取 0。

代码实现可以用独热编码（one-hot encoding）：

```text
entropy_logits = affinity @ one_hot_labels
```

示例：一共 3 个类别，新样本与 3 个熵缓存样本的相似度如下：

| 缓存样本 | 伪标签 | 相似度 |
|---|---:|---:|
| A | 0 | 0.6 |
| B | 2 | 0.8 |
| C | 2 | 0.3 |

则熵缓存得分为：

```text
S_H = [0.6, 0.0, 1.1]
```

能量缓存得分（energy cache logits）和对齐缓存得分（alignment cache logits）按同样方式计算，只是它们使用各自缓存中的样本。

---

## 11. 最终得分公式

### 11.1 手动参数版本

E7-A 第一版使用手动参数（manual weighting）：

```text
S_final =
alpha_ZS * S_zs
+ alpha_H  * S_H
+ alpha_E  * S_E
+ alpha_A  * S_A
- alpha_N  * S_N
```

其中：

| 符号 | 中文解释 |
|---|---|
| `S_zs` | 零样本得分（zero-shot logits） |
| `S_H` | 熵缓存得分（entropy cache logits） |
| `S_E` | 能量缓存得分（energy cache logits） |
| `S_A` | 对齐缓存得分（alignment cache logits） |
| `S_N` | 负缓存得分（negative cache logits） |

E7-A 不使用负缓存，因此：

```text
alpha_N = 0
```

建议初始值：

```text
alpha_ZS = 1.0
alpha_H  = 2.0
alpha_E  = 2.0
alpha_A  = 2.0 或 3.0
alpha_N  = 0.0
```

也可以对齐当前 Point-Cache 的常用缓存强度：

```text
alpha_H = alpha_E = alpha_A = 4.0
beta_H = beta_E = beta_A = 3.0
```

但建议第一轮先不要过强，否则熵缓存、能量缓存和后置对齐缓存可能重复放大同一批伪标签。

### 11.2 自动参数版本

E7-C 再引入自动参数（adaptive weighting）。

当前约定：E7-A4 第一版仍使用手动权重（manual weighting），并优先验证
候选池、对齐缓存前置、以及更严格可靠准入是否能提升缓存样本质量。自动
参数/自适应权重（adaptive weighting）暂不放入 A4 第一版，作为后续版本
单独设计和验证，避免同时改变缓存准入逻辑与最终融合权重。

自动参数思想：

```text
当前样本低熵、低能量时：
    更信任熵缓存、能量缓存和对齐缓存。

当前样本高熵或高能量时：
    降低正向缓存权重；
    若引入负缓存，则提高负缓存抑制权重。
```

一个可选形式：

```text
confidence_entropy = 1 - normalized_entropy
confidence_energy  = sigmoid(-energy_z)
confidence_align   = confidence_entropy * confidence_energy
```

然后：

```text
alpha_H(x) = base_alpha_H * confidence_entropy
alpha_E(x) = base_alpha_E * confidence_energy
alpha_A(x) = base_alpha_A * confidence_align
```

如果 E7-B/C 加入负缓存：

```text
alpha_N(x) = base_alpha_N * (1 - confidence_align)
```

这里的 `energy_z` 是在线能量统计得到的 z-score。需要注意：自动参数必须只使用当前样本和历史在线统计，不能使用 clean/corruption 标签。

---

## 12. 实验阶段设计

### 12.1 E7-A：双并行缓存 + 后置对齐缓存，手动权重，无负缓存

目标：验证“熵缓存和能量缓存并行更新、对齐缓存后置汇合”的结构是否能替代原始全局+局部缓存结构。

配置：

| 项目 | 设置 |
|---|---|
| 局部缓存（local cache） | 不使用 |
| 熵缓存（entropy cache） | 使用 |
| 能量缓存（energy cache） | 使用 |
| 对齐缓存（alignment cache） | 使用 |
| 负缓存（negative cache） | 不使用 |
| 权重 | 手动参数 |
| 文本分类器 | `manual_full`（零样本 logits），不变 |
| 文本分布 | E1 LLM 描述仅用于各缓存的文本分布打分，不进入最终分类器 |
| backbone | ULIP |
| 数据集 | ModelNet-C |
| 首轮评估 | severity=2 七类 corruption + clean |

建议实验编号：

```text
E7-A0：entropy + energy + alignment, manual weights, no negative cache
```

第一轮必须跑：

| 数据 | 目的 |
|---|---|
| ModelNet-C S2 七类 corruption | 与 `02_9_2` 对齐 |
| ModelNet-C clean | 检查 clean 是否继续下降 |

若 S2 平均超过 `54.71`，且 clean 不低于 `63.86`，再扩展 all35。

### 12.2 E7-B：加入克制型负缓存

目标：验证负缓存是否能补充 E7-A 的弱点。

新增：

| 项目 | 设置 |
|---|---|
| 负缓存进入条件 | 高熵或高能量，满足一个即可 |
| 负缓存替换条件 | 高熵且更符合负缓存分布，或高能量且更符合负缓存分布 |
| 负缓存得分 | 只抑制 `topk_confusing_labels` |
| 建议 `topk` | 1 或 2 |

不建议恢复 Point-Cache 当前完整 `prob_map` 阈值抑制方式，因为它可能抑制过多类别。

### 12.3 E7-C：自动权重

目标：减少手动参数依赖。

新增：

| 模块 | 自动权重依据 |
|---|---|
| 熵缓存 | 当前样本归一化熵 |
| 能量缓存 | 当前样本能量 z-score |
| 对齐缓存 | 熵置信度与能量置信度的乘积 |
| 负缓存 | 不可信度，即 `1 - alignment confidence` |

E7-C 不应在 E7-A 失败前实现。需要先知道“双并行缓存 + 后置对齐缓存”是否有独立价值。

### 12.4 E7-D：全量 all35 与跨数据集验证

若 E7-A/B/C 中任一版本在 S2 + clean 同时表现良好，再进行：

| 扩展 | 目的 |
|---|---|
| ModelNet-C all35 | 检查 severity 稳定性 |
| ScanObjNN-C hardest S2/S4 | 检查跨数据集鲁棒性 |
| clean ScanObjNN hardest | 检查 clean generalization |

---

## 13. 诊断指标

E7 必须保存比准确率更细的诊断信息，否则无法判断多个缓存的作用。

每个 corruption 至少保存：

| 指标 | 中文解释 |
|---|---|
| `final_acc` | 最终准确率 |
| `entropy_cache_total` | 熵缓存总样本数 |
| `energy_cache_total` | 能量缓存总样本数 |
| `alignment_cache_total` | 对齐缓存总样本数 |
| `entropy_accept / reject / replace` | 熵缓存接受、拒绝、替换次数 |
| `energy_accept / reject / replace` | 能量缓存接受、拒绝、替换次数 |
| `alignment_accept / reject / replace` | 对齐缓存接受、拒绝、替换次数 |
| `per_cache_logits_norm` | 各缓存 logits 范数，检查是否某个缓存过强 |
| `zs_pred_vs_final_pred_change_rate` | 最终预测相比零样本预测的改变率 |
| `cache_agreement_rate` | 熵缓存与能量缓存的预测一致率，以及二者与后置对齐缓存的一致率 |

如果加入负缓存，还需要：

| 指标 | 中文解释 |
|---|---|
| `negative_cache_total` | 负缓存总样本数 |
| `negative_trigger_entropy_count` | 因高熵进入负缓存的样本数 |
| `negative_trigger_energy_count` | 因高能量进入负缓存的样本数 |
| `negative_topk_suppression_count` | 负缓存抑制类别次数 |
| `negative_flip_correct_to_wrong` | 负缓存导致正确预测变错的次数 |
| `negative_flip_wrong_to_correct` | 负缓存导致错误预测变对的次数 |

---

## 14. 可能的优化方案

### 14.1 缓存容量消融

当前初始容量：

```text
entropy K = 5
energy K = 5
alignment K = 3
```

可消融：

| 版本 | 熵缓存 | 能量缓存 | 对齐缓存 |
|---|---:|---:|---:|
| small | 3 | 3 | 2 |
| default | 5 | 5 | 3 |
| large | 7 | 7 | 3 |

### 14.2 对齐缓存进入条件与级联删除

默认规则：

```text
同一样本同时被熵缓存和能量缓存接受（acceptance event），
才进入对齐缓存。
进入后，即使上游缓存发生替换，对齐缓存中的该条目不受影响
（非级联删除）。
```

可选变体：

```text
1. 更宽松的进入条件：
   同一样本被熵缓存接受，且 energy 低于在线能量阈值；
   或被能量缓存接受，且 entropy 低于在线熵阈值。
   该版本更宽松，但也更容易引入伪标签错误。建议只在默认规则样本太少时尝试。

2. 级联删除（不推荐）：
   如果熵缓存或能量缓存中某样本被替换出去，同步从对齐缓存中删除。
   不推荐的原因是：这会打破对齐缓存的独立性和历史分布稳定性。
```

### 14.3 分布打分方式

E7-A 默认使用对角高斯式分布打分（diagonal Gaussian-style score）。

可选：

| 方案 | 说明 |
|---|---|
| cosine-to-mean | 只计算样本与历史均值的余弦相似度 |
| diagonal Gaussian | 使用历史均值和对角方差 |
| robust z-score | 使用中位数和 MAD，降低离群样本影响 |
| shared covariance | 后续结合 E5-C，但不建议第一版使用 |

### 14.4 最终 logits 归一化

多个缓存输出的尺度可能不同。可选归一化：

| 方案 | 说明 |
|---|---|
| none | 不归一化，最简单 |
| running_zscore | 沿用 E4-C 的在线 z-score |
| per-sample norm | 对每个 cache logits 做 L2 norm |
| temperature scaling | 每个缓存使用独立温度 |

E7-A 第一版建议先记录每个缓存 logits 的均值、方差和范数，再决定是否归一化。

### 14.5 负缓存抑制方式

如果 E7-B 加入负缓存，优先尝试：

```text
top1_confusing_labels
top2_confusing_labels
```

不优先尝试完整 `prob_map` 阈值抑制。

原因是：完整 `prob_map` 可能抑制太多类别，前面负缓存实验已经显示负缓存方向很容易误伤。

### 14.6 自动权重

E7-C 的自动权重可从最简单版本开始：

```text
alpha_H(x) = base_H * (1 - normalized_entropy)
alpha_E(x) = base_E * sigmoid(-energy_z)
alpha_A(x) = base_A * (1 - normalized_entropy) * sigmoid(-energy_z)
```

可能优化：

| 方案 | 说明 |
|---|---|
| hard gate | 低可信样本直接不使用某缓存 |
| soft gate | 连续权重，更平滑 |
| normalized gate | 保证各缓存权重和固定 |
| confidence floor | 设置最小权重，避免完全关闭 |

---

## 15. 风险与判定标准

### 15.1 主要风险

| 风险 | 说明 |
|---|---|
| 伪标签污染 | 低熵或低能量不等于一定正确 |
| 重复计数 | 对齐缓存由熵缓存和能量缓存共同推导，可能与两条并行缓存形成重复证据 |
| 对齐缓存过小 | 同时满足两套规则的样本可能太少 |
| 权重过强 | 多个缓存同时支持同一错误类别时会放大错误 |
| clean 下降 | 过强测试时适应可能损害干净数据准确率 |

### 15.2 成功标准

E7-A 若满足以下任意一种，即值得继续：

| 标准 | 说明 |
|---|---|
| S2 平均超过 `54.71`，clean 不低于 `63.86` | 最理想 |
| S2 平均接近 `54.71`，clean 明显高于 `63.86` | 说明 clean 问题可能缓解 |
| 某些 corruption 明显提升，且诊断显示缓存机制合理 | 可以作为局部优化方向 |

E7-A 若出现以下情况，应停止当前版本：

| 情况 | 处理 |
|---|---|
| S2 平均低于 `02_9_2` 超过 `0.5` | 暂停，分析缓存权重和接受率 |
| clean 低于 `63.5` | 暂停，说明过适应风险较高 |
| 对齐缓存几乎为空 | 放宽对齐缓存进入条件或降低容量 |
| 某缓存 logits 范数远高于零样本 logits | 降低该缓存权重或做归一化 |

---

## 16. E7-A 建议实现摘要

第一版 E7-A 应尽量小步、可解释：

```text
1. 不修改点云编码器，不删除 CLS token。
2. 不使用局部缓存。
3. 零样本分类器使用 `manual_full` 文本原型构造，E1 的 LLM 文本描述仅用于各缓存的 text distribution 打分，不替换最终文本分类器。这是 02-9-2 的核心经验。
4. 建立熵缓存、能量缓存两个并行缓存，以及后置对齐缓存。
5. 熵缓存和能量缓存并行更新。
6. 每个样本处理后做同步判断：若 entered_entropy=True AND entered_energy=True，则该样本有资格进入对齐缓存。
7. 对齐缓存独立维护，不因上游缓存替换而级联删除已有条目。
8. 每个缓存都通过相似度加权投票输出类别得分向量。
9. 最终得分使用手动权重融合：S_final = 1.0*S_zs + alpha_H*S_H + alpha_E*S_E + alpha_A*S_A。
10. E7-A 不加入负缓存。
11. 先跑 ModelNet-C S2 七类 corruption 和 clean。
```

初始公式：

```text
S_final =
1.0 * S_zs
+ alpha_H * S_H
+ alpha_E * S_E
+ alpha_A * S_A
```

建议第一组参数：

```text
alpha_H = 2.0
alpha_E = 2.0
alpha_A = 2.0
beta_H  = 3.0
beta_E  = 3.0
beta_A  = 3.0
```

如果缓存 logits 偏弱，再尝试：

```text
alpha_H = 4.0
alpha_E = 4.0
alpha_A = 4.0
```

---

## 17. 术语约定

| 中文术语 | English |
|---|---|
| 测试时适应 | Test-Time Adaptation, TTA |
| 零样本推理 | zero-shot inference |
| 点云编码器 | point cloud encoder |
| 文本编码器 | text encoder |
| 类别标记 | CLS token |
| 全局特征 | global feature |
| 局部缓存 | local cache |
| 熵 | entropy |
| 能量 | energy |
| 熵缓存 | entropy cache |
| 能量缓存 | energy cache |
| 对齐缓存 | alignment cache |
| 负缓存 | negative cache |
| 伪标签 | pseudo label |
| 独热编码 | one-hot encoding |
| 类别概率分布 | class probability distribution |
| 前 k 个混淆类别 | top-k confusing labels |
| 手动权重 | manual weighting |
| 自动权重 | adaptive weighting |

后续实现、日志、论文草稿中应尽量使用以上术语，避免同一概念多种叫法。
