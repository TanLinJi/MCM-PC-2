# DPC-Point E0-E3 Historical Narrative

本文件记录 DPC-Point 形成之前的 E0-E3 研究路线。早期文字中若出现
MCM-PC 表述，按历史上下文理解；当前论文题目和主线已更新为
**DPC-Point: Distribution-Guided Prototype Cache for Robust Point Cloud Test-Time Adaptation**。

更新时间：2026-06-07  
项目根目录：`/root/autodl-tmp/MCM-PC-2`  
核心代码库：`/root/autodl-tmp/MCM-PC-2/Point-Cache`  
当前主线：基于 Point-Cache 的三维点云视觉-语言模型测试时泛化方法研究  
当前阶段：E3 smoke 系列已经形成阶段性结论，当前应从单中心 GPA Cache 转向更能表达类内几何变化的后续方案  

---

## 0. 写作目的与阅读方式

这份文档不是论文初稿，而是项目研究过程的“完整路线说明书”。它面向一个完全没参与过本项目的人，目标是让读者读完后能回答以下问题：

1. 我们为什么从 Point-Cache 做起？
2. E0、E1、E2、E3 分别想验证什么？
3. 每个实验是怎么做的？路径、命名、设置、结果分别是什么？
4. 每个实验过程中遇到了哪些问题？这些问题为什么会发生？我们怎么解决？
5. 当前哪些结果只是最小烟雾测试，哪些结论可以暂时相信，哪些必须后续补完整实验？
6. 如果接手者继续做，应优先做什么，暂时不要做什么？

文档中所有实验阶段都遵循一个基本原则：

> 当前优先做纵向最小验证，把从 E0 到 E3 的整体研究链路先跑通；等主线跑通后，再回头补横向消融、完整数据集、完整 corruption、更多 backbone、更多文本/缓存设定。

这个原则非常重要，因为当前很多结果不能被理解为最终完整 benchmark，而应该被理解为“方向可行性检查”。

---

## 1. 文献与方法背景

### 1.1 为什么选择 Point-Cache 作为项目起点

Point-Cache 的目标是在测试时动态维护缓存，用在线测试样本中的可靠预测来补充大规模三维视觉-语言模型的 zero-shot 推理。它的核心特点是：不访问训练集、不进行反向传播、依赖在线测试流、使用全局缓存和局部缓存，并利用大规模 3D VLM 的 zero-shot 能力，同时引入测试时分布信息。

Point-Cache 原文强调了“全局结构 + 局部部件细节”的 hierarchical cache 设计：全局缓存存储点云整体特征，局部缓存存储局部 patch 聚类后的局部代表特征。原文还强调 Point-Cache 是 training-free、plug-and-play 的测试时方法，适合接到 ULIP、ULIP-2、OpenShape、Uni3D 等大规模 3D 模型上。

### 1.2 Point-Cache 的基础机制

Point-Cache 的零样本推理先由大模型得到：全局点云特征 `pc_feats`、文本类别原型 `clip_weights`、zero-shot logits、预测类别 `pred`、熵 `loss` 和局部 patch 聚类中心 `patch_centers`。

正缓存主要包括：

1. **Global Entropy Cache**：每个类别最多存 K 个全局样本。每个缓存项通常可以理解为 `[global feature, entropy]`，概念上是 `(点云全局特征, 伪标签, 熵)`。更新逻辑是低熵优先：未满时加入，满时用新低熵样本替换该类缓存中熵最高样本。
2. **Local Cache**：存储点云的局部结构代表，即 `patch_centers`。如果每个点云聚类为 `m` 个局部中心，每类全局缓存容量为 `K`，则某类局部缓存理论上最多有 `K×m` 个局部特征项。
3. **Negative Cache**：原文对 negative cache 的强调不如源码中明显，但源码中确实存在负缓存分支。后续仍要专门分析负缓存是否真实起作用、何时更新、如何参与最终 logits。

Point-Cache 的最终预测大体来自：

```text
zero-shot logits
+ global cache logits
+ local cache logits
- negative cache logits
```

当前 smoke test 中常见参数：

```text
positive.shot_capacity K = 3
n_cluster m = 3
alpha = 4.0
beta = 3.0
negative.shot_capacity = 2
```

### 1.3 MCP 给 E3 的启发

MCP 的核心问题意识是：仅用低熵选择样本并不一定能得到可靠原型。MCP 明确指出，已有 cache-enhanced TTA 方法依赖低熵准则来选择样本，但在 distribution shift 下，低熵样本也可能不可靠，且低熵样本不一定形成紧凑的类内分布。MCP 因此引入 Entropy Cache、Align Cache、Negative Cache。Align Cache 不只看低熵，还看样本到类别原型中心的距离，目标是提升类内紧凑性。

这直接启发了 E3：在 Point-Cache 里新增一个更严格的全局原型对齐缓存，让它控制 local cache 的写入，使 local cache 更干净。

但是，Point-Cache 是点云任务，且有 global/local hierarchical cache；MCP 是视觉语言模型的原型缓存框架，两者不能机械照搬。E3 中没有直接采用 MCP 的完整最终预测公式，也没有马上引入文本原型 residual 或 prototype residual tuning，而是先做最小测试：只改缓存构造逻辑，先看能否让 local cache 更可靠。

### 1.4 其他相关文献给后续方向的启发

项目中还上传并讨论过 BayesMM、Uni-Adapter、FreeTTA、ReTTA 等工作。当前没有把它们作为 E3 的直接实现路线，但它们对后续方向有启发：

- BayesMM 提醒我们，固定容量 cache 会丢失长期统计信息，启发后续从“离散缓存”走向“分布建模”。
- Uni-Adapter 指出高置信样本可能只覆盖部分类内模式，建议用多中心/cluster-based prototypes 捕捉 intra-class variability。
- ReTTA 提醒“低熵不等于可靠”，熵与能量/可观测性有区别，后续做熵-能量可靠性缓存时可参考。
- FreeTTA 强调在线目标分布建模，和后续从 cache 到 prototype/distribution 的扩展方向一致。

当前主线仍然先聚焦 Point-Cache + 文本原型 + 原型对齐缓存。

---

## 2. 总体研究路线

### 2.1 从 E0 到 E3 的逻辑链

整个项目是一个逐步推进的纵向链条：

```text
E0：先确认 Point-Cache 原始方法能完整复现。
E1：在不引入缓存的 zero-shot 条件下，验证文本描述增强是否有效。
E2：把 E1 中有效的文本增强迁移到 Point-Cache，验证文本收益是否能传递到缓存推理。
E3：在 E2 的基础上，不只改文本，而是改 Point-Cache 的缓存构造机制，引入全局原型对齐缓存，让 local cache 的来源更可靠。
```

其中 E1 和 E2 回答“文本能不能带来收益”，E3 回答“缓存自身能不能更干净、更原型对齐”。

### 2.2 当前阶段只是纵向 smoke test，不是最终完整实验

用户多次强调：当前先把纵向实验跑通，不要一开始就把所有横向消融做完。因此当前很多实验采用：

```text
Backbone: ULIP
Dataset: ModelNet-C
Severity: 2
Corruptions: 7 个 corruption
```

这是一种最小验证。它能帮助判断方向，但不能直接作为最终论文完整结论。跑通后必须回头补：完整 ModelNet-C 的 35 个设置、ScanObjectNN-C hardest、更多 backbone、clean/corrupted 对比、zero-shot/global-only/global+local 三种 cache setting、文本方法横向消融、GPA 关系/中心来源/初始化消融、测试流顺序敏感性、运行时间和显存统计。

换句话说，当前 E0-E3 是“研究链路打通”，后续才是“系统性补全”。

---

## 3. E0：Point-Cache 完整复现

### 3.1 目的

E0 的目标是复现原始 Point-Cache，建立后续所有改动的基准。如果 E0 不稳定，后续 E1/E2/E3 的增益或下降都无法解释。

### 3.2 做了什么

在 `/root/autodl-tmp/MCM-PC-2/Point-Cache` 中跑通原始 Point-Cache baseline 流程，并逐步建立脚本命名、结果目录、summary 汇总方式。E0 之后形成了这些基本理解：global cache 存样本级全局特征；local cache 存 patch 聚类后的局部特征；每个样本的 local item 必须和 global item 同步；negative cache 在源码中存在；summary.csv 是后续比较的核心依据；平均准确率必须单独计算或写入 summary。

### 3.3 关键源码问题与解决思路

#### 3.3.1 Local cache 是否最多 K×m

用户问：每类 global cache 最多 K 个样本，每个样本聚类出 m 个中心，那么 local cache 对某类是不是最多 K×m 个项？

结论是：是。若 `shot_capacity=K`，`n_cluster=m`，则每类进入 local cache 的局部中心最多来自 K 个点云，每个点云 m 个 patch centers，所以最多 K×m 个局部项。这个问题重要，是因为 E3 后来用 GPA Cache 控制 local cache 写入，如果 GPA Cache 每类少于 K 个样本，则 local cache 覆盖可能减少；如果 GPA Cache 选样过于中心化，则 local cache 的局部结构多样性可能下降。

#### 3.3.2 Global item 替换时 local item 如何同步

用户进一步追问：如果 global cache 替换某个样本，被替换样本对应的 local features 如何被定位和替换？

这个问题非常关键。原始 Point-Cache 的 local cache 如果只存 `(local feature, pseudo label)`，缺少“属于哪个 global item”的显式索引，那么同步替换会变复杂。E3 中吸取这个教训，所有 GPA 版本都要求：

```text
gpa_cache[pred][i] 与 gpa_local_cache[pred][i] 必须一一对应。
替换 gpa_cache[pred][i] 时，同步替换 gpa_local_cache[pred][i]。
排序时必须 global/local 一起排序。
```

这也是后面多次加入 `_sort_cache_and_local_together`、`length mismatch` 检查、candidate pool 状态检查的原因。

#### 3.3.3 负缓存问题

用户指出：Point-Cache 原文中没有明显强调负缓存，但源码中有负缓存，这个负缓存是否真实起作用？当时结论是：不能因为原文没强调就忽略源码中的 negative cache；后续所有复现和改动都必须确认 `negative.enabled` 是否为 True；在 E2/E3 中，负缓存仍然沿用原始逻辑，不作为当前变量；后续可以单独做 negative cache ablation。

---

## 4. E1：Text Prototype Enhancement

### 4.1 目的

E1 不引入 Point-Cache，只在 zero-shot 推理条件下研究文本原型增强。核心问题是：LLM 生成的类别描述能否提升 3D VLM 的 zero-shot 分类？

这个问题的背景是：大规模 3D VLM 不是纯 3D 模型，它的语义空间来自 point-image-text 三模态对齐。因此，文本模板对 zero-shot 分类非常关键。原始 Point-Cache 使用固定模板，而我们希望用 LLM 生成更丰富的类别描述。

### 4.2 关键假设

E1 的核心假设有两个：第一，不能只使用 3D 几何描述，因为 3D VLM 的文本语义空间继承了 2D 图像-文本预训练的语义表达，删除 2D 风格模板可能导致语义对齐变差。第二，LLM 描述不能单独替代人工模板，但可以补充人工模板，因为人工模板提供稳定锚点，LLM 生成描述提供更丰富的类别语义、多视角、功能、形状和上下文信息。

### 4.3 四种文本设定

| 方法 | 具体含义 | 要回答的问题 |
|---|---|---|
| `manual_full` | 原始完整人工模板 | 原始 zero-shot baseline 是多少？ |
| `manual_3d` | 过滤掉 2D-image-style prompt，只保留 3D 相关模板 | 2D 风格模板是否必要？ |
| `llm_only` | 只使用 LLM 生成的多视角 2D/3D 描述 | LLM 模板能否替代人工模板？ |
| `manual_full_llm_fusion` | 人工模板和 LLM 模板加权融合 | LLM 是否能作为人工模板的补充？ |

用户明确要求命名要体现“融合”，因此不用 `manual_full_add_llm`，统一用 `manual_full_llm_fusion`。

### 4.4 LLM prompt 生成与工程处理

LLM 生成描述时遇到的问题包括输出为空、部分类别失败、JSON 不完整、生成中断、重复生成浪费时间、prompt 文件在不同实验目录间复用困难。对应解决办法：生成脚本支持 partial cache；支持失败类别补生成；输出后检查 JSON 合法性；统一复制到 shared prompt 目录；后续所有实验优先读取 shared prompt，而不是重新生成。

当前共享路径：

```text
Point-Cache/results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json
```

后续建议改名为更短模型别名，例如：

```text
modelnet_c_ds_v4pro_multiview_2d3d_10_prompts.json
```

### 4.5 E1 smoke test 结果

当前 E1 采用 ULIP × ModelNet-C × severity=2 × 7 corruptions。

| 方法 | 平均准确率 | 相对 manual_full | 结论 |
|---|---:|---:|---|
| manual_full | 47.68 | 0.00 | 原始完整人工模板 baseline |
| manual_3d | 35.63 | -12.05 | 删除 2D 风格模板后严重下降 |
| llm_only | 39.30 | -8.38 | LLM-only 不能替代人工模板 |
| manual_full_llm_fusion | 48.88 | +1.20 | 人工模板 + LLM 融合有收益 |

### 4.6 E1 结果解释

`manual_3d` 大幅下降说明 2D 图像风格 prompt 不是冗余项。当前 3D VLM 的语义对齐空间并非只理解 point cloud geometry，它还依赖视觉语言预训练中形成的图像语义表达。`llm_only` 不好，说明 LLM 生成的描述更丰富但也更分散，可能包含不稳定表达、过度细节或与模型预训练模板分布不一致的语言风格。`manual_full_llm_fusion` 保留人工模板作为稳定锚点，同时引入 LLM 描述作为语义补充，因此略好。

### 4.7 E1 权重消融

| 融合权重 | 平均准确率 | 说明 |
|---|---:|---|
| manual 0.90 / LLM 0.10 | 48.41 | 小幅提升 |
| manual 0.85 / LLM 0.15 | 48.62 | 小幅提升 |
| manual 0.75 / LLM 0.25 | 48.88 | 当前最优 |
| manual 0.50 / LLM 0.50 | 48.37 | LLM 权重过高后下降 |

结论：LLM 贡献存在上限，过高权重会削弱人工模板锚点。当前 smoke test 最好是 `0.75/0.25`。

### 4.8 E1 后续补实验计划

E1 当前只是 smoke test，后续必须补：完整 35 corruption、ScanObjectNN-C hardest、多 backbone、不同 LLM 模板数量、不同描述类型、不同融合方式、prompt 质量统计，以及是否引入文本原型中心用于 E3 的 Center-D。

---

## 5. E2：Text Prototype Transfer to Point-Cache

### 5.1 目的

E2 的目标是验证：E1 中 zero-shot 的文本增强收益，能否传递到 Point-Cache 的完整缓存推理中。这一步非常重要，因为 E1 只证明文本原型能改善 zero-shot logits，但 Point-Cache 的缓存构建依赖初始预测、熵和伪标签。如果文本融合改变初始预测分布，它可能带来正向效果，也可能让某些错误预测被更高置信地缓存并被缓存机制放大。

### 5.2 命名修正

用户希望 E2 名称明确体现“E1 文本原型收益迁移到 Point-Cache”，因此最终命名为：

```text
E2_text_prototype_transfer_to_pointcache
```

对应路径：

```text
Point-Cache/runners/E2_text_prototype_transfer_to_pointcache
Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache
Point-Cache/results/E2_text_prototype_transfer_to_pointcache
docs/experiments/E2_text_prototype_transfer_to_pointcache
```

### 5.3 E2 实验设定

```text
Backbone: ULIP
Dataset: ModelNet-C
Severity: 2
Corruptions: 7 个 corruption
Text: manual_full / manual_full_llm_fusion
Cache: zs_global / zs_global_local
```

### 5.4 E2 当前主要结果

| 方法 | Cache 设置 | 平均准确率 | 说明 |
|---|---|---:|---|
| manual_full | zs_global | 52.66 | 只使用 global cache |
| manual_full_llm_fusion | zs_global | 53.18 | 文本融合传递到 global cache，有提升 |
| manual_full | zs_global_local | 54.00 | 原始 full Point-Cache smoke baseline |
| manual_full_llm_fusion | zs_global_local | 54.21 | 文本融合在 full Point-Cache 下仍有小幅收益 |

### 5.5 E2 结果解释

E2 说明 E1 的文本收益可以传递到 Point-Cache 中。尤其在 `zs_global_local` 下，`manual_full` 是 54.00，`manual_full_llm_fusion` 是 54.21，提升 +0.21。提升不大，但方向是正的，说明 LLM 融合没有破坏 Point-Cache 的缓存推理。

### 5.6 用户曾问的关键问题：为什么文本改变会影响缓存

Point-Cache 不是在真实标签上建缓存，而是在模型预测上建缓存。每个测试样本先 zero-shot 推理得到类别 `pred` 和熵。如果文本模板改变了 zero-shot logits，那么 `pred`、熵、是否进入 cache、进入哪个类别 cache 都可能改变。后续样本查询 cache 时，会被这些缓存项影响。所以文本融合不是只改变最后一次分类，而是改变整个测试流中的缓存轨迹。

### 5.7 E2 过程中的 bug

E2 后两组实验曾报：

```text
ValueError: not enough values to unpack (expected 6, got 5)
```

原因是调用的 runner 期望 `get_logits()` 返回 6 个值：`pc_feats, patch_centers, clip_logits, loss, prob_map, pred`，但某个文本融合路径下返回值不一致，只有 5 个。解决思路是统一 `get_logits` 返回协议，确保 hierarchical cache 路径需要的 `patch_centers`、`prob_map` 等都存在。这个 bug 的背景很重要：E2 不是只改文本文件，它必须兼容 Point-Cache 的 hierarchical cache runner。

### 5.8 E2 后续补实验计划

E2 当前只是 smoke test，后续要补：完整 35 corruption；manual_full 与 manual_full_llm_fusion 在所有 cache settings 下比较；E1 权重消融迁移到 Point-Cache；分析文本融合是否改变进入缓存样本的类别分布；分析文本融合是否降低 entropy cache 的平均熵；检查 global cache 和 local cache 的命中/替换事件；在 ScanObjectNN-C 和更多 backbone 上验证。

---

## 6. E3：Global Prototype-Alignment Cache

### 6.1 E3 的核心目的

E3 的目标是：在 Point-Cache 中新增一个更严格的全局原型对齐缓存，用它控制 local cache 的写入，从而让 local cache 更干净。原始 Point-Cache 主要基于低熵准入。MCP 指出低熵不一定可靠，低熵样本也不一定让类内分布紧凑。因此我们希望加入“到原型中心的距离”作为第二个约束。

E3 的中文概念：

```text
全局原型对齐缓存
Global Prototype-Alignment Cache
简称 GPA Cache
```

用户明确认可这个命名，并将 E3 路径定为：

```text
E3_global_prototype_alignment_cache
```

### 6.2 E3 的初始设计

最小验证阶段先不改最终预测加权公式，只改缓存构造逻辑。即：

```text
zero-shot logits + global cache logits + local cache logits - negative cache logits
```

最终公式暂不变。E3 初始设定是：保留原始 Global Entropy Cache，仍用于 global cache logits；新增 GPA Cache；GPA Cache 维护每类原型中心；只有进入 GPA Cache 的样本，其 `patch_centers` 才写入 local cache；local cache 暂时不单独构造局部原型中心；当前先做视觉原型，文本原型后续再引入。

### 6.3 用户围绕 MCP 规则提出的问题

#### 6.3.1 高熵样本是不是距离一定大？

不是。熵衡量预测分布不确定性，距离衡量特征到类中心的几何接近程度。一个样本可能熵低但距离远，也可能熵高但距离近。这就是为什么 E3 要同时记录替换时的新旧熵、新旧距离，方便后续分析熵与距离的关系。

#### 6.3.2 MCP 的 entropy cache 和 align cache 是顺序关系还是并列关系？

回头分析 MCP 后，我们理解 MCP 的 Entropy Cache 和 Align Cache 更接近并列更新，而不是“必须先进入 entropy cache，再进入 align cache”的严格顺序。两个缓存的更新都主要用自身缓存状态判断，而不是互相替换。

#### 6.3.3 我们应该借鉴 MCP 的并列方案，还是采用用户提出的顺序方案？

用户最初提出的是顺序方案：先满足 Global Entropy Cache 条件，再在这个基础上满足原型距离约束进入 GPA Cache，只有进入 GPA Cache 的样本写 local cache。我们决定先做顺序式 E3-V1，因为它最符合用户原始直觉，且是最小改动；如果效果不好，再切换到 MCP 更接近的并列式 E3-V2。后来实验结果证明并列式更好。

---

## 7. E3-V1：顺序式 GPA Cache

### 7.1 设计

E3-V1 是顺序式：

```text
Global Entropy Cache -> GPA Cache -> GPA-controlled Local Cache
```

样本必须先通过 entropy cache 的低熵准入，才有资格进入 GPA Cache。GPA Cache 再根据原型距离进一步筛选。只有进入 GPA Cache 的样本，其 local item 才写入 local cache。

### 7.2 中心来源消融

| 编号 | 名称 | 中心来源 |
|---|---|---|
| V1-A | GPA-only center | 只用 GPA Cache 计算中心 |
| V1-B | Entropy-only center | 只用 Global Entropy Cache 计算中心 |
| V1-C | Entropy+GPA union center | 用 Entropy Cache 和 GPA Cache 并集计算中心 |

### 7.3 结果

| 方法 | 关系 | 中心来源 | 平均准确率 |
|---|---|---|---:|
| E2 baseline | 原始 full Point-Cache | 无 GPA | 54.00 |
| E3-V1-A | 顺序式 | GPA-only | 53.44 |
| E3-V1-B | 顺序式 | Entropy-only | 52.43 |
| E3-V1-C | 顺序式 | Entropy+GPA union | 53.01 |

### 7.4 结果解释

顺序式整体不如 E2 baseline。可能原因包括：GPA Cache 只是 Global Entropy Cache 的后置筛选，独立性不足；GPA Cache 未满时与 entropy cache 差异很小；前 K 个样本直接进入 GPA Cache，中心仍然受早期样本影响；local cache 由更严格 GPA 控制后，覆盖可能下降；顺序过滤使候选空间过窄，容易错过虽然没进入 entropy cache 但对原型结构有用的样本。

### 7.5 E3-V1 过程中的重要 bug 和修复

#### bug 1：预构建 GPA Cache 没有传递到测试阶段

预构建阶段形成的 GPA Cache 曾没有正确传到正式测试阶段，测试阶段重新维护了一个 `runtime_gpa_cache`。这样会导致测试时原型中心和预构建 local cache 来源不一致。修复方式：`build_cache_in_advance` 返回 `entropy_cache, gpa_cache, gpa_local_cache, stats`；`run_test_tda` 接收 `gpa_cache`；测试阶段沿用预构建好的 `gpa_cache`；删除 `runtime_gpa_cache` 的空壳逻辑。

#### bug 2：GPA stats 没有保存

结果目录没有 `gpa_stats`，用户怀疑之前说的 bug 没修好。后续加入更明确的 stats 保存和 event_records。

#### bug 3：替换事件没有记录新旧熵和距离

用户要求记录替换时被替换样本的熵和距离、新样本的熵和距离。后来加入 `gpa_replacement_events_*.jsonl`，记录 `phase, class_index, decision, new_entropy, old_entropy, new_distance, old_distance`。这个日志对分析“高熵是否一定距离大”“低熵但距离远怎么办”非常重要。

---

## 8. E3-V2：并列式 GPA Cache

### 8.1 为什么进入 V2

E3-V1 结果说明顺序式不理想。我们判断问题不只是中心来源，而是缓存关系：GPA Cache 不应该依附于 Global Entropy Cache，而应当和 Entropy Cache 并列更新。

E3-V2 的关系：

```text
Global Entropy Cache 和 GPA Cache 并列更新。
两者互不替换、互不依赖。
Global Entropy Cache 仍用于 global logits。
GPA Cache 控制 local cache 写入。
```

### 8.2 三种中心来源

| 编号 | 名称 | 中心来源 |
|---|---|---|
| V2-A | GPA-only center | GPA Cache |
| V2-B | Entropy-only center | Global Entropy Cache |
| V2-C | Entropy+GPA union center | Entropy Cache ∪ GPA Cache |

### 8.3 结果

| 方法 | 关系 | 中心来源 | 平均准确率 | 相对 E2 baseline |
|---|---|---|---:|---:|
| E2 baseline | 原始 full Point-Cache | 无 GPA | 54.00 | 0.00 |
| E3-V2-A | 并列式 | GPA-only | 53.70 | -0.30 |
| E3-V2-B | 并列式 | Entropy-only | 53.15 | -0.85 |
| E3-V2-C | 并列式 | Entropy+GPA union | 54.04 | +0.04 |

分 corruption 对比：

| corruption | E2 baseline | V2-A | V2-B | V2-C |
|---|---:|---:|---:|---:|
| add_global | 47.81 | 48.58 | 47.37 | 46.84 |
| add_local | 46.68 | 48.34 | 48.14 | 50.49 |
| dropout_global | 59.20 | 57.78 | 57.78 | 58.31 |
| dropout_local | 56.69 | 55.92 | 56.65 | 56.04 |
| rotate | 62.07 | 60.98 | 59.76 | 61.67 |
| scale | 55.23 | 55.59 | 54.46 | 55.06 |
| jitter | 50.32 | 48.74 | 47.89 | 49.88 |
| 平均 | 54.00 | 53.70 | 53.15 | 54.04 |

### 8.4 V1 与 V2 对比

| 中心来源 | V1 顺序式 | V2 并列式 | 提升 |
|---|---:|---:|---:|
| GPA-only | 53.44 | 53.70 | +0.26 |
| Entropy-only | 52.43 | 53.15 | +0.72 |
| Entropy+GPA union | 53.01 | 54.04 | +1.03 |

结论：三种中心来源下，并列式都比顺序式更好。

### 8.5 为什么 V2-C 最好

V2-C 同时利用 Global Entropy Cache 的低熵稳定性和 GPA Cache 的原型对齐信息。GPA-only center 太依赖 GPA 自身，受初始化影响大；Entropy-only center 又太接近原始低熵逻辑，不能表达 GPA 的对齐结构。Union center 更稳定。

但 V2-C 只比 baseline 高 +0.04，说明方向有潜力但不稳定，不能作为最终方法。尤其它主要靠 add_local 提升拉动，其他多数 corruption 仍低于 baseline。

### 8.6 V2 后续补实验计划

完整 35 corruption；E2 的 manual_full_llm_fusion 版本是否也适合 V2-C；V2-C 在更多 backbone 上是否稳定；分析替换事件中熵和距离的关系；分析 local cache 数量和覆盖；检查 add_local 为什么提升明显；判断是否需要调整 final logits 权重，而不是只改缓存构造。

---

## 9. E3-V3：GPA Cache 初始化问题

### 9.1 为什么进入 V3

V2-C 说明并列式和 union center 有潜力，但收益太小。进一步怀疑：GPA Cache 的初始化仍然不合理。当前问题是：未满时，前 K 个样本直接进入 GPA Cache；距离约束只有满后才真正生效；而 K=3，很小；因此早期样本强烈决定 GPA 中心和 local cache 初始内容。

如果前 K 个样本伪标签错误、熵低但距离远、局部结构覆盖不足，就会污染后续 GPA 和 local cache。

### 9.2 原失败/不稳定初始化方式

原方式是：GPA Cache 未满直接加入；GPA Cache 已满后，低熵 + 距离更近才替换。问题包括：前 K 个样本没有真正筛选；前 K 个样本直接决定 GPA center；前 K 个样本的 local item 直接进入 local cache；后续替换围绕早期中心展开，可能自我强化；local cache 可能被早期样本污染或覆盖不足。

### 9.3 三种初始化改进方案

#### 9.3.1 Init-A：Entropy-bootstrap initialization

用户最终同意的 Init-A 版本：初始化阶段只用 Global Entropy Cache 启动 GPA Cache；初始化完成后使用 Entropy Cache + GPA Cache union center 做后续更新。准确名称是：

```text
Entropy-bootstrap initialization with Entropy+GPA union center
```

流程：build 阶段只构建 Global Entropy Cache；同步保存 entropy cache 样本对应的 local item；build 结束后，用 entropy cache 的低熵样本初始化 GPA Cache；同步把这些样本的 local item 写入 GPA-controlled Local Cache；test 阶段继续并列更新 Entropy Cache 和 GPA Cache；后续中心使用 Entropy+GPA union center。

优点是保守、不引入 candidate pool 状态机、避免 GPA 自己前 K 个样本冷启动、保留 V2-C 的 union center 优势。风险是 GPA 初始化几乎等于复制 entropy cache，可能和原始 Point-Cache 太像；如果 entropy cache 本身有高置信错误，GPA 也会继承；local cache 仍来自 entropy cache，不一定真正更“原型对齐”。

当前实现状态：Init-A 文件已生成并通过标识检查，但运行时曾出现 `_loss_value(entropy_cache[pred][-1])` 的 bug，应该改成 `_loss_value(entropy_cache[pred][-1][1])`。这说明 `_update_entropy_cache_with_local` 中取 worst entropy 时不能把整个 `[pc_feats, loss]` 当作 loss。

#### 9.3.2 Init-B：Delayed Local Cache Writing

用户批评之前没有讲清楚 Init-B 的筛选机制。完整版本如下。

Init-B 的核心不是重建 GPA Cache，而是保护 local cache：GPA Cache 可以照常更新，但 local cache 不立即写入，而是等候选样本筛选后再写入。

**Init-B0 弱版本**：GPA Cache 仍按原逻辑未满直接加入；local cache 暂时不写；当某类 GPA Cache 达到 K 后，再把这 K 个样本的 local item 写入 local cache。问题是没有真正筛选，只是延迟写入。

**Init-B1 推荐版本**：每类维护 local candidate pool；先收集 M 个候选，例如 M=2K；根据熵和距离筛出 K 个；只有这 K 个样本的 local item 写入 local cache；GPA Cache 本身可以仍然用原逻辑或 Entropy-bootstrap 逻辑。

筛选触发时机：`candidate_pool[c]` 达到 2K 时筛选；build 阶段结束时，如果 `candidate_pool[c] >= K`，也可以退化筛选。

筛选准则：计算每个候选的 `entropy_i`；计算到中心的距离 `distance_i`；可以使用 `rank_score = entropy_rank + distance_rank`；选择 rank_score 最小的 K 个。

Init-B 与 Init-C 的区别是：Init-B 重点控制 local cache 写入；Init-C 同时控制 GPA Cache 和 local cache 初始化。

#### 9.3.3 Init-C：Candidate Pool Initialization

Init-C 是最直接解决前 K 个样本无筛选进入的方法。设计是每类先收集 2K 或 3K 个候选，用 Entropy Cache ∪ GPA candidate pool 计算中心，根据熵和距离筛出 K 个，这 K 个进入正式 GPA Cache，只有这 K 个写入 local cache。

当前代码第一版的筛选规则：

```text
entropy_order = 按 entropy 升序排序；
distance_order = 按 distance 升序排序；
entropy_rank_i = 熵排名；
distance_rank_i = 距离排名；
rank_score_i = entropy_rank_i + distance_rank_i；
选 rank_score 最小的 K 个。
```

这个规则不是严格的“熵低且距离近”，而是排序折中。如果某个样本熵低但距离远，它可能被 distance_rank 惩罚，不一定被选中。

当前 Init-C 第一版失败现象：

```text
entropy cache total: 89
gpa cache total: 87
gpa local cache total: 87
gpa candidate pool total: 2
测试累计准确率：0.00 -> 30.14 -> 36.26
用户手动停止。
```

判断：不能直接说 Init-C 思路失败，但当前第一版策略/实现过于激进或存在逻辑风险。可能原因：只选中心样本，local cache 多样性下降；local cache 条目变少，覆盖不足；candidate pool/formal GPA/local cache 三状态机复杂；筛选规则没有处理好熵和距离冲突；selected 样本适合作为全局原型，但不一定适合作为 local cache 来源；现有 final logits 权重没有随 local cache 数量变化调整。

因此当前先暂停 Init-C，记录失败分析，转向更保守的 Init-A。

### 9.4 E3-V3 当前状态

截至 2026-06-07，E3-V3 初始化系列已经从早期 Init-A/Init-C 调试推进到候选池距离初始化消融。已经完成的关键结果包括：

| 方法 | 初始化/更新规则 | 平均准确率 | 相对 E2 baseline 54.00 |
|---|---|---:|---:|
| E3-V3-B | Entropy-bootstrap initialization | 53.25 | -0.75 |
| E3-V3-C1-Ub | Candidate-only center + distance-only update | 53.39 | -0.61 |
| E3-V3-C1-Ua2 | Candidate-only center + low-entropy gate + replace farthest | 53.68 | -0.32 |
| E3-V3-C1-Ua1 | Candidate-only center + low-entropy gate + replace highest-entropy | 54.02 | +0.02 |
| E3-V3-C2-Ua1 | Candidate+Entropy center + low-entropy gate + replace highest-entropy | 54.00 | 0.00 |

最新判断是：candidate pool 初始化可以缓解部分前 K 个样本直接进入的问题，但没有超过 E3-V2-C 的 54.04；纯距离更新不稳定，低熵门控仍然必要；替换最高熵样本比替换最远样本更稳。E3-V3-C1-Ua1 基本追平 E2 baseline，但仍略低于 E3-V2-C。

---

## 10. 当前实验结果总表

### 10.1 E1 zero-shot 文本方法

| 阶段 | 方法 | 平均准确率 |
|---|---|---:|
| E1 | manual_full | 47.68 |
| E1 | manual_3d | 35.63 |
| E1 | llm_only | 39.30 |
| E1 | manual_full_llm_fusion | 48.88 |

### 10.2 E1 fusion weight

| 权重 | 平均准确率 |
|---|---:|
| manual 0.90 / LLM 0.10 | 48.41 |
| manual 0.85 / LLM 0.15 | 48.62 |
| manual 0.75 / LLM 0.25 | 48.88 |
| manual 0.50 / LLM 0.50 | 48.37 |

### 10.3 E2 文本迁移到 Point-Cache

| 文本方法 | Cache | 平均准确率 |
|---|---|---:|
| manual_full | zs_global | 52.66 |
| manual_full_llm_fusion | zs_global | 53.18 |
| manual_full | zs_global_local | 54.00 |
| manual_full_llm_fusion | zs_global_local | 54.21 |

### 10.4 E3 V1/V2/V3

| 方法 | 关系 | 中心来源 | 平均准确率 |
|---|---|---|---:|
| E2 baseline | 原始 full Point-Cache | 无 GPA | 54.00 |
| E3-V1-A | 顺序式 | GPA-only | 53.44 |
| E3-V1-B | 顺序式 | Entropy-only | 52.43 |
| E3-V1-C | 顺序式 | Entropy+GPA union | 53.01 |
| E3-V2-A | 并列式 | GPA-only | 53.70 |
| E3-V2-B | 并列式 | Entropy-only | 53.15 |
| E3-V2-C | 并列式 | Entropy+GPA union | 54.04 |
| E3-V3-B | 并列式 + 初始化改进 | Entropy-bootstrap | 53.25 |
| E3-V3-C1-Ub | 候选池初始化 | Candidate-only + distance-only | 53.39 |
| E3-V3-C1-Ua2 | 候选池初始化 | Candidate-only + 低熵门控 + 替换最远样本 | 53.68 |
| E3-V3-C1-Ua1 | 候选池初始化 | Candidate-only + 低熵门控 + 替换最高熵样本 | 54.02 |
| E3-V3-C2-Ua1 | 候选池初始化 | Candidate+Entropy + 低熵门控 + 替换最高熵样本 | 54.00 |

---

## 11. 当前最重要的判断与后续规划

1. E1 的文本增强方向成立，但 LLM 只能补充人工模板，不能替代人工模板。
2. E2 说明文本增强收益能迁移到 Point-Cache，但提升较小，后续要完整验证。
3. E3 的顺序式 GPA 不理想，并列式更合理。
4. E3 当前最佳仍是 V2-C：并列式 + Entropy+GPA union center，平均 54.04，相对 E2 baseline 只有 +0.04，不足以作为最终方法。
5. E3-V3 证明 candidate pool 初始化不是当前主要突破口：C1-Ua1 达到 54.02，C2-Ua1 达到 54.00，但都没有超过 V2-C。
6. 纯距离更新不稳定，低熵门控仍然必要；替换最高熵样本比替换最远样本更稳。
7. E3 的正收益主要集中在 add_global / add_local，说明单中心原型对齐更擅长过滤添加型外点噪声。
8. E3 在 dropout、rotate、scale、jitter 上不稳定，说明单中心原型会损失类内几何变化和 local cache 覆盖。
9. 当前瓶颈不再是单纯的中心来源或初始化方式，而是单中心原型无法表达类别内部几何变化范围。
10. 未来更有潜力的方向包括：多中心/聚类覆盖、类别概率分布、local cache 多样性、文本原型中心、熵-能量联合可靠性。

纵向主线下一步：不要继续围绕 E3 单中心初始化做小修小补。应以 E3-V2-C 作为 E3 阶段最优参考，进入能表达类内多模式或类别概率分布的下一阶段；如果继续保留 E3 线索，优先做多中心/分布式原型和 local cache 多样性，而不是继续单中心 candidate pool 变体。

回补横向实验包括：E1 完整文本消融；E2 完整文本迁移；E3 V1/V2/V3 完整消融；GPA relation 消融；center source 消融；initialization 消融；final logits 权重调参；negative cache ablation；local cache 覆盖度分析。

---

## 12. 专家自查：一个陌生专家读完还会问什么？

### 12.1 这些结果是不是完整 benchmark？

不是。当前多数是 ULIP × ModelNet-C severity=2 × 7 corruption 的最小烟雾测试。文档已明确：这是纵向打通实验，不是最终 benchmark。

### 12.2 为什么不直接用 BayesMM 或 Uni-Adapter？

因为当前项目主线是基于 Point-Cache 的渐进式改造，先验证文本和 cache 机制。如果直接跳到分布建模或聚类原型，会打断当前从 E0 到 E3 的逻辑链。BayesMM/Uni-Adapter 作为后续方向保留。

### 12.3 为什么 E3 只提升 +0.04 还继续分析？

因为 +0.04 本身不是目标，重要的是 E3 揭示了两个边界：并列式 + union center 比顺序式更合理；单中心原型对 add_global/add_local 这类添加型噪声有效，但对 dropout、rotate、scale、jitter 等几何变化不稳定。这个发现直接支持后续从单中心 GPA 转向多中心或类别概率分布。

### 12.4 为什么 Init-C 暂停而不是放弃？

因为 Init-C 的第一版同时改变了 GPA 初始化、local cache 写入、候选池状态机和筛选规则，下降可能来自策略过激或代码状态复杂，不足以否定候选池思想。

### 12.5 后续最先该做什么？

以 E3-V2-C 作为 E3 阶段最优参考，优先设计下一阶段：从“样本是否更靠近单一类别中心”转向“样本是否符合类别内部多模式/概率分布”。如果继续补 E3，应补 local cache 覆盖度、多中心原型、候选池多样性和 final logits 权重，而不是继续单中心初始化细枝末节。

---

## 13. 附录：当前重要路径与文件

```text
/root/autodl-tmp/MCM-PC-2
/root/autodl-tmp/MCM-PC-2/Point-Cache
Point-Cache/runners/E1_text_prototype_enhancement
Point-Cache/runners/E2_text_prototype_transfer_to_pointcache
Point-Cache/runners/E3_global_prototype_alignment_cache
Point-Cache/scripts/E3_global_prototype_alignment_cache
docs/experiments/E3_global_prototype_alignment_cache
```

当前 E3 重要文件：

```text
model_with_hierarchical_caches_parallel_gpa_entropy_gpa_union_center.py
run_e3_ulip_modelnetc_s2_parallel_gpa_entropy_gpa_union_center.py
model_with_hierarchical_caches_parallel_gpa_candidate_pool_init.py
run_e3_ulip_modelnetc_s2_parallel_gpa_candidate_pool_init.py
model_with_hierarchical_caches_parallel_gpa_entropy_bootstrap_init.py
run_e3_ulip_modelnetc_s2_parallel_gpa_entropy_bootstrap_init.py
03_1_ulip_modelnetc_s2_zs_global_local_parallel_gpa_candidate_pool_init_manual_full.sh
03_2_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_bootstrap_init_manual_full.sh
```

---

## 14. 结束语

本项目目前最重要的不是某一个 smoke test 的小幅数值，而是已经形成了一条清晰的研究路线：

```text
复现 Point-Cache
-> 验证文本增强
-> 验证文本收益迁移到缓存
-> 改造缓存构造
-> 发现原型对齐缓存的关系、中心来源和初始化问题
-> 继续改进 GPA 初始化和 local cache 覆盖
```

当前应避免频繁换大方向，先把 Init-A 调通，再系统补实验。
