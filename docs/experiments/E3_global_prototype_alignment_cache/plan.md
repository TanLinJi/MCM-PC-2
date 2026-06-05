# E3：全局原型对齐缓存实验计划

更新日期：2026-06-04

## 1. 实验名称

英文目录名：

    E3_global_prototype_alignment_cache

中文名称：

    E3：全局原型对齐缓存

核心模块名称：

    Global Prototype-Alignment Cache

中文名称：

    全局原型对齐缓存

简称：

    GPA Cache

## 2. 实验背景

E1 已经验证了文本原型增强在 zero-shot 设置下有效。

E2 已经验证了 E1 的文本原型融合收益可以传递到 Point-Cache 的 global cache 和 full Point-Cache 流程中。

E2 当前最重要结论为：

| 阶段 | manual_full | manual_full_llm_fusion | 提升 |
|---|---:|---:|---:|
| zero-shot | 47.68 | 48.88 | +1.20 |
| global cache | 52.66 | 53.18 | +0.52 |
| full Point-Cache | 54.00 | 54.21 | +0.21 |

E2 说明文本原型融合可以作为附加增益，但完整 Point-Cache 的主要增益仍然来自 cache 机制本身。

因此，E3 开始从文本原型增强转向 Point-Cache 缓存机制本身的改进。

## 3. E3 核心目标

E3 的目标是：

    在完整 Point-Cache 设置中，引入全局原型对齐缓存，
    通过更严格的全局样本筛选，提高 local cache 的样本质量，
    从而提升整体分类准确率。

E3 当前关注的是整体平均准确率，而不是单个损坏类型上的特殊现象。

## 4. E3 与前序实验的关系

| 阶段 | 主要目标 |
|---|---|
| E1 | 验证文本原型增强是否提升 zero-shot |
| E2 | 验证文本原型收益是否能传递到 Point-Cache 流程 |
| E3 | 改进 Point-Cache 缓存机制本身，重点提升 local cache 样本质量 |

E3 不再继续横向扩展 E1 文本消融，而是继续纵向推进 Point-Cache 机制改进。

## 5. 当前 E3-V1：顺序式全局原型对齐缓存方案

E3-V1 对应当前优先实现的方案。

它采用顺序式设计：

    原始 Global Entropy Cache
        ↓
    Global Prototype-Alignment Cache
        ↓
    Local Cache

也就是说：

1. 新样本先按照原始 Point-Cache 的低熵规则尝试进入 Global Entropy Cache；
2. 只有满足全局低熵准入的样本，才有资格继续尝试进入 Global Prototype-Alignment Cache；
3. Global Prototype-Alignment Cache 为每个类别维护自己的原型中心；
4. 只有进入 Global Prototype-Alignment Cache 的样本，它对应的 local patch centers 才写入 Local Cache；
5. 当前最小验证阶段暂不修改最终预测加权公式。

## 6. E3-V1 中 GPA Cache 的原型中心

当前 E3-V1 采用：

    GPA-only center

也就是：

    每个类别的原型中心
    =
    该类别 Global Prototype-Alignment Cache 中全局特征的均值

注意：

当前 E3-V1 不使用 Global Entropy Cache 来计算原型中心。

Global Entropy Cache 仍然保留原始 Point-Cache 的作用，主要用于 global cache 分支；Global Prototype-Alignment Cache 则用于维护更严格的对齐样本，并控制 local cache 的样本来源。

## 7. GPA Cache 未形成中心之前如何处理

这是 E3-V1 的关键启动规则。

由于 Global Prototype-Alignment Cache 一开始为空，因此无法立即计算类别原型中心。

当前采用如下规则：

### 7.1 启动阶段

当某个类别的 Global Prototype-Alignment Cache 中样本数量不足以形成稳定类别中心时：

    暂时不启用“到原型中心距离”的判断。

此时，如果一个新样本已经满足原始 Point-Cache 的全局低熵准入条件，则允许它进入该类别的 Global Prototype-Alignment Cache。

这些样本用于建立该类别最初的 GPA 原型中心。

### 7.2 中心形成阶段

当某个类别的 Global Prototype-Alignment Cache 中样本数量达到 `min_center_size` 后：

    使用该类别 GPA Cache 中已有全局特征的均值作为类别原型中心。

当前建议：

    min_center_size = 2

原因是当前 Point-Cache 每类正缓存容量 K 较小，若要求过多初始样本，可能导致 GPA Cache 启动过慢。

### 7.3 严格更新阶段

当某个类别的 GPA Cache 已经形成原型中心后，新的样本需要满足：

1. 低熵条件；
2. 原型距离条件。

更具体地：

如果该类 GPA Cache 未满：

    满足低熵准入的样本可以加入。

如果该类 GPA Cache 已满：

    找到该类 GPA Cache 中当前熵最高的样本；
    若新样本熵更低，
    且新样本到 GPA 原型中心的距离
    小于这个最高熵样本到 GPA 原型中心的距离，
    则替换 GPA Cache 中这个最高熵样本。

## 8. E3-V1 对 local cache 的影响

原始 Point-Cache 中：

    只要一个样本进入 global cache，
    它对应的 local patch centers 就可以进入 local cache。

E3-V1 中：

    一个样本即使进入了 Global Entropy Cache，
    也不一定能贡献 local cache。

只有当该样本进一步进入 Global Prototype-Alignment Cache 后，它对应的 local patch centers 才写入 local cache。

因此，E3-V1 的本质是：

    不改变 zero-shot 分支；
    不改变 global cache 主体逻辑；
    当前最小验证阶段不改变最终预测加权公式；
    只改变 local cache 的样本来源。

这样可以隔离变量，优先验证：

    更严格的全局样本筛选是否能够提升 local cache 的样本质量。

## 9. E3-V1 当前实验矩阵

E3-V1 当前只新增两组实验。

这两组都在完整 Point-Cache 设置下运行：

    zs_global_local

### 9.1 不使用 E2 文本融合

| 项目 | 设置 |
|---|---|
| 文本方法 | manual_full |
| cache 设置 | zs_global_local |
| 新机制 | 顺序式 Global Prototype-Alignment Cache |
| 对比对象 | E2 00_3：manual_full + zs_global_local = 54.00 |

### 9.2 使用 E2 文本融合

| 项目 | 设置 |
|---|---|
| 文本方法 | manual_full_llm_fusion |
| cache 设置 | zs_global_local |
| 新机制 | 顺序式 Global Prototype-Alignment Cache |
| 对比对象 | E2 00_4：manual_full_llm_fusion + zs_global_local = 54.21 |

## 10. E3-V1 预期回答的问题

E3-V1 需要回答两个问题：

1. 在不使用 E2 文本融合时，GPA Cache 是否能单独提升完整 Point-Cache？
2. 在使用 E2 文本融合时，GPA Cache 是否能与文本原型融合兼容，并继续提升完整 Point-Cache？

对应对比为：

| 对比 | 原始 Point-Cache | E3-V1 GPA Cache |
|---|---:|---:|
| manual_full | 54.00 | 待实验 |
| manual_full_llm_fusion | 54.21 | 待实验 |

## 11. E3-V2：MCP-style 并列更新方案

E3-V2 对应后续要做的并列更新方案。

该方案中：

    Global Entropy Cache 和 Global Prototype-Alignment Cache 并列更新。

也就是说，新样本到来后：

    一路尝试更新 Global Entropy Cache；
    另一路尝试更新 Global Prototype-Alignment Cache。

两个缓存互不替换对方内部样本。

这与 E3-V1 的区别是：

| 项目 | E3-V1 顺序式方案 | E3-V2 并列式方案 |
|---|---|---|
| Entropy Cache 与 GPA Cache 关系 | 先 Entropy 后 GPA | 两者并列更新 |
| GPA 是否依赖先进入 Entropy Cache | 是 | 不一定 |
| local cache 来源 | GPA Cache 控制 | 后续设计 |
| 是否更接近 MCP 源码逻辑 | 否，属于 Point-Cache 最小改造 | 是，更接近 MCP 的并列缓存思想 |
| 改动范围 | 小 | 更大 |

E3-V2 不一定等 E3-V1 失败才做。

即使 E3-V1 有效，E3-V2 也需要作为后续对照实验，用于验证并列式缓存是否更优。

## 12. 原型中心来源消融计划

当前 E3-V1 采用：

    Center-A：GPA-only center

但是原型中心来源不一定唯一，后续需要做消融。

### 12.1 视觉原型中心来源

| 名称 | 原型中心来源 | 含义 |
|---|---|---|
| Center-A | GPA-only center | 只使用 Global Prototype-Alignment Cache 中样本计算中心 |
| Center-B | Entropy-only center | 只使用 Global Entropy Cache 中样本计算中心 |
| Center-C | Entropy+GPA union center | 使用 Global Entropy Cache 和 Global Prototype-Alignment Cache 的并集计算中心 |

说明：

- Center-A 最符合当前 E3-V1 设计；
- Center-B 启动更稳定，但可能包含低熵离群样本；
- Center-C 更充分利用两个缓存，但变量更多，解释更复杂。

### 12.2 引入文本原型后的中心来源

后续会引入文本原型中心。

引入文本原型后，可以形成更多组合：

| 名称 | 原型中心来源 |
|---|---|
| Center-D | Text-only center |
| Center-E | GPA+Text center |
| Center-F | Entropy+Text center |
| Center-G | Entropy+GPA+Text center |

其中：

- Text-only center：只使用文本原型；
- GPA+Text center：使用 GPA Cache 视觉中心与文本原型融合；
- Entropy+Text center：使用 Entropy Cache 视觉中心与文本原型融合；
- Entropy+GPA+Text center：同时使用 Entropy Cache、GPA Cache 和文本原型。

这些组合暂时不在 E3-V1 中实现，只写入后续计划。

## 13. GPA Cache 准入规则消融计划

当前 E3-V1 采用类似“熵优先 + 原型距离校验”的规则：

    新样本熵更低；
    且新样本到原型中心的距离
    小于 GPA Cache 中最高熵样本到原型中心的距离。

后续可以做准入规则消融。

| 名称 | 规则 | 含义 |
|---|---|---|
| Rule-A | entropy-first distance-check | 低熵优先，再检查是否比最高熵样本更靠近中心 |
| Rule-B | distance-priority max-distance | 要求新样本距离小于当前 GPA Cache 中最大距离 |
| Rule-C | entropy-only | 只使用低熵，不使用距离约束 |
| Rule-D | distance-only | 只使用距离，不使用熵约束 |

其中，Rule-B 对应之前提出的“小于当前最大距离”的方案。

当前 E3-V1 先不做这些消融，只在后续完整实验后再验证。

## 14. Entropy Cache 与 GPA Cache 关系消融计划

后续需要比较：

| 名称 | 关系 | 说明 |
|---|---|---|
| Relation-A | 顺序式 | 先进入 Global Entropy Cache，再尝试进入 GPA Cache |
| Relation-B | 并列式 | Entropy Cache 和 GPA Cache 并列更新 |
| Relation-C | GPA 替代 Entropy | GPA Cache 直接替代 Global Entropy Cache |
| Relation-D | Entropy + GPA 双分支 | 两个缓存都参与最终预测，并可设置不同权重 |

当前 E3-V1 采用：

    Relation-A：顺序式

后续 E3-V2 采用：

    Relation-B：并列式

## 15. 最终预测加权公式的后续扩展

E3-V1 当前最小验证阶段暂时不修改最终预测加权公式。

这不是长期限制，而是为了隔离变量，先验证 GPA Cache 对 local cache 样本质量的影响。

后续可以继续研究：

1. zero-shot logits、Global Entropy Cache logits、GPA Cache logits、Local Cache logits 的独立加权；
2. 根据熵动态调整 cache 分支权重；
3. 根据到原型中心的距离动态调整 cache 分支权重；
4. 为 GPA Cache 单独设计 global logits；
5. 使用文本原型中心参与最终加权。

## 16. 论文表述与内部说明的区别

在论文中，不主动强调与 MCP 的关系。

论文中可以表述为：

    为提高测试时局部缓存的可靠性，引入全局原型对齐缓存，
    通过低熵和原型距离约束筛选更具代表性的全局样本，
    并仅允许这些样本的局部特征进入局部缓存。

在内部实验说明文档中，可以详细记录与 MCP 的对应关系、差异和后续消融计划，便于代码实现和实验复现。

## 17. 当前不做的事情

E3-V1 当前不做：

1. 不做并列式缓存；
2. 不引入文本原型中心；
3. 不改变最终预测加权公式；
4. 不做多原型中心；
5. 不做准入规则消融；
6. 不跑 all35；
7. 不跑多数据集；
8. 不跑多 backbone。

这些内容全部列入后续计划。

## 18. 下一步任务

### E3 准备检查

记录 MCP 原文与源码中 Entropy Cache、Align Cache 的真实规则。

注意：

    checks/ 不使用实验编号。

### E3-1：实现顺序式 GPA Cache

新增 E3 专用 runner，不直接修改原始 Point-Cache runner。

建议路径：

    Point-Cache/runners/E3_global_prototype_alignment_cache/

### E3-2：运行两组 smoke test

| 编号 | 文本方法 | cache 设置 |
|---|---|---|
| 00_1 | manual_full | zs_global_local |
| 00_2 | manual_full_llm_fusion | zs_global_local |

### E3-3：写结果分析

与 E2 的 full Point-Cache 两组结果对比：

| 文本方法 | E2 原始 full Point-Cache | E3 GPA Cache |
|---|---:|---:|
| manual_full | 54.00 | 待实验 |
| manual_full_llm_fusion | 54.21 | 待实验 |

### E3-4：Git 提交

提交 E3 文档、脚本和代码。

## 19. E3-V1 初版结果后的修正计划

E3-V1 初版采用：

    Relation-A：顺序式
    Center-A：GPA-only center

初版结果显示：

    manual_full + zs_global_local + GPA-only center = 53.44
    E2 原始 manual_full + zs_global_local = 54.00

整体下降：

    -0.56

因此，当前 Center-A 不能作为主方法。

### 19.1 当前发现的问题

预构建阶段中，GPA-controlled Local Cache 的数量与 Global Entropy Cache 数量几乎一致，说明 GPA Cache 在当前规则下没有形成足够筛选。

主要原因是：

    GPA Cache 未形成中心前直接加入；
    GPA Cache 形成中心但未满时仍直接加入；
    只有 GPA Cache 满了以后才启用距离约束。

在 K 较小的情况下，这会导致距离约束发挥不足。

### 19.2 统计日志修正

后续必须保存 GPA 统计文件：

    gpa_stats/

并额外保存替换事件日志：

    gpa_replacement_events_<cor_type>.jsonl

每次替换或距离拒绝时，应记录：

- phase；
- class_index；
- new_entropy；
- old_entropy；
- new_distance；
- old_distance；
- decision。

该日志用于分析：

    高熵样本是否一定更远；
    距离约束是否真正发挥作用；
    被拒绝样本和被替换样本之间的熵/距离关系。

### 19.3 下一步中心来源消融

在顺序式关系不变的前提下，优先测试：

| 名称 | 原型中心来源 |
|---|---|
| Center-B | Entropy-only center |
| Center-C | Entropy+GPA union center |

这两组实验用于判断当前下降是否来自 GPA-only center 不稳定。

### 19.4 后续并列式方案

如果 Center-B / Center-C 仍然不理想，或即使其中之一有效，后续仍需测试：

    E3-V2：MCP-style 并列更新方案

即：

    Global Entropy Cache 和 Global Prototype-Alignment Cache 并列更新，
    二者互不替换，
    再进一步设计它们如何参与 local cache 或最终预测。
