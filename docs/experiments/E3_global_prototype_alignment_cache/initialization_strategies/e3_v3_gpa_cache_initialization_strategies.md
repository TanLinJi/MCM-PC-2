# E3-V3：GPA Cache 初始化机制改进方案记录

更新日期：2026-06-05

## 1. 背景

E3 当前已经完成两个阶段：

- E3-V1：顺序式 Global Prototype-Alignment Cache；
- E3-V2：并列式 Global Prototype-Alignment Cache。

其中 E3-V2-C，即：

    并列式 GPA Cache
    + Entropy+GPA union center
    + manual_full
    + zs_global_local

取得当前最好结果：

    54.04

略高于 E2 原始 full Point-Cache baseline：

    54.00

但是该提升幅度很小，仅为：

    +0.04

这说明并列式 GPA Cache 和 Entropy+GPA union center 是有潜力的，但当前方法仍然不够稳定。

进一步分析后认为，当前主要问题可能来自：

    GPA Cache 初始化机制不够可靠。

## 2. 当前 GPA Cache 初始化问题

当前 E3-V2 中，Global Entropy Cache 和 GPA Cache 是并列更新的。

样本到来后：

    一路尝试更新 Global Entropy Cache；
    另一路独立尝试更新 GPA Cache。

GPA Cache 的当前更新规则为：

    如果某类 GPA Cache 未满：
        直接加入样本；
        对应局部特征也写入 local cache。

    如果某类 GPA Cache 已满：
        找到该类 GPA Cache 中当前最高熵样本；
        若新样本熵更低；
        且新样本到原型中心的距离
        小于该最高熵样本到原型中心的距离；
        则替换该最高熵样本。

该规则的问题是：

    GPA Cache 未满阶段没有真正的距离筛选。

因此，在每类容量 K 较小的情况下，前 K 个样本会较容易直接进入 GPA Cache。

当前实验中 K 通常为：

    K = 3

这意味着每个类别中，最早进入 GPA Cache 的 3 个样本会直接决定：

1. GPA Cache 的初始内容；
2. GPA 原型中心；
3. GPA-controlled Local Cache 的初始内容；
4. 后续样本是否能够替换已有缓存项。

如果前 K 个样本中存在伪标签错误、高置信错误、离群样本或局部结构覆盖不足，那么后续 GPA Cache 可能会沿着错误中心继续更新，导致局部缓存质量不稳定。

因此，E3-V3 的目标是：

    改进 GPA Cache 初始化机制，
    避免前 K 个样本几乎无筛选地进入 GPA Cache 和 local cache。

## 3. 相关概念说明

### 3.1 Global Entropy Cache

Global Entropy Cache 是原始 Point-Cache 中的全局低熵缓存。

它存储样本级全局特征，主要根据预测熵进行更新。

其作用是：

    为 global cache logits 提供全局样本级检索信息。

### 3.2 Global Prototype-Alignment Cache

Global Prototype-Alignment Cache 简称 GPA Cache。

它是 E3 新增的全局原型对齐缓存。

其目标是：

    筛选更可靠、更接近类别原型中心的全局样本，
    并用这些样本控制 local cache 的写入。

### 3.3 Local Cache 与局部特征

Point-Cache 的 local cache 存储的是测试样本的局部结构特征。

在代码中，一个点云样本经过局部特征聚类后会得到若干局部代表特征，通常称为：

    patch_centers

可以理解为：

    一个点云样本的若干局部结构中心。

例如，对于椅子类点云，局部特征可能对应椅背、椅面、椅腿等局部结构。

E3 中的设计是：

    只有进入 GPA Cache 的样本，
    它的局部特征才写入 local cache。

因此，如果 GPA Cache 初始化阶段样本质量不高，那么 local cache 也可能被早期样本影响。

## 4. E3-V3 总体目标

E3-V3 需要解决的问题是：

    如何在 GPA Cache 未形成可靠中心之前，
    避免前 K 个样本无筛选地进入 GPA Cache 和 local cache。

当前总结出三类可行方案：

| 方案 | 名称 | 核心思想 | 优先级 |
|---|---|---|---|
| Init-A | Global Entropy Center Initialization | 先用 Global Entropy Cache 初始化中心 | 中 |
| Init-B | Delayed Local Cache Writing | 延迟 local cache 写入，先筛 GPA 后写 local | 中 |
| Init-C | Candidate Pool Initialization | 先收集 2K/3K 候选，再筛 K 个进入 GPA | 高 |

当前建议优先实现：

    Init-C：候选池初始化。

原因是它最直接地解决“前 K 个样本直接进入 GPA Cache”的问题。

## 5. Init-A：先用 Global Entropy Cache 初始化中心

### 5.1 核心思想

Init-A 的核心思想是：

    先构建 Global Entropy Cache；
    再使用 Global Entropy Cache 中每类样本的全局特征均值作为初始中心；
    最后使用该中心指导 GPA Cache 的初始化和更新。

也就是说，GPA Cache 不再完全冷启动，而是借助原始低熵缓存提供一个初始视觉中心。

### 5.2 流程设计

流程可以写成：

    Step 1：按原始 Point-Cache 规则预构建 Global Entropy Cache；
    Step 2：对每个类别 c，使用 Global Entropy Cache[c] 中的全局特征计算初始中心；
    Step 3：GPA Cache 初始化时，不再完全无条件接收前 K 个样本；
    Step 4：新样本尝试进入 GPA Cache 时，根据熵和到 Entropy center 的距离进行筛选；
    Step 5：进入 GPA Cache 的样本，其局部特征写入 local cache。

### 5.3 中心计算方式

对于类别 c：

    entropy_center_c = mean(Global Entropy Cache[c])

然后归一化：

    entropy_center_c = normalize(entropy_center_c)

### 5.4 准入规则

当 GPA Cache 未满时，可以设置为：

    样本熵较低；
    且样本到 entropy_center_c 的距离较小。

具体实现上可以先采用候选排序：

    score = entropy_rank + distance_rank

选择综合排名较好的样本进入 GPA Cache。

或者保守一点：

    GPA Cache 未满时仍允许加入；
    但只使用 entropy_center 作为满后替换时的中心。

不过如果仍然未满直接加入，则 Init-A 对初始化问题的改善有限。因此更推荐在 Init-A 中也加入候选筛选机制。

### 5.5 优点

Init-A 的优点是：

1. 实现相对简单；
2. 利用已有 Global Entropy Cache；
3. 避免 GPA Cache 完全依赖自身前 K 个样本计算初始中心；
4. 与 E3-V2-C 中 union center 的思想兼容。

### 5.6 风险

Init-A 的风险是：

1. Global Entropy Cache 本身也可能包含高置信错误样本；
2. Entropy center 可能偏向原始 Point-Cache 的低熵分布；
3. 如果只用 Entropy center，可能无法充分体现 GPA Cache 的对齐样本分布；
4. 仍然没有彻底解决“前 K 个样本直接写入 local cache”的问题。

### 5.7 适合回答的问题

Init-A 主要回答：

    用 Global Entropy Cache 提供初始中心，
    是否能缓解 GPA Cache 冷启动问题？

## 6. Init-B：延迟 local cache 写入

### 6.1 核心思想

Init-B 的核心思想是：

    GPA Cache 初始化阶段可以先收集候选样本，
    但不要立即把这些样本的局部特征写入 local cache。

等 GPA Cache 完成筛选后：

    只有最终保留下来的 K 个样本，
    它们的局部特征才写入 local cache。

### 6.2 为什么要延迟 local cache 写入

当前 E3 的设计中：

    样本进入 GPA Cache
        ->
    该样本的局部特征立即进入 local cache。

问题是：

    GPA Cache 在未满阶段并没有经过充分筛选，
    因此前 K 个样本的局部特征也会直接进入 local cache。

这可能导致：

    local cache 被早期不稳定样本影响。

Init-B 要避免这种情况。

### 6.3 流程设计

以每类容量 K=3 为例：

当前做法：

    前 3 个进入 GPA Cache 的样本
        ->
    立即写入 local cache。

Init-B 改为：

    Step 1：先为每个类别收集 GPA 候选样本；
    Step 2：这些候选样本暂时只存在 GPA candidate pool；
    Step 3：不立即写入 local cache；
    Step 4：等候选池达到一定数量后，筛选出最可靠的 K 个；
    Step 5：这 K 个样本进入正式 GPA Cache；
    Step 6：只有这 K 个样本的局部特征写入 local cache。

### 6.4 “筛选”是什么意思

筛选可以基于两个指标：

1. 熵；
2. 到原型中心的距离。

候选样本需要满足：

    预测熵较低；
    到类别中心较近。

更具体地，可以先计算每个候选样本的：

    entropy_i
    distance_i

然后构造排序指标，例如：

    rank_score_i = rank(entropy_i) + rank(distance_i)

选择 rank_score 最小的 K 个。

也可以使用加权指标：

    score_i = alpha * normalized_entropy_i + beta * normalized_distance_i

选择 score 最小的 K 个。

### 6.5 优点

Init-B 的优点是：

1. 直接保护 local cache；
2. 避免未经筛选的早期样本污染 local cache；
3. 可以与 Init-A 或 Init-C 组合；
4. 不一定需要改变最终预测加权公式。

### 6.6 风险

Init-B 的风险是：

1. 实现复杂度高于 Init-A；
2. 需要维护 GPA candidate pool；
3. 在 local cache 延迟写入期间，local cache 可能较小；
4. 如果候选池形成太慢，早期预测可能缺少 local cache 支持。

### 6.7 适合回答的问题

Init-B 主要回答：

    当前 E3 的下降是否来自早期 GPA 样本的局部特征过早进入 local cache？

## 7. Init-C：候选池初始化

### 7.1 核心思想

Init-C 的核心思想是：

    每个类别先不急着选 K 个样本进入 GPA Cache；
    而是先收集超过 K 个候选样本；
    再从候选池中筛出最可靠的 K 个；
    这 K 个进入 GPA Cache；
    只有这 K 个样本的局部特征进入 local cache。

例如当前 K=3，可以先收集：

    2K = 6 个候选样本

或者：

    3K = 9 个候选样本

再筛出：

    K = 3 个正式缓存样本。

### 7.2 为什么 Init-C 优先级最高

当前 GPA Cache 最大问题是：

    前 K 个样本容易直接进入 GPA Cache。

Init-C 直接解决这个问题：

    不再让前 K 个样本直接成为 GPA Cache；
    而是先观察 2K 或 3K 个候选样本；
    再选出最可靠的 K 个。

这比 Init-A 和 Init-B 更直接。

### 7.3 候选池来源

候选池可以这样构建：

    每个样本根据预测类别 c 放入 GPA candidate pool[c]；
    每个类别最多先收集 candidate_capacity 个候选。

其中：

    candidate_capacity = r * K

例如：

    r = 2，则 candidate_capacity = 2K；
    r = 3，则 candidate_capacity = 3K。

当前建议先用：

    r = 2

即：

    每类收集 2K 个候选，再筛 K 个。

### 7.4 中心来源：推荐 C-3

Init-C 中需要计算候选样本到中心的距离。

当前更推荐 C-3：

    Global Entropy Cache + GPA candidate pool union center

即：

    对类别 c，
    使用 Global Entropy Cache[c] 和 GPA candidate pool[c] 的全局特征共同计算中心。

形式为：

    center_c = mean(Global Entropy Cache[c] ∪ GPA candidate pool[c])

然后归一化：

    center_c = normalize(center_c)

### 7.5 为什么推荐 C-3

C-3 的优点是：

1. Global Entropy Cache 提供低熵稳定样本；
2. GPA candidate pool 提供当前待筛选样本集合；
3. 两者联合中心比单独候选池中心更稳定；
4. 它继承了 E3-V2-C 的经验：Entropy+GPA union center 当前效果最好。

因此，Init-C 的首选版本应为：

    Candidate Pool Initialization
    + Entropy Cache and GPA Candidate Pool Union Center

### 7.6 筛选规则

对每个候选样本 i，计算：

    entropy_i
    distance_i = distance(feature_i, center_c)

然后根据熵和距离共同筛选。

可以采用排序融合：

    rank_score_i = rank(entropy_i) + rank(distance_i)

选择 rank_score 最小的 K 个。

也可以采用加权融合：

    score_i = alpha * normalized_entropy_i + beta * normalized_distance_i

选择 score 最小的 K 个。

第一版建议采用排序融合。

原因：

    不需要额外调 alpha / beta；
    对熵和距离的数值尺度不敏感；
    更适合 smoke test。

### 7.7 正式写入

筛选完成后：

    最可靠的 K 个候选样本进入正式 GPA Cache；
    它们的局部特征写入 local cache。

没有被选中的候选样本：

    不进入 GPA Cache；
    不写入 local cache。

### 7.8 后续动态更新

GPA Cache 初始化完成后，正式测试阶段可以继续使用 E3-V2-C 的并列式更新逻辑：

    Global Entropy Cache 与 GPA Cache 并列更新；
    GPA Cache 已满后，根据熵和距离替换；
    local cache 跟随 GPA Cache 中最终样本更新。

需要注意：

    如果 GPA Cache 中某个样本被替换，
    其对应的局部特征也应同步替换。

### 7.9 优点

Init-C 的优点是：

1. 直接解决前 K 个样本无筛选进入的问题；
2. 充分利用更多候选样本；
3. 与 E3-V2-C 的 union center 结论一致；
4. 可以减少早期样本对 GPA Cache 和 local cache 的影响；
5. 逻辑清楚，便于论文解释。

### 7.10 风险

Init-C 的风险是：

1. 需要额外维护 candidate pool；
2. 候选池容量 rK 的选择需要实验验证；
3. 初始化阶段可能需要更多样本；
4. 如果某些类别预测样本较少，候选池可能不容易填满；
5. 需要处理候选池未满时的退化逻辑。

### 7.11 候选池未满时的处理

如果某个类别一直无法收集到 2K 个候选，可以设置退化规则：

    如果候选数 >= K：
        直接在已有候选中筛 K 个；
    如果候选数 < K：
        暂时不形成该类 GPA Cache，
        或者使用已有候选并标记为不完整初始化。

第一版建议：

    候选数 >= K 时即可筛选；
    不强制必须达到 2K。

这样可以避免某些类别长期没有 GPA Cache。

### 7.12 适合回答的问题

Init-C 主要回答：

    比前 K 更多的候选样本筛选，
    是否能让 GPA Cache 初始化更稳定，
    并进一步放大 E3-V2-C 的收益？

## 8. 三种方案对比

| 方案 | 解决的问题 | 优点 | 风险 | 优先级 |
|---|---|---|---|---|
| Init-A | GPA 中心冷启动 | 简单，利用 Entropy Cache 初始化中心 | 仍可能让早期样本写入 local cache | 中 |
| Init-B | local cache 过早污染 | 直接保护 local cache | 需要维护候选和延迟写入逻辑 | 中 |
| Init-C | 前 K 个样本无筛选进入 GPA | 最直接解决初始化问题 | 实现复杂度最高 | 高 |

## 9. 推荐实施顺序

当前推荐：

    先实现 Init-C。

理由：

1. 当前 E3-V2-C 已经证明 Entropy+GPA union center 有潜力；
2. Init-C 可以继承 union center 思想；
3. Init-C 直接解决前 K 个样本直接进入 GPA Cache 的问题；
4. Init-C 同时可以自然包含 Init-B 的延迟 local cache 写入思想。

第一版 E3-V3 可以设计为：

    E3-V3-A：parallel GPA + candidate pool initialization
    center source：Global Entropy Cache + GPA candidate pool union center
    candidate pool size：2K
    final GPA size：K
    local cache：只写入最终筛出的 K 个样本
    text method：manual_full
    cache setting：zs_global_local

## 10. 暂时不引入文本原型

当前暂时不引入文本原型。

原因：

    E3-V2-C 已经说明视觉缓存内部仍有改进空间。
    应先解决 GPA Cache 初始化问题，
    再考虑使用文本原型作为语义锚点。

文本原型可以作为后续方向：

    E3-V4：Visual-Text Prototype Center

其核心思想是：

    visual center = Global Entropy Cache + GPA Cache
    text center = class text prototype
    final center = visual-text fused center

但这不是 E3-V3 的当前重点。

## 11. 当前结论

E3-V3 的核心任务是：

    改进 GPA Cache 初始化机制。

三种方法中：

    Init-C 候选池初始化最值得优先实现。

它将当前 E3-V2-C 的最佳发现：

    并列式 GPA Cache
    + Entropy+GPA union center

进一步扩展为：

    并列式 GPA Cache
    + 候选池初始化
    + Entropy Cache 与 GPA candidate pool 联合中心
    + 先筛 K 个，再写入 GPA Cache 和 local cache。

该方向最有可能解决当前 GPA Cache 初始化不稳定的问题。
