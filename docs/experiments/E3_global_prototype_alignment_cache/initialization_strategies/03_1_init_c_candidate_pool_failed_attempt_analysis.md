# E3-V3-A Init-C 第一版失败分析：Candidate Pool Initialization

更新日期：2026-06-06

## 1. 实验背景

E3-V3 旨在解决 GPA Cache 初始化不稳定的问题。

在 E3-V2-C 中，当前最优方案为：

    parallel GPA Cache
    + Entropy+GPA union center
    + manual_full
    + zs_global_local

其平均准确率为：

    54.04

略高于 E2 原始 full Point-Cache baseline：

    54.00

但是提升幅度很小，说明 GPA Cache 初始化仍然存在改进空间。

因此，E3-V3 首先尝试 Init-C：

    Candidate Pool Initialization

## 2. Init-C 第一版设计

Init-C 第一版设计为：

    parallel GPA
    + candidate pool initialization
    + Entropy Cache and GPA candidate pool union center
    + candidate pool size = 2K
    + final GPA size = K
    + local cache 只写入最终筛出的 K 个样本

当前 K 为：

    K = 3

因此候选池默认大小为：

    2K = 6

## 3. Init-C 第一版筛选规则

每个类别先收集候选样本。

对候选样本 i，计算：

    entropy_i
    distance_i = distance(feature_i, center_c)

其中中心为：

    center_c = mean(Global Entropy Cache[c] ∪ GPA candidate pool[c])

然后计算排名：

    entropy_rank_i：熵越低排名越靠前
    distance_rank_i：距离越近排名越靠前

最终得分为：

    rank_score_i = entropy_rank_i + distance_rank_i

选择 rank_score 最小的 K 个样本进入正式 GPA Cache。

对应 local cache 只写入这 K 个样本的局部特征。

## 4. 实验现象

在 add_global_2 上，预构建阶段输出：

    len(entropy_cache): 31
    len(gpa_cache): 29
    len(gpa_local_cache): 29
    entropy cache total: 89
    gpa cache total: 87
    gpa local cache total: 87
    gpa candidate pool total: 2

这说明候选池初始化机制确实生效：

    GPA Cache 和 GPA-controlled Local Cache 的条目数少于 Global Entropy Cache；
    有部分样本留在 candidate pool 中，没有进入正式 GPA/local cache。

但是正式测试阶段累计准确率异常偏低：

    0.00
    30.14
    36.26

该结果明显低于 E2 baseline 和 E3-V2-C 在 add_global_2 上的水平。

因此实验被手动停止。

## 5. 当前判断

该现象不能直接说明 Init-C 思路无效。

更准确的判断是：

    Init-C 第一版实现或策略过于激进，导致准确率急剧下降；
    需要暂停该方向，先进行诊断。

## 6. 可能原因分析

### 6.1 筛选规则过于中心化

当前筛选依据是：

    低熵 + 距离中心近

这可能选出全局上更接近类别中心的样本，但这些样本不一定适合作为 local cache 的来源。

local cache 可能不仅需要“干净”，还需要覆盖不同局部结构。

如果选出的 K 个样本过于相似，局部结构覆盖不足，local cache logits 可能变差。

### 6.2 local cache 覆盖下降

Init-C 会减少进入 local cache 的早期样本。

虽然这可以避免污染，但也可能导致：

    local cache 条目变少；
    local cache 覆盖不足；
    局部分支贡献变差。

由于当前最终加权公式没有调整，local cache 权重仍沿用原始 Point-Cache 设置，因此 local cache 质量或覆盖下降可能被放大。

### 6.3 候选池状态机复杂

Init-C 引入了三个状态：

    GPA candidate pool
    formal GPA Cache
    GPA-controlled Local Cache

这三者必须严格同步。

当前第一版实现已经出现过参数错位和空 formal cache 的问题，虽然已修复，但仍不能排除状态逻辑存在潜在问题。

需要进一步检查：

1. gpa_cache 与 gpa_local_cache 每类长度是否始终一致；
2. candidate selected 后是否按正确类别写入；
3. rejected candidate 是否错误影响后续；
4. online update 时是否正确处理尚未正式初始化的类别；
5. local cache logits 是否因为条目不足而异常。

### 6.4 熵和距离冲突未被合理处理

当前 rank_score 是：

    entropy_rank + distance_rank

该方法不需要设置权重，但它不是严格的“低熵且近中心”准则。

如果出现低熵但距离远、距离近但熵高的样本，rank_score 可能选出折中样本，但这些折中样本未必最适合作为 local cache 来源。

## 7. 结论

当前 Init-C 第一版暂时不继续推进。

记录为：

    候选池初始化第一版出现异常下降，需要暂停并诊断。

后续如果重新启用 Init-C，应考虑：

1. 更保守的候选筛选规则；
2. local cache 多样性约束；
3. local cache 权重调整；
4. 分离 GPA Cache 选择与 local cache 选择；
5. 更详细的 candidate selected / rejected 日志分析。

## 8. 下一步

当前先转向更保守的 Init-A：

    Entropy-bootstrap initialization with Entropy+GPA union center

Init-A 的思想是：

    初始化阶段用 Global Entropy Cache 启动 GPA Cache；
    初始化完成后继续使用 Entropy+GPA union center。
