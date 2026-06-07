# E3 实验阶段总结：从单中心原型改进到 E4 类别概率分布引导 Cache

## 1. E3 实验目标

E3 阶段的目标是在 Point-Cache 的 hierarchical cache 框架中引入 GPA-Cache，即 Global Prototype-Alignment Cache，用原型对齐机制改善进入 local cache 的样本质量。

原始 Point-Cache 的缓存结构包括：

Global Entropy Cache：
    每类缓存低熵样本的全局特征，用于 global cache logits。

Local Cache：
    存储对应样本的局部 patch features，用于 local cache logits。

Negative Cache：
    用于负向修正。

E3 的基本思想是：

    不仅依赖低熵筛选，
    还引入类别原型中心，
    让更靠近类别中心、更可靠的样本进入 GPA-Cache，
    并让这些样本对应的局部 patch features 进入 GPA-controlled local cache。

E3 的核心问题是：

    如何让进入 GPA-Cache 和 GPA-local-cache 的样本更干净？

---

## 2. E3 的主要实验路线

E3 实验主要经历了三个阶段。

E3-V1：
    顺序式 GPA-Cache。
    即样本先通过 Entropy Cache，再进入 GPA-Cache。

E3-V2：
    并列式 GPA-Cache。
    即 Entropy Cache 和 GPA-Cache 并列维护。

E3-V3：
    围绕 GPA-Cache 初始化问题进行改进。
    包括 Entropy-bootstrap initialization、candidate pool initialization、candidate-only center、candidate+entropy center 等。

其中，E3-V2-C 是 E3 阶段目前效果最好的版本：

    E3-V2-C：
        并列式 GPA-Cache
        + Entropy Cache 与 GPA-Cache 联合中心
        + 低熵门控
        + 替换 GPA-Cache 中最高熵样本

E3-V3 进一步尝试解决 E3-V2-C 中的初始化问题：

    E3-V2-C 在 GPA-Cache 未满时，前 K 个样本会直接进入。
    如果前 K 个样本不干净，GPA-Cache 和 GPA-local-cache 会被早期样本污染。

因此，E3-V3 尝试引入 candidate pool：

    每类先收集 2K 个候选样本；
    再从候选池中选择 K 个样本进入 GPA-Cache；
    从而避免前 K 个样本直接进入。

---

## 3. E3 关键实验结果汇总

当前主要结果如下：

| 方法 | 完整含义 | 平均准确率 |
|---|---|---:|
| E2 baseline | manual_full + 原始 full Point-Cache | 54.00 |
| E3-V2-C | 并列式 GPA-Cache + Entropy/GPA 联合中心 | 54.04 |
| E3-V3-B | Entropy-bootstrap 初始化 GPA-Cache | 53.25 |
| E3-V3-C1-Ub | Candidate-only center + 无熵距离更新 | 53.39 |
| E3-V3-C1-Ua2 | Candidate-only center + 低熵门控 + 替换最远样本 | 53.68 |
| E3-V3-C1-Ua1 | Candidate-only center + 低熵门控 + 替换最高熵样本 | 54.02 |
| E3-V3-C2-Ua1 | Candidate+Entropy center + 低熵门控 + 替换最高熵样本 | 54.00 |

从整体结果看：

    E3-V2-C 仍然是 E3 阶段整体最好的版本，平均准确率 54.04。

    E3-V3-C1-Ua1 基本追平 E2 baseline，平均准确率 54.02，
    但仍略低于 E3-V2-C。

    E3-V3-C2-Ua1 平均准确率为 54.00，
    基本等于 E2 baseline，
    但没有超过 E3-V2-C 和 C1-Ua1。

---

## 4. E3-V2-C 与 E3-V3-C2-Ua1 对比

E3-V2-C 和 E3-V3-C2-Ua1 的平均准确率非常接近：

    E3-V2-C：       54.04
    E3-V3-C2-Ua1： 54.00

但分 corruption 看，它们的差异很有意义。

| Corruption | E3-V2-C | E3-V3-C2-Ua1 | C2-Ua1 - V2-C |
|---|---:|---:|---:|
| add_global | 46.84 | 48.99 | +2.15 |
| add_local | 50.49 | 50.57 | +0.08 |
| dropout_global | 58.31 | 57.54 | -0.77 |
| dropout_local | 56.04 | 55.55 | -0.49 |
| rotate | 61.67 | 61.06 | -0.61 |
| scale | 55.06 | 54.54 | -0.52 |
| jitter | 49.88 | 49.72 | -0.16 |
| Average | 54.04 | 54.00 | -0.04 |

这个结果说明：

    E3-V3-C2-Ua1 仍然主要提升 add_global 和 add_local；
    但在 dropout_global、dropout_local、rotate、scale、jitter 上仍然低于 E3-V2-C。

因此，candidate pool 初始化与 Candidate+Entropy 联合中心并没有从根本上解决几何结构变化问题。

---

## 5. E3-V2-C 的规则回顾

E3-V2-C 的完整含义是：

    并列式 GPA-Cache
    + Entropy Cache 与 GPA-Cache 联合中心
    + 低熵门控
    + 替换最高熵样本

### 5.1 初始化规则

对于某个类别 c：

    如果 GPA-Cache[c] 未满：
        新样本直接进入 GPA-Cache[c]
        其 local patch features 同步进入 GPA-local-cache[c]

也就是说，E3-V2-C 没有 candidate pool。

它的初始化方式是：

    前 K 个被 zero-shot 预测为类别 c 的样本直接进入 GPA-Cache[c]。

### 5.2 中心构造规则

E3-V2-C 使用联合中心：

    center_c = mean(EntropyCache[c] ∪ GPACache[c])

也就是说，类别中心由 Entropy Cache 和 GPA-Cache 共同构造。

### 5.3 GPA-Cache 满后的替换规则

当 GPA-Cache[c] 已经满时：

    1. 找到 GPA-Cache[c] 中最高熵样本 x_high；
    2. 计算新样本 x_new 到 center_c 的距离 d_new；
    3. 计算 x_high 到 center_c 的距离 d_high；
    4. 如果：

           entropy(x_new) < entropy(x_high)
           and
           d_new < d_high

       则用 x_new 替换 x_high；
    5. 同步替换 local cache；
    6. 重新计算中心。

总结：

    E3-V2-C 的替换对象是最高熵样本；
    替换条件是低熵 + 距离联合中心更近。

---

## 6. E3-V3-C2-Ua1 的规则回顾

E3-V3-C2-Ua1 的完整含义是：

    候选池初始化 GPA-Cache
    + Candidate Pool 与 Entropy Cache 联合中心
    + 低熵门控
    + 替换最高熵样本

### 6.1 初始化规则

对于类别 c：

    1. 如果 GPA-Cache[c] 尚未初始化，新样本不会直接进入 GPA-Cache；
    2. 新样本先进入 candidate_pool[c]；
    3. 当 candidate_pool[c] 达到 2K 个样本后，触发初始化；
    4. 使用 candidate_pool[c] + EntropyCache[c] 构造临时中心；
    5. 只从 candidate_pool[c] 中选择距离临时中心最近的 K 个样本进入 GPA-Cache；
    6. 这 K 个样本对应的 local patch features 同步进入 GPA-local-cache。

注意：

    Entropy Cache 只参与初始化中心构造；
    Entropy Cache 中的样本不会直接进入 GPA-Cache。

### 6.2 初始化中心构造

    temporary_center_c = mean(candidate_pool[c] ∪ EntropyCache[c])

然后从 candidate_pool[c] 中选：

    距离 temporary_center_c 最近的 K 个候选样本

进入 GPA-Cache。

### 6.3 GPA-Cache 满后的替换规则

E3-V3-C2-Ua1 的满后替换规则与 E3-V2-C 很接近：

    1. 找到 GPA-Cache[c] 中最高熵样本 x_high；
    2. 计算新样本 x_new 到当前 GPA-Center 的距离 d_new；
    3. 计算 x_high 到当前 GPA-Center 的距离 d_high；
    4. 如果：

           entropy(x_new) < entropy(x_high)
           and
           d_new < d_high

       则用 x_new 替换 x_high；
    5. 同步替换 local cache；
    6. 立即重新计算 GPA-Center。

总结：

    C2-Ua1 的替换对象仍然是最高熵样本；
    替换条件仍然是低熵 + 距离更近。

---

## 7. E3-V2-C 和 E3-V3-C2-Ua1 的关键差异

| 对比项 | E3-V2-C | E3-V3-C2-Ua1 |
|---|---|---|
| 是否使用 candidate pool | 否 | 是 |
| GPA-Cache 未满时 | 前 K 个样本直接进入 | 先收集 2K 个候选 |
| 初始化中心 | Entropy Cache + GPA-Cache | Candidate Pool + Entropy Cache |
| 初始化时谁能进入 GPA-Cache | 前 K 个样本 | Candidate Pool 中距离临时中心最近的 K 个 |
| Entropy Cache 是否参与中心 | 是 | 是 |
| Entropy Cache 是否直接进入 GPA-Cache | 否，不是直接规则 | 否，只参与中心构造 |
| 满后替换对象 | 最高熵样本 | 最高熵样本 |
| 满后替换条件 | 低熵 + 距离更近 | 低熵 + 距离更近 |
| 是否同步替换 local cache | 是 | 是 |
| 是否更新中心 | 是 | 是 |

可以看出：

    E3-V3-C2-Ua1 主要改的是初始化；
    满后的替换规则基本沿用 E3-V2-C 的思想。

---

## 8. E3 实验得到的关键发现

### 8.1 候选池初始化没有带来整体提升

E3-V3-C 系列尝试通过 candidate pool 改善初始化问题，但整体结果没有超过 E3-V2-C：

    E3-V2-C：54.04
    E3-V3-C1-Ua1：54.02
    E3-V3-C2-Ua1：54.00

说明：

    前 K 个样本直接进入确实有风险；
    但 candidate pool 初始化并没有成为当前主要瓶颈的解决方案。

### 8.2 恢复低熵门控是必要的

C1 系列结果：

    C1-Ub：  53.39
    C1-Ua2：53.68
    C1-Ua1：54.02

说明：

    纯距离更新不稳定；
    低熵门控仍然必要；
    替换最高熵样本比替换最远样本更稳。

### 8.3 E3 的几种改进本质上仍属于“改进原型中心”的方法

E3 阶段虽然做了多种改进，包括：

    1. GPA-only center；
    2. Entropy-only center；
    3. Entropy/GPA union center；
    4. Entropy-bootstrap initialization；
    5. Candidate-only center；
    6. Candidate+Entropy center；
    7. 替换最高熵样本；
    8. 替换最远样本。

但从更高层看，这些方法的本质大多仍然属于：

    围绕单一类别原型中心进行改进。

具体来说，它们主要在改变：

    1. 类别中心由谁来构造；
    2. 哪些样本可以参与中心构造；
    3. 哪些样本更靠近这个中心；
    4. 是否用低熵门控辅助中心距离判断；
    5. 替换最高熵样本还是替换离中心最远样本。

也就是说，E3 的多数方法仍然建立在一个核心假设上：

    每个类别可以被一个中心表示；
    越靠近这个中心的样本越可靠；
    越靠近这个中心的样本越适合进入 GPA-Cache 和 GPA-local-cache。

这个假设对一部分噪声扰动成立，但对几何结构变化不成立。

### 8.4 单中心原型改进对 add_global / add_local 有效

E3-V3-C1 和 E3-V3-C2 的实验结果都显示：

    add_global 和 add_local 往往有正收益。

原因是：

    add_global / add_local 的本质是向点云中添加全局或局部外点；
    原始物体主体结构仍然存在；
    外点会把受污染严重的样本特征拉离类别主体中心；
    因此，选择更靠近原型中心的样本，实际上起到了过滤外点噪声的作用。

也就是说，E3 这类“改进原型中心”的方法，在 add_global 和 add_local 上有效，是因为它们本质上在做：

    外点噪声过滤 / 去噪样本选择。

这可以作为论文中的一个重要发现：

    单中心原型对齐机制更擅长处理添加型噪声扰动。
    当原始几何主体仍然保留、只是额外加入噪声点时，中心距离可以有效识别受噪声影响较小的样本，从而提升缓存质量。

### 8.5 单中心原型改进面对几何变化会失效

与 add_global / add_local 不同，以下 corruption 的本质不是简单添加外点：

    dropout_global：
        删除全局结构。

    dropout_local：
        删除局部部件。

    rotate：
        改变整体姿态。

    scale：
        改变整体尺度。

    jitter：
        扰动局部点坐标，破坏局部几何细节。

这些损坏类型改变的是：

    1. 结构完整性；
    2. 整体几何分布；
    3. 局部 patch 稳定性；
    4. 类内特征模式；
    5. 查询样本与 local cache 的匹配关系。

因此，一个样本离中心远，不一定说明它是坏样本。它可能只是：

    1. 该类别在某种旋转下的正常样本；
    2. 该类别在某种尺度下的正常样本；
    3. 该类别在某种结构缺失下的正常样本；
    4. 该类别中另一个有效结构模式；
    5. 一个局部形态不同但类别仍然正确的样本。

所以，单中心原型改进在这些几何变化下容易失效。其底层原因是：

    单中心方法只知道“离中心近不近”，
    但不知道“这个偏移是否属于该类别允许的几何变化范围”。

这也可以作为论文中的一个重要分析点：

    单中心原型对齐方法虽然能提升加噪声场景下的缓存洁净度，
    但它会把类内几何变化误判为远离中心的异常样本，
    从而牺牲 cache 的多样性和几何覆盖能力。

### 8.6 中心来源不是主瓶颈

C2-Ua1 将初始化中心从 Candidate-only 改成 Candidate+Entropy：

    C1-Ua1：54.02
    C2-Ua1：54.00

结果没有提升。

这说明：

    问题不主要是 candidate-only center 不稳定；
    也不主要是 Entropy Cache 是否作为中心锚点；
    真正的问题是单中心原型本身表达能力不足。

---

## 9. 为什么进入 E4：类别概率分布引导 Cache

E3 的核心局限是：

    每个类别只维护一个中心；
    用“样本是否更靠近中心”来判断是否进入 GPA-Cache。

但点云在几何变化下，一个类别不一定是一个紧凑点簇。

一个类别可能存在：

    不同姿态；
    不同尺度；
    不同局部缺失模式；
    不同局部扰动模式；
    多个结构子模式。

因此，样本离中心远，不一定说明它是坏样本。它可能只是该类别在某种几何变化下的正常样本。

这就是 E3 单中心 GPA 对 add_global / add_local 有用，但对 dropout / rotate / scale / jitter 不稳定的根本原因。

因此，E4 准备正式引入 BayesMM 的启发：

    为每个类别维护一个概率分布。

但 E4 不是复现 BayesMM，也不是替代 Point-Cache。

E4 的目标是：

    继续保留 Cache；
    继续保留 Point-Cache 的最终推理结构；
    但用“样本是否更符合类别分布”来帮助维护更干净的 GPA-Cache 和 GPA-local-cache。

换句话说，E4 要把 E3 的判断标准从：

    样本是否更靠近类别中心

升级为：

    样本是否更符合该类别的概率分布

---

## 10. E4 初步设想

E4 可以先沿用 E3-V2-C 的整体框架，因为 E3-V2-C 是当前 E3 最优版本。

也就是说：

    沿用 E3-V2-C 的初始化方式：
        GPA-Cache 未满时，样本直接进入 GPA-Cache。

    沿用 E3-V2-C 的替换方式：
        替换 GPA-Cache 中最高熵样本。

    但把“距离中心更近”改为“更符合类别概率分布”。

E4 的初步规则可以写成：

    对于类别 c，维护一个类别分布 Dist[c]。

    当 GPA-Cache[c] 未满：
        按 E3-V2-C 的方式直接初始化；
        同时更新 Dist[c]。

    当 GPA-Cache[c] 已满：
        找到 GPA-Cache[c] 中最高熵样本 x_high。

        如果：
            entropy(x_new) < entropy(x_high)
            and
            prob_score(x_new, Dist[c]) > prob_score(x_high, Dist[c])

        则：
            用 x_new 替换 x_high；
            同步替换 local cache；
            更新 Dist[c]。

其中：

    prob_score(x, Dist[c])

表示样本 x 是否符合类别 c 的概率分布。

这就是 E4 的核心：

    Cache 仍然存在；
    替换最高熵样本的逻辑仍然存在；
    低熵门控仍然存在；
    只是把“距离单中心”换成“符合类别分布”。

---

## 11. E3 阶段最终结论

E3 阶段得到的主要结论如下：

    1. E3-V2-C 是当前 E3 最优版本，平均准确率 54.04。
    2. 候选池初始化可以缓解部分初始化问题，但没有超过 E3-V2-C。
    3. 纯距离更新不稳定，低熵门控仍然必要。
    4. 替换最高熵样本比替换最远样本更稳。
    5. E3 的多种改进本质上仍然属于单中心原型改进。
    6. 单中心原型改进对 add_global / add_local 这类添加型外点噪声有效。
    7. 单中心原型改进对 dropout、rotate、scale、jitter 等几何结构变化不稳定。
    8. Candidate+Entropy 中心没有显著超过 Candidate-only 中心。
    9. 当前瓶颈不是中心来源，而是单中心原型无法表达类别内部几何变化范围。
    10. E4 应正式引入类别概率分布，用分布符合度帮助维护更干净的 Cache。

最终判断：

    E3 证明了原型对齐和低熵门控的必要性，
    也发现了单中心原型改进的适用边界：

        它对添加型噪声有用；
        但对几何结构变化不足。

    因此，E4 应从“单中心 GPA-Cache”转向“类别概率分布引导的 GPA-Cache 净化”。
