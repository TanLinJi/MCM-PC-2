这个创新点可以一句话理解：

> **普通 Negative Cache 只知道“这个样本不可靠”；Confusion-Aware Negative Cache 进一步知道“这个样本主要是在 A 类和 B 类之间混淆”。**

也就是说，它不是简单存高熵样本，而是存**类别之间的混淆关系**。

------

# 1. 先理解普通 cache 的问题

以 Point-Cache 为例，它的核心做法是：

1. 来一个测试点云；
2. 模型预测一个伪标签；
3. 如果这个样本熵低，就认为它可靠；
4. 把它的特征和伪标签存进 cache；
5. 之后遇到新样本，就用 cache 里的相似样本辅助预测。

Point-Cache 的 global cache 存的是：

[
(e_p^g, \hat{L}, h)
]

其中 (e_p^g) 是点云全局特征，(\hat{L}) 是伪标签，(h) 是预测熵。它根据熵判断样本质量，低熵样本更容易进入 cache 。

问题是：**低熵不一定正确。**

比如一个真实类别是 table 的点云，因为局部缺失或者角度问题，模型非常自信地预测成 chair：

[
p(\text{chair}) = 0.86,\quad p(\text{table}) = 0.08
]

这个样本熵很低，但伪标签是错的。如果它被放进 chair 的 positive cache，后面很多 table 样本可能都会被错误地拉向 chair。

这就是 cache pollution，缓存污染。

------

# 2. MCP 的 Negative Cache 解决了什么？

MCP 发现，只用低熵样本构建 cache 不够。它提出了 negative cache，用高熵样本作为负参考，帮助校准预测。MCP 里明确说，高熵样本本身有不确定性，不能直接使用伪标签，所以要经过 recalibration / reflecting mechanism，再决定是否放进 negative cache 。

普通 negative cache 的思想是：

> 高熵样本虽然不能当正样本，但它们包含“模型容易犯错的边界信息”。

例如：

[
p(\text{chair}) = 0.42,\quad p(\text{stool}) = 0.39,\quad p(\text{table}) = 0.10
]

这个样本不能安全地放进 chair cache，因为 chair 和 stool 很接近。但它很有价值，因为它告诉我们：

> chair 和 stool 在这个特征区域很容易混淆。

普通 negative cache 会说：

> 这个样本是不确定样本，用来抑制错误预测。

但它还不够细。

------

# 3. Confusion-Aware Negative Cache 的核心区别

普通 Negative Cache 存的是：

[
\text{uncertain sample}
]

Confusion-Aware Negative Cache 存的是：

[
\text{uncertain sample} + \text{confusing class pair}
]

也就是：

[
x \rightarrow (\text{top-1 class},\ \text{top-2 class})
]

例如：

[
x \rightarrow (\text{chair},\ \text{stool})
]

或者：

[
x \rightarrow (\text{table},\ \text{desk})
]

或者：

[
x \rightarrow (\text{cup},\ \text{vase})
]

所以它不是简单地说：

> 这个样本不可靠。

而是说：

> 这个样本位于 chair 和 stool 的决策边界附近。以后如果遇到相似样本，不要轻易让 chair 或 stool 中某一类过度占优。

------

# 4. 为什么点云中特别需要这个？

因为点云分类里有很多类别本来就局部结构相似。

例如：

[
\text{chair} \leftrightarrow \text{stool}
]

[
\text{table} \leftrightarrow \text{desk}
]

[
\text{cup} \leftrightarrow \text{vase}
]

[
\text{dresser} \leftrightarrow \text{cabinet}
]

而点云还会受到 corruption 影响，比如：

- local dropout：局部结构缺失；
- jitter：点坐标扰动；
- rotate：旋转；
- add local / global：加入噪声点；
- scale：尺度变化。

Point-Cache 论文也强调，点云全局结构和局部 part 信息都很重要，所以它设计了 global cache 和 local cache；local cache 用来处理 global feature 难以区分的细粒度差异 。

但是 Point-Cache 的 local cache 主要还是正向增强：

# [ \hat{y}

\hat{y}_{zs}
+
\alpha_g \hat{y}_g
+
\alpha_l \hat{y}_l
]

也就是 zero-shot、global cache、local cache 三部分相加 。

它没有显式建模：

> 哪两个类别正在互相混淆。

Confusion-Aware Negative Cache 就是补这个缺口。

------

# 5. 一个具体例子

假设来了一个点云，真实类别是 stool，但是模型预测：

[
p(\text{chair}) = 0.46
]

[
p(\text{stool}) = 0.43
]

[
p(\text{table}) = 0.04
]

这个样本的 top-1 是 chair，top-2 是 stool，而且两者差距很小：

# [ m(x)

# p(\text{chair}) - p(\text{stool})

0.03
]

这个 (m(x)) 就是 margin。margin 很小说明模型在 chair 和 stool 之间非常犹豫。

如果我们把它当作 chair 的正样本存入 cache，会有风险：

[
x \in \mathcal{M}^{+}_{\text{chair}}
]

这样之后很多 stool 样本可能会被拉向 chair。

更合理的做法是：不要把它放进 chair positive cache，而是放进 chair-stool 的 confusion cache：

[
x \in \mathcal{B}_{\text{chair},\text{stool}}
]

其中 (\mathcal{B}_{\text{chair},\text{stool}}) 表示 chair 和 stool 之间的边界样本缓存。

它的含义是：

> 这个样本不能证明它是 chair，也不能证明它是 stool；但它证明 chair 和 stool 在这个区域容易混淆。

------

# 6. 它到底怎么参与预测？

假设之后又来了一个新点云 (q)。

模型初始预测：

[
p(\text{chair}) = 0.51
]

[
p(\text{stool}) = 0.45
]

它也很像之前那个 chair-stool 边界样本。此时如果只用 positive cache，模型可能会直接偏向 chair。

但 Confusion-Aware Negative Cache 会提醒模型：

> 这个区域以前出现过 chair-stool 混淆，不要过度相信 top-1。

于是我们可以对相关类别加入一个负向校准项：

# [ \tilde{z}_c(q)

## z_c^{zs}(q) + \alpha z_c^{pos}(q)

\eta z_c^{conf}(q)
]

其中：

- (z_c^{zs}(q))：zero-shot 文本预测；
- (z_c^{pos}(q))：positive cache 给类别 (c) 的支持；
- (z_c^{conf}(q))：confusion-aware negative cache 给类别 (c) 的惩罚；
- (\eta)：负向校准强度。

更具体地，confusion penalty 可以写成：

# [ z_c^{conf}(q)

\sum_{j \ne c}
\sum_{u \in \mathcal{B}_{c,j}}
A(f_q, f_u) \cdot w(u)
]

这里：

- (\mathcal{B}_{c,j})：类别 (c) 和类别 (j) 的混淆缓存；
- (u)：混淆缓存里的样本；
- (A(f_q, f_u))：当前样本 (q) 和混淆样本 (u) 的相似度；
- (w(u))：这个混淆样本的权重；
- (z_c^{conf}(q))：如果当前样本很像类别 (c) 的混淆边界样本，就降低对类别 (c) 的盲目信任。

直观理解：

> 如果一个新样本落在历史上经常混淆的区域，就不要让某个类别的 cache 证据过度支配最终预测。

------

# 7. 它和普通 Negative Cache 的区别

普通 Negative Cache：

# [ \mathcal{N}

{x_1, x_2, x_3, \ldots}
]

只知道这些是“不确定样本”。

Confusion-Aware Negative Cache：

# [ \mathcal{B}_{\text{chair},\text{stool}}

{x_1, x_2, \ldots}
]

# [ \mathcal{B}_{\text{table},\text{desk}}

{x_3, x_4, \ldots}
]

# [ \mathcal{B}_{\text{cup},\text{vase}}

{x_5, x_6, \ldots}
]

它知道每个不确定样本具体发生在哪两个类别之间。

所以它的信息量更大。

------

# 8. 为什么这可能是创新点？

因为现有方法通常有三类：

### Point-Cache

Point-Cache 有 global cache 和 local cache，但主要是 positive retrieval。它根据低熵样本存储高质量特征，然后用 global cache 和 local cache 修正 zero-shot logits 。它没有显式建模类别对之间的混淆关系。

### MCP

MCP 有 negative cache，但它是 2D VLM TTA 场景。它关注高熵样本如何作为负参考，帮助校准预测 。但它没有针对 3D 点云的 global-local 结构，也没有显式构建 pairwise class confusion。

### BayesMM

BayesMM 批评 cache-based 方法容量有限、logit 融合经验化，于是用文本和几何 Gaussian distribution 做 Bayesian fusion 。但它不是 cache 结构，也没有显式记录“chair 和 stool 经常混淆”这种 pairwise boundary evidence。

所以我们的 Confusion-Aware Negative Cache 可以写成：

> We extend negative caching from sample-level uncertainty modeling to class-pair-level confusion modeling, explicitly preserving boundary evidence between confusing 3D categories.

中文就是：

> 我们把 negative cache 从“样本级不确定性”扩展到“类别对级混淆关系”，显式保存点云类别之间的边界证据。

------

# 9. 这个模块可以怎么设计？

我建议第一版设计得简单一点。

对于每个测试样本 (x)，先计算 top-1 和 top-2 类别：

[
a = \arg\max_c p_c(x)
]

[
b = \arg\max_{c \ne a} p_c(x)
]

然后计算 margin：

[
m(x) = p_a(x) - p_b(x)
]

如果：

[
m(x) < \tau_m
]

说明 top-1 和 top-2 很接近，这个样本是边界样本。

再结合熵：

[
H_{\text{low}} < H(x) < H_{\text{high}}
]

说明它不是完全可靠，也不是完全混乱，而是有一定可用信息的“中等不确定样本”。

那么就把它存入：

[
\mathcal{B}_{a,b}
]

也就是类别 (a) 和类别 (b) 的 confusion cache。

------

# 10. 为什么不存极高熵样本？

因为极高熵样本通常是完全混乱的：

[
p_1 \approx p_2 \approx \cdots \approx p_C
]

这种样本没有明确告诉我们“哪两个类别混淆”，它只是说明模型完全不知道。

所以我们只存中等不确定样本：

[
H_{\text{low}} < H(x) < H_{\text{high}}
]

这和 MCP 的思想一致：MCP 也不是无脑保存所有高熵样本，而是通过 recalibration 后，只有仍处于合适不确定区间的样本才进入 negative cache 。

------

# 11. 它和 Global-Local Consistency 可以结合

这是它在 3D 点云里最有价值的地方。

对于一个样本 (x)，我们可以分别得到：

[
p^{global}(x)
]

[
p^{local}(x)
]

如果 global 预测 chair，local 预测 stool：

[
\hat{y}^{global} = \text{chair}
]

[
\hat{y}^{local} = \text{stool}
]

那么这个样本也应该进入：

[
\mathcal{B}_{\text{chair},\text{stool}}
]

因为它不仅 top-1 和 top-2 接近，而且 global 和 local 结构本身发生了冲突。

这比 2D negative cache 更有 3D 特点：

> 点云的类别混淆不仅来自分类概率，还来自全局形状和局部结构的不一致。

比如 chair 和 stool：

- global shape 可能像 stool；
- local part 发现有 backrest，支持 chair。

这时它就是非常典型的 boundary evidence。

------

# 12. 最终它的作用是什么？

Confusion-Aware Negative Cache 主要有三个作用。

第一，**防止错误样本进入 positive cache**。

边界样本不再被粗暴放进 top-1 类别，而是进入 pairwise confusion cache：

[
x \notin \mathcal{M}^{+}_{a}
]

[
x \in \mathcal{B}_{a,b}
]

第二，**帮助抑制易混类别的过度激活**。

如果一个新样本非常像历史上的 chair-stool 边界样本，那么模型会降低对单一类别的盲目信任。

第三，**提供可解释性**。

我们可以统计：

[
|\mathcal{B}_{\text{chair},\text{stool}}|
]

如果这个值很大，就说明 chair 和 stool 是模型最容易混淆的一对类别。

这可以写进实验分析：

> Our confusion-aware negative cache reveals that most erroneous predictions concentrate on semantically and geometrically similar class pairs, such as chair-stool and table-desk.

------

# 13. 论文里可以怎么写？

可以放在 **Method** 的一个小节：

## Confusion-Aware Negative Cache

英文草稿：

> Instead of treating all uncertain samples uniformly, we explicitly preserve their class-pair confusion structure. For each incoming test sample, we identify the top-1 and top-2 predicted classes and measure their prediction margin. Samples with moderate entropy and small margins are not inserted into the positive cache. Instead, they are stored in a pairwise boundary cache associated with the corresponding class pair. During inference, this cache provides negative calibration signals for samples located near historically ambiguous decision regions, reducing the risk of over-amplifying confusing classes through cache retrieval.

这段可以翻译成中文理解为：

> 我们不把所有不确定样本一视同仁，而是保存它们具体在哪两个类别之间混淆。对于每个测试样本，我们找到 top-1 和 top-2 类别，并计算它们的 margin。中等熵、小 margin 的样本不进入正 cache，而是进入对应类别对的边界 cache。在推理时，这个 cache 为处于历史混淆区域的新样本提供负向校准信号，减少 cache retrieval 过度放大错误类别的风险。

------

# 14. 最简单可实现版本

第一版不用做复杂。

只需要三个判断：

[
a = \mathrm{Top1}(p(x))
]

[
b = \mathrm{Top2}(p(x))
]

[
m(x) = p_a(x) - p_b(x)
]

如果：

[
H_{\text{low}} < H(x) < H_{\text{high}}
\quad \text{and} \quad
m(x) < \tau_m
]

那么：

[
x \rightarrow \mathcal{B}_{a,b}
]

推理时：

# [ \tilde{z}(q)

## z^{zs}(q) + \alpha z^{pos}(q) + \beta z^{local}(q)

\eta z^{conf}(q)
]

这个就是第一版 Confusion-Aware Negative Cache。

------

# 15. 一句话总结

**Confusion-Aware Negative Cache 不是为了记住“错误样本”，而是为了记住“模型容易在哪些类别对之间犯错”。**

它把普通 negative cache 从：

[
\text{sample uncertainty}
]

升级为：

[
\text{class-pair boundary uncertainty}
]

这在 3D 点云里很有意义，因为很多错误都发生在全局形状相似、局部结构差异细微的类别之间，比如 chair / stool、table / desk、cup / vase。