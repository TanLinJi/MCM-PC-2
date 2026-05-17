## 核心创新点

### 1 Reliability-Aware Multi-Cache Matrix

Point-Cache 已经有 global cache 和 local cache，用在线测试样本构建动态层次化缓存，并且只依赖测试数据，不访问训练集 。MCP 已经有 entropy / align / negative 三缓存，并指出低熵样本不一定可靠，类内紧凑性对 cache 效果很重要。我们的创新不是简单叠加，而是把它做成一个**矩阵结构**：

|                       | Entropy Cache | Compact Cache      | Boundary / Negative Cache   |
| --------------------- | ------------- | ------------------ | --------------------------- |
| Global level          | 全局低熵样本  | 全局靠近类中心样本 | 全局易混 / 高不确定样本     |
| Local level           | 局部低熵 part | 局部结构一致 part  | 局部冲突 / 缺失 / 噪声 part |
| Text / Semantic level | 文本先验      | 文本-几何一致性    | 文本-几何冲突类别           |



### 2 Compactness-Margin Dual Reliability Criterion

MCP 只强调类内紧凑性，MF-CLIP 强调类间 margin / discriminability。我们可以把这两个思想结合起来，提出一个更强的 cache admission 标准。

具体而言，原来的 `PointCache` 思路是： 低熵样本 = 高质量样本。但这个判断不够。MCP 已经指出低熵样本在分布偏移下可能不可靠，并且类内不紧凑会影响 cache 原型质量。MF-CLIP 也说明，单有表征能力不够，特定任务中类间 margin 被压缩会削弱判别性；它通过 margin-aware 方法增强类间分离 。

所以我们可以提出：

```
可靠样本 ≠ 低熵样本
可靠样本 = 低熵 + 类内紧凑 + 类间有 margin + global/local 一致
```

定义一个 Reliability Score:
$$
R(x) = - H(x) 
       + \lambda_1 \cdot \text{sim}(f_x, \mu_{\hat{y}}) 
       - \lambda_2 \cdot \max_{j \neq \hat{y}} \text{sim}(f_x, \mu_j) 
       + \lambda_3 \cdot C_{\text{global-local}}(x)
$$
其中：

-  `H(x)` 是预测熵； 
-  `sim(fx, μŷ)` 衡量样本是否靠近预测类别中心； 
-  `max sim(fx, μj)` 衡量它是否也靠近其他类别； 
-  两者差值就是 margin； 
-  `Cglobal-local(x)` 衡量全局预测和局部 part 预测是否一致。



### 3 Global-Local Consistency Guided Cache Update

Point-Cache 已经有 local cache，但它的 local cache 主要跟随 global cache 的样本选择；也就是说，一个样本能不能进入 local cache，主要看它的 global fingerprint 是否通过筛选 。这给我们留下一个很好的改进空间：点云的全局形状可能是错的，但局部结构可能提供纠正信号；反过来，局部 corruptions 也可能误导模型。比如：

- 椅子整体像 stool，但局部 backrest patch 能说明它是 chair； 

- airplane 被 drop local 后，全局轮廓仍对，但局部 part 缺失导致不确定； 

- table 和 desk 全局相似，但局部结构可以区分。

所以可以提出：**Global-Local Consistency-Aware Cache Update**，具体做法是：

```
1. 全局特征给出 global prediction；
2. local parts 分别投票得到 local prediction；
3. 如果 global 和 local 高度一致，样本进入 positive / compact cache；
4. 如果 global 正确但 local 冲突，进入 uncertain cache；
5. 如果 local 强烈支持另一个类别，进入 boundary / negative cache；
6. final logits 由 global evidence、local evidence、text evidence 动态融合。
```



### 4 Adaptive Evidence Fusion instead of Fixed Logit Weights

PointCache给出的预测是：
$$
\text{zero-shot logits} 
+ \alpha_g \cdot \text{global cache logits} 
+ \alpha_l \cdot \text{local cache logits}
$$
这里的 αg、αl 是固定超参数。BayesMM 批评过 cache-based 方法的 logit fusion 比较 heuristic，而且依赖手工超参数；它提出用 Bayesian model averaging 自动平衡文本和几何模态。我们不一定要完整复现 BayesMM，但可以提出一个轻量版：`Reliability-Gated Evidence Fusion`，也就是每个测试样本动态决定：

```
该相信 zero-shot text 多少；
该相信 global cache 多少；
该相信 local cache 多少；
该相信 negative cache suppression 多少。
```

最终的得分是：
$$
\begin{align*}
\mathrm{logits} &= w_\text{text} \cdot \mathrm{logits}_\text{text} \\
&+ w_\text{global} \cdot \mathrm{logits}_\text{global_cache} \\
&+ w_\text{local} \cdot \mathrm{logits}_\text{local_cache} \\
&- w_\text{neg} \cdot \mathrm{logits}_\text{negative_cache}
\end{align*}
$$
这个创新可以直接针对 BayesMM 和 Point-Cache 的痛点：**不用固定 α，改成样本自适应权重**。



## 辅助创新点

### 1 vMF / Spherical Text Prior

BayesMM 用 Gaussian 建模文本分布，并通过 LLM 生成 paraphrases 后计算均值和协方差 。但 CLIP / ULIP / OpenShape 这类模型的特征通常经过归一化，分类时也使用 cosine similarity。因此文本特征更自然地位于单位球面上。于是用 vMF：
$$
\begin{aligned}
z_{c,m} &\in S^{d-1} \\
z_{c,m} &\sim \text{vMF}(\mu_c, \kappa_c)
\end{aligned}
$$
其中：

-  `μ_c` 是类别文本方向； 
-  `κ_c` 是文本 prompt 的集中程度； 
-  `κ_c` 越大，说明这个类别的文本描述越稳定； 
-  `κ_c` 越小，说明文本 prompt 分歧大，需要更多依赖几何 cache。

它可以自然接入 adaptive fusion：

```
文本越稳定，w_text 越大；
文本越分散，w_text 越小。
```

### 2 Class-wise Adaptive Cache Budget

Point-Cache 默认每类 cache size K 固定，比如 K=3。它的消融也说明 K 会影响性能，不同数据集趋势不一样 。这说明固定 K 不是最优。我们可以提出：**Adaptive Cache Budget Allocation**不是每个类别都存 K 个样本，而是：

-  高不确定类别多存； 
-  类内分散类别多存； 
-  易混类别多存； 
-  稳定类别少存。 

例如：
$$

\kappa_c = \kappa_{\text{base}} + \Delta \left( \text{uncertainty}_c + \text{dispersion}_c + \text{confusion}_c \right)
$$
这可以解决长尾类别、难分类类别和大规模类别集的问题。尤其 Objaverse-LVIS 有 1,156 类，固定每类 K 可能浪费显存。这个创新实现难度中等，但很适合做消融。



### 3 Confusion-Aware Negative Cache

MCP 的 negative cache 主要用高熵样本作为负参考。我们可以让它更 3D、更细：不是存“高熵样本”，而是存“类别对之间的混淆证据”。例如当前样本 top-1 是 chair，top-2 是 stool，而且 margin 很小，那么我们记录：

```
chair ↔ stool confusion
```

最终预测 chair 时，如果它也强烈匹配 stool 的 boundary cache，就抑制 chair 或提醒重新加权。

可以 叫**Pairwise Boundary Cache** 或 **Confusion-Aware Negative Cache**，这个点比普通 negative cache 更有创新，因为它建模的是**类别对混淆关系**，而不是单纯高熵。