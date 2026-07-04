# E5-B/C/D 修正方案：从 standalone GDA 转向文本先验锚定的 posterior prototype residual

日期：2026-06-10  
项目根目录：`/root/autodl-tmp/MCM-PC-2`  
实验目录：`docs/experiments/E5_adapt_inspired_gaussian_alignment_cache`  
当前基线：`02_9_2`，即 `E4-C-A0+E1-textdist-only`，`E4_TEXT_SCORE_WEIGHT=0.15`

---

## 1. 写作目的

这份文档记录 E5 的修正方案。它不是 E5-A 的结果报告，而是基于 E5-A 初步负向结果、ADAPT/PGA 论文思想、以及我们已有 E4-C 正向经验后，对下一步 E5-B/C/D 的重新设计。

核心判断是：

```text
ADAPT/PGA 的思想可以迁移到我们任务中，
但不能把 shared-covariance GDA 当成独立分类器或直接最终专家。
更合理的迁移方式是：
文本原型作为 prior，
高置信 test-time 样本 bank 修正 prior，
最终只加入相对于文本 prior 的 posterior residual。
```

换句话说，E5 后续不再把 GDA 作为一个新的强分类器，而是把它拆成三个更保守、可解释、可逐步验证的模块：

| 编号 | 名称 | 作用 |
|---|---|---|
| E5-B | Text-prior posterior prototype residual | 用高置信视觉 bank 修正文本原型，并把修正量作为最终 logits 的残差证据 |
| E5-C | Shared-covariance metric for replacement | shared covariance 只作为 cache replacement 的距离度量，不直接进入最终分类 |
| E5-D | Dynamic text-visual fusion weight | 用有效样本数动态决定文本 prior 和视觉统计的融合权重 |

---

## 2. 当前事实基础

### 2.1 当前最强基线

当前 ModelNet-C severity=2 的最好完整七类 corruption 实验是：

```text
实验编号：02_9_2
方法：E4-C-A0+E1-textdist-only
文本分布权重：E4_TEXT_SCORE_WEIGHT=0.15
数据集：ModelNet-C
severity：2
backbone：ULIP
cache setting：zs_global_local
七类 corruption 平均准确率：约 54.71
```

逐项结果如下：

| corruption | accuracy |
|---|---:|
| add_global_2 | 47.89 |
| add_local_2 | 50.85 |
| dropout_global_2 | 59.12 |
| dropout_local_2 | 57.21 |
| rotate_2 | 61.30 |
| scale_2 | 55.92 |
| jitter_2 | 50.65 |

clean 对比也必须纳入后续判断：

| 方法 | clean accuracy |
|---|---:|
| 原始 Point-Cache `ZS + Global + Local Cache` | 64.18 |
| `02_9_2` clean | 63.86 |

这说明 `02_9_2` 是 corruption severity=2 上的当前最好方法，但在 clean 上有轻微回退。后续 E5 不能只看 corrupted 平均，还必须同时检查 clean。

### 2.2 E5-A 的结果与结论

E5-A 已经完成的两个 corruption：

结果目录：

```text
Point-Cache/results/E5_adapt_inspired_gaussian_alignment_cache/00_1_ulip_modelnetc_s2_zs_global_local_e5_a0_a1_adapt_gda_diagnostics_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

| corruption | 原始 Point-Cache/E4 分支 | standalone GDA | 结论 |
|---|---:|---:|---|
| add_global_2 | 47.89 | 47.40 | 负向 |
| add_local_2 | 50.85 | 49.39 | 负向 |

E5-A 的 standalone GDA 和原始 Point-Cache/E4 分支的一致率只有约 68.90% 到 71.46%，说明 GDA 不是一个稳定替代分类器。如果把它作为最终 logits 的强专家，风险很高。

因此当前结论是：

```text
E5-A 作为 standalone GDA 方向失败。
但 E5-A 没有否定 ADAPT/PGA 的整体启发，
它只说明“直接把 shared-covariance GDA 当分类器或最终证据”不适合当前 Point-Cache 框架。
```

---

## 3. 为什么 E5-A 的迁移方式不合理

### 3.1 ADAPT/PGA 的前提和我们的前提不同

ADAPT/PGA 的核心是一个完整的概率对齐框架。它的 Gaussian 模型、文本 prior、test-time bank 和闭式更新在同一个推理目标中共同工作。

我们的基础框架是 Point-Cache。Point-Cache 已经有自己的最终预测公式：

```text
zero-shot logits
+ global cache logits
+ local cache logits
- negative cache logits
```

E4-C 又在 cache replacement 阶段加入了文本-视觉分布打分。也就是说，我们不是一个空白的 GDA 分类器，而是一个已经存在多个相关证据项的 cache-based TTA 系统。

如果在这个系统上直接叠加 standalone GDA，就容易出现两个问题：

1. GDA 和 Point-Cache 使用的是同一批在线测试样本，证据高度相关，容易重复计数；
2. GDA 的 pseudo-label bank 由当前模型预测产生，一旦早期伪标签错误，Gaussian 均值和协方差会被污染。

### 3.2 shared covariance 稳定，但不等于最终分类更好

shared covariance 的优势是减少每类协方差估计的自由度。它比每类独立协方差更稳，尤其适合样本少的 online test-time bank。

但这只说明它适合作为度量：

\[
d_c^2(x)
=
(f(x)-\mu_c)^\top
\Sigma^{-1}
(f(x)-\mu_c)
\]

其中：

- \(f(x)\)：当前测试点云的归一化视觉特征；
- \(\mu_c\)：类别 \(c\) 的均值；
- \(\Sigma^{-1}\)：共享逆协方差矩阵；
- \(d_c^2(x)\)：样本到类别 \(c\) 的 Mahalanobis 距离。

它不自动保证下面的 GDA 判别式能直接超过 Point-Cache：

\[
g_c(x)
=
\mu_c^\top \Sigma^{-1} f(x)
-
\frac{1}{2}
\mu_c^\top \Sigma^{-1} \mu_c
\]

其中：

- 第一项 \(\mu_c^\top \Sigma^{-1} f(x)\) 表示样本和类别均值在 shared covariance metric 下的匹配；
- 第二项 \(-\frac{1}{2}\mu_c^\top \Sigma^{-1} \mu_c\) 是类别相关偏置；
- \(g_c(x)\) 越大，GDA 越支持类别 \(c\)。

E5-A 的结果说明：在我们当前的 online pseudo-label bank 条件下，\(g_c(x)\) 本身不足以成为强分类器。

### 3.3 当前样本不能更新自己的分布

E5 后续仍应保留 delayed update 原则：

```text
先用历史 bank 预测当前样本；
当前样本预测结束后，再决定是否写入 bank。
```

如果当前样本先写入 bank 再参与自己的预测，会形成自强化：

```text
当前样本被错分
-> 写入错误类别 bank
-> 分布模型支持错误类别
-> 最终 logits 更自信但更错误
```

这条原则对 E5-B/C/D 都成立。

---

## 4. E5-B：文本先验锚定的 posterior prototype residual

### 4.1 核心思想

E5-B 不再让 Gaussian/GDA 独立分类。它只做一件事：

```text
用高置信视觉样本 bank 修正文本原型，
再把“修正后原型相对于原文本原型的增量”加入最终 logits。
```

这样可以避免一个常见错误：把测试时视觉统计当成新的分类器，压过原本稳定的文本/Point-Cache 证据。

### 4.2 文本 prior prototype

对每个类别 \(c\)，先有一个固定文本先验原型：

\[
t_c \in \mathbb{R}^{d}
\]

其中：

- \(t_c\)：类别 \(c\) 的文本 prior prototype；
- \(d\)：ULIP/CLIP 对齐空间的特征维度；
- \(t_c\) 在一个实验过程中固定，不随测试流更新。

第一版建议使用 `manual_full` 对应的类别文本原型作为 \(t_c\)。原因是：`02_7` 已经表明，把 E1 prompt fusion 直接强行带入最终分类会导致明显下降；而 `02_9_2` 的正收益来自 E1 只进入 text distribution，不是替换最终 classifier。

因此 E5-B 的默认原则是：

```text
最终分类器的文本 prior 不要被 LLM prompt fusion 粗暴替代。
E1 缓存 prompt 可以继续作为 text distribution 的辅助信息，
但 posterior residual 的锚点优先使用稳定 manual_full prior。
```

### 4.3 高置信视觉 bank

为每个类别维护一个独立的视觉统计 bank：

\[
B_c = \{(v_i, q_i)\}_{i=1}^{n_c}
\]

其中：

- \(B_c\)：类别 \(c\) 的视觉统计 bank；
- \(v_i\)：第 \(i\) 个被接受样本的归一化视觉特征；
- \(q_i\)：该样本的可靠性权重；
- \(n_c\)：类别 \(c\) 当前 bank 中的样本数。

第一版可以令 \(q_i\) 来自当前稳定基线的置信度，例如负熵或 softmax top-1 confidence。需要注意：进入 \(B_c\) 的类别标签应来自稳定基线分支，不应来自 E5-B 自己增强后的预测，否则会形成反馈污染。

### 4.4 可靠性加权视觉均值

类别 \(c\) 的视觉均值定义为：

\[
\bar{v}_c
=
\frac{
\sum_{i \in B_c} q_i v_i
}{
\sum_{i \in B_c} q_i + \epsilon
}
\]

其中：

- \(\bar{v}_c\)：类别 \(c\) 的可靠性加权视觉均值；
- \(v_i\)：类别 \(c\) 的第 \(i\) 个视觉样本特征；
- \(q_i\)：样本 \(v_i\) 的可靠性权重；
- \(\epsilon\)：防止分母为 0 的数值稳定项。

这个均值不是简单平均，而是让更可靠的样本贡献更大。

### 4.5 有效样本数

定义类别 \(c\) 的有效视觉样本数：

\[
n_c^{\mathrm{eff}}
=
\sum_{i \in B_c} q_i
\]

其中：

- \(n_c^{\mathrm{eff}}\)：类别 \(c\) 的有效样本量；
- \(q_i\)：第 \(i\) 个样本的可靠性权重。

它比普通计数 \(n_c\) 更合理，因为 16 个低置信样本不应该等价于 16 个高置信样本。

### 4.6 动态视觉权重

用有效样本数决定视觉统计应该在 posterior prototype 中占多大权重：

\[
\lambda_c
=
\frac{
n_c^{\mathrm{eff}}
}{
n_c^{\mathrm{eff}} + \kappa
}
\]

其中：

- \(\lambda_c\)：类别 \(c\) 的视觉统计权重；
- \(n_c^{\mathrm{eff}}\)：类别 \(c\) 的有效视觉样本数；
- \(\kappa\)：文本 prior strength，表示文本先验相当于多少个有效样本；
- \(\lambda_c \in [0,1]\)。

当 \(n_c^{\mathrm{eff}}\) 很小，\(\lambda_c\) 接近 0，posterior prototype 主要依赖文本 prior。  
当 \(n_c^{\mathrm{eff}}\) 很大，\(\lambda_c\) 增大，posterior prototype 更多吸收测试时视觉统计。

第一版建议：

```text
kappa = 8 或 16
```

如果 `StatsBank capacity L=16`，那么 `kappa=8` 表示中等强度文本先验，`kappa=16` 表示更保守的文本先验。

### 4.7 Posterior prototype

类别 \(c\) 的 posterior prototype 定义为：

\[
m_c
=
\operatorname{normalize}
\left(
(1-\lambda_c)t_c
+
\lambda_c \bar{v}_c
\right)
\]

其中：

- \(m_c\)：类别 \(c\) 的 posterior prototype；
- \(t_c\)：类别 \(c\) 的文本 prior prototype；
- \(\bar{v}_c\)：类别 \(c\) 的可靠性加权视觉均值；
- \(\lambda_c\)：类别 \(c\) 的动态视觉权重；
- \(\operatorname{normalize}(\cdot)\)：L2 归一化。

这个式子的意义是：posterior prototype 不是纯文本，也不是纯视觉，而是由文本先验和测试时高置信视觉统计共同决定。

### 4.8 Residual evidence

对当前样本 \(x\)，提取视觉特征 \(f(x)\)。E5-B 不直接使用 \(\cos(f(x),m_c)\) 作为新分类器，而只使用它相对于文本 prior 的增量：

\[
\Delta_{\mathrm{posterior}}(c,x)
=
\operatorname{Norm}
\left(
\cos(f(x),m_c)
-
\cos(f(x),t_c)
\right)
\]

其中：

- \(\Delta_{\mathrm{posterior}}(c,x)\)：posterior prototype 对类别 \(c\) 的残差证据；
- \(\cos(f(x),m_c)\)：样本和 posterior prototype 的余弦相似度；
- \(\cos(f(x),t_c)\)：样本和原始文本 prior prototype 的余弦相似度；
- \(\operatorname{Norm}(\cdot)\)：归一化函数。

这个残差项的解释是：

```text
如果测试时视觉统计把类别原型移动到了更适合当前样本的位置，
则 cos(f(x), m_c) - cos(f(x), t_c) 为正；
如果 posterior prototype 没有比文本 prior 更支持当前样本，
该项接近 0 或为负。
```

这比直接加 \(\cos(f(x),m_c)\) 更稳，因为它不会重复计算文本 prior 已经提供的基础语义证据。

### 4.9 最终 logits

设原始 Point-Cache/E4 分支的最终 logits 为：

\[
z_{\mathrm{pc}}(c)
\]

E5-B 的新 logits 为：

\[
z_{\mathrm{new}}(c)
=
z_{\mathrm{pc}}(c)
+
\gamma
\Delta_{\mathrm{posterior}}(c,x)
\]

其中：

- \(z_{\mathrm{pc}}(c)\)：原始 Point-Cache/E4 对类别 \(c\) 的最终 logits；
- \(\gamma\)：posterior residual 的融合强度；
- \(\Delta_{\mathrm{posterior}}(c,x)\)：posterior residual evidence。

第一版建议在一次实验中同时输出：

```text
original_acc
e5_b_gamma_0.05_acc
e5_b_gamma_0.10_acc
e5_b_gamma_0.20_acc
```

因为 E5-B 只改最终 logits，不改变输入特征和缓存写入，所以多个 \(\gamma\) 可以在同一次 forward 中并行计算，不需要重复跑多个脚本。

### 4.10 推荐归一化

E5-B 的 \(\operatorname{Norm}\) 建议使用 sample-wise class z-score：

\[
\operatorname{Norm}(a_c)
=
\operatorname{clip}
\left(
\frac{
a_c - \mu_a
}{
\sigma_a + \epsilon
},
-\tau,
\tau
\right)
\]

其中：

\[
\mu_a
=
\frac{1}{C}
\sum_{j=1}^{C} a_j
\]

\[
\sigma_a
=
\sqrt{
\frac{1}{C}
\sum_{j=1}^{C}
(a_j-\mu_a)^2
}
\]

并且：

- \(a_c=\cos(f(x),m_c)-\cos(f(x),t_c)\)；
- \(C\)：类别数；
- \(\epsilon\)：数值稳定项，建议 `1e-6`；
- \(\tau\)：clip 阈值，建议 `3.0`。

选择这个归一化的原因：

1. 最终分类只关心同一样本内部的类别竞争；
2. sample-wise 归一化不会被 early pseudo-label running statistics 污染；
3. residual 的尺度通常很小，直接加 logits 可能不起作用；
4. clip 可以防止少数异常 posterior prototype 产生过强扰动。

### 4.11 三类别示例

假设当前只有三个类别：

```text
c1 = airplane
c2 = chair
c3 = table
```

对当前样本 \(x\)，原始文本 prior 相似度为：

| 类别 | \(\cos(f(x),t_c)\) |
|---|---:|
| airplane | 0.28 |
| chair | 0.35 |
| table | 0.33 |

如果只看文本 prior，`chair` 稍高。

测试流中已经为每个类积累了高置信视觉 bank。设 posterior prototype 计算后，相似度变为：

| 类别 | \(\cos(f(x),m_c)\) |
|---|---:|
| airplane | 0.29 |
| chair | 0.36 |
| table | 0.40 |

则 posterior residual 为：

\[
a_c
=
\cos(f(x),m_c)
-
\cos(f(x),t_c)
\]

逐类展开：

\[
a_{\mathrm{airplane}}
=
0.29 - 0.28
=
0.01
\]

\[
a_{\mathrm{chair}}
=
0.36 - 0.35
=
0.01
\]

\[
a_{\mathrm{table}}
=
0.40 - 0.33
=
0.07
\]

这个例子的含义是：

```text
posterior prototype 并不是直接说 table 的绝对相似度最高就完全改判，
而是说：相对于原始文本 prior，高置信视觉 bank 对 table 的支持增加最多。
```

如果原始 Point-Cache logits 为：

| 类别 | \(z_{\mathrm{pc}}(c)\) |
|---|---:|
| airplane | 1.20 |
| chair | 1.55 |
| table | 1.50 |

则 `chair` 原本略高于 `table`。E5-B 会对 \(a_c\) 做 sample-wise class z-score 后得到 \(\Delta_{\mathrm{posterior}}(c,x)\)，再计算：

\[
z_{\mathrm{new}}(c)
=
z_{\mathrm{pc}}(c)
+
\gamma
\Delta_{\mathrm{posterior}}(c,x)
\]

如果 `table` 的 residual 明显高于其他类，并且 \(\gamma\) 不过大，那么 `table` 可能被温和提升；如果 residual 差距很小，最终预测基本仍由原始 Point-Cache 决定。这就是 E5-B 相比 standalone GDA 更保守的地方。

---

## 5. E5-C：shared covariance 只作为 replacement metric

E5-C 保留 ADAPT/PGA 的 shared covariance 思想，但只用于判断 cache replacement，不直接参与最终 logits。

对类别 \(c\)，使用 posterior prototype \(m_c\) 作为类别中心，定义 Mahalanobis 距离：

\[
d_c^2(x)
=
(f(x)-m_c)^\top
\Sigma^{-1}
(f(x)-m_c)
\]

其中：

- \(f(x)\)：当前样本视觉特征；
- \(m_c\)：E5-B 定义的 posterior prototype；
- \(\Sigma^{-1}\)：由独立 StatsBank 估计的 shared inverse covariance；
- \(d_c^2(x)\)：样本到类别 \(c\) 的 shared-covariance 距离。

replacement 规则建议：

```text
只在 predicted class c 内比较。
新样本必须满足：
1. 熵更低，说明分类置信更高；
2. d_c^2(x) 更小，说明更符合 posterior prototype 的 shared-covariance metric；
3. 当前样本不能先进入 bank 再参与自己的 replacement 评分。
```

如果要把两者写成一个综合分数，可以定义：

\[
R(c,x)
=
-H(x)
-
\eta d_c^2(x)
\]

其中：

- \(R(c,x)\)：replacement score；
- \(H(x)\)：当前样本预测熵；
- \(d_c^2(x)\)：Mahalanobis 距离；
- \(\eta\)：距离项权重。

但第一版更建议使用门控规则，而不是立即引入新的 \(\eta\) 超参数：

```text
entropy 更低 AND Mahalanobis 距离更小 -> 才替换
```

---

## 6. E5-D：动态文本-视觉融合权重

E4-C / `02_9_2` 中，文本分布权重 `0.15` 是通过经验消融得到的。它有效，但仍然是固定超参数。

E5-D 的目标是把这个固定权重改成由不确定性决定的动态权重。

定义视觉权重：

\[
w_{\mathrm{visual},c}
=
\frac{
n_c^{\mathrm{eff}}
}{
n_c^{\mathrm{eff}} + \kappa
}
\]

定义文本权重：

\[
w_{\mathrm{text},c}
=
\frac{
\kappa
}{
n_c^{\mathrm{eff}} + \kappa
}
\]

其中：

- \(w_{\mathrm{visual},c}\)：类别 \(c\) 的视觉统计权重；
- \(w_{\mathrm{text},c}\)：类别 \(c\) 的文本先验权重；
- \(n_c^{\mathrm{eff}}\)：类别 \(c\) 的有效视觉样本数；
- \(\kappa\)：文本先验强度。

二者满足：

\[
w_{\mathrm{visual},c}
+
w_{\mathrm{text},c}
=
1
\]

这个设计的含义是：

```text
当某类测试时可靠样本很少，文本 prior 更可信；
当某类已积累足够可靠样本，视觉 posterior 更可信。
```

E5-D 可以用于两个位置：

1. posterior prototype 的文本/视觉融合；
2. E4-C text distribution 与 visual distribution 的联合打分。

第一版建议先用于 posterior prototype，不要同时改太多位置。

---

## 7. 实验顺序

### 7.1 E5-B0：posterior prototype 诊断

目的：

```text
只构建 posterior prototype 和 residual statistics，不改变最终预测。
```

输出：

```text
original_acc
posterior residual mean/std
posterior residual top1-top2 margin
per-class bank coverage
effective sample count distribution
lambda_c distribution
clean/corrupted residual scale comparison
```

判断问题：

```text
posterior residual 是否有稳定类别区分能力？
是否在 clean 上产生异常大扰动？
是否只对少数类有效？
```

### 7.2 E5-B1：posterior residual final logits

目的：

```text
在不改变 cache replacement 的情况下，把 posterior residual 接入最终 logits。
```

公式：

\[
z_{\mathrm{new}}(c)
=
z_{\mathrm{pc}}(c)
+
\gamma
\Delta_{\mathrm{posterior}}(c,x)
\]

一次实验同时输出：

```text
original_acc
gamma=0.05
gamma=0.10
gamma=0.20
```

必须同时评估：

```text
ModelNet-C severity=2 七类 corruption
ModelNet-C clean
```

### 7.3 E5-C1：Mahalanobis replacement

目的：

```text
验证 shared covariance metric 是否能改进 cache replacement。
```

与 E5-B1 的区别：

```text
E5-B1 改最终 logits；
E5-C1 改 cache replacement；
两者第一轮不要混在一起。
```

### 7.4 E5-D1：动态文本-视觉权重

目的：

```text
用 n_eff 和 kappa 代替固定 text_weight=0.15。
```

建议设置：

```text
kappa = 8
kappa = 16
```

第一轮可以只跑 `kappa=16`，因为它更保守，更不容易破坏 clean。

---

## 8. 推荐初始配置

| 配置项 | 推荐值 | 说明 |
|---|---|---|
| base method | `02_9_2` | 当前 severity=2 最强基线 |
| dataset | ModelNet-C severity=2 + ModelNet-C clean | 必须同时看 corrupted 和 clean |
| backbone | ULIP | 保持和当前主线一致 |
| StatsBank capacity | `L=16` | 独立于 Point-Cache `shot_capacity=3` |
| delayed update | enabled | 当前样本不能更新自己的 posterior |
| text prior | `manual_full` prototype | 稳定锚点，避免重复 `02_7` 问题 |
| E1 component | text distribution only | 保留 `02_9_2` 的有效设定 |
| `kappa` | `8` 或 `16` | 文本先验强度 |
| gamma list | `0.05, 0.10, 0.20` | B1 一次实验同时输出 |
| residual norm | sample-wise class z-score + clip | 避免 running 统计污染 |
| output | original + enhanced 双结果 | 不覆盖原公式结果 |

---

## 9. 成功标准

E5-B/C/D 不能只看某一个 corruption，也不能只看 corrupted。

最低成功标准：

```text
ModelNet-C severity=2 平均准确率 > 02_9_2 的约 54.71
并且 clean accuracy 不低于 02_9_2 clean 的 63.86
```

更强成功标准：

```text
ModelNet-C severity=2 平均准确率 > 54.71
并且 clean accuracy 接近或超过原始 Point-Cache clean 的 64.18
```

如果 corrupted 提升但 clean 明显下降，需要视为不稳定，不能作为主方法。

---

## 10. 主要风险与应对

### 10.1 伪标签污染

风险：

```text
StatsBank 由在线预测产生，错误预测会污染 posterior prototype。
```

应对：

```text
使用 delayed update；
使用高置信准入；
E5-B 增强后的预测不要反向更新自己的 bank；
保存 per-class bank purity 诊断，如果可获得标签则离线分析。
```

### 10.2 单 prototype 仍然不足以表达多峰类别

风险：

```text
chair/table/lamp 等类别本身可能多形态。
单一 posterior prototype 可能仍然过度平滑。
```

应对：

```text
E5-B 先做单 prototype 验证；
如果有效但受限，再做 multi-mode posterior prototype；
不要第一版同时引入 K=3 modes，避免变量过多。
```

### 10.3 clean 退化

风险：

```text
在 clean 上，Point-Cache 本身已经很强，posterior residual 可能引入不必要扰动。
```

应对：

```text
不采用 clean fallback，因为 fallback 会牺牲 clean 上的提升机会；
改用 residual 而不是 standalone classifier；
使用小 gamma；
使用 sample-wise z-score + clipping；
必须保存 original 和 enhanced 双结果。
```

### 10.4 residual 被归一化放大

风险：

```text
如果所有类别 residual 都很小，z-score 可能把噪声放大。
```

应对：

```text
保存 residual raw std；
当 raw std 低于阈值时可跳过增强；
第一版先记录诊断，不急于加入跳过规则。
```

### 10.5 与 E4-C 分布得分重复

风险：

```text
E4-C 已经用 text/visual distribution 控制 GPA replacement，
E5-B posterior residual 也来自 high-confidence bank，
两者可能重复利用相似证据。
```

应对：

```text
E5-B1 第一版只改 final logits；
E5-C1 第一版只改 replacement；
不要一开始把 B/C/D 全部混合。
```

---

## 11. 和 E4-C / 02_9_2 的关系

E5 不是推翻 `02_9_2`，而是在它基础上做更有理论约束的改进。

`02_9_2` 的有效点是：

```text
E1 prompt 信息只进入 text distribution；
text distribution 权重 0.15；
running z-score 归一化；
accepted-history visual distribution；
最终 Point-Cache logits 公式不被大幅破坏。
```

E5-B 保留这些经验，但做两个改变：

1. 不再只把分布信息用于 replacement，而是以 residual 形式进入最终 logits；
2. 不直接加入新分类器分数，而是加入相对文本 prior 的 posterior 修正量。

因此 E5-B 的理论定位更准确：

```text
不是 GDA classifier fusion，
而是 prior-to-posterior prototype correction。
```

---

## 12. 专家自查

### 12.1 这个方案是否仍然依赖超参数？

是。主要超参数包括：

```text
StatsBank capacity L
kappa
gamma
residual norm clip
bank admission threshold
```

但相比 E5-A 直接加入 GDA logits，E5-B 的超参数更可解释：

- \(L\)：最多保留多少测试时视觉证据；
- \(\kappa\)：文本 prior 有多强；
- \(\gamma\)：posterior residual 对最终 logits 的影响有多大。

### 12.2 为什么不是继续直接优化 GDA 判别式？

因为 E5-A 已经显示 standalone GDA 在 add_global/add_local 上弱于原始 Point-Cache/E4 分支。继续把 \(g_c(x)\) 作为分类器主证据，风险大于收益。

### 12.3 为什么 residual 比直接加 posterior similarity 更合理？

直接加 \(\cos(f(x),m_c)\) 会重复计算文本 prior 中已有的基础类别语义。residual：

\[
\cos(f(x),m_c)-\cos(f(x),t_c)
\]

只表达 posterior prototype 相对原文本 prior 新增了什么，因此更符合“修正”而不是“替代”的目标。

### 12.4 为什么不让 clean 自动退回 Point-Cache？

因为用户明确要求：clean 时直接退回 Point-Cache 虽然安全，但没有提升空间。当前目标不是做 conservative fallback，而是设计一个在 clean 和 corrupted 上都可能带来正收益的可靠残差项。

### 12.5 如果 E5-B 仍失败，怎么判断失败原因？

优先检查：

1. posterior residual raw scale 是否太小；
2. z-score 是否放大噪声；
3. bank 中伪标签是否污染；
4. \(\lambda_c\) 是否过大，导致视觉统计过早压过文本 prior；
5. clean 和 corrupted 的 residual 分布是否不同；
6. 哪些 corruption 受益，哪些 corruption 受损。

如果失败主要来自单 prototype 表达不足，再考虑 multi-mode posterior prototype；如果失败主要来自 clean 扰动过强，则优先调小 \(\gamma\) 或增加 residual skip gate。

---

## 13. 当前结论

E5 的下一步不应继续强化 standalone GDA，也不应直接把 ADAPT/PGA 公式机械接入 Point-Cache。

更合理的推进路线是：

```text
E5-B：先做 posterior prototype residual，只改 final logits，双结果输出。
E5-C：再用 shared covariance metric 改 replacement，不改 final logits。
E5-D：最后把固定 text_weight=0.15 改为有效样本数驱动的动态权重。
```

第一优先级是 E5-B0/B1。只有当 E5-B 证明 posterior residual 有稳定信号后，才值得继续做 E5-C 和 E5-D。

---

## 14. E5-B0/B1 当前实现文件

当前已经新增 E5-B0/B1 的独立实现，不覆盖 E5-A，也不覆盖 E4-C `02_9_2`。

核心 model：

```text
Point-Cache/runners/E5_adapt_inspired_gaussian_alignment_cache/model_e5_b0_b1_posterior_prototype_residual.py
```

ModelNet-C severity=2 runner：

```text
Point-Cache/runners/E5_adapt_inspired_gaussian_alignment_cache/run_e5_b0_b1_ulip_modelnetc_s2_posterior_prototype_residual.py
```

通用脚本：

```text
Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache/01_run_e5_b0_b1_ulip_modelnetc_s2_common.sh
```

入口脚本：

```text
Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache/01_1_ulip_modelnetc_s2_zs_global_local_e5_b0_b1_posterior_prototype_residual_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh
```

运行方式：

```bash
bash Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache/01_1_ulip_modelnetc_s2_zs_global_local_e5_b0_b1_posterior_prototype_residual_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh 0
```

默认设置：

```text
E5_STATSBANK_CAPACITY=16
E5_POSTERIOR_KAPPA=16
E5_POSTERIOR_MIN_TOTAL=8
E5_POSTERIOR_MIN_CLASSES=2
E5_POSTERIOR_NORM_EPS=1e-6
E5_POSTERIOR_NORM_CLIP=3.0
E5_POSTERIOR_GAMMAS=0.05,0.10,0.20
E4_TEXT_SCORE_WEIGHT=0.15
E4_SCORE_NORM_MODE=running_zscore
E4_TEXT_DIST_PROMPT_SOURCE=manualfull_llm_dynamic_init
```

输出设计：

```text
summary.csv 中同一个 corruption 会写多行：
1. original_pointcache_formula
2. posterior_gamma_0.05
3. posterior_gamma_0.1
4. posterior_gamma_0.2
```

每个 corruption 还会保存：

```text
gpa_stats/
e5_posterior_stats/
```

其中 `e5_posterior_stats/` 保存 StatsBank 覆盖度、\(\lambda_c\)、有效样本数、posterior residual 统计、每个 gamma 的 changed/fixes/breaks/net_fixes 等诊断信息。
