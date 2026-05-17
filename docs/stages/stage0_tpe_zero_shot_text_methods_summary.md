# E0-TPE 文本原型实验阶段性说明文档：四种文本端方法的思路与结果

生成时间：2026-05-17 06:40:17

## 0. 文档目的

本文档用于归档 MCM-PC / Point-Cache 项目中 **E0-TPE 文本原型增强** 阶段的 zero-shot 诊断实验。

这一阶段的核心问题是：

> 在不改变 Point-Cache cache 机制的前提下，仅改变文本原型或文本模板得分聚合方式，是否能提升 zero-shot 预测？

我们已经确认：原始 Point-Cache 源码并不是只使用一个固定模板，而是使用 64 个文本模板。每个模板先经过 CLIP / SLIP 文本编码器，得到一个文本特征向量；这些向量逐个 L2 归一化，然后对 64 个模板特征求平均，最后再 L2 归一化，得到每个类别的文本原型。

因此，本阶段不是从“单模板”升级到“多模板”，而是在 Point-Cache 已有 64 模板机制基础上，尝试几种轻量文本处理方式。

本次记录包含四种方法：

| 编号 | 方法名称 | 简称 | 是否保留单一文本原型 |
|---|---|---|---|
| 方法 0 | 轻量文本原型收缩 | E0-TPE-v1-lite shrinkage | 是 |
| 方法 1 | 模板得分截尾平均 | E0-TPE-v2 trimmean10 | 否 |
| 方法 2 | 模板得分普通平均 | E0-TPE-v2 meanlogit | 否 |
| 方法 3 | 模板得分中位数 | E0-TPE-v2 median | 否 |

本阶段实验结果表明：**四种方法均未超过原始 ZS-BASE。**

---

## 1. 实验设置

### 1.1 数据与 corruption 范围

本次只跑 ModelNet-C 的 7 个 corruption，全部为 severity 2。根据当前本地数据命名规则，本地后缀为：

```text
_1
```

因此实际运行的 corruption 为：

```text
add_global_1
add_local_1
dropout_global_1
dropout_local_1
jitter_1
rotate_1
scale_1
```

### 1.2 为什么只跑 zero-shot？

本阶段的目标是判断 **文本端处理本身是否有效**。

如果直接跑完整 Point-Cache hierarchical baseline，最终结果会受到以下因素共同影响：

```text
文本原型变化
global cache 更新
local cache 更新
negative cache 更新
测试流顺序
cache admission 规则
```

这样很难判断收益是否来自文本端。

因此本阶段只做 zero-shot，即：

```text
不使用 cache
不做 test-time adaptation
只用点云 global feature 和文本原型 / 模板得分进行分类
```

### 1.3 Zero-shot baseline 的定义

原始 Point-Cache 文本原型构造为：

$$
\mathbf t_c^{PC}
=
\operatorname{Normalize}
\left(
\frac{1}{64}
\sum_{m=1}^{64}
\mathbf z_c^{(m)}
\right)
$$

其中：

- $c$：类别编号；
- $m$：文本模板编号；
- $\mathbf z_c^{(m)}$：类别 $c$ 的第 $m$ 个模板经过文本编码器后得到的单位文本特征；
- $\mathbf t_c^{PC}$：Point-Cache 默认文本原型。

对输入点云 $x$，点云编码器得到单位特征 $\mathbf f_x$，zero-shot logit 为：

$$
s_c^{PC}(x)
=
\mathbf f_x^\top \mathbf t_c^{PC}
$$

最终预测为：

$$
\hat y
=
\arg\max_c s_c^{PC}(x)
$$

---

## 2. 对比基线：ZS-BASE

ZS-BASE 使用 Point-Cache 原始 64 模板归一化平均文本原型，不使用 cache。

本次 ZS-BASE 的 severity 2 结果为：

| Corruption | ZS-BASE Accuracy |
|---|---:|
| add_global_1 | 66.05 |
| add_local_1 | 57.86 |
| dropout_global_1 | 70.30 |
| dropout_local_1 | 63.98 |
| jitter_1 | 34.52 |
| rotate_1 | 72.24 |
| scale_1 | 69.08 |
| **Average** | **62.00** |

这个结果作为后续四种 E0-TPE 方法的对照。

---

## 3. 方法 0：E0-TPE-v1-lite 轻量文本原型收缩

### 3.1 方法动机

最初我们考虑的问题是：Point-Cache 虽然使用了 64 个模板，但不同类别的 64 个模板经过文本编码器后，方向一致性可能不同。

例如：

```text
类别 A 的 64 个模板方向高度一致；
类别 B 的 64 个模板方向比较分散；
类别 C 的某些模板可能产生语义偏移。
```

原始 Point-Cache 对所有类别都直接使用 64 模板平均方向：

$$
\mathbf m_c
=
\frac{
\sum_{m=1}^{64} \mathbf z_c^{(m)}
}{
\left\|
\sum_{m=1}^{64} \mathbf z_c^{(m)}
\right\|_2
}
$$

这等价于假设：每个类别都应该完全相信 64 个模板的平均方向。

方法 0 尝试加入一个非常轻量的模板一致性机制：如果 64 个模板方向一致，则基本沿用原始平均方向；如果 64 个模板方向分散，则将文本原型轻微拉回一个基础模板方向。

### 3.2 $\bar R_c$：平均合向量长度

方法 0 使用 $\bar R_c$ 衡量模板一致性：

$$
\bar R_c
=
\left\|
\frac{1}{64}
\sum_{m=1}^{64}
\mathbf z_c^{(m)}
\right\|_2
$$

直观理解：

| $\bar R_c$ | 含义 |
|---|---|
| 接近 1 | 64 个模板特征方向高度一致 |
| 中等 | 模板之间有一定分散 |
| 接近 0 | 模板方向高度分散，平均方向不稳定 |

由于所有 $\mathbf z_c^{(m)}$ 都是单位向量，根据三角不等式，有：

$$
0 \le \bar R_c \le 1
$$

### 3.3 收缩公式

选取基础模板方向：

$$
\mathbf z_c^{base}
$$

这里通常使用 Point-Cache 64 个模板中的第一个模板，例如：

```text
a point cloud model of {class}.
```

方法 0 的最终文本原型为：

$$
\mathbf a_c^T
=
\operatorname{Normalize}
\left(
(1-\bar R_c)\mathbf z_c^{base}
+
\bar R_c\mathbf m_c
\right)
$$

其中：

- $\mathbf a_c^T$：新的文本原型；
- $\mathbf z_c^{base}$：基础模板经过文本编码器后的单位特征；
- $\mathbf m_c$：Point-Cache 原始 64 模板平均方向；
- $\bar R_c$：模板一致性权重。

### 3.4 直观解释

如果某个类别模板非常一致：

$$
\bar R_c \approx 1
$$

则：

$$
\mathbf a_c^T \approx \mathbf m_c
$$

也就是几乎沿用原始 Point-Cache 文本原型。

如果某个类别模板比较分散：

$$
\bar R_c < 1
$$

则文本原型会向基础模板方向收缩。

该方法的本质是：

```text
模板一致 → 信任 64 模板平均；
模板分散 → 向基础模板收缩。
```

### 3.5 实验结果

| corruption | ZS-BASE | E0-TPE-v1-lite shrinkage | Δ E0-TPE-v1-lite shrinkage |
| --- | ---: | ---: | ---: |
| add_global_1 | 66.05 | 65.92 | -0.13 |
| add_local_1 | 57.86 | 57.86 | 0.00 |
| dropout_global_1 | 70.30 | 70.14 | -0.16 |
| dropout_local_1 | 63.98 | 63.78 | -0.20 |
| jitter_1 | 34.52 | 34.44 | -0.08 |
| rotate_1 | 72.24 | 71.19 | -1.05 |
| scale_1 | 69.08 | 68.88 | -0.20 |
| Average | 62.00 | 61.74 | -0.26 |


### 3.6 分析

方法 0 平均准确率为 **61.74**，相比 ZS-BASE 的 **62.00** 下降 **0.26**。

该结果说明：

```text
将 64 模板平均文本原型向基础模板收缩，并没有提升 zero-shot 表现。
```

可能原因是：

1. Point-Cache 原始 64 模板平均方向已经比较稳定；
2. 基础模板虽然更接近 3D 任务，但单个模板的信息不如 64 模板平均充分；
3. 模板方向分散不一定意味着应该向基础模板收缩；
4. 对 rotate、dropout 等 corruption，文本原型轻微改变无法解决点云几何特征扰动问题。

---

## 4. 方法 1：E0-TPE-v2-trimmean10 模板得分截尾平均

### 4.1 方法动机

方法 0 仍然构造单一文本原型，只是在原始文本原型方向上做轻量收缩。方法 1 则尝试换一种处理方式：不再先把 64 个模板特征平均成一个文本原型，而是在推理时让点云分别和 64 个模板计算相似度，再对 64 个模板得分做聚合。

对输入点云 $x$，先计算每个类别、每个模板的相似度：

$$
s_c^{(m)}(x)
=
\mathbf f_x^\top \mathbf z_c^{(m)}
$$

然后再对模板维度 $m$ 聚合。

### 4.2 截尾平均公式

方法 1 使用截尾平均：

$$
s_c^{trim}(x)
=
\operatorname{TrimMean}_{m=1}^{64}
\left(
s_c^{(m)}(x)
\right)
$$

本次实验使用 `trimmean10`，即去掉最高 10% 和最低 10% 的模板得分，再对剩余模板得分求平均。

由于 64 个模板的 10% 约为 6 个，因此近似可以理解为：

```text
对每个类别：
先得到 64 个模板得分；
去掉约 6 个最高得分；
去掉约 6 个最低得分；
对中间约 52 个得分求平均。
```

### 4.3 为什么可能有效？

这个方法的假设是：某些模板可能产生异常高分或异常低分。异常低分可能来自不合适模板；异常高分可能来自某些模板偶然与错误类别高度匹配。截尾平均试图减少这些异常模板的影响。

### 4.4 为什么也可能失败？

截尾平均的风险是：最高得分模板可能恰好是最有判别力的模板；最低得分模板也可能提供有用的类别区分信息。因此，去掉两端模板得分可能会损失有效信息。

### 4.5 实验结果

| corruption | ZS-BASE | E0-TPE-v2 trimmean10 | Δ E0-TPE-v2 trimmean10 |
| --- | ---: | ---: | ---: |
| add_global_1 | 66.05 | 66.37 | 0.32 |
| add_local_1 | 57.86 | 56.44 | -1.42 |
| dropout_global_1 | 70.30 | 69.12 | -1.18 |
| dropout_local_1 | 63.98 | 62.32 | -1.66 |
| jitter_1 | 34.52 | 33.39 | -1.13 |
| rotate_1 | 72.24 | 70.34 | -1.90 |
| scale_1 | 69.08 | 68.15 | -0.93 |
| Average | 62.00 | 60.88 | -1.13 |


### 4.6 分析

方法 1 平均准确率为 **60.88**，相比 ZS-BASE 下降 **1.13**。

只有 `add_global_1` 有小幅提升，其他 6 个 corruption 全部下降。这说明模板得分的最高/最低部分并不一定是噪声，简单截尾可能删除了有用模板信号。

---

## 5. 方法 2：E0-TPE-v2-meanlogit 模板得分普通平均

### 5.1 方法动机

方法 2 仍然保留“先算每个模板得分，再聚合”的思路，但不做截尾。它更温和地测试：原始 Point-Cache 的“先平均模板特征再算 logit”是否不如“先算每个模板 logit 再平均 logit”。

### 5.2 数学公式

对每个类别 $c$，先计算 64 个模板得分：

$$
s_c^{(m)}(x)
=
\mathbf f_x^\top \mathbf z_c^{(m)}
$$

然后普通平均：

$$
s_c^{meanlogit}(x)
=
\frac{1}{64}
\sum_{m=1}^{64}
s_c^{(m)}(x)
$$

代入可得：

$$
s_c^{meanlogit}(x)
=
\mathbf f_x^\top
\left(
\frac{1}{64}
\sum_{m=1}^{64}
\mathbf z_c^{(m)}
\right)
$$

与 ZS-BASE 的区别是：ZS-BASE 使用的是归一化后的平均文本原型：

$$
s_c^{PC}(x)
=
\mathbf f_x^\top
\operatorname{Normalize}
\left(
\frac{1}{64}
\sum_{m=1}^{64}
\mathbf z_c^{(m)}
\right)
$$

而 meanlogit 不对平均向量再归一化。因此：

$$
s_c^{meanlogit}(x)
=
\bar R_c
\cdot
\mathbf f_x^\top \mathbf m_c
$$

这意味着：如果某个类别 64 个模板方向分散，$\bar R_c$ 较小，该类别 logit 会自然变小。

### 5.3 为什么可能有效？

这个方法的假设是：如果一个类别的模板方向很分散，那么模型对这个类别的文本定义不稳定，因此这个类别的 logit 不应过度自信。meanlogit 保留了平均向量长度信息，相当于用 $\bar R_c$ 对类别 logit 做隐式降权。

### 5.4 为什么可能失败？

模板方向分散不一定代表类别不可靠。某些类别本身可能需要多个不同文本视角描述。此时方向分散可能是语义丰富性，而不是噪声。如果直接用 $\bar R_c$ 降权，可能会错误压低这些类别。

### 5.5 实验结果

| corruption | ZS-BASE | E0-TPE-v2 meanlogit | Δ E0-TPE-v2 meanlogit |
| --- | ---: | ---: | ---: |
| add_global_1 | 66.05 | 66.45 | 0.40 |
| add_local_1 | 57.86 | 56.36 | -1.50 |
| dropout_global_1 | 70.30 | 69.04 | -1.26 |
| dropout_local_1 | 63.98 | 62.28 | -1.70 |
| jitter_1 | 34.52 | 33.71 | -0.81 |
| rotate_1 | 72.24 | 70.38 | -1.86 |
| scale_1 | 69.08 | 68.11 | -0.97 |
| Average | 62.00 | 60.90 | -1.10 |


### 5.6 分析

方法 2 平均准确率为 **60.90**，相比 ZS-BASE 下降 **1.10**。

它与 trimmean10 非常接近，说明在 logit 层聚合模板得分，不如 Point-Cache 原始的归一化平均文本原型稳定。

---

## 6. 方法 3：E0-TPE-v2-median 模板得分中位数

### 6.1 方法动机

方法 3 是更强的鲁棒聚合方式。相比 meanlogit 和 trimmean10，median 不关心得分的均值，而是取中间位置的模板得分：

$$
s_c^{median}(x)
=
\operatorname{Median}_{m=1}^{64}
\left(
s_c^{(m)}(x)
\right)
$$

它的想法是：如果少数模板出现异常高分或异常低分，中位数不容易被这些极端值影响。

### 6.2 为什么可能有效？

如果 64 个模板里有少数模板发生明显语义漂移，那么 median 可以避免这些异常模板影响类别得分。

### 6.3 为什么可能失败？

median 的问题是过于保守。在分类任务中，有些模板可能特别适合当前点云视角或腐蚀类型，它们的高分可能是有价值的信号。median 会弱化这些高响应模板，因此可能降低类别区分度。

### 6.4 实验结果

| corruption | ZS-BASE | E0-TPE-v2 median | Δ E0-TPE-v2 median |
| --- | ---: | ---: | ---: |
| add_global_1 | 66.05 | 66.13 | 0.08 |
| add_local_1 | 57.86 | 56.40 | -1.46 |
| dropout_global_1 | 70.30 | 69.04 | -1.26 |
| dropout_local_1 | 63.98 | 62.12 | -1.86 |
| jitter_1 | 34.52 | 33.06 | -1.46 |
| rotate_1 | 72.24 | 69.94 | -2.30 |
| scale_1 | 69.08 | 67.46 | -1.62 |
| Average | 62.00 | 60.59 | -1.41 |


### 6.5 分析

方法 3 平均准确率为 **60.59**，相比 ZS-BASE 下降 **1.41**，是本阶段最差的方法。

这说明：过度鲁棒的模板得分聚合会损失有效模板信号。特别是 `rotate_1` 下降 **2.30**，说明 median 在旋转扰动下明显损害了文本匹配结果。

---

## 7. 总体结果汇总

### 7.1 完整对比表

| corruption | ZS-BASE | E0-TPE-v1-lite shrinkage | Δ E0-TPE-v1-lite shrinkage | E0-TPE-v2 trimmean10 | Δ E0-TPE-v2 trimmean10 | E0-TPE-v2 meanlogit | Δ E0-TPE-v2 meanlogit | E0-TPE-v2 median | Δ E0-TPE-v2 median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_global_1 | 66.05 | 65.92 | -0.13 | 66.37 | 0.32 | 66.45 | 0.40 | 66.13 | 0.08 |
| add_local_1 | 57.86 | 57.86 | 0.00 | 56.44 | -1.42 | 56.36 | -1.50 | 56.40 | -1.46 |
| dropout_global_1 | 70.30 | 70.14 | -0.16 | 69.12 | -1.18 | 69.04 | -1.26 | 69.04 | -1.26 |
| dropout_local_1 | 63.98 | 63.78 | -0.20 | 62.32 | -1.66 | 62.28 | -1.70 | 62.12 | -1.86 |
| jitter_1 | 34.52 | 34.44 | -0.08 | 33.39 | -1.13 | 33.71 | -0.81 | 33.06 | -1.46 |
| rotate_1 | 72.24 | 71.19 | -1.05 | 70.34 | -1.90 | 70.38 | -1.86 | 69.94 | -2.30 |
| scale_1 | 69.08 | 68.88 | -0.20 | 68.15 | -0.93 | 68.11 | -0.97 | 67.46 | -1.62 |
| Average | 62.00 | 61.74 | -0.26 | 60.88 | -1.13 | 60.90 | -1.10 | 60.59 | -1.41 |


### 7.2 平均准确率排序

| Method | Average Accuracy | Δ vs ZS-BASE |
| --- | ---: | ---: |
| ZS-BASE | 62.00 | 0.00 |
| E0-TPE-v1-lite shrinkage | 61.74 | -0.26 |
| E0-TPE-v2 meanlogit | 60.90 | -1.10 |
| E0-TPE-v2 trimmean10 | 60.88 | -1.13 |
| E0-TPE-v2 median | 60.59 | -1.41 |


---

## 8. 阶段性结论

### 8.1 四种方法均未超过原始 ZS-BASE

本阶段尝试了四种文本端轻量方法：

```text
方法 0：轻量文本原型收缩
方法 1：模板得分截尾平均
方法 2：模板得分普通平均
方法 3：模板得分中位数
```

结果显示，四种方法都没有超过 Point-Cache 原始 64 模板归一化平均文本原型。

因此可以得出结论：

> Point-Cache 原始 64 模板归一化平均文本原型已经是一个较强且稳定的文本端基线。

### 8.2 文本原型端目前不是主要瓶颈

从 severity 2 的 zero-shot 诊断结果来看，文本端轻量修改并不能带来增益。尤其是三种模板得分聚合方法均明显下降，说明：

```text
将 64 个模板保留到 logit 层再聚合，不如先平均并归一化为一个稳定文本原型。
```

### 8.3 为什么原始方法可能更好？

原始方法有三个优势：

1. **归一化平均方向稳定**  
   它保留了 64 个模板的整体语义方向，同时消除了模板分散导致的模长差异。

2. **避免类别 logit 被 $\bar R_c$ 错误降权**  
   meanlogit 会让模板分散类别的得分变小，但模板分散不一定代表类别不可靠。

3. **不过度删除模板信息**  
   trimmean 和 median 都会弱化一部分模板信号，而这些信号可能对分类有用。

### 8.4 对后续工作的影响

E0-TPE 文本原型增强不应继续投入过多复杂工作。

建议将其作为一个阶段性负结果归档：

```text
We evaluated several lightweight text prototype modifications over the original Point-Cache 64-template text prototype, including prototype shrinkage and template-score aggregation. None of them improved zero-shot performance on ModelNet-C severity 2, indicating that the original normalized 64-template text prototype is already a strong textual baseline.
```

中文表述：

```text
我们评估了多种基于 Point-Cache 原始 64 模板文本原型的轻量文本端改进，包括文本原型收缩和模板得分聚合。结果显示，这些方法均未提升 ModelNet-C severity 2 上的 zero-shot 性能，说明原始 64 模板归一化平均文本原型已经是较强的文本端基线。
```

---

## 9. 是否保留这些实验代码？

建议保留，但不要作为主线方法继续推进。

推荐保留内容：

```text
runners/zs_infer_e0_tpe_v1_lite.py
runners/zs_infer_e0_tpe_v2_template_score_agg.py
scripts/recur-pc/run_e0_tpe_v1_lite_zs_modelnetc_severity2_corruptions_dual_gpu.sh
scripts/recur-pc/run_e0_tpe_v2_trimmean10_zs_modelnetc_severity2_corruptions_dual_gpu.sh
scripts/recur-pc/run_e0_tpe_v2_meanlogit_zs_modelnetc_severity2_corruptions_dual_gpu.sh
scripts/recur-pc/run_e0_tpe_v2_median_zs_modelnetc_severity2_corruptions_dual_gpu.sh
```

但提交时应明确说明这是 **diagnostic / negative result**，而不是主方法。

建议 commit message：

```text
exp: add e0 tpe zero-shot diagnostic runners
docs: archive e0 tpe zero-shot negative results
```

---

## 10. 后续建议

E0-TPE 阶段可以收尾。下一步应回到 MCM-PC 主线：

```text
cache reliability
negative cache admission
global-local conflict diagnosis
soft suppression
multi-cache matrix integration
```

也就是说，当前阶段给我们的结论是：

> 文本原型不是当前主要瓶颈，继续优化文本端的性价比较低；后续应将精力放回 test-time cache 可靠性与 conflict-aware negative / suppression 机制上。
