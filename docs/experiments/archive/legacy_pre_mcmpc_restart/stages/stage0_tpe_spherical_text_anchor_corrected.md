# Stage 0 / E0-TPE：基于源码更正后的文本原型增强说明文档

生成时间：2026-05-15 09:29:27

> 本文档是对原 `stage0_tpe_spherical_text_anchor.md` 和 `stage0_tpe_spherical_text_anchor_report.html` 的更正版。  
> 更正重点：后续分析不能只看论文主文中的流程示意，还必须结合 Point-Cache 源码实现。根据源码，Point-Cache 实验实现并不是只使用一个固定文本模板，而是默认使用 `datasets/templates.py` 中的 64 个手工文本模板；每个模板先经过 CLIP/SLIP 文本编码器，得到该类别在该模板下的文本特征，再对这些文本特征进行归一化、平均、再归一化，得到最终每个类别的文本原型。

---

## 0. 关键更正结论

### 0.1 之前需要更正的地方

原文档中曾经把 Point-Cache 的文本端描述为：

```text
原始 Point-Cache 通常只用一个固定模板，例如 “a point cloud object of a {class}”。
```

这个说法不准确。

更准确的结论是：

```text
Point-Cache 主文中出现的单模板表达主要是符号说明和流程示意；
但源码中的实际实验实现使用 64 个手工文本模板。
```

也就是说，Point-Cache baseline 的文本原型已经不是 single-prompt text prototype，而是：

```text
64-template normalized mean text prototype
```

中文统一称为：

```text
64 模板归一化平均文本原型
```

### 0.2 源码依据

在 Point-Cache 源码中，`datasets/templates.py` 定义了 `text_prompts`，长度为 64。典型模板包括：

```python
text_prompts = [
    "a point cloud model of {}.",
    "There is a {} in the scene.",
    "There is the {} in the scene.",
    "a photo of a {} in the scene.",
    ...
    "a painting of a {}."
]
```

在 `utils/utils.py` 的 `clip_classifier(args, classnames, template, clip_model)` 中，实际流程是：

```python
for classname in classnames:
    classname = classname.replace('_', ' ')
    texts = [t.format(classname) for t in template]

    if args.lm3d == 'uni3d' or args.lm3d == 'ulip':
        texts = clip.tokenize(texts).cuda()
    elif args.lm3d == 'openshape':
        texts = open_clip.tokenizer.tokenize(texts).cuda()

    class_embeddings = clip_model.encode_text(texts)
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

    class_embedding = class_embeddings.mean(dim=0)
    class_embedding /= class_embedding.norm()

    clip_weights.append(class_embedding)

clip_weights = torch.stack(clip_weights, dim=1).cuda()
```

因此，真实流程是：

```text
64 个文本模板
→ 每个模板填入类别名
→ tokenize
→ CLIP/SLIP text encoder
→ 得到 64 个文本特征向量
→ 每个文本特征向量 L2 归一化
→ 对 64 个单位文本特征向量求平均
→ 再 L2 归一化
→ 得到该类别最终文本原型
```

---

## 1. E0-TPE 的实验名字更正

### 1.1 E0-TPE 是什么？

`E0-TPE` 的完整英文是：

```text
Experiment 0 - Text Prototype Enhancement
```

中文统一称为：

```text
第 0 个实验：文本原型增强
```

其中：

| 缩写 | 英文 | 中文 |
|---|---|---|
| E0 | Experiment 0 | 第 0 个实验 |
| T | Text | 文本 |
| P | Prototype | 原型 |
| E | Enhancement | 增强 |

### 1.2 为什么叫 E0？

因为它在逻辑上位于所有 cache 实验之前。

此前路线是：

```text
E1-BASE → E2-EMR → E3-GLC → E4-CANC
```

这些实验都默认了一个前提：

```text
文本原型已经足够可靠。
```

但 Point-Cache 的 zero-shot prediction、pseudo-label、entropy、margin、cache update 都依赖文本原型。如果文本原型存在问题，后续所有 cache 规则都会被影响。

因此，文本端应该补成更上游的阶段：

```text
E0-TPE → E1-BASE → E2-EMR → E3-GLC → E4-CANC → E5-MCM
```

### 1.3 更正后的 E0-TPE 子实验命名

由于 Point-Cache baseline 已经使用 64 个模板，所以不能再把“多模板平均”当成新方法。

更正后的实验命名应为：

| 实验编号 | 英文名称 | 中文名称 | 是否已被 Point-Cache baseline 包含 |
|---|---|---|---|
| E0-TPE-BASE | Point-Cache 64-template Normalized Mean Text Prototype | Point-Cache 64 模板归一化平均文本原型 | 是 |
| E0-TPE-DIAG | Text Prototype Dispersion Diagnosis | 文本原型离散度诊断 | 否 |
| E0-TPE-v1 | vMF-Shrunk Text Prototype over Point-Cache 64 Templates | 基于 Point-Cache 64 模板的 vMF 收缩式文本原型 | 否 |
| E0-TPE-v2 | DeepSeek Paraphrase Expansion with vMF-Shrunk Text Prototype | DeepSeek 改写扩充 + vMF 收缩式文本原型 | 否，后续再做 |

---

## 2. 为什么必须结合源码而不是只看主文？

### 2.1 主文中的单模板是符号示意

Point-Cache 主文为了说明 zero-shot inference，会写类似：

```text
a point cloud object of a {class}
```

这类表达主要是为了公式和流程图简洁。它说明文本输入的形式，但不等价于实际实验中只用了一个模板。

因此，不能仅根据主文示意推断实际实现是 single prompt。

### 2.2 源码才决定实验真实设置

实际实验跑的是代码，不是论文里的简化符号。源码中 `clip_classifier` 明确对 `template` 中的所有模板逐个编码，然后平均。因此，我们后续做实验必须以源码为准。

### 2.3 后续工作规则

后续分析必须遵守：

```text
先看论文主文理解方法目标；
再看 Supplement / Appendix 理解实验细节；
最后看源码确认真实实现。
```

尤其是以下内容不能只看论文：

1. 文本模板数量；
2. 文本特征是否先归一化；
3. 平均后是否再归一化；
4. cache update 的真实顺序；
5. local cache 是否独立更新；
6. summary 里 corruption suffix 的真实映射；
7. runner 是否使用了同一个 `clip_classifier`。

---

## 3. Point-Cache 当前文本原型构造的严格数学表达

### 3.1 符号说明

| 符号 | 含义 |
|---|---|
| \(c\) | 类别编号 |
| \(C\) | 类别总数 |
| \(m\) | 文本模板编号 |
| \(M\) | 文本模板数量。Point-Cache 默认 \(M=64\) |
| \(t_m(\cdot)\) | 第 \(m\) 个文本模板 |
| \(p_c^{(m)}\) | 将类别名填入第 \(m\) 个模板后得到的完整 prompt |
| \(\mathcal E_{text}\) | CLIP/SLIP/OpenCLIP 文本编码器 |
| \(\mathbf u_c^{(m)}\) | 第 \(m\) 个 prompt 编码后的原始文本特征 |
| \(\mathbf z_c^{(m)}\) | 第 \(m\) 个 prompt 的 L2 归一化文本特征 |
| \(\mathbf m_c\) | Point-Cache 默认的 64 模板平均方向 |
| \(\mathbf t_c^{PC}\) | Point-Cache 默认最终文本原型 |
| \(\mathbf a_c^T\) | E0-TPE-v1 生成的新文本原型 |

### 3.2 从模板字符串到 prompt

Point-Cache 不是直接对类别名编码，而是先构造模板化文本。

对类别 \(c\)，第 \(m\) 个模板是：

\[
t_m(\cdot)
\]

填入类别名后得到：

\[
p_c^{(m)} = t_m(c)
\]

例如类别为 `chair` 时：

```text
a point cloud model of chair.
There is a chair in the scene.
a photo of a chair.
a toy chair.
...
```

### 3.3 每个 prompt 先经过文本编码器

每个 prompt 字符串必须先经过文本编码器：

\[
\mathbf u_c^{(m)}
=
\mathcal E_{text}(p_c^{(m)})
\in \mathbb R^d
\]

这一步非常关键。

E0-TPE 不是对文本字符串做数学处理，而是对文本编码器输出的特征向量做数学处理。

### 3.4 每个文本特征先 L2 归一化

源码中：

```python
class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
```

数学上是：

\[
\mathbf z_c^{(m)}
=
\frac{\mathbf u_c^{(m)}}{\|\mathbf u_c^{(m)}\|_2}
\]

因此：

\[
\|\mathbf z_c^{(m)}\|_2=1
\]

所有模板特征都位于单位超球面：

\[
\mathbf z_c^{(m)}\in \mathbb S^{d-1}
\]

### 3.5 Point-Cache 默认 64 模板归一化平均文本原型

源码中：

```python
class_embedding = class_embeddings.mean(dim=0)
class_embedding /= class_embedding.norm()
```

数学上是：

\[
\bar{\mathbf z}_c
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbf z_c^{(m)}
\]

然后：

\[
\mathbf t_c^{PC}
=
\frac{\bar{\mathbf z}_c}{\|\bar{\mathbf z}_c\|_2}
=
\frac{\sum_{m=1}^{M}\mathbf z_c^{(m)}}{
\left\|\sum_{m=1}^{M}\mathbf z_c^{(m)}\right\|_2
}
\]

也就是：

\[
\mathbf t_c^{PC}=\mathbf m_c
\]

其中：

\[
\mathbf m_c
=
\frac{\sum_{m=1}^{M}\mathbf z_c^{(m)}}{
\left\|\sum_{m=1}^{M}\mathbf z_c^{(m)}\right\|_2
}
\]

所以，Point-Cache baseline 已经在做：

```text
normalized mean direction on the unit hypersphere
```

中文：

```text
单位超球面上的归一化平均方向
```

---

## 4. 这意味着 E0-TPE-v0 要取消或改名

### 4.1 原 E0-TPE-v0 的问题

原文档把 E0-TPE-v0 定义为：

```text
固定模板 prompt ensemble
```

但源码说明 Point-Cache baseline 已经做了固定模板 prompt ensemble，而且用了 64 个模板。

因此，原来的 E0-TPE-v0 不能作为新实验。

### 4.2 更正后的说法

应改为：

```text
E0-TPE-BASE = Point-Cache default 64-template normalized mean text prototype.
```

中文：

```text
E0-TPE-BASE = Point-Cache 默认 64 模板归一化平均文本原型。
```

它不是新方法，而是 baseline 的文本端实现。

### 4.3 E0-TPE-v1 的真正新意

E0-TPE-v1 不再是“多模板平均”，而是：

```text
在同样 64 个模板、同样文本编码器、同样归一化特征的基础上，
进一步利用 64 个模板特征的方向一致性，构造类别自适应的 vMF 收缩式文本原型。
```

也就是说，变量控制非常干净：

| 项目 | E0-TPE-BASE | E0-TPE-v1 |
|---|---|---|
| 文本模板 | 同一组 64 templates | 同一组 64 templates |
| 文本编码器 | 相同 | 相同 |
| 每个模板特征归一化 | 是 | 是 |
| 是否使用 64 个模板特征 | 是 | 是 |
| 最终聚合方式 | 归一化平均 | vMF 收缩式聚合 |
| 是否改 cache 规则 | 否 | 否 |
| 是否引入 DeepSeek | 否 | 否 |

---

## 5. E0-TPE-v1：基于 Point-Cache 64 模板的 vMF 收缩式文本原型

### 5.1 核心问题

Point-Cache 默认方法对所有类别都同等相信 64 个模板的平均方向：

\[
\mathbf t_c^{PC}=\mathbf m_c
\]

但不同类别的 64 个模板特征可能有不同的方向一致性。

例如：

- 类别 A 的 64 个模板特征方向高度一致；
- 类别 B 的 64 个模板特征方向分散；
- 类别 C 的某些模板可能产生语义偏移。

Point-Cache 默认方法没有显式区分这些情况。

E0-TPE-v1 的核心目标是：

```text
如果 64 个模板特征方向一致，则继续信任平均方向；
如果方向分散，则将文本原型部分收缩回基础 prompt 的方向。
```

### 5.2 平均合向量长度

定义：

\[
\bar R_c
=
\left\|
\frac{1}{M}
\sum_{m=1}^{M}
\mathbf z_c^{(m)}
\right\|_2
\]

由于每个 \(\mathbf z_c^{(m)}\) 都是单位向量，有：

\[
0\le \bar R_c\le 1
\]

解释：

| \(\bar R_c\) | 含义 |
|---|---|
| 接近 1 | 64 个模板特征方向高度一致 |
| 中等 | 模板之间存在一定语义分散 |
| 接近 0 | 模板方向高度分散，平均方向不稳定 |

证明：

由三角不等式：

\[
\left\|\sum_{m=1}^{M}\mathbf z_c^{(m)}\right\|_2
\le
\sum_{m=1}^{M}\|\mathbf z_c^{(m)}\|_2
=
M
\]

所以：

\[
\bar R_c
=
\frac{1}{M}
\left\|
\sum_{m=1}^{M}\mathbf z_c^{(m)}
\right\|_2
\le 1
\]

又因为范数非负：

\[
0\le \bar R_c\le 1
\]

### 5.3 vMF 浓度估计

vMF 分布，即 von Mises-Fisher distribution，中文可写作冯·米塞斯-费舍尔分布。它是单位超球面上的方向分布。

其密度为：

\[
p(\mathbf z;\boldsymbol\mu,\kappa)
=
C_d(\kappa)\exp(\kappa\boldsymbol\mu^\top\mathbf z)
\]

其中：

| 符号 | 含义 |
|---|---|
| \(\mathbf z\) | 单位超球面上的随机向量 |
| \(\boldsymbol\mu\) | 均值方向，\(\|\boldsymbol\mu\|_2=1\) |
| \(\kappa\) | 浓度参数 |
| \(C_d(\kappa)\) | 归一化常数 |
| \(d\) | 特征维度 |

直观解释：

- \(\kappa=0\)：接近球面均匀分布；
- \(\kappa\) 越大：样本越集中在 \(\boldsymbol\mu\) 附近；
- \(\kappa\to\infty\)：所有样本几乎集中到一个方向。

根据平均合向量长度，可以近似估计浓度：

\[
\hat\kappa_c
\approx
\frac{\bar R_c(d-\bar R_c^2)}{1-\bar R_c^2}
\]

实现时必须加入数值稳定：

\[
\bar R_c \leftarrow \operatorname{clip}(\bar R_c,\epsilon,1-\epsilon)
\]

否则当 \(\bar R_c\to 1\) 时，分母 \(1-\bar R_c^2\to0\)，会出现数值爆炸。

### 5.4 基础 prompt 的选择

由于 Point-Cache 64 个模板中第一个模板是：

```text
a point cloud model of {}.
```

它比大量 “a photo of ...” 模板更符合点云任务，因此可以将它作为基础 prompt：

\[
\mathbf z_c^{base}=\mathbf z_c^{(1)}
\]

注意这里的基础 prompt 也必须经过同一个文本编码器，并且经过 L2 归一化。

也就是说：

```text
基础 prompt 不是原始字符串，而是第一个模板编码后的单位文本特征。
```

### 5.5 自适应收缩权重

定义：

\[
w_c
=
\frac{\hat\kappa_c}{\hat\kappa_c+\kappa_0}
\]

其中：

- \(w_c\in[0,1]\)；
- \(\hat\kappa_c\) 越大，说明 64 个模板越一致，越相信平均方向 \(\mathbf m_c\)；
- \(\hat\kappa_c\) 越小，说明 64 个模板越分散，越收缩回基础 prompt \(\mathbf z_c^{base}\)；
- \(\kappa_0\) 是基础 prompt 的先验强度。

不要直接固定：

\[
\kappa_0=2
\]

因为在高维特征空间中 \(\hat\kappa_c\) 可能非常大。更稳妥的做法是按所有类别的 \(\hat\kappa_c\) 标定：

\[
\kappa_0
=
\frac{\rho}{1-\rho}
\operatorname{median}_{c}(\hat\kappa_c)
\]

其中 \(\rho\) 表示希望在“中等模板一致性类别”上基础 prompt 所占的平均权重。

例如：

\[
\rho=0.3
\]

表示希望基础 prompt 在中位类别上约占 30%。

### 5.6 最终 vMF 收缩式文本原型

最终文本原型定义为：

\[
\mathbf a_c^T
=
\frac{
(1-w_c)\mathbf z_c^{base}
+
w_c\mathbf m_c
}{
\left\|
(1-w_c)\mathbf z_c^{base}
+
w_c\mathbf m_c
\right\|_2
}
\]

其中：

- \(\mathbf m_c\)：Point-Cache 默认 64 模板平均方向；
- \(\mathbf z_c^{base}\)：基础 prompt 的文本特征；
- \(w_c\)：类别自适应权重；
- \(\mathbf a_c^T\)：E0-TPE-v1 生成的新文本原型。

---

## 6. E0-TPE-v1 与 Point-Cache baseline 的数学关系

Point-Cache baseline：

\[
\mathbf t_c^{PC}=\mathbf m_c
\]

E0-TPE-v1：

\[
\mathbf a_c^T
=
\operatorname{Normalize}
\left(
(1-w_c)\mathbf z_c^{base}
+
w_c\mathbf m_c
\right)
\]

当：

\[
w_c=1
\]

时：

\[
\mathbf a_c^T
=
\mathbf m_c
=
\mathbf t_c^{PC}
\]

所以 Point-Cache baseline 是 E0-TPE-v1 的一个特殊情况。

当：

\[
0<w_c<1
\]

时，E0-TPE-v1 在基础 prompt 和 64 模板平均方向之间做类别自适应收缩。

当：

\[
w_c=0
\]

时：

\[
\mathbf a_c^T=\mathbf z_c^{base}
\]

也就是只使用基础 prompt。

因此，E0-TPE-v1 不是推翻 Point-Cache 的多模板平均，而是在它之上增加了：

```text
类别级文本模板一致性建模。
```

---

## 7. MAP 形式与闭式解证明

### 7.1 后验目标

将基础 prompt 看作先验方向，将 64 模板平均方向看作数据方向，可以构造：

\[
\mathbf a_c^T
=
\arg\max_{\mathbf z\in\mathbb S^{d-1}}
\left[
\kappa_0(\mathbf z_c^{base})^\top\mathbf z
+
\hat\kappa_c\mathbf m_c^\top\mathbf z
\right]
\]

令：

\[
\mathbf q_c
=
\kappa_0\mathbf z_c^{base}
+
\hat\kappa_c\mathbf m_c
\]

则目标变为：

\[
\mathbf a_c^T
=
\arg\max_{\|\mathbf z\|_2=1}
\mathbf q_c^\top\mathbf z
\]

### 7.2 Cauchy-Schwarz 证明

根据 Cauchy-Schwarz 不等式：

\[
\mathbf q_c^\top\mathbf z
\le
\|\mathbf q_c\|_2\|\mathbf z\|_2
\]

因为：

\[
\|\mathbf z\|_2=1
\]

所以：

\[
\mathbf q_c^\top\mathbf z
\le
\|\mathbf q_c\|_2
\]

当：

\[
\mathbf z
=
\frac{\mathbf q_c}{\|\mathbf q_c\|_2}
\]

时取等号。

因此闭式解为：

\[
\mathbf a_c^T
=
\frac{
\kappa_0\mathbf z_c^{base}
+
\hat\kappa_c\mathbf m_c
}{
\left\|
\kappa_0\mathbf z_c^{base}
+
\hat\kappa_c\mathbf m_c
\right\|_2
}
\]

等价写成权重形式就是：

\[
\mathbf a_c^T
=
\frac{
(1-w_c)\mathbf z_c^{base}
+
w_c\mathbf m_c
}{
\left\|
(1-w_c)\mathbf z_c^{base}
+
w_c\mathbf m_c
\right\|_2
}
\]

### 7.3 为什么输出仍在单位超球面上？

因为最后做了 L2 归一化：

\[
\|\mathbf a_c^T\|_2
=
\left\|
\frac{\mathbf q_c}{\|\mathbf q_c\|_2}
\right\|_2
=
1
\]

所以 E0-TPE-v1 不会改变类别间文本原型的模长，不会引入模长偏置。

---

## 8. 这不是标准 SLERP

之前文档中把该操作写成 SLERP 不够严谨。

更准确表述是：

```text
它是归一化球面加权均值，也可以解释为 vMF 后验自然参数方向的 MAP 解。
```

标准 SLERP 公式是：

\[
\operatorname{slerp}(\mathbf u,\mathbf v;t)
=
\frac{\sin((1-t)\theta)}{\sin\theta}\mathbf u
+
\frac{\sin(t\theta)}{\sin\theta}\mathbf v
\]

其中：

\[
\theta=\arccos(\mathbf u^\top\mathbf v)
\]

而 E0-TPE-v1 使用的是：

\[
\frac{(1-w)\mathbf u+w\mathbf v}{\|(1-w)\mathbf u+w\mathbf v\|_2}
\]

所以不要写“严格 SLERP”，应写：

```text
normalized spherical weighted mean, related to SLERP but not identical to standard SLERP.
```

中文：

```text
归一化球面加权均值，与 SLERP 有关，但不是标准 SLERP。
```

---

## 9. 与 BayesMM 的关系

BayesMM 的思想对我们有启发：它不是只用固定文本原型，而是把文本 prompt 扩展成多个 paraphrases，并用这些 embeddings 建立 textual distribution，捕捉 prompt variants 的语义多样性。

但 BayesMM 使用的是欧氏 Gaussian textual distribution。

我们的 E0-TPE-v1 只借鉴其核心思想：

```text
多个文本表达可以刻画文本原型的不确定性。
```

但不直接照搬 Gaussian 建模，而是结合 Point-Cache / CLIP 的余弦推理几何，在单位超球面上建模。

对比：

| 项目 | BayesMM | E0-TPE-v1 |
|---|---|---|
| 输入 | LLM paraphrases | Point-Cache 已有 64 templates |
| 处理对象 | 编码后的文本 embeddings | 编码后的文本 embeddings |
| 分布空间 | 欧氏空间 \(\mathbb R^d\) | 单位超球面 \(\mathbb S^{d-1}\) |
| 分布假设 | Gaussian | vMF / 方向统计 |
| 新增变量 | textual covariance | 平均合向量长度 \(\bar R_c\)、浓度 \(\hat\kappa_c\) |
| 输出 | MAP textual prototype | vMF 收缩式文本原型 |
| 是否改 cache | 是完整分布学习框架 | 初期不改 cache，只替换文本原型 |

---

## 10. 实验路线更正

### 10.1 不再做“固定模板多模板平均”作为新实验

因为 baseline 已经做了。

### 10.2 先做 E0-TPE-DIAG

目的：

```text
不改推理，只分析 64 个模板 embedding 的类别级分散程度。
```

输出指标：

| 指标 | 含义 |
|---|---|
| \(\bar R_c\) | 类别 c 的 64 模板方向一致性 |
| \(\hat\kappa_c\) | 类别 c 的 vMF 浓度 |
| \(\angle(\mathbf z_c^{base},\mathbf m_c)\) | 基础 prompt 与 64 模板平均方向的夹角 |
| nearest text prototype | 文本原型之间最相近的类别 |
| inter-class cosine margin | 类别间文本原型间隔 |

先做诊断可以判断：

```text
64 模板是否已经高度一致；
vMF 收缩是否可能产生足够大的变化。
```

如果所有类别 \(\bar R_c\) 都接近 1，且 \(\mathbf z_c^{base}\) 与 \(\mathbf m_c\) 夹角很小，那么 vMF 收缩可能几乎不改变结果。

如果部分类别 \(\bar R_c\) 明显低，说明 E0-TPE-v1 有发挥空间。

### 10.3 再做 E0-TPE-v1-ZS

只替换文本原型，不使用 cache。

对比：

| 实验 | 文本原型 | cache |
|---|---|---|
| ZS-BASE | Point-Cache 64 模板归一化平均文本原型 | 否 |
| E0-TPE-v1-ZS | vMF 收缩式文本原型 | 否 |

观察：

- zero-shot accuracy；
- entropy；
- top-1/top-2 margin；
- class-wise accuracy；
- prediction changed rate；
- changed-and-correct rate；
- changed-and-wrong rate。

### 10.4 最后做 E0-TPE-v1 + E1-BASE

只替换 `clip_weights`，不改 cache update 规则。

对比：

| 实验 | 文本原型 | cache |
|---|---|---|
| E1-BASE | Point-Cache 64 模板归一化平均文本原型 | 原始 hierarchical cache |
| E0-TPE-v1 + E1 | vMF 收缩式文本原型 | 原始 hierarchical cache |

这样可以判断：

```text
文本原型聚合方式变化是否能改善原始 Point-Cache hierarchical baseline。
```

---

## 11. 代码实现要求

### 11.1 不能只对最终 `clip_weights` 做处理

因为最终 `clip_weights` 已经是平均后的文本原型，里面丢失了 64 个模板之间的分散信息。

E0-TPE-v1 必须在 `clip_classifier` 内部或替代函数中实现，保留：

```text
class_embeddings: [num_templates, dim]
```

也就是：

```python
class_embeddings = clip_model.encode_text(texts)
class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
```

之后立即计算：

```python
mean_vec = class_embeddings.mean(dim=0)
R_bar = mean_vec.norm()
m_c = mean_vec / mean_vec.norm()
z_base = class_embeddings[base_template_index]
```

再构造：

```python
kappa_hat = R_bar * (d - R_bar ** 2) / (1 - R_bar ** 2)
w = kappa_hat / (kappa_hat + kappa_0)

text_proto = (1 - w) * z_base + w * m_c
text_proto = text_proto / text_proto.norm()
```

### 11.2 新函数建议命名

建议新增：

```python
clip_classifier_tpe_v1(args, classnames, template, clip_model)
```

不要直接覆盖原始 `clip_classifier`，这样便于对比 baseline。

### 11.3 runner 命名建议

zero-shot 诊断：

```text
run_e0_tpe_v1_zs_modelnetc_suffix2_corruptions_dual_gpu.sh
```

Point-Cache hierarchical：

```text
run_e0_tpe_v1_hierarchical_modelnetc_suffix2_corruptions_dual_gpu.sh
```

all35：

```text
run_e0_tpe_v1_hierarchical_modelnetc_all35_corruptions_dual_gpu.sh
```

---

## 12. 文档中需要删除或改写的旧说法

### 12.1 删除

```text
原始 Point-Cache 只使用一个固定模板。
```

### 12.2 改成

```text
Point-Cache 主文使用单模板作为公式和流程示意，但源码实际实现使用 64 个手工文本模板，并对每个模板编码后的文本特征进行归一化、平均、再归一化，得到每个类别的文本原型。
```

### 12.3 删除

```text
E0-TPE-v0 = 固定模板 prompt ensemble。
```

### 12.4 改成

```text
E0-TPE-BASE = Point-Cache 已有的 64 模板归一化平均文本原型。
E0-TPE-v1 = 在同一组 64 模板的编码特征上进行 vMF 收缩式文本原型构造。
```

### 12.5 删除或弱化

```text
高斯假设不成立。
```

### 12.6 改成

```text
欧氏 Gaussian 建模不是绝对错误，但当实际推理使用 L2-normalized embeddings 和 cosine similarity 时，单位超球面上的方向统计与该推理几何更加一致。
```

---

## 13. 更正后的最终结论

更正后，E0-TPE 的定位不是：

```text
从单模板升级到多模板。
```

而是：

```text
在 Point-Cache 已有 64 模板文本原型的基础上，利用每个类别 64 个模板 embedding 的方向一致性，构建类别自适应的 vMF 收缩式文本原型。
```

因此，下一步不是做普通 prompt ensemble，而是做：

```text
E0-TPE-DIAG：先统计 64 模板 embedding 的方向分散程度；
E0-TPE-v1-ZS：只替换文本原型，测试 zero-shot；
E0-TPE-v1 + E1-BASE：替换文本原型后跑原始 hierarchical Point-Cache。
```

这条路线同时满足：

1. 与 Point-Cache 源码实现一致；
2. 与 BayesMM 的文本分布思想相关；
3. 不混入 DeepSeek 新变量；
4. 不直接改 cache 规则；
5. 能清楚判断收益是否来自文本原型处理。
