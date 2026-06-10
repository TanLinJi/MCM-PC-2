- 这里先定义：
  - `P1`：Probe 1，探针实验 1。Probe 的意思是“诊断性小实验”，不是正式主实验。
  - `feature`：特征向量。点云输入经过 OpenShape / PointBERT 编码器后，会变成一个高维向量，用来表示这个物体的语义。
  - `feature failure`：特征失效。意思是同一个物体被 corruption 损坏后，它的特征向量离原来的干净特征太远，导致模型认不出来。
  - `anchor`：锚点、参照点。可以理解成“模型判断当前样本时拿来对照的历史样本或类别原型”。
  - `anchor pollution`：锚点污染。意思是 cache 里保存的参照样本本身已经被错误预测污染了，后续样本再去参考这些错误锚点，就会把错误放大。
- jitter 是点坐标随机抖动
- regime 的意思是状态区间、工作区间、机制区域
- 所以 drift regime 可以翻译成：特征漂移状态区间



D = Decision，决策记录
P = Probe，探针实验
F = Finding，实验发现
G = Gap，论文漏洞 / 证据缺口
W = Week / Work milestone，阶段任务

```
D19 = 第 19 条重要决策
P1 = 第 1 个探针实验
F1 = 第 1 条关键发现
G1 = 第 1 个待修论文漏洞
W2 = 第 2 阶段工作
```

pivot 的意思是路线转向。D20 pivot 的意思是：基于 P1 结果，我们把方法路线从“ICP-CD 几何补偿”转向“conditional anchor switching”。



**conditional anchor switching** ：有条件地切换锚点来源



entropy(q)
= 当前预测的不确定性，越高表示越不确定

max_text_cos(q)
= 当前点云特征和最相似文本类别 anchor 的余弦相似度

top1_top2_margin(q)
= 最像的类别和第二像的类别之间的差距





## 实验

### 1. EMR = Entropy-Margin Reliability

中文可以叫：

> **熵-间隔可靠性**

对应实验：

```
E2-EMR
```

它的意思是：用 **entropy** 和 **margin** 一起判断一个样本是否可靠，能不能进入 positive cache。

其中：

Hg(x)H_g(x)Hg(x)

表示 global prediction 的 entropy，熵越低，说明模型越自信。

Mg(x)=pg(1)(x)−pg(2)(x)M_g(x)=p_g^{(1)}(x)-p_g^{(2)}(x)Mg(x)=pg(1)(x)−pg(2)(x)

表示 top-1 和 top-2 类别概率的差，也就是 margin。margin 越大，说明模型越不犹豫。

所以 E2-EMR 的核心思想是：

R(x)=−Hg(x)+λmMg(x)R(x)=-H_g(x)+\lambda_m M_g(x)R(x)=−Hg(x)+λmMg(x)

也就是：

> 熵低、top-1/top-2 差距大的样本，更可能是可靠样本。

它试图解决的问题是：

```
哪些样本适合进入 positive cache？
```

### 2. GLC = Global-Local Consistency

中文可以叫：

> **全局-局部一致性**

对应实验：



```
E3-GLC
```



它的意思是：不仅看 global prediction 是否自信，还看 local parts 是否支持 global prediction。

例如 global branch 认为这个点云是：



```
chair
```



如果 local part features 也支持：



```
chair
```



那说明 global-local 一致。

如果 global 说是：



```
chair
```



但 local parts 更像：



```
stool / table / sofa
```



那说明 global-local 不一致。

E3-GLC 的最初想法是：

> 如果 global 和 local 一致，这个样本可能更可靠，可以更放心地进入 positive cache。

所以它试图解决的问题仍然是：



```
哪些样本适合进入 positive cache？
```



但后面实验发现：把 GLC 用作 positive cache promotion 不稳定，所以我们没有继续把它作为主线。

------

### 3. CANC = Conflict-Aware Negative Cache

中文可以叫：

> **冲突感知负缓存**

对应实验：

```
E4-CANC
```

这里的 Conflict 指的是：

```
global-local conflict
```

也就是 global prediction 和 local evidence 发生冲突。

E4 和 E3 的关键区别是：

| 实验    | local 信息怎么用                                             |
| ------- | ------------------------------------------------------------ |
| E3-GLC  | local 支持 global 时，把样本当作更可靠的 positive cache      |
| E4-CANC | local 反对 global 时，认为 global pseudo-label 可能错，要 suppress 它 |

所以 CANC 的意思是：

> 如果 global 和 local 发生冲突，不把 local alternative 当正确答案，而是把 global predicted class 当成需要抑制的对象。

例如：

```
global prediction: chair
local evidence: 不支持 chair
```

E4-CANC 不会直接说：

```
那它一定是 stool
```

而是说：

```
chair 这个 global pseudo-label 可能不可靠，应该被 negative suppression
```

所以 E4-CANC 试图解决的问题是：

```
如何发现可能错误的 global pseudo-label，并避免它污染 cache？
```

最关键的阶段转变是：

EMR / GLC: 找可靠 positive samples\text{EMR / GLC: 找可靠 positive samples}EMR / GLC: 找可靠 positive samples

变成：

CANC: 找不可靠 global pseudo-label 并 suppress\text{CANC: 找不可靠 global pseudo-label 并 suppress}CANC: 找不可靠 global pseudo-label 并 suppress

也就是从：



```
怎么更好地相信模型？
```

转向：

```
什么时候不应该相信模型？
```



| 缩写 | 全称                          | 中文            | 核心作用                                                     |
| ---- | ----------------------------- | --------------- | ------------------------------------------------------------ |
| EMR  | Entropy-Margin Reliability    | 熵-间隔可靠性   | 用熵和 top-1/top-2 间隔判断 positive cache 样本可靠性        |
| GLC  | Global-Local Consistency      | 全局-局部一致性 | 用 global 和 local 是否一致判断 positive cache 样本可靠性    |
| CANC | Conflict-Aware Negative Cache | 冲突感知负缓存  | 用 global-local conflict 发现错误 global pseudo-label，并进行负向抑制 |



## 4 我们的 E4-CANC 属于这条路径

E4-CANC 的全称是：

Conflict-Aware Negative Cache\text{Conflict-Aware Negative Cache}Conflict-Aware Negative Cache

中文是：

> 冲突感知负缓存

它的核心是 global-local conflict：

Dglmean(x)=[max⁡k≠y^gplmean(k∣x)−plmean(y^g∣x)]+D_{gl}^{mean}(x) = \left[ \max_{k\ne \hat y_g}p_l^{mean}(k|x) - p_l^{mean}(\hat y_g|x) \right]_+Dglmean(x)=[k=y^gmaxplmean(k∣x)−plmean(y^g∣x)]+

这个公式的意思是：

-  global predicted class 是 y^g\hat y_gy^g； 

- local mean probability 对其他类别的最大支持是：

  max⁡k≠y^gplmean(k∣x)\max_{k\ne \hat y_g}p_l^{mean}(k|x)k=y^gmaxplmean(k∣x)

- local 对 global predicted class 的支持是：

  plmean(y^g∣x)p_l^{mean}(\hat y_g|x)plmean(y^g∣x)

-  如果 local 更支持别的类别，而不是 global 类别，那么 DglD_{gl}Dgl 就大。 

所以：

Dgl(x)>0D_{gl}(x)>0Dgl(x)>0

表示：



```
local evidence 反对 global prediction
```