**E4-CANC: Confusion-Aware Negative Cache**

**实验 E4-CANC：混淆感知负缓存实验**

但 E4 第一版不要直接改 accuracy。我们先做一个诊断实验：

**E4-CANC-DIAG: Global-Local Conflict Diagnosis**

它不改变模型结果，只统计：

1.  每个 corruption 里 global-local 冲突比例是多少； 
2.  top-1/top-2 margin 小的边界样本有多少； 
3.  这些样本是否集中出现在 local corruption； 
4.  现有 negative cache 是否覆盖了这些冲突样本。 

这样可以先验证 E4 的前提是否成立。



## 四、E4 的核心思想

E3 失败说明：

local evidence 不适合直接给 positive cache 加分\text{local evidence 不适合直接给 positive cache 加分}local evidence 不适合直接给 positive cache 加分

但如果 local evidence 和 global prediction 冲突，它仍然很有价值：

y^global≠y^local\hat{y}_{global} \ne \hat{y}_{local}y^global=y^local

这说明样本可能位于类别边界，或者全局判断和局部结构不一致。

E4 的目标是把这种样本放进 negative / boundary cache，而不是 positive cache。





## 五、E4 初始数学定义

global prediction：

y^g=arg⁡max⁡kpg(k∣x)\hat{y}_g = \arg\max_k p_g(k \mid x)y^g=argkmaxpg(k∣x)

local prediction：

y^l=arg⁡max⁡kpl(k∣x)\hat{y}_l = \arg\max_k p_l(k \mid x)y^l=argkmaxpl(k∣x)

global margin：

Mg(x)=pg(1)(x)−pg(2)(x)M_g(x) = p_g^{(1)}(x) - p_g^{(2)}(x)Mg(x)=pg(1)(x)−pg(2)(x)

global-local conflict score：

Dgl(x)=[max⁡k≠y^gpl(k∣x)−pl(y^g∣x)]+D_{gl}(x) = \left[ \max_{k \ne \hat{y}_g} p_l(k \mid x) - p_l(\hat{y}_g \mid x) \right]_+Dgl(x)=[k=y^gmaxpl(k∣x)−pl(y^g∣x)]+

如果：

Dgl(x)>τdD_{gl}(x) > \tau_dDgl(x)>τd

说明 local evidence 明确不支持 global prediction。

E4-CANC-v0 的候选 negative cache 条件可以写成：

Ineg(x)=I[τl<Hg(x)<τu]∨I[Mg(x)<τm]∨I[Dgl(x)>τd]\mathbb{I}_{neg}(x) = \mathbb{I}[\tau_l < H_g(x) < \tau_u] \lor \mathbb{I}[M_g(x)<\tau_m] \lor \mathbb{I}[D_{gl}(x)>\tau_d]Ineg(x)=I[τl<Hg(x)<τu]∨I[Mg(x)<τm]∨I[Dgl(x)>τd]

也就是三类样本进入 negative / boundary cache：

1.  原始 Point-Cache 的中等熵 negative 样本； 
2.  top-1/top-2 很接近的边界样本； 
3.  global-local 明显冲突的样本。 





## 六、E4 第一阶段不要直接改方法

先做诊断版：



```
E4-CANC-DIAG
```



新增文件命名：



```
Point-Cache/runners/model_with_hierarchical_caches_e4_canc_diag.py
```



脚本命名：



```
Point-Cache/scripts/recur-pc/run_e4_canc_diag_hierarchical_modelnetc_all_corruptions_dual_gpu.sh
```



这个版本只统计，不改变 final logits，不改变 cache update。

输出 summary 里要有：

| corruption | accuracy | conflict_rate | boundary_rate | negative_candidate_rate |
| ---------- | -------- | ------------- | ------------- | ----------------------- |
|            |          |               |               |                         |

如果诊断结果显示 local corruption 里 conflict_rate 明显更高，那么 E4-CANC 就有充分理由继续做。







## 当前阶段总结

我们已经完成了一个很重要的阶段：

### 已排除的方向

positive cache admission with entropy-margin / GLC\text{positive cache admission with entropy-margin / GLC}positive cache admission with entropy-margin / GLC

实验表明：



```
E2-EMR、E3-GLC-v0、E3-GLC-v1 都没有超过原始 hierarchical baseline。
```



所以不再继续把 GLC 用作 positive cache promotion。

### 已验证的新方向

global-local conflict⇒wrong global pseudo-label diagnostic signal\text{global-local conflict} \Rightarrow \text{wrong global pseudo-label diagnostic signal}global-local conflict⇒wrong global pseudo-label diagnostic signal

E4-CANC-DIAG-v2 证明：



```
global-local conflict 在 7 个 corruption 上都显著提高 global pseudo-label 出错概率。
```



但同时：



```
local alternative class 正确率低。
```



所以后续方法应该是：

y^g suppression\hat y_g \text{ suppression}y^g suppression

而不是：

kl⋆ positive correctionk_l^\star \text{ positive correction}kl⋆ positive correction







这里有三个重要发现。

第一，`rotate` 和 `scale` 仍然很稳。说明几何整体变换下，Point-Cache / ULIP-2 本身比较鲁棒。

第二，`add_local` 和 `dropout_local` 的 conflict rate 明显高于 global corruption，说明 E4 的 global-local conflict 假设是合理的：

local corruption⇒more global-local conflict\text{local corruption} \Rightarrow \text{more global-local conflict}local corruption⇒more global-local conflict

第三，`jitter` 仍然是最大瓶颈。它的平均只有：

34.286934.286934.2869

而且 conflict rate 最高：

13.387413.387413.3874

但 extra negative 反而最低：

1.81521.81521.8152

这说明 jitter 下虽然冲突很多，但当前 v1 规则没有有效利用这些冲突。很可能是 jitter 让模型置信度整体下降，导致很多 conflict 样本过不了：

pg(1)>τpp_g^{(1)}>\tau_ppg(1)>τp

这个门槛。

------

## 4. suffix `_2` 与之前 7 corruption 对比

之前我们一直跑的是本地后缀 `_2`，现在它对应 paper severity 3。

在 `_2` 这 7 个 corruption 上，E4-CANC-v1 平均是：

62.850262.850262.8502

之前对比是：

| Method     | suffix `_2` Avg |
| ---------- | --------------- |
| E1-BASE    | 62.81           |
| E4-CANC-v0 | 62.83           |
| E4-CANC-v1 | 62.85           |

所以在这个固定 setting 上，v1 仍然是目前最高的，但提升非常小：

62.85−62.81=+0.0462.85 - 62.81 = +0.0462.85−62.81=+0.04

逐项看：

| Corruption       | E4-CANC-v1 |
| ---------------- | ---------- |
| add_global_2     | 68.0713    |
| add_local_2      | 61.4263    |
| dropout_global_2 | 73.2577    |
| dropout_local_2  | 63.5332    |
| rotate_2         | 73.5008    |
| scale_2          | 70.5835    |
| jitter_2         | 29.5786    |

所以 v1 不是完全没用，但目前的收益非常弱。

------

## 5. 我现在怎么看这个结果？

我觉得你“不甘心”仍然是合理的，因为这不是一个“理论假设失败”的结果。

E4-DIAG 已经证明：

Dgl⇒y^g likely wrongD_{gl} \Rightarrow \hat y_g \text{ likely wrong}Dgl⇒y^g likely wrong

这个诊断信号是强的。

但 E4-CANC-v1 的结果说明：

strong diagnostic signal⇏hard negative cache insertion gives strong accuracy gain\text{strong diagnostic signal} \not\Rightarrow \text{hard negative cache insertion gives strong accuracy gain}strong diagnostic signal⇒hard negative cache insertion gives strong accuracy gain

也就是说，问题可能不在 conflict signal，而在当前使用方式：



```
把 conflict 样本硬加入长期 negative cache
```



这个方式可能太粗糙。

它会带来两个抵消效应：

| 正向作用                     | 负向作用                                 |
| ---------------------------- | ---------------------------------------- |
| 抑制错误 global pseudo-label | 误抑制一部分原本正确的 global prediction |
| 增强 negative cache          | 引入 false negative pollution            |
| 对严重 corruption 有潜力     | 对轻度 corruption 可能过度干预           |

所以当前 v1 的状态可以总结为：

> E4-CANC-v1 成功增强了 negative intervention，但没有显著转化为 accuracy gain。

------

## 6. 当前最关键的问题

### 问题 1：新增 negative 样本比例仍然不高

平均 extra negative 只有：

2.77%2.77\%2.77%

而 conflict rate 是：

9.39%9.39\%9.39%

说明大量 conflict 样本仍然没有真正进入 E4 negative cache。

### 问题 2：jitter 中 conflict 很多，但利用很少

jitter：

conflict_rate=13.39%conflict\_rate=13.39\%conflict_rate=13.39%

但：

extra_negative=1.82%extra\_negative=1.82\%extra_negative=1.82%

这说明 v1 对最困难的 jitter 没有充分干预。

### 问题 3：hard cache insertion 可能不是最合适的形式

negative cache 一旦写入，会影响后续测试样本。如果新增样本里混入 false negative，它的影响不是一次性的，而是持续污染。

------

## 7. 下一步我建议不要马上继续 v1.2 硬加 cache

我建议分两步。

第一步，先补一个公平基线：

# E1-BASE all35

因为现在只有 E4-CANC-v1 all35。没有 E1-BASE all35，我们不能正式判断 v1 all35 是否比原始 Point-Cache 好。

第二步，开始设计：

# E4-CANC-v2: Soft Conflict Suppression

不要把 conflict 样本写进长期 negative cache，而是对当前样本的 global predicted class 做软抑制：

zy^g′=zy^g−α⋅Dgl(x)z'_{\hat y_g} = z_{\hat y_g} - \alpha \cdot D_{gl}(x)zy^g′=zy^g−α⋅Dgl(x)

或者更平滑：

zy^g′=zy^g−α⋅σ(γ(Dgl(x)−τd))z'_{\hat y_g} = z_{\hat y_g} - \alpha \cdot \sigma(\gamma(D_{gl}(x)-\tau_d))zy^g′=zy^g−α⋅σ(γ(Dgl(x)−τd))

它的好处是：

| Hard negative cache   | Soft conflict suppression |
| --------------------- | ------------------------- |
| 会长期影响后续样本    | 主要影响当前样本          |
| 容易 cache pollution  | 不污染 cache              |
| 二值决策              | 连续强度                  |
| false negative 风险大 | 风险更可控                |
| 受 cache 容量影响     | 不依赖 cache 容量         |

这条路更符合我们现在的现象：conflict 是强诊断信号，但硬写 cache 不够高效。

------

## 8. 现在最合理的下阶段路线

### 先做 E1-BASE all35

这是必须的。没有它，all35 结果无法写进论文主表。

然后根据 E1-BASE all35 对比：

| 情况                   | 下一步                                |
| ---------------------- | ------------------------------------- |
| v1 明显高于 baseline   | 做 v1 参数扫描                        |
| v1 只小幅高于 baseline | 做 v2 soft suppression                |
| v1 低于 baseline       | 停止 hard insertion，直接做 v2        |
| jitter 拖累严重        | 单独分析 jitter，不要让它误导整体结论 |







> E4-CANC-v1 completes the corrected all35 evaluation with all 35 corruptions successfully processed. It increases the effective negative-cache intervention compared with v0, but its hard negative insertion mechanism does not yet show a clearly strong accuracy gain. The result suggests that global-local conflict is better exploited as a soft suppression signal rather than a hard long-term cache insertion rule.