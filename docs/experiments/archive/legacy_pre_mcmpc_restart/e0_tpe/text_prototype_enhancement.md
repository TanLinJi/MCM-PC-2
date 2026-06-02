### Experiment 0 - Text Prototype Enhancement（E0-TPE）

`文本原型增强实验`

### 1 E0实验在整个实验中的位置

| 编号    | 名称                          | 中文            | 作用                              |
| ------- | ----------------------------- | --------------- | --------------------------------- |
| E0-TPE  | Text Prototype Enhancement    | 文本原型增强    | 改进文本类别锚点                  |
| E1-BASE | Baseline Reproduction         | 基线复现        | 复现原始 Point-Cache              |
| E2-EMR  | Entropy-Margin Reliability    | 熵-间隔可靠性   | 判断 global pseudo-label 是否可靠 |
| E3-GLC  | Global-Local Consistency      | 全局-局部一致性 | 判断 local 是否支持 global        |
| E4-CANC | Conflict-Aware Negative Cache | 冲突感知负缓存  | 抑制可疑 global pseudo-label      |
| E5-MCM  | Multi-Cache Matrix            | 多缓存矩阵      | 整合完整方法                      |



vMF 分布是用来描述单位球面上方向集中程度的。



$\bar{R}_c$: 类别 c 的平均合向量长度，表示的是 类别 c 的 64 个模板编码后的文本特征方向有多一致。

| 情况                 | $\bar R_c$ | 含义               |
| -------------------- | ---------- | ------------------ |
| 所有模板方向完全一致 | 接近 1     | 文本原型很稳定     |
| 模板方向有一定分散   | 中等       | 文本原型有不确定性 |
| 模板方向高度分散     | 接近 0     | 平均方向不太可靠   |







----

## 2. 如果要“完全不同”，应该怎么做？

真正不同的方向是：

> 不再把 64 个模板 embedding 先压缩成一个平均文本原型，而是在分类时保留 64 个模板的分布信息。

也就是说，Point-Cache 原来是：

zc(1),…,zc(64)→mc→sc(x)=x⊤mc\mathbf z_c^{(1)},\dots,\mathbf z_c^{(64)} \rightarrow \mathbf m_c \rightarrow s_c(x)=\mathbf x^\top \mathbf m_czc(1),…,zc(64)→mc→sc(x)=x⊤mc

新的方法可以改成：

sc(x)=Agg⁡m=164(x⊤zc(m))s_c(x) = \operatorname{Agg}_{m=1}^{64} \left( \mathbf x^\top \mathbf z_c^{(m)} \right)sc(x)=Aggm=164(x⊤zc(m))

也就是：

> 对每个类别，不再先平均模板特征，而是让点云特征分别和 64 个模板特征计算相似度，再对这 64 个相似度做聚合。

这才是和原 Point-Cache 文本原型构造不同的处理方式。







> We use the mean resultant length of the 64 prompt embeddings as a lightweight consistency score. When prompt embeddings are directionally consistent, the original Point-Cache averaged text prototype is preserved; otherwise, the prototype is softly shrunk toward a base 3D-aware prompt.







## 4. 后续文档中统一这样写

实验名称：



```
E0-TPE-v1-lite
```



英文全称：



```
Lightweight Text Prototype Shrinkage
```



中文名称：



```
轻量文本原型收缩
```



论文方法小节可以叫：



```
Lightweight Text Prototype Calibration
```



中文仍然写：



```
轻量文本原型校准
```



核心表述：

> We use the mean resultant length of the 64 prompt embeddings as a lightweight consistency score. When prompt embeddings are directionally consistent, the original Point-Cache averaged text prototype is preserved; otherwise, the prototype is softly shrunk toward a base 3D-aware prompt.

中文解释：

> 我们使用 64 个模板 embedding 的平均合向量长度作为轻量一致性分数。当模板方向一致时，保留 Point-Cache 原有的平均文本原型；当模板方向分散时，将文本原型轻微收缩回基础 3D prompt 方向。