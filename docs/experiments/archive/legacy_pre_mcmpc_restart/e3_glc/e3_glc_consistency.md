这一步我们新增两个文件，不覆盖已有实验：

```
Point-Cache/runners/model_with_hierarchical_caches_reliability_v2_glc.py
Point-Cache/scripts/recur-pc/run_reliability_v2_glc_hierarchical_modelnetc_all_corruptions_gpu1.sh
```

实验名：

```
E3-GLC: Global-Local Consistency Reliability Cache Admission
```

核心公式：
$$
R(x)=−H(x)+λmM(x)+λcCglobal-local(x)R(x) = -H(x) + \lambda_m M(x) + \lambda_c C_{\text{global-local}}(x)R(x)=−H(x)+λmM(x)+λcCglobal-local(x)
$$
其中：
$$
M(x)=p(1)(x)−p(2)(x)M(x)=p_{(1)}(x)-p_{(2)}(x)M(x)=p(1)(x)−p(2)(x)
$$

$$
C_{global-local(x)}=plocal(y^global)C_{\text{global-local}}(x)=p_{\text{local}}(\hat{y}_{\text{global}})Cglobal-local(x)=plocal(y^global)
$$

意思是：全局预测认为样本属于某类时，局部 part 的文本相似性也应该支持这个类别。







Experiment: E3-GLC-v0
Full name: Naive Global-Local Consistency Reliability Cache Admission
Result:
- Average accuracy: 62.50
- Delta vs Baseline: -0.31
- Delta vs E2-EMR: -0.01

Observation:
- Improves add_local_2 over E2-EMR by +0.61.
- Improves rotate_2 over E2-EMR by +0.37.
- Does not improve overall average.
Next:
- Replace local support with agreement-aware local margin.





> E3-GLC-v0 successfully integrates local part evidence into cache admission, but naive local support averaging does not improve the overall average accuracy. It slightly improves add_local_2 and rotate_2 compared with E2-EMR, suggesting that global-local consistency is useful but requires a more discriminative formulation.





> E3-GLC-v1 fixes the mathematical risk of additive local consistency by using a non-expansive gate. However, it still fails to improve the overall average accuracy, suggesting that local part evidence is not reliable enough to directly control positive cache admission under local corruptions.

它说明我们当前这条 **global-local consistency 作为 positive cache admission score** 的路线并不稳。

E3-v1 的确修复了 v0 的数学问题：

R(x)=Sg(x)⋅Ggl(x)R(x)=S_g(x)\cdot G_{gl}(x)R(x)=Sg(x)⋅Ggl(x)

0≤R(x)≤Sg(x)0\le R(x)\le S_g(x)0≤R(x)≤Sg(x)

但实验上没有提升平均值，说明问题不只是公式形式，而是更根本：

> 当前 local evidence 本身可能不适合直接参与 positive cache admission 排序。

也就是说，local part evidence 在某些 corruption 下不是可靠信号，尤其是：

| Corruption      | E3-v1 相比 baseline |
| --------------- | ------------------- |
| add_local_2     | -0.85               |
| dropout_local_2 | -0.64               |
| scale_2         | -1.05               |
| rotate_2        | -0.24               |

最讽刺的是，我们本来希望 GLC 改善 local corruption，但它对 `add_local_2` 和 `dropout_local_2` 仍然低于 baseline。这说明：

> local corruption 下，local part 特征本身被污染，用它做 positive cache 准入反而可能引入错误筛选。







我建议：**停止把 GLC 作为 positive cache admission 主项。**

但不要完全丢掉 GLC。它可以换一个位置使用。

当前实验已经证明：

| 用法                              | 结果       |
| --------------------------------- | ---------- |
| GLC 作为加法奖励                  | 不稳       |
| GLC 作为 non-expansive gate       | 仍不提升   |
| GLC 用于 positive cache admission | 暂时不适合 |

所以后面更合理的用法是：

> GLC 不用于决定“谁进入 positive cache”，而用于识别 global-local 冲突样本，把它们送入 boundary / negative cache。

这就转向我们之前的另一个主创新：

**Confusion-Aware Negative Cache**