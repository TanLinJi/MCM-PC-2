# E4-A 实验计划：视觉类别概率分布引导的 GPA-Cache 净化

## 1. 背景

E3 阶段的实验发现：

    E3-V2-C 是当前 E3 阶段最好的版本，平均准确率 54.04。
    E3-V3 的候选池初始化方法没有超过 E3-V2-C。
    E3 的多种改进，本质上仍然属于“改进单中心原型”的方法。
    这类方法对 add_global 和 add_local 这类添加型外点噪声有效。
    但面对 dropout、rotate、scale、jitter 等几何结构变化时不稳定。

这个发现很重要：

    单中心原型方法可以过滤外点噪声；
    但它无法判断一个远离中心的样本到底是坏样本，还是该类别在几何变化下的正常样本。

因此，E4 引入类别概率分布。当前第一版只引入视觉类别分布，命名为 E4-A。

## 2. E4 的核心思想

E4 不再只问：

    样本离类别中心近不近？

而是问：

    样本是否符合这个类别允许的变化范围？

也就是说，每个类别不只维护一个中心，还维护一个分布。

第一版 E4-A 使用最简单的视觉类别对角分布：

    mean_c：
        类别 c 的平均特征。

    var_c：
        类别 c 每个特征维度的变化范围。

    count_c：
        类别 c 已经用于分布统计的样本数量。

注意：E4-A 的分布只来自测试阶段进入 GPA-Cache 的视觉全局特征，不包含文本 prompt 分布。

## 3. 为什么沿用 E3-V2-C

E3-V2-C 是当前 E3 最优版本，所以 E4-A 不再继续改初始化方式。

E4-A 沿用：

    GPA-Cache 未满时：
        新样本直接进入 GPA-Cache；
        对应 local patch features 进入 GPA-local-cache。

    GPA-Cache 满后：
        替换对象仍然是 GPA-Cache 中最高熵样本；
        仍然要求新样本熵更低。

E4-A 改变的是第二个判断条件：

    E3-V2-C：
        新样本距离中心更近。

    E4-A：
        新样本更符合类别分布。

## 4. E4-A 更新规则

对于新样本 x，zero-shot 预测类别为 c。

如果 GPA-Cache[c] 未满：

    1. x 直接进入 GPA-Cache[c]；
    2. x 的 local patch features 进入 GPA-local-cache[c]；
    3. 用 x 更新类别分布 Dist[c]。

如果 GPA-Cache[c] 已满：

    1. 找到 GPA-Cache[c] 中最高熵样本 x_high；
    2. 计算新样本分布符合度 prob_score(x, c)；
    3. 计算最高熵样本分布符合度 prob_score(x_high, c)；
    4. 如果：

           entropy(x) < entropy(x_high)
           and
           prob_score(x, c) > prob_score(x_high, c)

       则：

           用 x 替换 x_high；
           同步替换 GPA-local-cache；
           更新类别分布 Dist[c]。

## 5. 分布符合度

E4-A 使用一个简单的分布分数：

    score(x, c) = - mean((x - mean_c)^2 / (var_c + eps))

这个分数越大，说明样本越符合类别 c 的分布。

这个分数不是最终分类概率，只是 Cache 更新时使用的判断指标。

## 6. E4-A 与 E3-V2-C 的区别

| 对比项 | E3-V2-C | E4-A |
|---|---|---|
| Cache 是否保留 | 是 | 是 |
| 初始化方式 | 未满直接进入 | 沿用 |
| 替换对象 | 最高熵样本 | 沿用 |
| 低熵门控 | 有 | 有 |
| 第二判断条件 | 距离中心更近 | 更符合类别分布 |
| 类别建模方式 | 单中心 | 概率分布 |
| 最终预测公式 | 原 Point-Cache 公式 | 暂时不改 |

## 7. 预期观察

如果 E4-A 在 dropout、rotate、scale、jitter 上优于 E3-V2-C，说明类别分布确实比单中心更适合几何变化。

如果 E4-A 仍然只在 add_global、add_local 上有效，说明单个类别分布仍然不足，后续可能需要多分布或多中心。

如果 E4-A 低于 E3-V2-C，说明分布估计可能不稳定，需要检查方差、更新门控和伪标签污染。

## 8. 当前首跑结果

当前旧命名首跑目录：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/00_1_ulip_modelnetc_s2_zs_global_local_distribution_guided_gpa_manual_full
```

该目录属于 E4-A，但目录名未显式包含 `e4_a`。规范化后新脚本会输出到包含 `e4_a_distribution_guided_gpa` 的目录。

首跑结果为 ULIP × ModelNet-C severity=2 × 7 corruptions：

| 方法 | 平均 Acc |
|---|---:|
| E2 full Point-Cache baseline | 54.0000 |
| E3-V2-C visual union center | 54.0414 |
| E4-A visual distribution guided | 53.2171 |

E4-A 相比 E3-V2-C：

| Corruption | Delta |
|---|---:|
| add_global_2 | +3.89 |
| add_local_2 | -2.64 |
| dropout_global_2 | -1.75 |
| dropout_local_2 | +0.44 |
| rotate_2 | -1.70 |
| scale_2 | -1.58 |
| jitter_2 | -2.43 |
| avg | -0.8243 |

这个结果说明：视觉单分布裁判在 add_global 上有强正收益，但整体显著低于 E3-V2-C。它没有解决 E3 在几何变化上的核心问题，尤其 rotate、scale、jitter 下降明显。

## 9. 规范化后的 E4-A 检查项

后续重跑 E4-A 时，需要重点检查：

1. 结果目录名是否包含 `e4_a_distribution_guided_gpa`；
2. `gpa_stats/` 是否正常生成；
3. `class_distribution_summary` 中各类别 `count/var_mean/var_min/var_max` 是否异常塌缩；
4. `gpa_replacement_events_*.jsonl` 中 `replace_distribution` 与 `reject_distribution` 的比例；
5. add_global 的收益是否来自分布裁判真实替换，而不是偶然 cache 路径差异。

## 10. E4-B 候选方向

如果继续做 E4-B，建议不要只调 E4-A 的 `eps/min_var`。

E4-B 应定义为：

```text
Text-Visual Distribution Guided Cache
```

核心变化：

1. 文本端不再只用 prompt 平均后的单点 `text mean`；
2. 每类多条 prompt embedding 形成固定 `text_dist[c]`；
3. 视觉端继续维护在线 `visual_dist[c]`；
4. GPA-Cache 替换分数由 `visual_score` 和 `text_distribution_score` 共同决定；
5. 文本分布作为语义先验，视觉分布作为当前 corruption 下的经验估计。

E4-B 的目标不是把文本均值和视觉均值线性混合，而是用分布一致性减少跨模态单点中心错配。

## 11. E4-B 当前实现方案

E4-B 已规范为：

```text
Text-Visual Distribution Guided GPA-Cache
```

它相对 E4-A 的核心变化是：

1. 文本端由 `clip_weights` 的类别均值单点，扩展为每类 prompt-level embedding 分布；
2. 视觉端不再使用 E4-A 的历史累加 Welford 分布；
3. 视觉分布改为从当前缓存状态动态重建：

```text
visual_dist[c] = Dist(unique(current EntropyCache[c] ∪ current GPACache[c]))
```

这条规则非常重要：

    cache 中被替换淘汰的旧样本不能继续参与视觉分布；
    同一个样本同时存在于 EntropyCache 和 GPACache 时不能重复计数；
    当前候选样本不能先进入 EntropyCache 后再参与自己的 GPA 替换评分。

因此 E4-B 的正缓存更新顺序调整为：

```text
1. get_logits 得到 x / pred / entropy；
2. 用当前旧的 EntropyCache[c] ∪ GPACache[c] 构建 visual_dist[c]；
3. 用 visual_dist[c] 与 text_dist[c] 计算 joint_score；
4. 更新 GPACache / GPA-local-cache；
5. 再按原 Point-Cache 规则更新 EntropyCache；
6. 用更新后的 cache 计算 final_logits。
```

未满阶段仍沿用 E3-V2-C / E4-A：

```text
如果 GPACache[c] 未满：
    直接加入 GPACache[c]
    同步加入 GPA-local-cache[c]
```

满缓存阶段：

```text
找到 GPACache[c] 中最高熵样本 x_old。

如果：
    entropy(x_new) < entropy(x_old)
    and
    joint_score(x_new, c) > joint_score(x_old, c)

则：
    用 x_new 替换 x_old
    同步替换 GPA-local-cache 中对应 local item
```

联合分数：

```text
visual_score(x, c) = -mean((x - visual_mean_c)^2 / (visual_var_c + eps_v))
text_score(x, c)   = -mean((x - text_mean_c)^2 / (text_var_c + eps_t))
joint_score(x, c)  = visual_score(x, c) + lambda_text * text_score(x, c)
```

默认参数：

```text
E4_DIST_EPS=1e-4
E4_DIST_MIN_VAR=1e-4
E4_TEXT_DIST_EPS=${E4_DIST_EPS}
E4_TEXT_DIST_MIN_VAR=${E4_DIST_MIN_VAR}
E4_TEXT_SCORE_WEIGHT=0.1
```

当前实现路径：

```text
Point-Cache/runners/E4_distribution_guided_cache/model_e4_b_text_visual_distribution_guided_gpa.py
Point-Cache/runners/E4_distribution_guided_cache/run_e4_b_ulip_modelnetc_s2_text_visual_distribution_guided_gpa.py
Point-Cache/scripts/E4_distribution_guided_cache/01_1_ulip_modelnetc_s2_zs_global_local_e4_b_text_visual_distribution_guided_gpa_manual_full.sh
```

## 12. E4-C 当前实现方案

E4-B 首跑 `add_global_2` 后发现准确率明显下降。事件统计显示，当前缓存截面分布在 `shot_capacity=3` 下过窄，导致大量低熵候选样本被 `visual_score` 拒绝。

因此 E4-C 不覆盖 E4-B，而是新增分支：

```text
E4-C：Accepted-History Text-Visual Distribution Guided GPA-Cache
```

E4-C 的核心思想：

```text
visual_dist[c] = Dist(history(samples accepted by EntropyCache or GPACache))
```

也就是说：

1. 样本只要曾经被 Global Entropy Cache 接受，就进入 `visual_dist[c]`；
2. 样本只要曾经被 GPA-Cache 接受，也进入 `visual_dist[c]`；
3. 同一个样本如果被两个正缓存同时接受，只计一次；
4. 被两个正缓存都拒绝的候选样本，不参与视觉分布；
5. 因容量限制后续从 cache 中淘汰的样本，仍保留在历史分布中。

这样做的目的：

    保留可信样本的多样性；
    避免 E4-B 当前缓存截面样本太少导致的分布过窄；
    仍然阻止低质量、未被正缓存接受的样本污染类别分布。

E4-C 仍沿用：

```text
GPA 未满：直接加入；
GPA 满后：替换最高熵样本；
替换门控：entropy(new) < entropy(old)；
第二准则：joint_score(new) > joint_score(old)；
最终 logits：不改；
Negative Cache：不改。
```

当前实现路径：

```text
Point-Cache/runners/E4_distribution_guided_cache/model_e4_c_accepted_history_text_visual_distribution_guided_gpa.py
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_ulip_modelnetc_s2_accepted_history_text_visual_distribution_guided_gpa.py
Point-Cache/scripts/E4_distribution_guided_cache/02_1_ulip_modelnetc_s2_zs_global_local_e4_c_accepted_history_text_visual_distribution_guided_gpa_manual_full.sh
```
