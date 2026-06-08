# E4：类别概率分布引导的 GPA-Cache 净化

E4 的目标是：在保留 Point-Cache / GPA-Cache 框架的前提下，引入 BayesMM 启发的“类别概率分布”思想，用来维护更干净的 Cache。

E4 不是复现 BayesMM，也不是替代 Cache。

E4 要做的是：

    继续保留 Global Entropy Cache；
    继续保留 GPA-Cache；
    继续保留 GPA-local-cache；
    继续保留 Negative Cache；
    最终预测公式暂时不改；
    只改变 GPA-Cache 的样本准入和替换判断标准。

E3 的判断标准是：

    新样本是否更靠近类别原型中心。

E4 的判断标准是：

    新样本是否更符合该类别的概率分布。

## E4-A：视觉类别分布引导的 GPA-Cache

当前已经完成的第一组实验属于 E4-A。

E4-A 的准确定义是：

    基于 E3-V2-C；
    沿用 E3-V2-C 的未满直接初始化；
    沿用 E3-V2-C 的替换最高熵样本；
    保留低熵门控；
    将“距离视觉单中心更近”替换为“更符合视觉类别分布”；
    分布只由进入 GPA-Cache 的测试视觉全局特征在线估计；
    暂不引入文本 prompt 分布；
    最终预测公式暂时不改。

E4-A 的分布形式是最小版对角高斯式统计：

    visual_dist[c] = {
        count_c,
        mean_c,
        var_c
    }

分布符合度为：

    score(x, c) = - mean((x - mean_c)^2 / (var_c + eps))

这个分数只用于 GPA-Cache 的准入/替换判断，不是最终分类概率。

## 当前命名约定

E4-A 代码：

    /root/autodl-tmp/MCM-PC-2/Point-Cache/runners/E4_distribution_guided_cache/

E4-A 脚本：

    /root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache/

规范化后的 E4-A 新结果目录名应包含：

    e4_a_distribution_guided_gpa

旧首跑结果目录未删除，旧目录名为：

    /root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/00_1_ulip_modelnetc_s2_zs_global_local_distribution_guided_gpa_manual_full

该旧目录仍属于 E4-A，只是命名中没有显式写出 e4_a。

## 与 E4-B 的边界

E4-A 只建立视觉测试特征分布，没有建立文本 prompt 分布。

如果继续做 E4-B，建议定义为：

    E4-B：Text-Visual Distribution Guided Cache

核心区别是：

    text_dist[c]：
        由该类多条 prompt embedding 构成固定文本分布；

    visual_dist[c]：
        由当前 Global Entropy Cache[c] 与 GPA-Cache[c] 的并集构成动态视觉分布；
        只使用此时此刻仍在缓存中的样本，不累加已经被替换淘汰的历史样本；

    joint_score(x, c)：
        同时考虑视觉分布符合度和文本分布一致性。

也就是说，E4-B 不再把 text mean 直接混进 visual mean，而是把文本端也作为分布先验来约束 cache 更新。

## E4-B：文本-视觉类别分布引导的 GPA-Cache

E4-B 已按以下边界创建：

    代码：
        /root/autodl-tmp/MCM-PC-2/Point-Cache/runners/E4_distribution_guided_cache/model_e4_b_text_visual_distribution_guided_gpa.py
        /root/autodl-tmp/MCM-PC-2/Point-Cache/runners/E4_distribution_guided_cache/run_e4_b_ulip_modelnetc_s2_text_visual_distribution_guided_gpa.py

    脚本：
        /root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache/01_1_ulip_modelnetc_s2_zs_global_local_e4_b_text_visual_distribution_guided_gpa_manual_full.sh

    结果目录命名：
        01_1_ulip_modelnetc_s2_zs_global_local_e4_b_text_visual_distribution_guided_gpa_manual_full

E4-B 的视觉分布不是历史累计分布，而是按需从当前缓存重建：

```text
visual_dist[c] = Dist(unique(current EntropyCache[c] ∪ current GPACache[c]))
```

这样 cache 替换后，旧样本自然不再参与分布估计。

E4-B 的文本分布来自每类 prompt-level embeddings：

```text
text_dist[c] = Dist(prompt_embeddings[c])
```

当前 `manual_full` 版本中，每个类别使用 64 条手工 prompt 的文本特征构建固定文本分布。该分布在测试期间不更新。

满缓存后的替换规则为：

```text
entropy(x_new) < entropy(x_old)
and
joint_score(x_new, c) > joint_score(x_old, c)

joint_score = visual_score + E4_TEXT_SCORE_WEIGHT * text_score
```

默认 `E4_TEXT_SCORE_WEIGHT=0.1`。最终 logits、Negative Cache 和 Point-Cache 主预测公式暂时不改。

## E4-C：可信历史累计文本-视觉分布引导的 GPA-Cache

E4-C 是在 E4-B 首跑发现 `add_global_2` 大幅下降后创建的新分支。它不修改 E4-B，而是作为独立对照：

```text
E4-B:
    visual_dist[c] = Dist(unique(current EntropyCache[c] ∪ current GPACache[c]))
    只使用当前缓存截面。

E4-C:
    visual_dist[c] = Dist(history(samples accepted by EntropyCache or GPACache))
    累计曾被正缓存接受过的历史可信样本。
```

E4-C 的核心判断是：

    被正缓存接受过的样本是相对可信的；
    即使后续因为容量限制被淘汰，它仍可作为类别分布的历史证据；
    但完全没有资格进入正缓存的候选样本，不能参与构建分布。

因此 E4-C 使用 accepted-history visual distribution：

```text
if sample accepted by GPACache:
    update visual_dist[c]

if sample accepted by EntropyCache:
    update visual_dist[c]

if sample rejected by both:
    do not update visual_dist[c]
```

同一个样本若同时被 GPACache 和 EntropyCache 接受，只计一次。

E4-C 代码：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/runners/E4_distribution_guided_cache/model_e4_c_accepted_history_text_visual_distribution_guided_gpa.py
/root/autodl-tmp/MCM-PC-2/Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_ulip_modelnetc_s2_accepted_history_text_visual_distribution_guided_gpa.py
```

E4-C 脚本：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache/02_1_ulip_modelnetc_s2_zs_global_local_e4_c_accepted_history_text_visual_distribution_guided_gpa_manual_full.sh
```

E4-C 结果目录命名：

```text
02_1_ulip_modelnetc_s2_zs_global_local_e4_c_accepted_history_text_visual_distribution_guided_gpa_manual_full
```
