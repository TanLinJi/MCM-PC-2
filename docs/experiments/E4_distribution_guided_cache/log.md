# E4 实验日志

## 2026-06-07：创建 E4-A 实验计划

E4 正式开始。

E4 的核心思想是：不复现 BayesMM，也不替代 Cache，而是在 Point-Cache / GPA-Cache 框架中引入类别概率分布，用“是否更符合类别分布”来维护更干净的 GPA-Cache 和 GPA-local-cache。

E4-A 第一版沿用 E3-V2-C 的初始化方式和替换最高熵样本规则，只把“距离中心更近”替换为“更符合类别分布”。

E4-A 的目标是验证：类别分布是否能缓解 E3 单中心原型方法在 dropout、rotate、scale、jitter 等几何结构变化上的失效问题。

## 2026-06-08：规范化 E4-A 命名、诊断与文档边界

对当前 E4 实验线进行规范化处理。

当前已经完成的实验明确归类为：

```text
E4-A：Visual Distribution-Guided GPA-Cache
```

E4-A 的定义：

1. 基于 E3-V2-C；
2. 沿用未满直接初始化；
3. 沿用最高熵样本作为替换对象；
4. 保留低熵门控；
5. 将“距离视觉单中心更近”替换为“更符合视觉类别分布”；
6. 分布只来自进入 GPA-Cache 的测试视觉全局特征；
7. 不包含文本 prompt 分布。

首跑旧结果目录为：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/00_1_ulip_modelnetc_s2_zs_global_local_distribution_guided_gpa_manual_full
```

该目录仍属于 E4-A，只是旧命名没有显式包含 `e4_a`。

首跑结果：

| 方法 | 平均 Acc |
|---|---:|
| E2 full Point-Cache baseline | 54.0000 |
| E3-V2-C visual union center | 54.0414 |
| E4-A visual distribution guided | 53.2171 |

观察：

1. E4-A 在 `add_global_2` 上有明显正收益，相比 E3-V2-C 为 `+3.89`；
2. 但在 `add_local_2/dropout_global_2/rotate_2/scale_2/jitter_2` 上下降；
3. 平均准确率相比 E3-V2-C 为 `-0.8243`；
4. 当前视觉单分布裁判不是稳定正收益方法。

本次规范化内容：

1. 新脚本输出目录名改为包含 `e4_a_distribution_guided_gpa`；
2. runner 写回 `baseline_exp_id` 与 `baseline_result_root`，便于 `gpa_stats/` 落盘；
3. runner 日志记录 `E4_DIST_EPS` 与 `E4_DIST_MIN_VAR`；
4. model 的 stats payload 增加 `e4_variant/distribution_scope/text_distribution_enabled`；
5. 文档中明确 E4-A 与后续 E4-B 的边界。

后续如果继续，应进入 E4-B：

```text
E4-B：Text-Visual Distribution Guided Cache
```

E4-B 要解决的问题是：E4-A 只把视觉端从单中心升级为分布，尚未把文本端从 prompt 平均单点升级为 prompt 分布。

## 2026-06-08：实现 E4-B 文本-视觉分布引导 GPA-Cache

根据新的设计确认，E4-B 不采用历史累加视觉分布，而是只使用当前缓存中的样本：

```text
visual_dist[c] = Dist(unique(current EntropyCache[c] ∪ current GPACache[c]))
```

这个规则保证：

1. 被 cache 替换淘汰的旧样本不会继续参与视觉分布；
2. 同一个样本同时存在于 EntropyCache 和 GPACache 时不会重复计数；
3. 当前候选样本不会通过先进入 EntropyCache 来参与自己的 GPA 替换评分。

为避免自我增强，E4-B 的正缓存更新顺序改为：

```text
先用旧 EntropyCache ∪ GPACache 评分并更新 GPACache；
再更新 EntropyCache；
最后用更新后的 cache 计算 final_logits。
```

新增代码：

```text
Point-Cache/runners/E4_distribution_guided_cache/model_e4_b_text_visual_distribution_guided_gpa.py
Point-Cache/runners/E4_distribution_guided_cache/run_e4_b_ulip_modelnetc_s2_text_visual_distribution_guided_gpa.py
```

新增脚本：

```text
Point-Cache/scripts/E4_distribution_guided_cache/01_run_e4_b_ulip_modelnetc_s2_common.sh
Point-Cache/scripts/E4_distribution_guided_cache/01_1_ulip_modelnetc_s2_zs_global_local_e4_b_text_visual_distribution_guided_gpa_manual_full.sh
```

默认实验目录：

```text
Point-Cache/results/E4_distribution_guided_cache/01_1_ulip_modelnetc_s2_zs_global_local_e4_b_text_visual_distribution_guided_gpa_manual_full
```

关键默认参数：

```text
E4_TEXT_SCORE_WEIGHT=0.1
E4_TEXT_DIST_EPS=${E4_DIST_EPS}
E4_TEXT_DIST_MIN_VAR=${E4_DIST_MIN_VAR}
```

本版仍保持最终预测公式、Global Entropy Cache logits、GPA-controlled Local Cache logits 和 Negative Cache 逻辑不变。

## 2026-06-08：新增 E4-C 可信历史累计分布分支

E4-B 首跑 `add_global_2` 时出现明显下降。对 `gpa_replacement_events_add_global_2.jsonl` 的初步检查显示：

```text
new_visual_score 明显低于 old_visual_score；
text_score 新旧差异很小；
主要问题来自 visual_dist 的当前缓存截面估计。
```

原因判断：

    当前 `EntropyCache[c] ∪ GPACache[c]` 在 `shot_capacity=3` 下样本很少；
    当前缓存截面分布过窄；
    新低熵样本容易被判定为不符合旧缓存小分布；
    导致 GPA-local-cache 更新不足。

因此新建 E4-C，而不是直接修改 E4-B。

E4-C 定义：

```text
E4-C：Accepted-History Text-Visual Distribution Guided GPA-Cache
```

核心规则：

```text
visual_dist[c] = history of samples accepted by EntropyCache or GPACache
```

具体约束：

1. 曾被 Global Entropy Cache 接受的样本进入 `visual_dist[c]`；
2. 曾被 GPA-Cache 接受的样本进入 `visual_dist[c]`；
3. 同一样本被两个正缓存接受时只计一次；
4. 被两个正缓存都拒绝的样本不进入分布；
5. 后续被 cache 容量机制淘汰的历史可信样本仍保留在分布中。

新增代码：

```text
Point-Cache/runners/E4_distribution_guided_cache/model_e4_c_accepted_history_text_visual_distribution_guided_gpa.py
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_ulip_modelnetc_s2_accepted_history_text_visual_distribution_guided_gpa.py
```

新增脚本：

```text
Point-Cache/scripts/E4_distribution_guided_cache/02_run_e4_c_ulip_modelnetc_s2_common.sh
Point-Cache/scripts/E4_distribution_guided_cache/02_1_ulip_modelnetc_s2_zs_global_local_e4_c_accepted_history_text_visual_distribution_guided_gpa_manual_full.sh
```

E4-B 保留为“当前缓存截面分布”对照，不再覆盖。
