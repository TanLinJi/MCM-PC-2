# 02_10_1：Distribution-Final Score

日期：2026-06-15
状态：已中止，单扰动结果显示该纯分布最终得分版本不成立

---

## 1. 实验目的

本实验基于当前最好版本 `02_9_2`：

```text
E4-C-A0+E1-textdist-only tw0.15
```

`02_9_2` 的核心特点是：

```text
分布得分只用于缓存替换；
最终预测仍然使用 zero-shot 文本原型点积 + 缓存投票。
```

本实验只改一个关键点：

```text
最终得分全部改成分布得分。
```

也就是说，最终分类不再直接使用：

```text
clip_logits
global entropy cache voting logits
GPA local cache voting logits
negative cache voting logits
```

而是使用：

```text
文本分布得分（text distribution score）
视觉历史分布得分（accepted-history visual distribution score）
```

其中：

1. `clip_logits` 指零样本文本原型点积得分（zero-shot text prototype dot-product logits）。
2. `global entropy cache voting logits` 指全局熵缓存投票得分（global entropy cache voting logits）。
3. `GPA local cache voting logits` 指 GPA 局部缓存投票得分（GPA local cache voting logits）。
4. `negative cache voting logits` 指负缓存投票得分（negative cache voting logits）。

---

## 2. 与 02_9_2 的关系

保持不变：

1. backbone 仍为 ULIP。
2. 数据仍为 ModelNet-C severity=2。
3. 文本分布仍来自 E1 LLM prompt cache。
4. GPA cache 的替换规则仍使用 accepted-history text-visual joint score。
5. Global entropy cache 和 GPA-controlled local cache 仍照常维护，用于建立视觉历史分布。

改变：

1. 最终预测不再使用缓存投票。
2. 最终预测不再直接使用 zero-shot 原型点积 logits。
3. 最终预测改为每类分布得分的归一化融合。

---

## 3. 分布得分计算

### 3.1 文本分布得分

对每个类别 `c`，E1 的多个文本提示经过文本编码器后形成：

```text
μ_text(c), σ_text^2(c)
```

当前点云特征为 `f`，则文本分布得分为：

```text
S_text_dist(c) = -mean((f - μ_text(c))^2 / (σ_text^2(c) + eps))
```

得分越高，表示当前点云特征越符合该类别的文本提示分布。

### 3.2 视觉历史分布得分

视觉历史分布来自 02_9_2 中已经存在的 accepted-history visual distribution。

只有成功进入或替换以下正缓存的样本才会更新视觉分布：

```text
Global entropy cache
GPA cache
```

对每个类别 `c`，视觉历史样本形成：

```text
μ_visual(c), σ_visual^2(c)
```

视觉分布得分为：

```text
S_visual_dist(c) = -mean((f - μ_visual(c))^2 / (σ_visual^2(c) + eps))
```

### 3.3 分数归一化

文本分布得分和视觉分布得分数值尺度不同，不能直接相加。

本实验第一版使用样本内 z-score：

```text
norm(S) = (S - mean(S)) / (std(S) + eps)
```

最终得分：

```text
S_final(c) =
    w_text * norm(S_text_dist(c))
  + w_visual * norm(S_visual_dist(c))
```

默认：

```text
w_text = 1.0
w_visual = 1.0
```

---

## 4. 重要边界

本实验仍是免训练测试时适应（training-free TTA）：

1. 不更新点云编码器。
2. 不更新文本编码器。
3. 不反向传播。
4. 真实标签只用于离线结果统计，不参与任何测试时决策。

为了和 `02_9_2` 保持可比，本实验第一版保留 `02_9_2` 的测试时更新顺序：

```text
先更新缓存/分布；
再用当前分布计算最终得分。
```

如果结果显示明显的当前样本自增强问题，后续需要做一个 old-distribution scoring 消融：

```text
先用旧分布打分；
再更新缓存/分布。
```

---

## 5. 实验参数

```text
E4_TEXT_SCORE_WEIGHT = 0.15
E4_SCORE_NORM_MODE = running_zscore
E4_FINAL_TEXT_DIST_WEIGHT = 1.0
E4_FINAL_VISUAL_DIST_WEIGHT = 1.0
E4_FINAL_DIST_NORM_MODE = per_sample_zscore
```

注意：`E4_TEXT_SCORE_WEIGHT=0.15` 仍只用于 GPA 缓存替换阶段的 text-visual joint score，不是最终分类权重。
最终分类权重由 `E4_FINAL_TEXT_DIST_WEIGHT` 和 `E4_FINAL_VISUAL_DIST_WEIGHT` 控制。

---

## 6. 实现文件

核心模型：

```text
Point-Cache/runners/E4_distribution_guided_cache/model_e4_c_a0_e1_distribution_final_score.py
```

扰动数据集 runner：

```text
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a0_e1_distribution_final_score_ulip_modelnetc_s2.py
```

干净数据集 runner：

```text
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a0_e1_distribution_final_score_ulip_modelnetc_clean.py
```

关键实现约束：

```text
final_logits =
    E4_FINAL_TEXT_DIST_WEIGHT * norm(S_text_dist)
  + E4_FINAL_VISUAL_DIST_WEIGHT * norm(S_visual_dist)
```

最终分类不再加：

```text
clip_logits
cache voting logits
negative cache voting logits
```

保存的诊断字段包括：

```text
final_score_type = distribution_only
final_score_uses_clip_logits = False
final_score_uses_cache_voting_logits = False
test_clip_proto_acc
test_text_dist_norm_acc
test_visual_dist_norm_acc
test_distribution_final_acc
```

---

## 7. 运行脚本

扰动数据集：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/02_10_1_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_distfinal_tw0p15_text1_visual1_score_norm_manualfull_llm_dynamic_init_textdist.sh 0
```

干净数据集：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/02_10_2_clean_ulip_modelnetc_clean_zs_global_local_e4_c_a0_e1_distfinal_tw0p15_text1_visual1_score_norm_manualfull_llm_dynamic_init_textdist.sh 0
```

---

## 8. 判断标准

需要同时报告百分比和样本数量，例如：

```text
准确率：52.51%（9071/17276）
```

主要比较对象：

1. `02_9_2`：54.71%（9451/17276）
2. PointCache baseline：54.00%（9329/17276）
3. clean `02_9_2`：63.86%（1576/2468）

如果本实验低于 `02_9_2`，需要进一步分析：

1. 文本分布得分单独准确率；
2. 视觉分布得分单独准确率；
3. text + visual 融合是否互相冲突；
4. 是否存在当前样本先更新分布再给自己打分造成的偏置。

---

## 9. 中止结果与诊断

运行日期：2026-06-16

用户在 `add_global_2` 完成后中止实验。中止是正确的，因为单扰动结果已经明显异常：

```text
02_10_1 add_global_2: 21.31%（526/2468）
02_9_2 add_global_2: 47.89%（约 1182/2468）
PointCache baseline add_global_2: 47.81%（约 1180/2468）
```

`02_10_1` 在 `add_global_2` 上不仅没有超过 `02_9_2`，还大幅低于零样本文本原型点积诊断项：

```text
零样本文本原型点积得分（zero-shot text prototype dot-product logits）: 33.63%（830/2468）
文本分布得分（text distribution score）: 1.62%（40/2468）
视觉历史分布得分（accepted-history visual distribution score）: 39.83%（983/2468）
最终分布融合得分（distribution-final score）: 21.31%（526/2468）
```

关键判断：

1. 视觉历史分布得分本身并没有完全崩掉，单独可达到 39.83%（983/2468）。
2. 文本分布得分本身严重错误，只有 1.62%（40/2468）。
3. 由于最终得分把文本分布和视觉历史分布按 1:1 融合，错误的文本分布项把视觉历史分布项拉低到 21.31%（526/2468）。
4. 因此，本实验失败的主要原因不是缓存维护失败，而是“把 E1 prompt embeddings 构成的高斯文本分布直接作为最终分类分布得分”这个假设不成立。

更具体地说，文本分布得分使用：

```text
S_text_dist(c) = -mean((f - μ_text(c))^2 / (σ_text^2(c) + eps))
```

这里的 `μ_text(c), σ_text^2(c)` 来自同一类别的多条文本提示嵌入。它描述的是“文本提示在文本编码器空间里的离散程度”，不等价于“点云特征在该类别下的视觉-文本匹配分布”。因此，把点云特征 `f` 直接放进这个文本高斯分布中计算马氏距离式得分，会造成类别排序严重失真。

结论：

```text
纯文本分布 + 纯视觉历史分布的最终得分版本暂时不继续跑完整 S2。
```

后续更合理的方向：

1. 不再直接使用文本 prompt 高斯分布作为最终分类得分。
2. 可以保留文本分布得分作为缓存替换时的辅助质量约束，因为 `02_9_2` 已证明它在替换阶段有收益。
3. 如果继续做分布最终得分，应优先尝试：
   - 视觉历史分布得分 + 零样本文本原型点积得分；
   - 或用文本原型点积得分作为主分类项，视觉分布得分作为校正项；
   - 或学习/估计视觉-文本对齐后的类别分布，而不是直接把文本嵌入分布当成点云特征分布。
