# 02_11：Protective Soft-Rejection GPA Replacement

日期：2026-06-17

状态：实验设计已确认，待实现代码与运行

## 1. 实验目的

本实验回到当前最好版本 `02_9_2`，只改一个位置：

```text
GPA 缓存（Global Prototype-Alignment Cache）满后的替换规则
```

其余逻辑保持 `02_9_2` 不变。

当前 `02_9_2` 在 ModelNet-C all35 上相比原始 Point-Cache 有小幅提升：

```text
PointCache all35：53.01%（45793/86380）
02_9_2 all35：    53.22%（45973/86380）
变化：            +0.21
```

但逐损坏类型看，`02_9_2` 的收益不稳定：

```text
add_local：+1.69
jitter：   +1.22
rotate：   -1.08
add_global：-0.94
```

核心问题是：当前 GPA 缓存替换规则可能过于严格。它要求新样本同时满足：

```text
熵更低（lower entropy）
联合分布分数不低于被替换样本（joint score no worse than replaced sample）
```

这能防止明显错误样本进入 GPA 缓存，但也可能拒绝一些低熵、可信、但联合分布分数略低的样本。尤其在 `add_global` 和 `rotate` 中，分布分数可能受扰动形态影响，导致原本有用的样本不能进入 GPA 缓存和对应的局部缓存。

本实验的目标是：

```text
把联合分布分数从“必须优于被替换样本”的强门槛，
改成“不能明显差于当前缓存整体”的保护性软拒绝规则。
```

## 2. 与 02_9_2 的关系

保持不变：

1. 骨干模型（backbone）仍为 ULIP。
2. 仍是免训练测试时适应（training-free Test-Time Adaptation, training-free TTA）。
3. 不更新点云编码器、文本编码器或任何模型参数。
4. 不使用反向传播（backpropagation）。
5. 真实标签只用于离线统计，不参与测试时决策。
6. 文本端仍使用 E1 的 LLM prompt cache，但只用于 GPA 替换阶段的文本分布先验。
7. 最终分类仍使用 `02_9_2` 的缓存投票方式，不使用纯分布最终得分。
8. 全局熵缓存（Global Entropy Cache）、全局负缓存（Global Negative Cache）和局部缓存（Local Cache）的主要逻辑不变。
9. `E4_TEXT_SCORE_WEIGHT=0.15`、`E4_SCORE_NORM_MODE=running_zscore` 等主参数不变。

改变：

```text
只改变 GPA 缓存满后的替换判定。
```

## 3. 术语说明

| 术语 | 中文解释 |
|---|---|
| `pc_feats` | 点云全局特征（point-cloud global features） |
| `patch_centers` | 局部 patch 聚类中心特征（local patch cluster-center features） |
| `clip_logits` | 零样本文本原型点积得分（zero-shot text prototype dot-product logits） |
| `entropy` | 熵 / 预测不确定性（entropy / prediction uncertainty） |
| `joint_score` | 联合分布分数（joint distribution score） |
| GPA 缓存 | 全局原型对齐缓存（Global Prototype-Alignment Cache） |
| 局部缓存 | GPA 控制的局部缓存（GPA-controlled Local Cache） |

代码历史说明：

```text
旧代码中的变量名 loss 实际表示 prediction entropy，不是训练损失。
从 02_11 开始，新代码中应尽量改名为 entropy。
该改名只改变变量命名，不改变数学计算。
```

## 4. 02_9_2 当前 GPA 替换规则

对当前测试样本，先通过零样本预测得到伪标签类别 `c`。

若 `GPA cache[c]` 未满：

```text
直接加入 GPA cache[c]
同步更新该类别的局部缓存
```

若 `GPA cache[c]` 已满：

```text
old = GPA cache[c] 中熵最高的样本
```

然后执行：

```text
如果 entropy_new >= entropy_old：
    拒绝当前样本

否则：
    计算 joint_score_new
    计算 joint_score_old

    如果 joint_score_new >= joint_score_old：
        用当前样本替换 old
        同步替换该类别局部缓存中对应位置
    否则：
        拒绝当前样本
```

注意：

```text
这里使用 >=，不是 >。
```

也就是说，只要联合分布分数不低于被替换样本，就允许替换。

## 5. 02_11 新规则

02_11 仍然先找出当前类别 GPA 缓存中熵最高的样本：

```text
old = GPA cache[c] 中熵最高的样本
```

替换目标仍然是这个熵最高样本，不改成替换联合分布分数最低的样本。

完整规则：

```text
如果 entropy_new >= entropy_old：
    拒绝当前样本

否则：
    计算 joint_score_new
    计算 joint_score_old
    计算 GPA cache[c] 中所有已有样本的 joint_score
    min_joint_score = min(已有样本 joint_score)

    如果 joint_score_new >= joint_score_old：
        按 02_9_2 原始严格规则接收
        用当前样本替换 old
        同步替换对应局部缓存

    否则如果 joint_score_new >= min_joint_score：
        按保护性软拒绝规则接收
        用当前样本替换 old
        同步替换对应局部缓存

    否则：
        拒绝当前样本
```

这等价于把接收区域分成两段：

```text
严格接收区：
joint_score_new >= joint_score_old

软接收区：
min_joint_score <= joint_score_new < joint_score_old

拒绝区：
joint_score_new < min_joint_score
```

## 6. 为什么不引入新阈值

之前讨论过“略低”和“明显低于”的阈值问题，但这会引入新的超参数，例如：

```text
delta
z_delta
soft threshold
```

这些阈值需要额外调参，不利于当前论文主线。

02_11 使用当前缓存内部的相对标准：

```text
min_joint_score = 当前类别 GPA 缓存中最差的联合分布分数
```

因此不需要新增阈值。

直观含义是：

```text
当前样本不一定要比被替换样本的联合分布分数更高；
但它不能比当前类别 GPA 缓存里所有已有样本都差。
```

这让联合分布分数从“强替换门槛”变成“异常样本保护门槛”。

## 7. 为什么仍然替换最高熵样本

虽然 02_11 会计算所有已有样本的 `joint_score`，但替换对象仍然是：

```text
entropy 最大的 GPA 样本
```

原因：

1. `02_9_2` 的 GPA 缓存本质仍是低熵可信样本缓存。
2. 熵（entropy）直接来自当前模型预测不确定性，是 Point-Cache 主逻辑。
3. 联合分布分数（joint score）在本实验中只作为保护项，不作为主要排序准则。
4. 如果改为替换 `joint_score` 最低样本，就同时改变了缓存定义，实验解释会变复杂。

因此 02_11 是一个最小改动实验：

```text
保留低熵缓存定义；
只放松 GPA 替换时的联合分布分数门槛。
```

## 8. 最终得分保持不变

02_11 最终 logits 仍沿用 `02_9_2`：

```text
final_logits =
    clip_logits
  + global_entropy_cache_logits
  + gpa_local_cache_logits
  - negative_cache_logits
```

其中：

1. `clip_logits` 是零样本文本原型点积得分（zero-shot text prototype dot-product logits）。
2. `global_entropy_cache_logits` 是全局熵缓存投票得分（global entropy cache voting logits）。
3. `gpa_local_cache_logits` 是 GPA 控制的局部缓存投票得分（GPA-controlled local cache voting logits）。
4. `negative_cache_logits` 是全局负缓存投票惩罚得分（global negative cache voting penalty logits）。

本实验不再尝试 `02_10_1` 的纯分布最终得分，因为该方向在 `add_global_S2` 上已经明显失败：

```text
零样本文本原型点积得分：33.63%（830/2468）
文本分布得分：          1.62%（40/2468）
视觉历史分布得分：      39.83%（983/2468）
最终分布融合得分：      21.31%（526/2468）
```

## 9. 需要记录的诊断

为了判断 02_11 是否真正改善了 GPA 替换，实验需要记录以下统计。

GPA 替换统计：

```text
gpa_add_empty
gpa_replace_strict_original
gpa_replace_soft_accept
gpa_reject_entropy
gpa_reject_below_min_joint
gpa_reject_no_accepted_history_text_visual_distribution
```

其中：

1. `gpa_replace_strict_original` 表示满足 `joint_score_new >= joint_score_old` 的原始严格接收。
2. `gpa_replace_soft_accept` 表示满足 `min_joint_score <= joint_score_new < joint_score_old` 的保护性软接收。
3. `gpa_reject_below_min_joint` 表示新样本联合分布分数低于当前类别 GPA 缓存中所有已有样本，因此被拒绝。

单次事件日志建议记录：

```text
decision
pred
entropy_new
entropy_old
joint_score_new
joint_score_old
min_joint_score
new_minus_old
new_minus_min
```

其中：

```text
new_minus_old = joint_score_new - joint_score_old
new_minus_min = joint_score_new - min_joint_score
```

这些诊断只用于分析实验，不应成为最终论文方法的必要组件。

## 10. 预期验证点

主要验证：

```text
02_11 是否能在保持 02_9_2 优势的同时，缓解 add_global 和 rotate 上的下降。
```

需要重点比较：

1. ModelNet-C severity=2 平均准确率。
2. ModelNet-C all35 平均准确率。
3. `add_global` 各 severity 的变化。
4. `rotate` 各 severity 的变化。
5. `add_local` 和 `jitter` 的收益是否被保留。
6. clean 准确率是否进一步下降。

报告准确率时必须同时给出样本数，例如：

```text
54.71%（9451/17276）
```

## 11. 计划实现文件

本轮实际执行范围：

```text
仅 S2（severity=2）扰动数据集
```

核心模型文件：

```text
Point-Cache/runners/E4_distribution_guided_cache/model_e4_c_a0_e1_protective_soft_rejection_gpa.py
```

扰动数据集 runner：

```text
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a0_e1_protective_soft_rejection_gpa_ulip_modelnetc_s2.py
```

扰动数据集脚本：

```text
Point-Cache/scripts/E4_distribution_guided_cache/02_11_1_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_protective_soft_rejection_gpa_tw0p15_score_norm_manualfull_llm_dynamic_init_textdist.sh
```

后续若 S2 结果合理，再补 clean / all35 扩展。

## 12. 运行命令

扰动数据集 severity=2：

```bash
cd Point-Cache
bash scripts/E4_distribution_guided_cache/02_11_1_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_protective_soft_rejection_gpa_tw0p15_score_norm_manualfull_llm_dynamic_init_textdist.sh 0
```

## 13. 成功标准

第一阶段成功标准：

```text
S2 平均准确率不低于 02_9_2：
02_9_2 S2 = 54.71%（9451/17276）
```

同时观察：

1. `add_global_S2` 是否不低于 02_9_2 的 `47.89%`。
2. `rotate_S2` 是否不低于 02_9_2 的 `61.30%`。
3. `gpa_replace_soft_accept` 是否有非零触发。
4. `gpa_replace_soft_accept` 样本的离线真实准确率是否高于或接近普通 GPA 接收样本。

第二阶段成功标准：

```text
all35 平均准确率不低于 02_9_2：
02_9_2 all35 = 53.22%（45973/86380）
```

这一项仅作为后续扩展目标，不在本轮 S2 运行范围内。

更理想的情况是：

```text
add_global 和 rotate 的负收益缩小；
add_local 和 jitter 的正收益保留。
```

## 14. 风险

潜在风险：

1. 软接收可能让更多边界样本进入 GPA 缓存，从而污染局部缓存。
2. 如果 `min_joint_score` 本身来自错误样本，保护门槛会变弱。
3. 如果某类早期 GPA 缓存质量较差，02_11 可能延续错误分布。

因此本实验应先跑 S2，不直接跑 all35。

如果 S2 显示：

```text
gpa_replace_soft_accept 增加，但准确率下降
```

则说明问题不只是 GPA 替换过严，而是 GPA/local cache 的写入样本本身需要更强的可靠性控制。
