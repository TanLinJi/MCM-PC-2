# 02_15_1：E1-33 Fused Prototype Text Gate 修复方案

日期：2026-06-18

状态：已运行，S2 结果已记录。

## 1. 背景

当前最好 E4 载体是 `02_9_2`：

```text
E4-C-A0+E1-textdist-only
ModelNet-C all35
E4_TEXT_SCORE_WEIGHT = 0.15
E4_SCORE_NORM_MODE = running_zscore
final logits = Point-Cache voting logits
```

`02_9_2` 的 all35 结果为：

```text
S0 average   = 59.93
S1 average   = 56.70
S2 average   = 54.71
S3 average   = 50.26
S4 average   = 44.51
all35 average = 53.22
```

之后我们做了 `02_14_1`，目标是在完全保留 `02_9_2` 载体的前提下，把 text distribution 使用的 E1 prompt 配置替换成当前固定的 E1-33 配置。

`02_14_1` 的核心设置：

```text
载体实验：02_9_2
最终分类器：manual_full，不改
最终 logits：Point-Cache voting，不改
E1 描述用途：只用于 cache replacement 阶段的 text distribution
prompt cache：llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json
prompt count：15
prompt composition：10 image-style + 5 pointcloud-style
manual_full:LLM = 0.60:0.40
```

`02_14_1` 的 all35 结果：

```text
S0 average   = 59.85
S1 average   = 56.69
S2 average   = 54.32
S3 average   = 50.52
S4 average   = 44.52
all35 average = 53.18
```

与 `02_9_2` 对比：

```text
02_14_1 all35 = 53.18
02_9_2  all35 = 53.22
delta          = -0.04
```

结论：E1-33 prompt 替换进 `02_9_2` 的 text distribution 后，整体基本持平但略低，没有获得 E1-33 在 E1 独立实验中的收益。

## 2. 需要坚持的边界

本轮修复不应把 `02_9_2` 改成 distribution-final score。

`02_9_2` 的最终分类应该保持 Point-Cache 式 voting：

```text
final_logits =
    clip_logits
  + global entropy cache voting logits
  + GPA local cache voting logits
  - negative cache voting logits
```

也就是说：

```text
distribution score 只用于 GPA/local cache 的写入与替换决策；
最终预测仍然使用 zero-shot text prototype dot-product + cache voting。
```

`02_10_1` 已经验证过纯 distribution-final score 不成立，因此当前修复方向不是修改最终分类公式，而是修复 E1-33 文本信号进入 cache replacement gate 的方式。

## 3. 失败原因判断

### 3.1 E1-33 在 E1 中有效的原因

E1-33 的有效点不是“prompt 方差更好”，而是“类别文本原型更好”。

E1 中的融合方式是：

```text
manual prompts -> manual_full prototype
LLM prompts    -> LLM prototype

fused prototype =
    normalize(0.60 * manual_full prototype + 0.40 * LLM prototype)
```

然后 zero-shot 分类使用：

```text
point feature · fused prototype
```

因此 E1-33 的优势本质上是 improved class center，即一个更适合点云分类的类别文本中心。

### 3.2 02_14_1 中 E1-33 的进入方式

`02_14_1` 没有把 E1-33 fused prototype 作为最终分类器。

它仍然使用 `manual_full` 构建 `clip_logits`，最终 logits 仍是 Point-Cache voting。

E1-33 只进入了 text distribution：

```text
manual prompt embeddings + LLM prompt embeddings
        -> weighted mean + weighted variance
        -> text distribution score
        -> cache replacement gate
```

这与 E1 的原型融合机制不同。

E1 是先分别形成两个稳定原型，再融合原型；当前 E4 text distribution 是直接把多个 prompt embedding 当成一个分布云。

### 3.3 为什么会不升反降

E1-33 的 15 条描述包含 10 条 image-style 和 5 条 pointcloud-style。语义更丰富，但 embedding 空间中也更分散。

当这些 prompt 被直接当作一个高斯分布时，会出现两个副作用。

第一，text distribution 方差变大：

```text
旧 02_9_2 text variance average 约为 0.000998
新 02_14_1 text variance average 约为 0.001105
```

方差变大后，基于 Gaussian/Mahalanobis 的 text distribution score 会变钝，对类别一致性的约束能力下降。

第二，cache replacement 更保守：

```text
02_14_1 的 test GPA replacement 数量少于 02_9_2
02_14_1 的 joint-score reject 数量多于 02_9_2
```

因此 E1-33 没有帮助 cache 接纳更多高质量样本，反而挡掉了一部分 `02_9_2` 会接受的样本。

核心判断：

```text
E1-33 的优势 = 更好的 fused text prototype
02_14_1 的用法 = 更宽的 prompt embedding distribution

两者没有对齐。
```

## 4. 修复原则

本轮修复应该遵守三个原则。

第一，保持 Point-Cache final voting 不变。

```text
不修改 final_logits 公式；
不复活 02_10_1 的 distribution-final score 路线。
```

第二，解耦文本分支。

```text
不要把 manual_full prompts 和 LLM prompts 直接揉成一个 text_dist。
应该分别保留：
1. manual_full text branch
2. LLM text branch
3. E1 fused prototype branch
```

第三，让 E1-33 的 fused prototype 优势进入 cache replacement gate。

```text
E1-33 应该以 fused prototype score 的形式影响 GPA/local cache replacement，
而不是只作为一个更宽的 prompt-level Gaussian distribution。
```

## 5. 候选方案 02_15_1

候选实验名：

```text
02_15_1_e1_33_fused_prototype_text_gate
```

实验目的：

```text
在 02_9_2 载体上，只修改 cache replacement gate 中的文本端信号，
验证 E1-33 fused prototype score 是否能恢复并放大 E1-33 的收益。
```

基本设置：

```text
dataset：ModelNet-C severity=2
corruptions：7 类 corruption
backbone：ULIP
carrier：02_9_2
final logits：Point-Cache voting，不变
clip_logits：manual_full，先不变
cache：global entropy cache + GPA local cache + negative cache，保持 02_9_2
score normalization：running_zscore
```

E1-33 prompt 设置：

```text
prompt cache：llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json
dynamic prompt count：15
prompt composition：10 image-style + 5 pointcloud-style
manual_full:LLM = 0.60:0.40
```

## 5.1 已实现文件

模型侧改动：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/runners/E4_distribution_guided_cache/model_e4_c_accepted_history_text_visual_distribution_guided_gpa.py
```

新增开关：

```text
E4_TEXT_GATE_MODE=distribution       # 默认值，保持 02_9_2 / 02_14_1 旧逻辑
E4_TEXT_GATE_MODE=fused_prototype    # 02_15_1 新逻辑
E4_TEXT_PROTO_SCORE_SCALE=1.0        # fused prototype cosine score 的缩放，默认 1.0
```

runner 侧改动：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a0_e1_textdist_only_ulip_modelnetc_s2_accepted_history_text_visual_distribution_guided_gpa.py
```

当 `E4_TEXT_GATE_MODE=fused_prototype` 时，runner 会额外构建：

```text
e1_fused_text_prototype =
    normalize(0.60 * manual_full_prototype + 0.40 * LLM_prototype)
```

并把它保存到每类 `text_dist[class_index]["prototype"]` 中，供 cache replacement gate 使用。

脚本侧改动：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache/02_run_e4_c_ulip_modelnetc_s2_common.sh
```

该通用脚本现在支持通过环境变量覆盖 E1 prompt 设置：

```text
E4_PROMPT_CACHE_DIR
E4_PROMPT_CACHE_FILE
E4_LLM_PROMPT_MODE
E4_DYNAMIC_PROMPT_COUNT
E4_PROMPT_STATIC_WEIGHT
E4_PROMPT_DYNAMIC_WEIGHT
```

新增实验脚本：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache/02_15_1_ulip_modelnetc_s2_e1_33_fused_prototype_text_gate_manual60_llm40.sh
```

## 5.2 运行命令

在当前 `mcmpc` 环境中执行：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/02_15_1_ulip_modelnetc_s2_e1_33_fused_prototype_text_gate_manual60_llm40.sh 0
```

其中：

```text
0 = 使用物理 GPU 0
当前环境按单张 RTX 4090 设计
```

## 5.3 结果保存位置

结果目录：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/02_15_1_ulip_modelnetc_s2_e4_c_a0_e1_33_fused_prototype_text_gate_tw0p15_manual60_llm40/
```

预期输出：

```text
summary.csv
logs/
gpa_stats/
wandb/
```

最关键的诊断文件：

```text
gpa_stats/*_gpa_stats.json
gpa_stats/gpa_replacement_events_*.jsonl
```

这些文件会记录：

```text
e4_text_gate_mode = fused_prototype
e4_text_proto_score_scale = 1.0
new_text_score / old_text_score
new_text_score_for_joint / old_text_score_for_joint
joint_score_margin
GPA replacement / reject 统计
```

## 6. 建议的 joint score 设计

### 6.1 第一版：最小改动方案

第一版只加入 E1 fused prototype score。

```text
joint_score =
    visual_score
  + w_proto * fused_prototype_score
```

其中：

```text
fused_prototype_score(c) = point_feature · e1_fused_text_prototype(c)
```

推荐先用：

```text
w_proto = 0.15
```

这样可以最大程度对齐 `02_9_2` 的原始 text weight 设定，同时避免引入太多变量。

### 6.2 第二版：manual 与 LLM 分支解耦

如果第一版有效但不稳定，可以进一步拆成：

```text
joint_score =
    visual_score
  + w_manual * manual_text_score
  + w_llm    * llm_text_score
  + w_proto  * fused_prototype_score
```

其中：

```text
manual_text_score：manual_full branch 的 distribution 或 prototype score
llm_text_score：LLM branch 的 distribution 或 prototype score
fused_prototype_score：E1-33 fused prototype dot-product score
```

保守初始权重：

```text
w_manual = 0.05
w_llm    = 0.05
w_proto  = 0.05
```

或者更贴近 E1-33：

```text
w_manual = 0.06
w_llm    = 0.04
w_proto  = 0.05
```

但第二版变量较多，不建议作为第一步。

### 6.3 暂不推荐的方案

暂不推荐继续单独调高或调低 `E4_TEXT_SCORE_WEIGHT`。

原因是当前失败的主要矛盾不是 text weight 数值，而是 text signal representation 错位：

```text
现在的问题不是“文本分数太强或太弱”，
而是“把 E1-33 的 fused prototype 优势错误地表示成了 prompt-level Gaussian 方差”。
```

## 7. 需要记录的诊断指标

`02_15_1` 不能只看 accuracy，需要同步记录 cache 行为。

至少记录：

```text
1. final accuracy
2. test_gpa_replace_accepted_history_text_visual_distribution
3. test_gpa_reject_accepted_history_text_visual_distribution
4. test_gpa_reject_entropy_accepted_history_text_visual_distribution
5. text/prototype score 的 mean/std
6. visual score 的 mean/std
7. running_zscore 是否已经 ready
```

重点观察：

```text
1. GPA replacement 是否比 02_14_1 恢复
2. joint-score reject 是否低于 02_14_1
3. add_local_S2 是否回升
4. dropout_local_S2 是否回升
5. rotate_S2 是否不继续受伤
```

## 8. 预期结果与判定标准

S2 对比基线：

```text
02_9_2 S2 average  = 54.71
02_14_1 S2 average = 54.32
```

`02_15_1` 的最低有效判定：

```text
S2 average > 54.32
```

说明 fused prototype gate 至少修复了 `02_14_1` 的负迁移。

更强判定：

```text
S2 average >= 54.71
```

说明新 gate 至少追平原始 `02_9_2`。

理想判定：

```text
S2 average > 54.71
```

说明 E1-33 fused prototype 确实能在 E4 cache replacement 阶段提供额外收益。

## 9. 如果 02_15_1 有效，下一步

如果 S2 有效，下一步按顺序推进：

```text
1. 跑 ModelNet-C all35
2. 与 02_9_2 all35 对比
3. 与 02_14_1 all35 对比
4. 分析逐 corruption / 逐 severity 变化
5. 检查 clean ModelNet 是否受损
```

all35 后需要重点确认：

```text
1. all35 average 是否超过 53.22
2. S2 是否保持提升
3. S0 clean-like 场景是否不下降
4. add_local / dropout_local 是否恢复
5. rotate 是否不会明显受损
```

## 10. 如果 02_15_1 无效，下一步

如果 S2 仍然不升，说明 E1-33 可能更适合最终 zero-shot prototype，而不适合作为 cache replacement gate。

此时不应继续在 text_dist 方差或 text gate 上反复调参，而应切换到另一条路线：

```text
保持 Point-Cache voting；
但将 clip_logits 的 text prototype 从 manual_full 换成 E1-33 fused prototype；
cache voting 机制仍保持 Point-Cache；
再观察 E1-33 zero-shot prototype 与 cache voting 是否能叠加。
```

这条路线要单独编号，不应混入 `02_15_1`。

## 11. 当前结论

`02_14_1` 的失败不是 E1-33 失败，而是 E1-33 的信号进入 E4 的方式不合理。

当前最合理的修复是：

```text
把 E1-33 从 prompt-level Gaussian text distribution
改为 fused prototype score / branch-wise text gate，
只影响 GPA/local cache replacement，
保持最终 Point-Cache voting 不变。
```

因此下一步建议优先实现并运行：

```text
02_15_1_e1_33_fused_prototype_text_gate
ModelNet-C severity=2
```

运行前必须先给出：

```text
1. 完整实验配置
2. 执行命令
3. 结果保存目录
4. 预期输出文件
5. 与 02_9_2 / 02_14_1 的对比指标
```

## 12. 实际运行结果

运行命令：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/02_15_1_ulip_modelnetc_s2_e1_33_fused_prototype_text_gate_manual60_llm40.sh 0
```

结果目录：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/02_15_1_ulip_modelnetc_s2_e4_c_a0_e1_33_fused_prototype_text_gate_tw0p15_manual60_llm40/
```

S2 逐扰动结果：

| corruption | E0 | 02_9_2 | 02_14_1 | 02_15_1 | 02_15_1 - 02_9_2 | 02_15_1 - 02_14_1 |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 47.81 | 47.89 | 47.93 | 48.82 | +0.93 | +0.89 |
| add_local | 46.68 | 50.85 | 50.16 | 49.15 | -1.70 | -1.01 |
| dropout_global | 59.20 | 59.12 | 58.55 | 59.48 | +0.36 | +0.93 |
| dropout_local | 56.69 | 57.21 | 56.44 | 57.01 | -0.20 | +0.57 |
| rotate | 62.07 | 61.30 | 60.90 | 61.02 | -0.28 | +0.12 |
| scale | 55.23 | 55.92 | 56.16 | 55.55 | -0.37 | -0.61 |
| jitter | 50.32 | 50.65 | 50.12 | 49.96 | -0.69 | -0.16 |

S2 平均：

```text
E0      = 54.00
02_9_2  = 54.71
02_14_1 = 54.32
02_15_1 = 54.43
```

结论：

```text
02_15_1 比 02_14_1 高 +0.10，说明 fused prototype gate 修复了一部分 prompt-level distribution 替换造成的负迁移。
02_15_1 比 E0 高 +0.43，说明该分支不是完全无效。
02_15_1 比 02_9_2 低 -0.28，说明当前 text gate 强度或形式仍未达到原始最好 E4 配置。
```

主要问题：

```text
add_global 和 dropout_global 有收益；
add_local、jitter、scale 相对 02_9_2 下降。
```

诊断判断：

```text
E1-33 fused prototype 方向是可用的，但 E4_TEXT_SCORE_WEIGHT=0.15 对局部扰动可能偏强。
下一步先做同设置下的 E4_TEXT_SCORE_WEIGHT=0.10，也就是 02_15_2。
```
