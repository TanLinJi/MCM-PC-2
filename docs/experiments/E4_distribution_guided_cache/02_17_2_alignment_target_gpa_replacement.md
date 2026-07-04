# 02_17_2：Alignment-Target GPA Replacement

日期：2026-06-19

状态：代码已实现并通过静态检查，待手动运行。

## 1. 实验目的

本实验回到当前主线 `02_9_2`，只修改 GPA-cache 满后的替换规则。

目标不是让 GPA-cache 回退成普通低熵缓存，而是让它更明确地成为：

```text
更干净的对齐缓存库
alignment-purified memory
```

当前 `02_9_2` 的 GPA-cache 替换逻辑存在一个标准混用问题：

```text
替换对象：选择 entropy 最大的样本
替换判定：要求新样本 joint_score 更高
```

这会导致一个问题：GPA-cache 中真正对齐质量最差的样本，未必会被替换掉；而 entropy 最大的样本可能并不是 alignment 最差的样本。

因此 `02_17_2` 改为：

```text
替换对象由 alignment quality 决定：
谁的 joint_score 最低，谁优先被替换。
```

同时保留低熵安全门控，避免高熵不可靠样本污染 GPA/local cache。

## 2. 与已有实验的关系

### 2.1 与 02_9_2 的关系

保持不变：

1. backbone 仍为 ULIP。
2. 数据集先只跑 ModelNet-C severity=2。
3. 最终 logits 仍使用 `02_9_2` 的 Point-Cache voting：

```text
final_logits =
    clip_logits
  + global_entropy_cache_logits
  + gpa_local_cache_logits
  - negative_cache_logits
```

4. Global Entropy Cache 仍按原始 Point-Cache 低熵规则更新。
5. Negative Cache 仍按当前规则更新。
6. 文本分布仍只用于 GPA 替换阶段的 text-visual joint score。
7. `E4_TEXT_SCORE_WEIGHT=0.15`。
8. `E4_SCORE_NORM_MODE=running_zscore`。
9. LLM prompt cache 仍复用 `02_9_2` 的默认 prompt cache。

改变：

```text
只改变 GPA-cache / GPA-controlled local cache 的满缓存替换策略。
```

### 2.2 与 02_11 的关系

`02_11` 的规则：

```text
替换对象仍是 entropy 最大的样本；
如果 new_joint_score >= min_joint_score，则允许软接收。
```

`02_17_2` 的规则：

```text
替换对象直接改为 joint_score 最低的样本；
新样本只要比这个 alignment 最差样本更低熵、更高 joint_score，即可替换。
```

因此 `02_17_2` 不是简单放松 `02_9_2`，而是重新定义 GPA-cache 的维护目标：

```text
从低熵缓存 + 分布 veto
改成 alignment-targeted clean memory。
```

### 2.3 与 07_1 的关系

`07_1` 的规则：

```text
对低熵 + 极低能量可信样本，
允许 joint_score_new >= joint_score_old - margin。
```

其中 `margin=0.05` 是额外超参数。

`02_17_2` 不引入 margin，不使用：

```text
joint_score_new >= joint_score_old - delta
```

而是使用当前缓存内部已有样本的相对顺序：

```text
target = argmin(joint_score_old_i)
```

因此 `02_17_2` 的判断是“是否优于当前 alignment 最差样本”，不是“是否略低于某个 old 样本”。

## 3. 当前 02_9_2 替换规则

对当前样本，先得到 zero-shot 伪标签类别 `c`。

如果 `GPA cache[c]` 未满：

```text
直接加入 GPA cache[c]
同步加入 GPA-controlled local cache[c]
```

如果 `GPA cache[c]` 已满：

```text
old = GPA cache[c] 中 entropy 最大的样本
```

然后：

```text
如果 entropy_new >= entropy_old：
    reject_entropy

否则：
    计算 joint_score_new
    计算 joint_score_old

    如果 joint_score_new > joint_score_old：
        replace old
    否则：
        reject_joint
```

问题：

```text
old 是 entropy 最差样本，但不一定是 alignment 最差样本。
```

这让 GPA-cache 的含义不够干净。

## 4. 02_17_2 新规则

### 4.1 核心思想

GPA-cache 不再替换最高熵样本，而是替换当前类别 GPA-cache 中 alignment 最差的样本。

alignment quality 使用当前已有的 text-visual joint score：

```text
joint_score =
    visual_joint_score
  + E4_TEXT_SCORE_WEIGHT * text_joint_score
```

得分越高，表示该样本越符合当前类别的 accepted-history visual distribution 与 text distribution。

### 4.2 满缓存替换流程

对当前样本 `x_new`，伪标签为 `c`。

如果 `GPA cache[c]` 未满：

```text
加入 GPA cache[c]
加入 GPA-controlled local cache[c]
更新 visual distribution
```

如果 `GPA cache[c]` 已满：

第一步，计算新样本：

```text
entropy_new
joint_score_new
```

第二步，对 `GPA cache[c]` 中所有旧样本重新计算：

```text
entropy_old_i
joint_score_old_i
```

第三步，选择替换对象：

```text
target = argmin_i(joint_score_old_i)
```

也就是当前类别 GPA-cache 中 text-visual alignment 最差的样本。

第四步，执行宽松接收条件：

```text
如果 entropy_new < max_entropy_in_GPA_cache[c]
并且 joint_score_new > joint_score_target：
    replace target
否则：
    reject
```

这里使用 `max_entropy_in_GPA_cache[c]`，而不是 `entropy_target`，是因为本实验采用宽松版本 `02_17_2`：

```text
新样本不必须比被替换的 alignment 最差样本更低熵；
只需要比当前 GPA-cache 中最不可靠的最高熵样本更低熵。
```

这样可以保留低熵安全门控，同时让 GPA-cache 更积极地修复 alignment 最差位置。

## 5. 为什么选择 02_17_2 宽松版本

更严格的 `02_17_1` 可以定义为：

```text
target = argmin(joint_score_old_i)

如果 entropy_new < entropy_target
并且 joint_score_new > joint_score_target：
    replace target
```

该规则很干净，但可能触发过少。因为 target 是 alignment 最差样本，它未必也是高熵样本；要求新样本同时比 target 更低熵，可能过于保守。

`02_17_2` 改成：

```text
entropy_new < max_entropy_in_GPA_cache[c]
joint_score_new > joint_score_target
```

它的含义是：

1. 新样本必须足够可靠，至少比当前 GPA-cache 中最不可靠样本低熵。
2. 新样本必须改善 alignment 最差样本。
3. 不要求新样本比 target 本身更低熵。

因此 `02_17_2` 是一个更适合先跑的宽松诊断版本。

## 6. 决策类型

建议记录以下 decision：

```text
add_not_full_alignment_target
replace_alignment_target
reject_no_joint_score
reject_entropy_not_below_cache_max
reject_joint_not_better_than_target
```

含义：

| Decision | 含义 |
|---|---|
| `add_not_full_alignment_target` | 当前类别 GPA-cache 未满，直接加入 |
| `replace_alignment_target` | 新样本通过低熵安全门控，并且 joint score 高于 alignment 最差样本 |
| `reject_no_joint_score` | 无法计算新样本或旧样本 joint score |
| `reject_entropy_not_below_cache_max` | 新样本熵不低于当前 GPA-cache 中最大熵 |
| `reject_joint_not_better_than_target` | 新样本 joint score 未超过 alignment 最差样本 |

## 7. 必须记录的诊断字段

每次 GPA replacement event 建议记录：

```text
phase
sample_index
class_index / pred
target
pseudo_label_correct
decision

new_entropy
new_joint_score
new_visual_score
new_text_score

target_index_in_cache
target_entropy
target_joint_score
target_visual_score
target_text_score

cache_max_entropy
cache_min_joint_score
cache_mean_entropy_before
cache_mean_joint_score_before
cache_mean_entropy_after
cache_mean_joint_score_after

new_minus_target_joint
new_minus_cache_max_entropy
target_entropy_rank
target_joint_rank
```

其中：

```text
new_minus_target_joint = joint_score_new - joint_score_target
new_minus_cache_max_entropy = cache_max_entropy - entropy_new
```

注意：

```text
target_joint_rank 应该等于最低 rank；
target_entropy_rank 用于观察 alignment 最差样本是否也是高熵样本。
```

如果大量 target 的 `target_entropy_rank` 并不是最差，说明当前 `02_9_2` 的“替换最高熵样本”确实没有对准 alignment 最差位置。

## 8. 需要额外统计的缓存质量

每个 corruption 结束后，除了最终准确率，还应保存：

```text
gpa_cache_mean_entropy
gpa_cache_max_entropy
gpa_cache_mean_joint_score
gpa_cache_min_joint_score
gpa_cache_replacement_count
gpa_cache_alignment_target_replace_count
```

还应继续记录分支准确率，至少包括：

```text
zero_shot_text_proto_dot
global_entropy_cache
gpa_local_cache
positive_cache_total
cache_total_signed
final_logits
```

如果实现成本可控，也建议记录 `gpa_global_cache_diag`，因为 `02_16_1` 中该分支强于 `gpa_local_cache`：

```text
gpa_global_cache_diag = 49.02
gpa_local_cache = 46.96
```

## 9. 预期实现文件

模型文件：

```text
Point-Cache/runners/E4_distribution_guided_cache/model_e4_c_a4_alignment_target_gpa_replacement.py
```

runner：

```text
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a4_alignment_target_gpa_replacement_ulip_modelnetc_s2.py
```

脚本：

```text
Point-Cache/scripts/E4_distribution_guided_cache/02_17_2_ulip_modelnetc_s2_alignment_target_gpa_replacement.sh
```

结果目录：

```text
Point-Cache/results/E4_distribution_guided_cache/02_17_2_ulip_modelnetc_s2_alignment_target_gpa_replacement/
```

说明：

```text
代码文件名使用 A4，是因为它是 E4-C 系列下新的 alignment-target GPA-cache 变体。
实验编号仍使用 02_17_2。
```

## 10. 已实现代码

本实验已经新增以下文件：

```text
Point-Cache/runners/E4_distribution_guided_cache/model_e4_c_a4_alignment_target_gpa_replacement.py
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a4_alignment_target_gpa_replacement_ulip_modelnetc_s2.py
Point-Cache/scripts/E4_distribution_guided_cache/02_17_2_ulip_modelnetc_s2_alignment_target_gpa_replacement.sh
```

关键实现点：

1. runner 已确认调用 A4 模型文件，而不是复用 A0 模型。
2. `clip_weights` 与最终 logits 计算保持 `02_9_2` 的 Point-Cache voting。
3. `manual_full` 仍作为主分类器 prompt source。
4. `manualfull_llm_dynamic_init` 只用于 text distribution replacement score。
5. GPA-cache 未满时直接加入。
6. GPA-cache 满后选择当前类别中 `joint_score` 最低的样本作为 target。
7. 接收条件固定为：

```text
entropy_new < max_entropy_in_GPA_cache[c]
joint_score_new > joint_score_target
```

8. `running_zscore` 只在通过低熵门控后，用新样本和 target 样本这一对更新，避免对整个旧缓存反复更新 score norm 状态。
9. 脚本运行前会检查默认 E1 prompt cache 是否存在且完整，避免意外调用 LLM API。

## 11. 基本实验设置

```text
实验编号：02_17_2
数据集：ModelNet-C
扰动等级：severity=2
扰动类型：add_global, add_local, dropout_global, dropout_local, rotate, scale, jitter
Backbone：ULIP
GPU：单张 RTX 4090，物理 GPU 0
主分类器 prompt source：manual_full
text distribution prompt source：manualfull_llm_dynamic_init
E1 prompt cache：results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json
E4_TEXT_SCORE_WEIGHT：0.15
E4_SCORE_NORM_MODE：running_zscore
最终 logits：
    clip_logits
  + global_entropy_cache_logits
  + gpa_local_cache_logits
  - negative_cache_logits
```

## 12. 运行命令

在 `mcmpc` 环境中运行：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/02_17_2_ulip_modelnetc_s2_alignment_target_gpa_replacement.sh 0
```

当前环境说明：

```text
GPU：单张 RTX 4090
物理 GPU：0
```

结果保存位置：

```text
Point-Cache/results/E4_distribution_guided_cache/02_17_2_ulip_modelnetc_s2_alignment_target_gpa_replacement/
```

关键输出文件：

```text
summary.csv
logs/
gpa_stats/
gpa_stats/*_gpa_stats.json
gpa_stats/gpa_replacement_events_*.jsonl
```

## 13. 静态检查记录

已执行：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
python -m py_compile \
  runners/E4_distribution_guided_cache/model_e4_c_a4_alignment_target_gpa_replacement.py \
  runners/E4_distribution_guided_cache/run_e4_c_a4_alignment_target_gpa_replacement_ulip_modelnetc_s2.py

bash -n scripts/E4_distribution_guided_cache/02_17_2_ulip_modelnetc_s2_alignment_target_gpa_replacement.sh
```

检查结果：

```text
Python 编译通过。
Shell 语法检查通过。
默认 E1 prompt cache 检查通过：40 类，每类至少 10 条 prompt。
```

## 14. 成功标准

第一阶段只看 ModelNet-C severity=2。

主要比较对象：

| Method | S2 avg |
|---|---:|
| E0 Point-Cache | 54.00 |
| 02_9_2 | 54.7057 |
| 02_11_1 | 54.6757 |
| 07_1 | 54.7114 |
| 02_16_1 diagnostic reproduction | 54.6957 |

成功标准：

```text
02_17_2 S2 avg >= 02_9_2 S2 avg
```

更理想：

```text
02_17_2 S2 avg >= 07_1 S2 avg
```

逐扰动重点：

1. `add_local` 不能明显回落，因为这是 `02_9_2` 的主要收益来源。
2. `rotate` 不能继续下降，因为 `02_9_2` 与 `02_11` 在 rotate 上都有负作用。
3. `add_global` 若能提升，说明 alignment-target 替换修复了原先的过严或错位替换问题。
4. `jitter` 若能维持或提升，说明更干净的 GPA/local cache 对噪声扰动有帮助。

## 15. 风险与失败解释

潜在风险：

1. `joint_score` 最低样本可能是低熵且真实正确的边界样本，替换它可能降低类别多样性。
2. alignment-target replacement 可能让 GPA-cache 过度集中到分布中心，削弱对 shifted samples 的适应。
3. 如果 text distribution 本身误导，`joint_score` 最低不一定代表样本质量最低。
4. 宽松低熵门控使用 `max_entropy_in_GPA_cache`，可能接收一些比 target 更高熵的样本。

如果结果下降，需要区分两类失败：

```text
失败 A：GPA-cache 更干净了，但 final 下降。
说明 final logits 对 GPA/local cache 的使用方式不合适。

失败 B：GPA-cache joint score 没有改善，final 也下降。
说明 joint_score 本身不能稳定定义 alignment quality。
```

因此本实验必须同时看准确率和 GPA-cache 质量诊断。

## 16. 当前判断

`02_17_2` 是比 `02_11` 更符合 GPA-cache 定义的实验。

它不再问：

```text
新样本能不能替换最高熵样本？
```

而是问：

```text
新样本能不能修复当前类别中 alignment 最差的位置？
```

如果有效，论文叙事可以收敛为：

```text
GPA-cache is maintained as an alignment-purified memory by replacing the least aligned prototype under an entropy safety gate.
```

中文表述：

```text
DPC-Point 将 GPA-cache 维护为对齐净化记忆库：
在低熵安全门控下，优先替换当前类别中 text-visual alignment 最差的原型。
```
