# 02_8 E4-C-A0+E1 Text Distribution Only 实验分析

> 日期：2026-06-09
>
> 实验编号：`02_8`
>
> 结论：该实验取得稳定的小幅正收益，是当前 E4 系列中最好的 severity=2 结果。

## 1. 实验目的

本实验用于修正 `02_7` 的耦合问题，并验证一个更干净的假设：

> E1 的多模态 2D/3D LLM 描述不直接替换最终分类器的文本原型，而只作为 E4-C 的 text distribution prior，参与 GPA-Cache 替换准则。

`02_7` 使用 `manualfull_llm_dynamic_init` 作为整体 prompt source，导致以下模块全部被 E1 prompt fusion 改变：

- 最终 `clip_weights`；
- cache 初始化阶段的 pseudo-label；
- Global Entropy Cache；
- Negative Cache；
- 最终 logits 的 zero-shot 文本分类器。

这使得 `02_7` 不再是“在 E4-C-A0 中加入 E1 描述”的干净消融，而是改变了 Point-Cache 的基础分类器。已有 E2 结果也显示，直接把 `manual_full_llm_fusion` 用进完整 Point-Cache 会显著拉低 `add_global_2`。

因此 `02_8` 采用解耦设计：

- `clip_weights`：保持 `manual_full`，与 E4-C-A0 完全一致；
- final logits：保持 E4-C-A0 原公式；
- E1 cached LLM descriptions：只用于构建 E4-C 的 text distribution；
- GPA replacement：继续使用 running z-score 归一化后的 text-visual joint score。

## 2. 实验入口

运行脚本：

```text
Point-Cache/scripts/E4_distribution_guided_cache/02_8_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_textdist_only_score_norm_accepted_history_text_visual_distribution_guided_gpa_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh
```

Runner：

```text
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a0_e1_textdist_only_ulip_modelnetc_s2_accepted_history_text_visual_distribution_guided_gpa.py
```

结果目录：

```text
Point-Cache/results/E4_distribution_guided_cache/02_8_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_textdist_only_score_norm_accepted_history_text_visual_distribution_guided_gpa_manual_full_clip_manualfull_llm_dynamic_init_textdist/
```

E1 共享 prompt 缓存：

```text
Point-Cache/results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json
```

该脚本在运行前检查缓存完整性：ModelNet-C 40 类，每类至少 10 条 LLM 描述。缓存不完整时直接退出，避免重新调用 LLM API。

## 3. 实验设置

基础设置：

| 项目 | 设置 |
|---|---|
| Backbone | ULIP |
| Dataset | ModelNet-C |
| Severity | 2 |
| Corruptions | 7 种：`add_global`, `add_local`, `dropout_global`, `dropout_local`, `rotate`, `scale`, `jitter` |
| Cache method | `zs_global_local` |
| Cache type | hierarchical |
| Positive cache capacity | 3 |
| Positive cache alpha / beta | 4.0 / 3.0 |
| Negative cache capacity | 2 |
| Negative cache alpha / beta | 0.117 / 1.0 |

E4-C-A0 设置：

| 项目 | 设置 |
|---|---|
| `E4_SCORE_NORM_MODE` | `running_zscore` |
| `E4_SCORE_NORM_MIN_COUNT` | 8 |
| `E4_SCORE_NORM_EPS` | `1e-6` |
| `E4_SCORE_NORM_CLIP` | 0 |
| `E4_TEXT_SCORE_WEIGHT` | 0.1 |
| `E4_DIST_EPS` | `1e-4` |
| `E4_TEXT_DIST_EPS` | `1e-4` |
| `E4_DIST_MIN_VAR` | `1e-4` |
| `E4_TEXT_DIST_MIN_VAR` | `1e-4` |

Prompt 设置：

| 模块 | Prompt source |
|---|---|
| final `clip_weights` | `manual_full` |
| E4 text distribution | `manualfull_llm_dynamic_init` |

## 4. 文本分布构建方式

对于类别 \(c\)，`manual_full` 提供 64 条手工模板，E1 cache 提供 10 条 LLM 多模态 2D/3D 描述。

记手工模板文本嵌入为：

\[
S_c = \{s_{c,1}, s_{c,2}, \ldots, s_{c,64}\}.
\]

记 E1 LLM 描述文本嵌入为：

\[
D_c = \{d_{c,1}, d_{c,2}, \ldots, d_{c,10}\}.
\]

`02_8` 不是把 74 条文本等权混合，而是按 E1 的分支权重构建加权文本分布：

\[
w^S_i = \frac{0.75}{64},
\]

\[
w^D_j = \frac{0.25}{10}.
\]

文本均值为：

\[
\mu^t_c
=
\sum_{i=1}^{64} w^S_i s_{c,i}
+
\sum_{j=1}^{10} w^D_j d_{c,j}.
\]

文本方差使用对角加权方差：

\[
(\sigma^t_c)^2
=
\sum_{i=1}^{64} w^S_i (s_{c,i}-\mu^t_c)^2
+
\sum_{j=1}^{10} w^D_j (d_{c,j}-\mu^t_c)^2.
\]

其中：

- \(\mu^t_c\)：类别 \(c\) 的文本分布均值；
- \((\sigma^t_c)^2\)：类别 \(c\) 的文本分布对角方差；
- \(s_{c,i}\)：类别 \(c\) 的第 \(i\) 个手工模板嵌入；
- \(d_{c,j}\)：类别 \(c\) 的第 \(j\) 个 LLM 描述嵌入；
- \(w^S_i\)：手工模板分支内单条文本权重；
- \(w^D_j\)：LLM 描述分支内单条文本权重。

文本分布得分为：

\[
s_t(x,c)
=
-
\frac{1}{d}
\sum_{k=1}^{d}
\frac{(x_k-\mu^t_{c,k})^2}{(\sigma^t_{c,k})^2+\epsilon}.
\]

其中：

- \(x\)：当前测试样本的视觉特征；
- \(d\)：特征维度；
- \(x_k\)：样本特征第 \(k\) 维；
- \(\mu^t_{c,k}\)：类别 \(c\) 的文本均值第 \(k\) 维；
- \((\sigma^t_{c,k})^2\)：类别 \(c\) 的文本方差第 \(k\) 维；
- \(\epsilon\)：数值稳定项。

视觉分布得分仍沿用 E4-C-A0：

\[
s_v(x,c)
=
-
\frac{1}{d}
\sum_{k=1}^{d}
\frac{(x_k-\mu^v_{c,k})^2}{(\sigma^v_{c,k})^2+\epsilon}.
\]

归一化后的 joint score 为：

\[
s_{\text{joint}}(x,c)
=
\widetilde{s}_v(x,c)
+
0.1 \cdot \widetilde{s}_t(x,c).
\]

其中：

- \(\widetilde{s}_v(x,c)\)：running z-score 归一化后的视觉分布得分；
- \(\widetilde{s}_t(x,c)\)：running z-score 归一化后的文本分布得分；
- 0.1：当前文本分布权重，即 `E4_TEXT_SCORE_WEIGHT=0.1`。

## 5. 最终预测公式

`02_8` 不修改 E4-C-A0 的最终 logits 公式：

\[
z_{\text{final}}
=
z_{\text{clip}}
+
z_{\text{entropy-global-cache}}
+
z_{\text{gpa-local-cache}}
-
z_{\text{negative-cache}}.
\]

其中：

- \(z_{\text{clip}}\)：基于 `manual_full` 文本原型的 zero-shot logits；
- \(z_{\text{entropy-global-cache}}\)：Global Entropy Cache 的正缓存 logits；
- \(z_{\text{gpa-local-cache}}\)：由 GPA-Cache 控制写入的 local cache logits；
- \(z_{\text{negative-cache}}\)：negative cache logits；
- GPA global cache 仍不直接参与最终 logits。

这点非常重要：E1 描述只影响 GPA replacement，不影响最终分类器的文本原型。

## 6. 实验结果

`02_8` 完整结果：

| corruption | accuracy |
|---|---:|
| `add_global_2` | 47.77 |
| `add_local_2` | 51.30 |
| `dropout_global_2` | 58.87 |
| `dropout_local_2` | 57.05 |
| `rotate_2` | 61.06 |
| `scale_2` | 56.16 |
| `jitter_2` | 50.49 |
| **Average** | **54.6714** |

## 7. 与关键实验对比

| 实验 | 平均准确率 |
|---|---:|
| E2 manual_full global_local | 54.0000 |
| E2 manual_full_llm_fusion global_local | 54.2086 |
| E4-C | 54.4986 |
| E4-C-A0 | 54.5214 |
| E4-C-A0-c3 one-vs-rest GPA global odds, gamma=0.05 | 54.5271 |
| **02_8 E4-C-A0+E1 textdist-only** | **54.6714** |

相对 E4-C-A0：

| corruption | E4-C-A0 | 02_8 | delta |
|---|---:|---:|---:|
| `add_global_2` | 47.73 | 47.77 | +0.04 |
| `add_local_2` | 50.69 | 51.30 | +0.61 |
| `dropout_global_2` | 58.51 | 58.87 | +0.36 |
| `dropout_local_2` | 56.85 | 57.05 | +0.20 |
| `rotate_2` | 61.06 | 61.06 | +0.00 |
| `scale_2` | 56.40 | 56.16 | -0.24 |
| `jitter_2` | 50.41 | 50.49 | +0.08 |
| **Average** | **54.5214** | **54.6714** | **+0.1500** |

相对 `E4-C-A0-c3 one-vs-rest GPA global odds, gamma=0.05`：

| corruption | c3 gamma=0.05 | 02_8 | delta |
|---|---:|---:|---:|
| `add_global_2` | 47.73 | 47.77 | +0.04 |
| `add_local_2` | 50.69 | 51.30 | +0.61 |
| `dropout_global_2` | 58.55 | 58.87 | +0.32 |
| `dropout_local_2` | 56.81 | 57.05 | +0.24 |
| `rotate_2` | 61.10 | 61.06 | -0.04 |
| `scale_2` | 56.40 | 56.16 | -0.24 |
| `jitter_2` | 50.41 | 50.49 | +0.08 |
| **Average** | **54.5271** | **54.6714** | **+0.1443** |

## 8. 结果分析

### 8.1 该结果是明确正收益

`02_8` 相比 E4-C-A0 提升 `+0.1500`，相比之前 E4 系列最好结果 `E4-C-A0-c3 gamma=0.05` 提升 `+0.1443`。

这个提升幅度不大，不能夸大为显著突破，但它有两个重要意义：

1. 它修正了 `02_7` 的实验耦合问题；
2. 它证明 E1 的多模态描述作为 text distribution prior 是有效的。

### 8.2 正收益主要来自局部扰动和 dropout

主要提升项是：

```text
add_local_2       +0.61
dropout_global_2  +0.36
dropout_local_2   +0.20
```

这说明 E1 描述对“局部结构扰动”和“点缺失”场景更有帮助。一个合理解释是：LLM 生成的多视角 2D/3D 描述包含部件、结构、形状和语义线索，能够在 GPA replacement 阶段帮助过滤不符合类别文本分布的候选样本。

### 8.3 `scale_2` 是唯一明显负项

`scale_2` 从 `56.40` 下降到 `56.16`，差值为 `-0.24`。

这说明当前 E1 text distribution 对尺度变化不一定稳。可能原因包括：

- 文本描述更强调典型物体结构，而 scale corruption 改变的是整体尺度，不一定改变类别语义；
- text score 在 scale corruption 下可能对有效样本过度惩罚；
- 当前固定 `E4_TEXT_SCORE_WEIGHT=0.1` 可能对部分 corruption 偏大。

因此后续需要做 text weight sweep，而不是直接把该设置视为最终最优。

### 8.4 `02_7` 不再作为有效主实验

`02_7` 的问题在于它把 E1 prompt fusion 接入了最终 `clip_weights`，导致基础分类器、缓存初始化、负缓存和最终 logits 同时变化。

`02_8` 的结果说明：

```text
E1 描述可以作为 E4 text distribution prior；
但不应该直接替换 E4-C-A0 的 manual_full clip classifier。
```

这是一个重要阶段性结论。

## 9. 阶段性结论

`02_8` 是当前 E4 系列中最好的 severity=2 结果：

```text
E4-C-A0+E1 textdist-only average = 54.6714
```

可以将其作为当前 E4 的主正向结果之一。

该实验支持以下论文叙事：

> Multimodal textual descriptions are useful when used as a distributional prior for cache purification, but directly replacing the base textual classifier may hurt some corruptions. A decoupled design that preserves the original classifier while injecting multimodal descriptions into the cache replacement criterion yields better robustness.

中文表述：

> 多模态文本描述适合作为缓存净化阶段的分布先验，而不适合直接替换基础文本分类器。保持原始 `manual_full` 分类器不变，只将 E1 描述注入 GPA-Cache 替换准则，可以获得更稳定的正收益。

## 10. 后续建议

当前不建议继续扩大 `02_7`。

E4 内部的下一步建议是基于 `02_8` 做非常小的 text weight sweep：

```text
E4_TEXT_SCORE_WEIGHT = 0.05
E4_TEXT_SCORE_WEIGHT = 0.10
E4_TEXT_SCORE_WEIGHT = 0.15
```

目的：

1. 判断 `0.1` 是否为最优文本权重；
2. 检查 `scale_2` 的负收益能否缓解；
3. 确认 E1 text distribution 的收益是否稳定。

E5 应单独启动，用于 ADAPT / probabilistic Gaussian alignment 启发的 shared covariance 和 GDA logits，不应混入当前 E4 结论。

## 11. 建议单独提交文件

建议将 `02_8` 作为独立正收益实验提交，至少包含：

```text
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a0_e1_textdist_only_ulip_modelnetc_s2_accepted_history_text_visual_distribution_guided_gpa.py

Point-Cache/scripts/E4_distribution_guided_cache/02_8_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_textdist_only_score_norm_accepted_history_text_visual_distribution_guided_gpa_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh

docs/experiments/E4_distribution_guided_cache/02_8_e4_c_a0_e1_textdist_only_analysis.md
```

如果基础 E4-C-A0 的 shared runner/model/common 脚本还未提交，则需要先确认这些依赖是否已经在历史提交中存在，否则 `02_8` 单独提交后在干净环境中可能无法复现。
