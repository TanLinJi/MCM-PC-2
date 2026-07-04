# 02_18：D-vMF 方向分布原型计划

日期：2026-06-22

状态：`D-vMF-0` 诊断代码已实现，尚未运行正式结果。

当前已实现入口：

```text
Point-Cache/runners/E4_distribution_guided_cache/model_02_18_1_d_vmf0_directional_stats_diag.py
Point-Cache/runners/E4_distribution_guided_cache/run_02_18_1_ulip_modelnetc_s2_d_vmf0_directional_stats_diag.py
Point-Cache/scripts/E4_distribution_guided_cache/02_18_1_ulip_modelnetc_s2_d_vmf0_directional_stats_diag.sh
```

运行命令：

```bash
cd Point-Cache
bash scripts/E4_distribution_guided_cache/02_18_1_ulip_modelnetc_s2_d_vmf0_directional_stats_diag.sh 0
```

说明：该版本只增加方向分布原型（directional vMF-inspired prototype）诊断统计，不改变 `02_9_2` 的最终得分、缓存更新和 GPA 替换规则。

## 1. 本文目的

本文记录一个独立于 A5 的后续方向：

```text
D-vMF：方向分布原型
Directional von Mises-Fisher-inspired Prototype
```

它的目标不是重做 E3 的类别原型分类，也不是继续做高维高斯（Gaussian）或混合高斯（Gaussian Mixture Model, GMM），而是在归一化特征空间中维护一个轻量的方向分布统计模块，用于：

```text
1. 诊断类别方向分布是否可靠；
2. 辅助全局对齐缓存的替换排序；
3. 在确认有效后，辅助全局缓存分支路由；
4. 最后才考虑作为额外 logits 残差分支。
```

当前结论：

```text
A5 继续处理严格在线时序与分支融合安全化；
D-vMF 作为后续独立路线，从诊断开始；
二者不要混在同一个实验里。
```

## 2. 与已有实验的关系

### 2.1 与 02_9_2 的关系

`02_9_2` 是当前 E4 主线中 S2 / all35 上表现最稳定的版本。其最终得分仍是 Point-Cache 风格投票：

```text
final_logits =
    clip_logits
  + global_entropy_cache_logits
  + gpa_local_cache_logits
  - negative_cache_logits
```

D-vMF 第一阶段不改变这个最终公式。

### 2.2 与 02_16_1 的关系

`02_16_1` 诊断说明：

```text
1. cache-only 分支不够强；
2. GPA global cache diag 高于 GPA local cache；
3. reject_joint 中有大量伪标签正确样本；
4. 简单归一化融合 norm_fusion_offline 低于正式 final；
5. 后续不应直接增强 cache 权重，也不应直接做粗暴融合。
```

D-vMF 需要继承这些结论：

```text
不能把方向分布支持度直接强加到 final logits；
不能让方向分布成为更严格的硬准入条件；
必须先做诊断，确认其与样本正确性、分支可靠性有关。
```

### 2.3 与 A5 的关系

A5 的重点是：

```text
严格在线时序；
alpha 外置；
分支 logits 校准；
负缓存安全化；
global/local 分支融合安全化。
```

D-vMF 的重点是：

```text
方向分布统计；
缓存质量控制；
global 分支路由；
后续可能的方向分布 residual。
```

因此执行顺序建议为：

```text
先完成 A5-0 / A5-T / A5-1a-b-c；
再进入 D-vMF-0。
```

### 2.4 与 E3 类别原型路线的区别

E3 已经验证过“类别原型直接作为分类分支”的路线效果不理想。D-vMF 不能重复这个失败路线。

区别必须写清楚：

```text
E3：类别原型直接参与分类。
D-vMF：方向分布原型先只用于诊断、替换排序、路由，不直接作为强分类器。
```

D-vMF 第一版不使用：

```text
final_logits = clip_logits + alpha * vMF_logits
```

这一步只能放到 D-vMF-3，并且只有前面阶段有效后才考虑。

## 3. 为什么使用方向分布而不是高斯 / GMM

当前 ULIP 分支中，点云全局特征会被 L2 归一化：

```text
pc_feats = pc_feats / ||pc_feats||
```

zero-shot logits 由余弦相似度得到：

```text
clip_logits = 100 * pc_feats @ clip_weights
```

因此当前特征空间更接近：

```text
单位超球面上的方向空间
unit hypersphere / directional space
```

在这个前提下，方向分布比欧氏空间高斯更自然。本文使用的是：

```text
vMF-inspired directional support
受 von Mises-Fisher 分布启发的方向支持度
```

而不是严格的 class-wise vMF probability density。

原因：

```text
1. 第一版不估计 per-class kappa；
2. 第一版不计算 Bessel 归一化常数；
3. 第一版只复用当前 cache affinity 的 beta 作为方向核尺度；
4. 该支持度只用于排序、路由和诊断，不直接作为概率密度解释。
```

## 4. 方向分布统计定义

### 4.1 基本符号

对类别 `c`，维护：

| 符号 | 中文含义 | 英文说明 |
|---|---|---|
| `mu_T[c]` | 文本语义方向 | text semantic direction |
| `a_T[c]` | 文本先验方向累计向量 | text prior accumulator |
| `a_V[c]` | 视觉方向累计向量 | visual accumulator |
| `n_V[c]` | 视觉累计权重 | visual accumulated weight |
| `w2_V[c]` | 视觉权重平方和 | squared visual weights |
| `mu_V[c]` | 当前融合方向中心 | online visual direction center |
| `R_V[c]` | 视觉方向集中度诊断 | visual mean resultant length |
| `n_eff[c]` | 有效视觉样本数 | effective visual sample count |
| `q_dir[c]` | 方向分布可靠性 | directional reliability |

### 4.2 文本先验

文本方向直接来自当前 zero-shot 文本原型：

```text
mu_T[c] = normalize(clip_weights[:, c])
```

文本先验向量：

```text
a_T[c] = n_prior * mu_T[c]
```

第一版建议：

```text
n_prior = 1
```

但注意：

```text
n_prior 只表示文本先验方向；
不能把文本先验当成视觉分布可靠性。
```

### 4.3 视觉累计统计

当某个样本进入正缓存后，且该样本此前没有更新过方向统计，则允许更新其 zero-shot 伪标签类别 `c` 的视觉统计。

样本权重：

```text
w_t = 1 - H_zs(x_t) / log(C)
```

其中：

```text
H_zs：zero-shot entropy
C：类别数
```

更新：

```text
a_V[c] = a_V[c] + w_t * x_t
n_V[c] = n_V[c] + w_t
w2_V[c] = w2_V[c] + w_t^2
```

其中 `x_t` 是 L2 归一化后的点云全局特征。

融合方向中心：

```text
mu_V[c] = normalize(a_T[c] + a_V[c])
```

视觉方向集中度只看真实视觉部分：

```text
R_V[c] = ||a_V[c]|| / (n_V[c] + eps)
```

有效视觉样本数：

```text
n_eff[c] = n_V[c]^2 / (w2_V[c] + eps)
```

成熟度：

```text
m_c = max(0, n_eff[c] - 1) / (max(0, n_eff[c] - 1) + n_prior + eps)
```

方向可靠性：

```text
q_dir[c] = m_c * R_V[c]
```

这样可以避免两个问题：

```text
1. 没有视觉样本时，文本先验导致 R=1 的虚假可靠性；
2. 只有一个视觉样本时，R_V=1 的单样本虚高。
```

边界：

```text
n_V[c] = 0 时：R_V[c] = 0, q_dir[c] = 0
n_eff[c] <= 1 时：q_dir[c] = 0
```

## 5. 方向支持度

第一版不使用：

```text
exp(100 * cosine)
```

因为 `100` 是 zero-shot logit scale，不等价于测试流类别分布集中度。

第一版复用当前全局缓存的 affinity beta：

```text
beta_g = positive.beta
```

方向核：

```text
K_dir(x, c) = exp(-beta_g * (1 - cosine(x, mu_V[c])))
```

方向支持度：

```text
S_dir(x, c) = q_dir[c] * K_dir(x, c)
```

注意：

```text
S_dir 不是严格 vMF likelihood；
它是 vMF-inspired directional support。
```

## 6. 严格时序

D-vMF 必须遵守严格在线时序：

```text
1. 当前样本 x_t 到来；
2. 用 t-1 时刻的 cache 和方向统计计算诊断或 final 分支；
3. 输出当前样本预测；
4. 根据原有 zero-shot pred / entropy / prob_map 更新 cache；
5. 如果样本进入正缓存，且 sample_id 未更新过方向统计，则更新 a_V / n_V / w2_V。
```

禁止：

```text
先用当前样本更新方向统计，再用该统计给当前样本打分。
```

同一个样本如果同时进入多个正缓存：

```text
方向统计最多更新一次。
```

## 7. D-vMF-0：只构建与诊断

### 7.1 目的

`D-vMF-0` 不改变预测逻辑，不改变缓存更新规则，只回答：

```text
方向分布统计是否真的能区分可靠样本与不可靠样本？
```

### 7.2 不改变内容

保持不变：

```text
final logits 不变；
cache update 不变；
GPA replacement 不变；
negative cache 不变；
local cache 不变；
```

第一版 D-vMF-0 还应保持载体实验的原始时序：

```text
如果载体是 02_9_2，则沿用 02_9_2 的 legacy dynamic-init / test loop；
如果载体是 A5-T，则沿用 A5-T 的 strict test loop；
不要在 D-vMF-0 中额外改变时序。
```

这样 D-vMF-0 只诊断方向统计本身，不混入在线时序变量。

### 7.3 记录指标

每个样本记录：

```text
sample_id
corruption
target
zs_pred
final_pred
zs_correct
final_correct
H_zs
w_t
S_dir_pred
S_dir_top1
S_dir_top2
S_dir_margin = S_dir_pred - S_dir_top2
q_dir_pred
R_V_pred
n_V_pred
n_eff_pred
cos(mu_V_pred, mu_T_pred)
direction_stats_updated
positive_cache_accepted
```

每个类别记录：

```text
n_V[c]
n_eff[c]
R_V[c]
q_dir[c]
cos(mu_V[c], mu_T[c])
```

聚合分析：

```text
1. q_dir 非零类别数；
2. q_dir 非零样本比例；
3. S_dir_margin 与 pseudo-label correctness 的关系；
4. q_dir 与 pseudo-label correctness 的关系；
5. R_V 与类别准确率的关系；
6. n_eff 与类别可靠性的关系；
7. 方向统计是否随 corruption 稳定；
8. 前 25% / 中间 50% / 后 25% 测试流区间的方向统计变化。
```

### 7.4 成功标准

D-vMF-0 不是看 final accuracy，而是看诊断信号。

继续到 D-vMF-1 的最低条件：

```text
1. q_dir 非零样本比例不能太低；
2. S_dir_margin 高的样本应有更高伪标签正确率；
3. q_dir 高的类别或样本应更稳定；
4. 方向统计不能高度依赖测试顺序。
```

如果这些不成立：

```text
D-vMF 不应继续进入缓存替换或 final fusion。
```

## 8. D-A0：GPA global cache 分支诊断

### 8.1 目的

在 D-vMF-2 之前，必须确认 GPA global cache 是否具备独立价值。

诊断分支：

```text
align_global_logits
```

它使用 GPA cache 中的全局特征计算 global cache logits，但不参与正式 final。

需要记录：

```text
align global top-1 accuracy
align global 与 zero-shot 一致率
align global 与 entropy global 一致率
align global 在 zero-shot 错误样本上的正确率
align global 在 zero-shot 正确样本上的误伤比例
```

### 8.2 与 02_16_1 的关系

`02_16_1` 已经有 GPA global cache diag：

```text
GPA global cache diag = 49.02%（8469/17276）
GPA local cache = 46.96%（8113/17276）
```

但 D-A0 需要在 A5 的严格时序与 raw logits 规范下重新确认，避免 protocol 差异影响判断。

## 9. D-vMF-1：方向质量用于替换排序

### 9.1 目的

D-vMF-1 只改变 GPA / global alignment cache 的满缓存替换排序，不改变 final logits。

核心原则：

```text
方向分布不作为硬准入条件；
方向分布只参与“替换谁”的排序；
低熵替换触发条件仍保留。
```

### 9.2 替换流程草案

对当前样本 `x_new`，zero-shot 伪标签为 `c`。

如果 `GPA cache[c]` 未满：

```text
按原规则加入。
```

如果 `GPA cache[c]` 已满：

```text
第一步：计算 H_new。
第二步：如果 H_new >= H_max_in_cache[c]，拒绝。
第三步：候选旧样本集合 A = {old_i | H_old_i > H_new}。
第四步：对 A 中每个 old_i 计算 leave-one-out 方向质量 Q_i。
第五步：替换 Q_i 最低的 old_i。
```

方向质量：

```text
Q_i = q_H(i) * S_dir_leave_one_out(x_i, c)
q_H(i) = 1 - H_i / log(C)
```

新样本的方向支持度使用 t-1 时刻统计：

```text
S_dir_new = S_dir_t-1(x_new, c)
Q_new = q_H(new) * S_dir_new
```

旧样本的方向支持度使用 leave-one-out：

```text
a_V_minus_i = a_V[c] - w_i * x_i
n_V_minus_i = n_V[c] - w_i
w2_V_minus_i = w2_V[c] - w_i^2
```

然后重新计算：

```text
mu_V_minus_i
R_V_minus_i
n_eff_minus_i
q_dir_minus_i
S_dir_leave_one_out(x_i, c)
```

### 9.3 为什么需要 leave-one-out

如果旧样本已经参与了方向统计，那么它会增加自身与 `mu_V[c]` 的相似度。

这会造成：

```text
self-support
自我支持偏差
```

因此对旧缓存样本评分时必须扣除它自己的贡献。

### 9.4 重要限制

当前草案只保证：

```text
替换后缓存熵质量不下降；
替换对象由方向质量排序。
```

它不保证：

```text
新样本方向质量一定高于旧样本。
```

如果 D-vMF-1 下降，需要考虑安全版本：

```text
只替换 H_old > H_new 且 Q_old < Q_new 的旧样本。
```

这会更保守，但会降低更新率。

因此 D-vMF-1 必须记录：

```text
Q_new
Q_replaced_old
delta_Q = Q_new - Q_replaced_old
delta_Q < 0 的比例
delta_H = H_replaced_old - H_new
leave-one-out Q_old 与普通 Q_old 的差异
```

如果 `delta_Q < 0` 比例很高且准确率下降，说明 relaxed 替换过激，应切换到 safe 版本。

## 10. D-vMF-2：方向可靠性激活 align residual

### 10.1 前置条件

D-vMF-2 只能在以下条件满足后进行：

```text
1. D-vMF-0 显示方向统计有可靠诊断信号；
2. D-A0 显示 GPA global cache 分支有独立价值；
3. D-vMF-1 没有明显污染 GPA/local cache。
```

### 10.2 不使用简单凸组合

不使用：

```text
global_residual = (1 - r_V) * entropy_global_residual + r_V * align_global_residual
```

原因：

```text
当 r_V 高而 align global 不可靠时，会削弱原本有效的 entropy global 分支。
```

### 10.3 更安全的 residual 激活

使用：

```text
global_residual =
    entropy_global_residual
  + r_V * ReLU(align_global_residual - entropy_global_residual)
```

其中：

```text
entropy_global_residual = ReLU(calib(entropy_global_logits, zs_logits))
align_global_residual = ReLU(calib(align_global_logits, zs_logits))
```

`r_V` 为标量方向路由系数，建议：

```text
r_V = max(0, S_dir_pred - S_dir_top2) / (S_dir_pred + eps)
```

边界：

```text
如果 q_dir_pred = 0，则 r_V = 0。
如果已初始化方向统计类别数不足 2，则 r_V = 0。
```

该公式的含义：

```text
默认保留 entropy global；
只有当方向统计成熟且预测类别具有相对分布优势时，
才让 align global 补充它比 entropy global 更强的那部分类别证据。
```

## 11. D-vMF-3：方向支持度作为 logits residual

D-vMF-3 不是当前优先实验。

只有当 D-vMF-0/1/2 有稳定收益后，才考虑：

```text
final_logits =
    previous_final_logits
  + alpha_dir * ReLU(calib(S_dir_all_classes, zs_logits))
```

风险：

```text
这一步最容易退化回 E3 类别原型分类；
如果 S_dir 的类别区分度不强，会放大弱原型噪声；
alpha_dir 是新增超参数，第一阶段不建议引入。
```

## 12. 诊断与报告要求

报告准确率必须同时给出样本数量，例如：

```text
54.71%（9451/17276）
```

D-vMF 所有诊断必须记录：

```text
1. 方向统计更新样本数；
2. 方向统计更新样本离线伪标签正确率；
3. q_dir 非零比例；
4. r_V 非零比例；
5. S_dir_pred / S_dir_top2 / S_dir_margin 分布；
6. q_dir 高低分桶准确率；
7. D-vMF-1 中 ΔH、ΔQ、ΔQ < 0 比例；
8. leave-one-out 前后 Q_old 差异；
9. 不同 corruption 的逐类型收益；
10. 测试流顺序敏感性。
```

## 13. 后续实现建议

第一步只实现：

```text
D-vMF-0
```

如果 D-vMF-0 不满足诊断标准，不继续实现 D-vMF-1/2/3。

计划文件名：

```text
Point-Cache/runners/E4_distribution_guided_cache/run_02_18_1_ulip_modelnetc_s2_d_vmf0_directional_stats_diag.py
Point-Cache/scripts/E4_distribution_guided_cache/02_18_1_ulip_modelnetc_s2_d_vmf0_directional_stats_diag.sh
```

建议结果目录：

```text
Point-Cache/results/E4_distribution_guided_cache/02_18_1_ulip_modelnetc_s2_d_vmf0_directional_stats_diag/
```

## 14. 参考文献

本文只把以下文献作为方法动机，不直接照搬其完整算法：

```text
1. Banerjee et al., Clustering on the Unit Hypersphere using von Mises-Fisher Distributions.
2. CLIP-Enhance: Improving CLIP Zero-Shot Classification via von Mises-Fisher Clustering.
3. BayesMM: Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning.
```

解释：

```text
Banerjee et al. 支持单位超球面方向建模；
CLIP-Enhance 支持 CLIP 类归一化特征的 vMF 几何解释；
BayesMM 支持用在线分布统计缓解 cache 历史信息丢失和 heuristic fusion 不稳定。
```
