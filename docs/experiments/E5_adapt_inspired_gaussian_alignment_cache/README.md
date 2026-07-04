# E5：受 ADAPT 启发的 Gaussian Alignment Point-Cache 实验计划

日期：2026-06-10

## 0. 2026-06-10 阶段更新

E5-A0/A1 已经完成 `add_global_2` 和 `add_local_2` 的初步诊断，结果显示 standalone shared-covariance GDA 不能作为当前 Point-Cache/E4-C 框架的直接替代分类器或强最终 logits 专家。

已完成结果：

| corruption | 原始 Point-Cache/E4 分支 | standalone GDA | 判断 |
|---|---:|---:|---|
| add_global_2 | 47.89 | 47.40 | 负向 |
| add_local_2 | 50.85 | 49.39 | 负向 |

因此，E5 后续主线已从最初的 `standalone GDA / direct GDA logits fusion` 修正为：

```text
E5-B：text-prior posterior prototype residual
E5-C：shared covariance 只作为 cache replacement metric
E5-D：基于有效样本数的动态 text-visual fusion weight
```

新的中文设计文档见：

```text
docs/experiments/E5_adapt_inspired_gaussian_alignment_cache/E5_BCD_posterior_prototype_residual_design.md
```

当前 E5-B0/B1 已新增独立代码和脚本：

```text
Point-Cache/runners/E5_adapt_inspired_gaussian_alignment_cache/model_e5_b0_b1_posterior_prototype_residual.py
Point-Cache/runners/E5_adapt_inspired_gaussian_alignment_cache/run_e5_b0_b1_ulip_modelnetc_s2_posterior_prototype_residual.py
Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache/01_run_e5_b0_b1_ulip_modelnetc_s2_common.sh
Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache/01_1_ulip_modelnetc_s2_zs_global_local_e5_b0_b1_posterior_prototype_residual_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh
```

阅读本 README 时需要注意：下面第 1 到第 12 节记录的是 E5-A 初始设计和当时的数学推导，仍可作为历史背景和基础设施说明，但不再代表 E5 的当前优先实现路线。当前优先级应以 `E5_BCD_posterior_prototype_residual_design.md` 为准。

## 1. 实验定位

E5 参考的核心论文是：

```text
Backpropagation-Free Test-Time Adaptation via Probabilistic Gaussian Alignment
NeurIPS 2025
```

E5 的目标不是复现 ADAPT，而是在我们现有 Point-Cache / E4-C 框架中，引入一个受 ADAPT 启发的 Gaussian 分布证据模块。

更准确的定位是：

```text
ADAPT-inspired Gaussian auxiliary evidence for Point-Cache TTA
```

不能直接写成：

```text
full ADAPT reproduction
```

原因是：ADAPT 的最终预测来自一个完整的正则化概率目标，联合使用 CLIP prior、GDA likelihood 和 knowledge-bank consistency。我们的 Point-Cache 已经有自己的预测公式：

```text
CLIP logits
+ global entropy cache logits
+ local positive cache logits
- negative cache logits
```

因此，如果我们在 Point-Cache logits 上额外加入 GDA 分数，这只是一个新的辅助专家融合方案，而不是 ADAPT 原始闭式解的直接复现。

## 2. 当前基线

E5 必须优先和当前固定的 E4 最优结果比较。

当前主基线是：

```text
实验：02_9_2
方法：E4-C-A0+E1 text distribution only
E4_TEXT_SCORE_WEIGHT = 0.15
数据集：ModelNet-C
severity = 2
7 类 corruption 平均准确率 = 54.70595045
```

这也是目前 E1/E2/E3/E4 已完成完整 severity-2 结果中的最好记录。

## 3. 为什么需要 E5

E4-C 已经证明：文本-视觉分布引导的 cache replacement 有正收益。但是 E4-C 仍然有两个问题：

1. 分布信息主要用于判断 GPA-Cache 是否替换，尚未系统地进入最终预测。
2. 当前 E4-C 的视觉分布更接近每类 diagonal Gaussian；当每类样本较少时，统计量容易不稳定。

ADAPT 给我们的关键启发是：

```text
高置信 test-time 样本 bank
+ 文本 / CLIP 原型先验
+ shared covariance Gaussian class model
+ 无反传闭式推理
```

这条路线和我们当前工作高度相关，但必须分阶段验证，不能直接把 GDA score 加进最终 logits 后就认为方法成立。

## 4. 数学核心

### 4.1 共享协方差 Gaussian 类条件模型

对测试样本特征 $x \in \mathbb{R}^{d}$ 和类别 $c$，E5 假设：

$$
x \mid y=c \sim \mathcal{N}(\mu_c, \Sigma)
$$

其中：

- $x$：归一化后的 ULIP 点云特征；
- $\mu_c$：类别 $c$ 的均值向量；
- $\Sigma$：所有类别共享的协方差矩阵；
- $d$：特征维度。

在共享协方差假设下，GDA 判别函数为：

$$
g_c(x)
=
\mu_c^\top \Sigma^{-1} x
-
\frac{1}{2}
\mu_c^\top \Sigma^{-1} \mu_c
$$

其中：

- $\mu_c^\top \Sigma^{-1} x$ 衡量样本 $x$ 在 shared inverse covariance metric 下和类别均值 $\mu_c$ 的匹配程度；
- $-\frac{1}{2}\mu_c^\top \Sigma^{-1}\mu_c$ 是类别相关的二次偏置项；
- $g_c(x)$ 越大，表示 GDA 模型越支持类别 $c$。

独立 GDA 分类器的预测为：

$$
\hat{y}_{\mathrm{GDA}}(x)
=
\arg\max_c g_c(x)
$$

### 4.2 Shrinkage inverse covariance

直接求经验协方差矩阵的逆是不安全的，尤其在以下条件下：

```text
特征维度高
StatsBank 样本少
类别覆盖不均匀
部分 early pseudo-label 错误
```

因此 E5 不应使用普通经验协方差逆，而应使用 ADAPT 思路中的 shrinkage inverse：

$$
\Sigma^{-1}
=
d
\left(
(N_B - 1)\Sigma
+
\operatorname{tr}(\Sigma) I_d
\right)^{-1}
$$

其中：

- $N_B$：StatsBank 中的总样本数；
- $I_d$：$d \times d$ 单位矩阵；
- $\operatorname{tr}(\Sigma)$：协方差矩阵的迹；
- $d$：特征维度。

这一点是 E5 的关键约束：

```text
主实验必须使用 shared covariance + shrinkage inverse。
不要把 plain covariance inverse 作为默认方案。
```

### 4.3 文本原型作为 prior mean

E5 中，文本原型不应该替换最终分类器，而应该作为类别均值的先验：

$$
\hat{\mu}_c
$$

其中 $\hat{\mu}_c$ 可以来自：

```text
manual_full 文本原型
或 E1 cached prompt fusion 文本原型
```

StatsBank 给出的视觉经验均值记为：

$$
\mu'_c
=
\frac{
\sum_{i \in B_c} w_{i,c} x_i
}{
\sum_{i \in B_c} w_{i,c}
}
$$

其中：

- $B_c$：类别 $c$ 的 StatsBank；
- $x_i$：StatsBank 中的样本特征；
- $w_{i,c}$：样本 $i$ 对类别 $c$ 的置信权重，可以先从 base/manual_full 预测置信度开始。

最终类别均值写成：

$$
\mu_c
=
\alpha \mu'_c
+
(1-\alpha)\hat{\mu}_c
$$

其中：

- $\alpha$ 控制测试时视觉统计和文本先验之间的权衡；
- ADAPT 中视觉历史统计权重较高；
- E5 第一版建议使用：

```text
alpha = 0.9
```

但 $\alpha$ 后续应该作为可消融参数保留。

## 5. GDA 分数归一化

E5 默认使用：

```text
sample-wise class z-score + clipping
```

也就是：对同一个测试样本 $x$，先计算所有类别的 GDA 分数：

$$
\{g_1(x), g_2(x), \dots, g_C(x)\}
$$

然后计算该样本内部的类别均值：

$$
m(x)
=
\frac{1}{C}
\sum_{j=1}^{C} g_j(x)
$$

再计算该样本内部的类别标准差：

$$
s(x)
=
\sqrt{
\frac{1}{C}
\sum_{j=1}^{C}
\left(g_j(x)-m(x)\right)^2
+
\epsilon
}
$$

最后得到归一化后的 GDA 分数：

$$
\tilde{g}_c(x)
=
\operatorname{clip}
\left(
\frac{g_c(x)-m(x)}{s(x)},
-\kappa,
\kappa
\right)
$$

推荐默认值：

```text
epsilon = 1e-6
kappa = 3
```

选择这种归一化的原因：

1. 最终分类只关心同一样本内部的类别竞争关系；
2. sample-wise z-score 能抵消 shared covariance 估计尺度不稳定；
3. 它不引入新的历史状态，避免 running normalization 被 early error 污染；
4. 它比 min-max 更不容易被极端类别分数支配；
5. 它比 raw GDA score 更适合和 Point-Cache logits 做融合。

不建议作为第一版主方案的归一化：

```text
raw g_c                  尺度不稳定
log_softmax(g_c)         对最终 argmax 基本等价 raw g_c，不能解决尺度问题
min-max                  对极端值敏感
per-class running zscore early pseudo-label 污染风险高
global running zscore    类间结构解释不清晰
median/MAD               可做后续补充，不做第一版默认
```

## 6. 独立 StatsBank

E5 必须引入独立 StatsBank，不能直接用 Point-Cache 的正缓存作为 Gaussian 统计来源。

原因是：

```text
Point-Cache positive cache shot_capacity = 3
```

这个容量太小，不适合估计 shared covariance。

E5 的 StatsBank 应该独立维护：

```text
StatsBank capacity L = 16
```

后续可以消融：

```text
L = 8
L = 16
L = 32
```

StatsBank 只负责估计：

```text
类别均值
shared covariance
GDA discriminant score
```

Point-Cache 原本的 positive cache、negative cache 和 final logits 机制保持独立。

第一版 StatsBank 的准入规则建议：

```text
使用 base/manual_full 预测的 negative entropy 作为置信度
按 predicted class 分配到对应 StatsBank
每类保留最高置信度的 L 个样本
不要用 GDA-enhanced prediction 反过来更新 StatsBank
```

## 7. Delayed update 原则

ADAPT 的一个关键点是：当前样本不能立刻更新自己的类别分布。

E5 也必须遵守：

```text
1. 用历史 StatsBank 计算当前样本的 GDA 证据。
2. 输出当前样本的原始 Point-Cache 预测和 GDA 诊断。
3. 当前样本预测完成后，再考虑是否进入 StatsBank。
```

如果不做 delayed update，会出现自强化错误：

```text
当前样本被错误预测
-> 当前特征进入错误类别 StatsBank
-> GDA 分布开始支持这个错误类别
-> final logits 更自信但更错误
```

因此：

```text
delayed update 不是可选增强，而是 E5 的基础机制。
```

## 8. 实验分阶段设计

### 8.1 E5-A0：StatsBank 与 delayed-update 基础设施

目的：

```text
只建立 StatsBank 和 delayed-update 流程，不改变最终预测公式。
```

输出：

```text
original Point-Cache accuracy
StatsBank class coverage
per-class bank size
covariance condition diagnostics
skipped GDA count due to insufficient statistics
```

这一阶段只验证基础设施是否稳定。

### 8.2 E5-A1：standalone shared-covariance GDA 诊断

目的：

```text
判断 GDA 本身是否包含有用的分类信号。
```

独立 GDA 预测：

$$
\hat{y}_{\mathrm{GDA}}(x)
=
\arg\max_c g_c(x)
$$

输出：

```text
original Point-Cache accuracy
standalone GDA accuracy
normalized GDA score statistics
GDA one-vs-rest margin statistics
Point-Cache / GDA agreement rate
```

注意：sample-wise z-score 不改变 $g_c(x)$ 的类别排序，所以 normalized GDA accuracy 和 raw GDA accuracy 通常相同。这里更重要的是保存 normalized score 和 margin 的统计分布。

### 8.3 E5-A2：GDA score 只用于 GPA replacement

目的：

```text
验证 shared-covariance GDA margin 是否比 E4-C diagonal Gaussian score 更适合作为 cache replacement 裁判。
```

对 predicted class $\hat{c}$，定义 one-vs-rest normalized GDA margin：

$$
M_{\hat{c}}(x)
=
\tilde{g}_{\hat{c}}(x)
-
\log
\left(
\frac{1}{C-1}
\sum_{j \ne \hat{c}}
\exp(\tilde{g}_j(x))
\right)
$$

替换规则：

```text
如果 current sample 的 entropy 低于当前 GPA-Cache 中最差样本，
并且 current sample 的 GDA margin 高于最差样本的 GDA margin，
则替换。
```

输出：

```text
original Point-Cache accuracy
replacement count
GDA margin statistics
与 E4-C-A0+E1 02_9_2 对比
```

### 8.4 E5-A3：GDA final logits，双结果输出

目的：

```text
测试 GDA 证据是否应该直接进入最终预测。
```

原始 Point-Cache logits：

$$
z_{\mathrm{orig}}(c)
$$

GDA 增强后的 logits：

$$
z_{\mathrm{gda}}(c)
=
z_{\mathrm{orig}}(c)
+
\gamma \tilde{g}_c(x)
$$

必须同时保存：

```text
original_acc
gda_acc_gamma_0.05
gda_acc_gamma_0.10
gda_acc_gamma_0.20
```

不能直接用 GDA-enhanced logits 替换原公式。必须先双输出，确认它确实比原公式好。

### 8.5 E5-A4：StatsBank 容量消融

目的：

```text
验证 shared covariance 是否受益于更大的独立统计 bank。
```

建议设置：

```text
L = 8
L = 16
```

后续可选：

```text
L = 32
```

只有当 E5-A1/A2/A3 证明 GDA 信号有价值时，才值得继续扩大容量消融。

## 9. 主要风险

### 9.1 Gaussian 假设可能过强

点云类别特征可能是多峰分布。shared covariance 虽然稳定，但可能低估类别内部的多形态结构。

应对方式：

```text
第一版先使用 shared covariance 保证稳定性。
不要声称这是最优分布模型。
把 Gaussian mixture 或 multi-mode Gaussian 作为后续优化方向。
```

### 9.2 GDA 证据可能和 Point-Cache 证据重复计数

Point-Cache local cache 和 StatsBank 都来自 high-confidence test-time 样本。如果同时用于 cache logits 和 GDA logits，可能重复计算同一批证据。

应对方式：

```text
E5-A3 必须保存 original 和 GDA-enhanced 双结果。
gamma 从小值开始。
final logits fusion 只能作为消融，不能直接作为默认主公式。
```

### 9.3 Early pseudo-label 会污染统计量

如果早期预测错了，StatsBank 可能会把错误类别的特征纳入统计，从而污染均值和协方差。

应对方式：

```text
StatsBank admission 使用 base/manual_full 的置信度。
使用 delayed update。
记录 per-class bank coverage。
对样本数不足的类别跳过 GDA 证据。
```

### 9.4 评估协议容易混淆

ADAPT online 是严格历史样本协议。Point-Cache 经常有 build cache in advance 的 warmup 流程。

因此 E5 文档和论文表述必须写清楚当前协议到底是：

```text
online-style
transductive-style
Point-Cache warmup-style
```

如果代码没有严格排除未来样本，就不能声称是 strict online ADAPT。

## 10. 推荐第一步实现

第一步只做：

```text
E5-A0 + E5-A1
```

不要一开始就做 final logits fusion。

最小可运行设置：

```text
Dataset: ModelNet-C
Severity: 2
Corruptions: 与 E4 相同的 7 类 corruption
Baseline: E4-C-A0+E1 textdist-only 02_9_2
StatsBank L: 16
alpha: 0.9
normalization: sample-wise class z-score, clip=3
covariance: shared covariance with shrinkage inverse
update: delayed update
```

必须保存的诊断：

```text
original Point-Cache accuracy
standalone GDA accuracy
normalized GDA score statistics
GDA one-vs-rest margin statistics
Point-Cache / GDA agreement rate
StatsBank coverage per class
covariance condition diagnostics
```

只有当这些诊断显示 GDA 确实有独立信号时，才进入：

```text
E5-A2: GDA replacement
E5-A3: GDA final logits
```

## 11. 当前决策

E5 的基本原则固定为：

```text
先统计量，后预测融合。
先诊断 GDA 是否有独立信号，再考虑加入 final logits。
不允许当前样本更新自己的统计量。
不使用 raw GDA logits 直接加到最终 logits。
不声称完整复现 ADAPT。
```

## 12. 当前已实现版本：E5-A0/A1

实现日期：2026-06-10

当前已经新增第一版 E5-A0/A1 代码。它不是新的最终预测公式，而是在当前最好基线 `02_9_2` 上增加独立诊断分支。

### 12.1 实验名称

```text
00_1_ulip_modelnetc_s2_zs_global_local_e5_a0_a1_adapt_gda_diagnostics_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

### 12.2 基线保持不变

E5-A0/A1 继承当前最好实验 `02_9_2` 的关键设置：

```text
final clip_weights: manual_full
final logits: E4-C-A0 / Point-Cache 原公式
E1 cached descriptions: 只进入 text distribution，不进入 final classifier
E4_TEXT_SCORE_WEIGHT: 0.15
E4_SCORE_NORM_MODE: running_zscore
```

因此，E5-A0/A1 输出的 `summary.csv` 中 `acc` 仍然表示原始 Point-Cache/E4-C-A0+E1-textdist-only 的准确率。

GDA 诊断不会加入：

```text
z_orig(c)
```

也不会计算：

```text
z_new(c) = z_orig(c) + gamma * gda_score(c)
```

这一步留给后续 E5-A3。

### 12.3 新增代码位置

核心模型：

```text
Point-Cache/runners/E5_adapt_inspired_gaussian_alignment_cache/model_e5_a0_a1_adapt_gda_diagnostics.py
```

运行器：

```text
Point-Cache/runners/E5_adapt_inspired_gaussian_alignment_cache/run_e5_a0_a1_ulip_modelnetc_s2_adapt_gda_diagnostics.py
```

通用脚本：

```text
Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache/00_run_e5_a0_a1_ulip_modelnetc_s2_common.sh
```

主脚本：

```text
Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache/00_1_ulip_modelnetc_s2_zs_global_local_e5_a0_a1_adapt_gda_diagnostics_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh
```

### 12.4 运行命令

在项目根目录或 `Point-Cache` 目录下均可运行：

```bash
bash /root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache/00_1_ulip_modelnetc_s2_zs_global_local_e5_a0_a1_adapt_gda_diagnostics_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh 0
```

其中最后的 `0` 表示物理 GPU 编号。

### 12.5 输出目录

实验结果会保存到：

```text
Point-Cache/results/E5_adapt_inspired_gaussian_alignment_cache/00_1_ulip_modelnetc_s2_zs_global_local_e5_a0_a1_adapt_gda_diagnostics_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

主要文件：

```text
summary.csv
logs/*.log
gpa_stats/*_gpa_stats.json
e5_gda_stats/*_e5_gda_stats.json
e5_gda_stats/gda_sample_diagnostics_*.jsonl
```

其中：

```text
gpa_stats
```

记录 E4-C-A0+E1-textdist-only 基线的 GPA replacement 统计。

```text
e5_gda_stats
```

记录 E5-A0/A1 的 StatsBank 和 standalone GDA 诊断。

### 12.6 当前实现的 delayed update

在测试循环中，E5 分支的顺序是：

```text
1. 用当前历史 StatsBank 构建 shared-covariance GDA。
2. 对当前样本计算 standalone GDA 预测和诊断。
3. 记录当前样本的 GDA 是否正确、是否与原 Point-Cache 预测一致。
4. 当前样本预测和诊断完成后，再按 base/manual_full 预测类别写入 StatsBank。
```

因此，E5 的 StatsBank/GDA 分支满足：

```text
当前样本不会参与自己的 GDA 分布估计。
```

需要注意的是：当前代码仍保留 Point-Cache/E4-C 原有的 warmup/build cache 和测试时 cache update 顺序。也就是说，严格的 delayed-update 只作用于 E5 新增的 StatsBank/GDA 分支，不改变 E4 基线本身。

### 12.7 StatsBank 准入规则

当前 StatsBank 只由原始 manual_full / Point-Cache 预测驱动：

```text
pred = argmax(z_clip)
confidence = 1 - normalized_entropy
```

每类最多保存：

```text
L = 16
```

当某类 bank 未满时直接加入；当某类 bank 已满时，只用更高 confidence 的样本替换当前最低 confidence 样本。

StatsBank 不使用：

```text
GDA-enhanced prediction
```

来更新自己，因为当前阶段还没有启用 GDA final logits。

### 12.8 GDA 判别式

当前实现使用 shared covariance Gaussian discriminant：

$$
x \mid y=c \sim \mathcal{N}(\mu_c,\Sigma)
$$

$$
g_c(x)
=
\mu_c^\top \Sigma^{-1}x
-
\frac{1}{2}\mu_c^\top \Sigma^{-1}\mu_c
$$

其中类别均值为：

$$
\mu_c
=
\alpha \mu'_c
+
(1-\alpha)\hat{\mu}_c
$$

符号说明：

```text
x: 当前测试样本的归一化点云特征
mu'_c: StatsBank 中类别 c 的置信度加权视觉均值
hat_mu_c: manual_full 文本原型
alpha: 视觉统计和文本先验的融合权重，当前为 0.9
Sigma: 所有类别共享的协方差矩阵
```

协方差逆使用 shrinkage inverse：

$$
\Sigma^{-1}
=
d
\left(
(N_B-1)\Sigma
+
\operatorname{tr}(\Sigma)I_d
\right)^{-1}
$$

符号说明：

```text
d: 特征维度
N_B: StatsBank 中当前总样本数
tr(Sigma): 共享协方差矩阵的迹
I_d: d 维单位矩阵
```

### 12.9 GDA 归一化

当前实现的归一化只用于诊断统计和后续融合准备，不改变 standalone GDA 的 argmax 结果。

对同一个样本先计算全部类别的 raw GDA 分数：

$$
\{g_1(x),g_2(x),\dots,g_C(x)\}
$$

然后计算样本内部类别均值：

$$
m(x)
=
\frac{1}{C}
\sum_{j=1}^{C}g_j(x)
$$

再计算样本内部类别标准差：

$$
s(x)
=
\sqrt{
\frac{1}{C}
\sum_{j=1}^{C}
\left(g_j(x)-m(x)\right)^2
+
\epsilon
}
$$

归一化分数为：

$$
\tilde{g}_c(x)
=
\operatorname{clip}
\left(
\frac{g_c(x)-m(x)}{s(x)},
-\kappa,
\kappa
\right)
$$

当前默认：

```text
epsilon = 1e-6
kappa = 3.0
```

### 12.10 当前默认超参数

```text
E5_STATSBANK_CAPACITY = 16
E5_GDA_ALPHA = 0.9
E5_GDA_MIN_TOTAL = 8
E5_GDA_MIN_CLASSES = 2
E5_GDA_NORM_EPS = 1e-6
E5_GDA_NORM_CLIP = 3.0
E5_SAVE_SAMPLE_DIAGNOSTICS = 1
```

### 12.11 第一轮结果应该如何解读

第一轮 E5-A0/A1 的核心不是看 `summary.csv` 是否超过 `02_9_2`，因为 final logits 没有改变。

真正需要看的指标是：

```text
standalone GDA accuracy
Point-Cache / GDA agreement rate
GDA margin statistics
StatsBank coverage
covariance diagnostics
```

如果 standalone GDA 明显低于原 Point-Cache，或者 agreement 很低，说明 GDA 证据暂时不适合直接加入 final logits。

如果 standalone GDA 虽然不如 Point-Cache，但在某些 corruption 上能纠正 Point-Cache 的一部分错误，则 E5-A3 可以尝试小 gamma 的双输出融合。

如果 GDA 几乎没有独立信号，下一步应优先回到 E5-A2，把 GDA margin 只用于 cache replacement，而不是直接进入 final logits。

## 13. 2026-06-10 中止后修改：加入 gated override 诊断

首轮运行中，`add_global_2` 和 `add_local_2` 的 standalone GDA 都是负向：

```text
add_global_2: GDA 纠正 150 个原方法错误，但破坏 164 个原方法正确样本，净亏 14。
add_local_2: GDA 纠正 139 个原方法错误，但破坏 174 个原方法正确样本，净亏 35。
```

这说明 raw standalone GDA 不适合直接替换原 Point-Cache 预测，也不支持立刻做 E5-A3 final logits fusion。

因此当前代码新增一个更保守的离线诊断：

```text
gated override diagnostic
```

它不会改变真实 final logits，只是在 `e5_gda_stats/*_e5_gda_stats.json` 中额外模拟以下策略：

```text
如果 GDA 预测和原 Point-Cache 预测不同，
并且 GDA normalized one-vs-rest margin >= threshold，
则用 GDA 预测覆盖原预测；
否则保留原 Point-Cache 预测。
```

数学上，对当前样本 \(x\)，GDA 归一化分数为：

$$
\tilde{g}_c(x)
$$

GDA 预测类别为：

$$
\hat{y}_{GDA}(x)
=
\arg\max_c \tilde{g}_c(x)
$$

GDA 的 one-vs-rest margin 为：

$$
M_{GDA}(x)
=
\tilde{g}_{\hat{y}_{GDA}}(x)
-
\log
\left(
\frac{1}{C-1}
\sum_{j \ne \hat{y}_{GDA}}
\exp(\tilde{g}_j(x))
\right)
$$

门控覆盖规则为：

$$
\hat{y}_{gate}(x)
=
\begin{cases}
\hat{y}_{GDA}(x),
&
\hat{y}_{GDA}(x) \ne \hat{y}_{orig}(x)
\ \text{且}\
M_{GDA}(x) \ge \tau
\\
\hat{y}_{orig}(x),
&
\text{otherwise}
\end{cases}
$$

其中：

```text
hat_y_orig: 原 Point-Cache/E4-C 预测
hat_y_GDA: standalone GDA 预测
tau: 门控阈值
```

当前默认阈值：

```text
E5_GDA_OVERRIDE_THRESHOLDS = 0.5,0.75,1.0,1.25,1.5,2.0
```

输出字段：

```text
gated_override_diagnostics
```

每个阈值都会记录：

```text
acc
overrides
override_rate
fixes
breaks
net_fixes
```

这可以判断 GDA 是否只在高置信 margin 区间有正收益。

### 13.1 只跑指定 corruption

为了避免每次都跑完整 7 类 corruption，现在 runner 支持：

```text
E5_CORRUPTIONS
```

例如只跑 `add_global`：

```bash
E5_CORRUPTIONS=add_global \
bash /root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache/00_1_ulip_modelnetc_s2_zs_global_local_e5_a0_a1_adapt_gda_diagnostics_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh 0
```

只跑 `add_global` 和 `add_local`：

```bash
E5_CORRUPTIONS=add_global,add_local \
bash /root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache/00_1_ulip_modelnetc_s2_zs_global_local_e5_a0_a1_adapt_gda_diagnostics_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh 0
```

如果不设置 `E5_CORRUPTIONS`，默认仍然跑全部 7 类 corruption。
