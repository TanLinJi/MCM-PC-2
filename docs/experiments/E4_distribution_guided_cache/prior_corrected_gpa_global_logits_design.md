# E4-C-A0-c2 Prior-Corrected GPA Global Logits Design

> Created: 2026-06-09
>
> Status: user confirmed the design; implementation added on 2026-06-09; experiment not yet run.

## 1. Motivation

E4-C-A0-c1 tested whether GPA global cache should directly participate in the final prediction logits.

The original A0 formula is:

\[
z_{\mathrm{orig}}(c)
=
z_{\mathrm{clip}}(c)
+
z_{\mathrm{entropy\_global}}(c)
+
z_{\mathrm{gpa\_local}}(c)
-
z_{\mathrm{negative}}(c).
\]

E4-C-A0-c1 added raw GPA global logits:

\[
z_{\mathrm{raw}}(c)
=
z_{\mathrm{orig}}(c)
+
z_{\mathrm{gpa\_global\_raw}}(c).
\]

The raw GPA global term is:

\[
z_{\mathrm{gpa\_global\_raw}}(c)
=
\alpha
\sum_{i\in G_c}
\exp\left(\beta_G(\cos(x,g_i)-1)\right).
\]

Where:

- \(x\): current test sample global feature.
- \(c\): class index.
- \(G_c\): GPA global cache entries assigned to class \(c\).
- \(g_i\): the \(i\)-th global feature in GPA cache class \(c\).
- \(\alpha\): PointCache positive cache weight, currently \(4.0\).
- \(\beta_G\): cache affinity sharpness, currently inherited from PointCache positive cache \(\beta=3.0\).

E4-C-A0-c1 result:

\[
\mathrm{avg}(z_{\mathrm{orig}})=54.52,
\quad
\mathrm{avg}(z_{\mathrm{raw}})=54.39.
\]

The raw GPA global addition decreases average accuracy by \(0.13\). It improves `rotate_2` and `scale_2` slightly, but hurts `add_global_2`, `jitter_2`, and several dropout settings.

This suggests GPA global cache contains some useful evidence, but raw logits addition is mathematically under-calibrated.

## 2. Why Raw GPA Global Logits Are Not Well-Calibrated

The raw GPA global term uses a class-wise kernel sum:

\[
\sum_{i\in G_c}K(x,g_i),
\quad
K(x,g_i)
=
\exp\left(\beta_G(\cos(x,g_i)-1)\right).
\]

This is not yet a probability, log-probability, or calibrated logit. It has several problems.

### 2.1 It Is Not Class-Size Normalized

If class \(c\) has more cache entries, its raw score can be larger simply because \(|G_c|\) is larger:

\[
\sum_{i\in G_c}K(x,g_i)
\propto
|G_c|.
\]

This creates an implicit and uncontrolled class prior.

Even though current shot capacity limits each class to at most three GPA samples, some classes may still have fewer than three samples, especially early or under difficult corruptions. Raw summation therefore mixes class evidence with cache population size.

### 2.2 It Is Not Cross-Class Normalized

For a classification decision, GPA global evidence should answer:

\[
\text{which class is most likely under the GPA global cache?}
\]

Raw class scores do not form a distribution:

\[
\sum_c z_{\mathrm{gpa\_global\_raw}}(c)\neq 1.
\]

Thus, directly adding them to logits treats unnormalized positive affinity as if it were calibrated class evidence.

### 2.3 It Double-Counts Correlated Evidence

The original formula already uses:

\[
z_{\mathrm{entropy\_global}}(c)
\]

from global positive cache, and:

\[
z_{\mathrm{gpa\_local}}(c)
\]

from GPA-controlled local cache.

Adding raw GPA global logits assumes:

\[
z_{\mathrm{entropy\_global}},
z_{\mathrm{gpa\_global}},
z_{\mathrm{gpa\_local}}
\]

are independent evidence sources. They are not independent:

- They use the same backbone feature space.
- They are produced from the same test stream.
- They rely on pseudo-label assignments.
- GPA global and GPA local are generated from the same GPA-accepted samples.

Therefore direct addition can over-amplify pseudo-label evidence.

## 3. Proposed Prior-Corrected GPA Global Evidence

The proposed solution is to convert GPA global cache into a normalized auxiliary posterior \(q_G(c\mid x)\), then fuse it as a prior-corrected likelihood-ratio term.

Define the GPA global kernel:

\[
K_G(x,g_i)
=
\exp\left(\beta_G(\cos(x,g_i)-1)\right).
\]

Define the class-wise GPA global density estimator:

\[
\hat{p}_G(x\mid c)
=
\frac{1}{|G_c|}
\sum_{i\in G_c}
K_G(x,g_i).
\]

Where:

- \(\hat{p}_G(x\mid c)\): GPA global cache estimated class-conditional evidence.
- \(|G_c|\): number of GPA global cache entries for class \(c\).
- The factor \(1/|G_c|\) removes raw cache-count bias.

With a class prior \(\pi_c\), define:

\[
q_G(c\mid x)
=
\frac{
\pi_c \hat{p}_G(x\mid c)
}{
\sum_j \pi_j \hat{p}_G(x\mid j)
}.
\]

Expanding \(\hat{p}_G(x\mid c)\):

\[
q_G(c\mid x)
=
\frac{
\pi_c
\frac{1}{|G_c|}
\sum_{i\in G_c}
\exp\left(\beta_G(\cos(x,g_i)-1)\right)
}{
\sum_j
\pi_j
\frac{1}{|G_j|}
\sum_{i\in G_j}
\exp\left(\beta_G(\cos(x,g_i)-1)\right)
}.
\]

This makes GPA global cache a normalized auxiliary classifier.

## 4. Why Use \(\log \frac{q_G(c\mid x)}{\pi_c}\)

By Bayes' rule:

\[
q_G(c\mid x)
=
\frac{
\pi_c \hat{p}_G(x\mid c)
}{
\hat{p}_G(x)
},
\]

where:

\[
\hat{p}_G(x)
=
\sum_j \pi_j \hat{p}_G(x\mid j).
\]

Then:

\[
\frac{q_G(c\mid x)}{\pi_c}
=
\frac{
\hat{p}_G(x\mid c)
}{
\hat{p}_G(x)
}.
\]

Taking logarithm:

\[
\log\frac{q_G(c\mid x)}{\pi_c}
=
\log \hat{p}_G(x\mid c)
-
\log \hat{p}_G(x).
\]

For a fixed sample \(x\), \(\hat{p}_G(x)\) is constant with respect to class \(c\). Therefore this term preserves the class ranking of the GPA global class-conditional likelihood while removing the class prior.

This is the key difference from adding:

\[
\log q_G(c\mid x).
\]

Adding \(\log q_G(c\mid x)\) includes \(\log \pi_c\), which can double-count class prior. Adding:

\[
\log\frac{q_G(c\mid x)}{\pi_c}
\]

is a prior-corrected evidence term.

## 5. Final Fusion Formula

The proposed final formula is:

\[
z_{\mathrm{new}}(c)
=
z_{\mathrm{orig}}(c)
+
\gamma
\log
\frac{q_G(c\mid x)}
{\pi_c}.
\]

Equivalently:

\[
p_{\mathrm{new}}(c\mid x)
\propto
p_{\mathrm{orig}}(c\mid x)
\left(
\frac{q_G(c\mid x)}
{\pi_c}
\right)^\gamma.
\]

Where:

- \(p_{\mathrm{orig}}(c\mid x)=\mathrm{softmax}(z_{\mathrm{orig}})_c\).
- \(q_G(c\mid x)\): normalized GPA global posterior.
- \(\pi_c\): class prior.
- \(\gamma\): strength of GPA global auxiliary evidence.

This is a product-of-experts style fusion:

\[
\text{original posterior}
\times
\text{prior-corrected GPA global evidence}^{\gamma}.
\]

For implementation, it is cleaner to avoid computing \(q_G(c\mid x)\) and then dividing by \(\pi_c\). Define:

\[
r_c(x)
=
\hat{p}_G(x\mid c)
=
\frac{1}{|G_c|}
\sum_{i\in G_c}
K_G(x,g_i).
\]

Define the prior-weighted normalizer:

\[
m(x)
=
\sum_j \pi_j r_j(x).
\]

Then:

\[
\log
\frac{q_G(c\mid x)}
{\pi_c}
=
\log
\frac{r_c(x)}
{m(x)}.
\]

The numerically stable implementation should use:

\[
e_G(c,x)
=
\log(r_c(x)+\epsilon)
-
\log(m(x)+\epsilon).
\]

Finally:

\[
z_{\mathrm{new}}(c)
=
z_{\mathrm{orig}}(c)
+
\gamma e_G(c,x).
\]

## 6. Prior Choice

For ModelNet-C, the default choice should be uniform prior:

\[
\pi_c=\frac{1}{C}.
\]

Where:

- \(C=40\) for ModelNet40 / ModelNet-C.

Under uniform prior:

\[
\log\frac{q_G(c\mid x)}{\pi_c}
=
\log q_G(c\mid x)+\log C.
\]

Since \(\log C\) is the same for every class, it does not affect argmax classification. However, we should still keep the prior-corrected formula in the documentation and implementation, because it is mathematically clearer and supports non-uniform priors later.

Do not use raw cache class counts as \(\pi_c\) in the first experiment. Cache counts are themselves pseudo-label dependent and may introduce the bias this method is trying to remove.

## 7. Recommended Experiment Name

Use a new independent experiment variant:

```text
E4-C-A0-c2
```

Suggested full name:

```text
E4-C-A0-c2: prior-corrected normalized GPA global logits
```

It should be based on:

```text
E4-C-A0
```

not on:

```text
E4-C-A0b
```

Reason:

- A0 z-score is the current strongest E4-C normalization setting.
- A0b robust normalization was weaker.
- c2 should isolate the effect of prior-corrected GPA global fusion, not mix it with robust score normalization.

## 8. Proposed Experimental Settings

Backbone and dataset:

```text
backbone: ULIP
dataset: ModelNet-C
severity: 2
corruptions:
  add_global
  add_local
  dropout_global
  dropout_local
  rotate
  scale
  jitter
```

Base cache settings:

```text
positive.shot_capacity = 3
positive.alpha = 4.0
positive.beta = 3.0
negative.enabled = true
negative.alpha = 0.117
negative.beta = 1.0
```

E4 score normalization:

```text
E4_SCORE_NORM_MODE=running_zscore
E4_SCORE_NORM_MIN_COUNT=8
E4_SCORE_NORM_EPS=1e-6
E4_SCORE_NORM_CLIP=0
E4_TEXT_SCORE_WEIGHT=0.1
```

GPA global normalized evidence:

```text
GPA_GLOBAL_EVIDENCE_MODE=prior_corrected_log_q
GPA_GLOBAL_PRIOR=uniform
GPA_GLOBAL_EPS=1e-12
GPA_GLOBAL_BETA=positive.beta
```

Gamma sweep:

```text
gamma ∈ {0.1, 0.25, 0.5}
```

Optional diagnostic value, not primary:

```text
gamma = 1.0
```

I do not recommend making \(\gamma=1.0\) the main setting, because original logits and GPA global evidence are correlated rather than independent.

## 9. Output Requirements

The next implementation should evaluate multiple formulas in one pass if practical:

```text
formula_original
formula_raw_gpa_global
formula_prior_corrected_gamma_0.1
formula_prior_corrected_gamma_0.25
formula_prior_corrected_gamma_0.5
```

The raw GPA global formula is already known to be weaker, but keeping it as a diagnostic row makes the comparison self-contained.

Recommended output files:

```text
summary.csv
summary_original.csv
summary_raw_gpa_global.csv
summary_prior_corrected_gamma_0.1.csv
summary_prior_corrected_gamma_0.25.csv
summary_prior_corrected_gamma_0.5.csv
```

The combined `summary.csv` should include a `formula` column.

Recommended formula labels:

```text
original_formula
raw_gpa_global_logits
prior_corrected_gpa_global_gamma_0.1
prior_corrected_gpa_global_gamma_0.25
prior_corrected_gpa_global_gamma_0.5
```

## 10. Expected Interpretation

If the prior-corrected formula improves over original:

- GPA global cache contains useful class-conditional evidence.
- The previous c1 failure was mainly caused by raw unnormalized logits and over-amplification.
- The paper can claim that GPA global features help only after probabilistic calibration.

If it improves only on `rotate` / `scale`:

- GPA global cache is useful for geometric transformations.
- It may need corruption-adaptive gating or confidence weighting for noise corruptions.

If it still decreases average accuracy:

- GPA global cache is not reliable enough as a final prediction expert.
- It should remain a cache-selection mechanism rather than a final logits branch.
- Future work should prioritize local cache or distribution log-likelihood instead of GPA global logits.

## 11. Mathematical Risks

### 11.1 Pseudo-Label Bias

All GPA cache entries are pseudo-labeled. If a class cache is contaminated, then:

\[
\hat{p}_G(x\mid c)
\]

will be biased. Prior correction does not remove pseudo-label errors.

### 11.2 Evidence Correlation

The original logits and GPA global evidence share the same feature extractor. Therefore:

\[
p_{\mathrm{orig}}(c\mid x)
\quad\text{and}\quad
q_G(c\mid x)
\]

are not independent. This is why \(\gamma<1\) is necessary.

### 11.3 Numerical Stability

When all GPA kernel scores are extremely small:

\[
\sum_j \pi_j \hat{p}_G(x\mid j)
\approx 0.
\]

Implementation must use:

\[
\epsilon
\]

inside logarithms and denominators:

\[
\log\left(
\frac{q_G(c\mid x)+\epsilon}
{\pi_c+\epsilon}
\right).
\]

### 11.4 Missing Classes

If \(G_c\) is empty:

\[
|G_c|=0.
\]

The first implementation should set:

\[
\hat{p}_G(x\mid c)=0
\]

for empty classes, then rely on \(\epsilon\) for numerical stability. Do not invent synthetic cache samples.

## 12. Implementation Guardrails

When coding starts, it should not modify existing A0, A0b, or c1 files. It should create new independent files, for example:

```text
Point-Cache/runners/E4_distribution_guided_cache/model_e4_c_a0_c2_prior_corrected_gpa_global_logits.py
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a0_c2_ulip_modelnetc_s2_prior_corrected_gpa_global_logits.py
Point-Cache/scripts/E4_distribution_guided_cache/02_5_run_e4_c_a0_c2_ulip_modelnetc_s2_common.sh
Point-Cache/scripts/E4_distribution_guided_cache/02_5_ulip_modelnetc_s2_zs_global_local_e4_c_a0_c2_prior_corrected_gpa_global_logits_manual_full.sh
```

The implementation has been added as a separate E4-C-A0-c2 variant and should be reviewed with static checks before running.

## 13. Implementation Status

Implemented files:

```text
Point-Cache/runners/E4_distribution_guided_cache/model_e4_c_a0_c2_prior_corrected_gpa_global_logits.py
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a0_c2_ulip_modelnetc_s2_prior_corrected_gpa_global_logits.py
Point-Cache/scripts/E4_distribution_guided_cache/02_5_run_e4_c_a0_c2_ulip_modelnetc_s2_common.sh
Point-Cache/scripts/E4_distribution_guided_cache/02_5_ulip_modelnetc_s2_zs_global_local_e4_c_a0_c2_prior_corrected_gpa_global_logits_manual_full.sh
```

The runner evaluates these formulas in one pass:

```text
original_formula
raw_gpa_global_logits
prior_corrected_gpa_global_gamma_0.1
prior_corrected_gpa_global_gamma_0.25
prior_corrected_gpa_global_gamma_0.5
```

The expected output files are:

```text
summary.csv
summary_original.csv
summary_raw_gpa_global.csv
summary_prior_corrected_gamma_0.1.csv
summary_prior_corrected_gamma_0.25.csv
summary_prior_corrected_gamma_0.5.csv
```

The prior-corrected evidence computation is done in float32 to avoid half-precision underflow around \(\epsilon=10^{-12}\).
