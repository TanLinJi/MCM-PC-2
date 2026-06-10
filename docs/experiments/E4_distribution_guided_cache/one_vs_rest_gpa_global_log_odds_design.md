# E4-C-A0-c3 One-vs-Rest GPA Global Log-Odds Design

> Created: 2026-06-09
>
> Status: user confirmed the direction; implementation target is an independent E4-C-A0-c3 variant.

## 1. Motivation

E4-C-A0-c2 used prior-corrected GPA global evidence:

\[
e_G(c,x)
=
\log(r_c(x)+\epsilon)
-
\log(m(x)+\epsilon),
\]

where:

\[
m(x)=\sum_j \pi_j r_j(x).
\]

This formula is probabilistically interpretable, but \(m(x)\) is a sample-level scalar and does not depend on class \(c\). In final logits:

\[
z_{\mathrm{new}}(c)
=
z_{\mathrm{orig}}(c)
+
\gamma \log(r_c(x)+\epsilon)
-
\gamma \log(m(x)+\epsilon),
\]

the last term is the same for every class. Therefore it is removed by softmax / argmax:

\[
\arg\max_c
\left[
z_{\mathrm{orig}}(c)
+
\gamma \log(r_c(x)+\epsilon)
-
\gamma \log(m(x)+\epsilon)
\right]
=
\arg\max_c
\left[
z_{\mathrm{orig}}(c)
+
\gamma \log(r_c(x)+\epsilon)
\right].
\]

Thus E4-C-A0-c2 mainly tests class-size-normalized log KDE, but does not introduce a class-dependent competing background term.

## 2. GPA Global Class Density

For current test sample global feature \(x\), and GPA global cache entries of class \(c\):

\[
G_c=\{g_i\mid i=1,\dots,|G_c|\},
\]

define the kernel:

\[
K_G(x,g_i)
=
\exp\left(\beta_G(\cos(x,g_i)-1)\right).
\]

The class-wise GPA global density is:

\[
r_c(x)
=
\frac{1}{|G_c|}
\sum_{i\in G_c}
K_G(x,g_i).
\]

Expanded:

\[
r_c(x)
=
\frac{1}{|G_c|}
\sum_{i\in G_c}
\exp\left(\beta_G(\cos(x,g_i)-1)\right).
\]

Where:

- \(x\): current test sample global feature.
- \(g_i\): one GPA global cache feature.
- \(G_c\): GPA global cache entries assigned to class \(c\).
- \(\beta_G\): kernel sharpness, default inherited from PointCache positive cache \(\beta=3.0\).
- \(r_c(x)\): class-size-normalized GPA global support for class \(c\).

## 3. One-vs-Rest Background

Instead of using a global normalizer \(m(x)\), define a class-dependent rest density:

\[
r_{\neg c}(x)
=
\frac{1}{C-1}
\sum_{j\ne c}
r_j(x).
\]

Where:

- \(C\): number of classes, \(C=40\) for ModelNet-C.
- \(r_{\neg c}(x)\): average GPA global support from all classes except \(c\).

Because \(r_{\neg c}(x)\) excludes class \(c\), it changes with \(c\). Therefore it cannot be removed as a shared constant in softmax.

## 4. Log-Odds Evidence

Define one-vs-rest GPA global log-odds evidence:

\[
e_{\mathrm{odds}}(c,x)
=
\log(r_c(x)+\epsilon)
-
\log(r_{\neg c}(x)+\epsilon).
\]

Equivalently:

\[
e_{\mathrm{odds}}(c,x)
=
\log
\frac{
r_c(x)+\epsilon
}{
r_{\neg c}(x)+\epsilon
}.
\]

Fully expanded:

\[
e_{\mathrm{odds}}(c,x)
=
\log
\left(
\frac{1}{|G_c|}
\sum_{i\in G_c}
\exp(\beta_G(\cos(x,g_i)-1))
+
\epsilon
\right)
-
\log
\left(
\frac{1}{C-1}
\sum_{j\ne c}
\frac{1}{|G_j|}
\sum_{i\in G_j}
\exp(\beta_G(\cos(x,g_i)-1))
+
\epsilon
\right).
\]

The final logits are:

\[
z_{\mathrm{new}}(c)
=
z_{\mathrm{orig}}(c)
+
\gamma e_{\mathrm{odds}}(c,x).
\]

This term directly compares "how much \(x\) looks like class \(c\)" against "how much \(x\) looks like all other classes".

## 5. Why This Is Different From E4-C-A0-c2

E4-C-A0-c2:

\[
\log(r_c(x)+\epsilon)-\log(m(x)+\epsilon)
\]

has a class-independent second term.

E4-C-A0-c3:

\[
\log(r_c(x)+\epsilon)-\log(r_{\neg c}(x)+\epsilon)
\]

has a class-dependent second term, because \(r_{\neg c}(x)\) excludes class \(c\). Therefore it creates real class competition.

## 6. Empty-Class Policy

The first implementation keeps the same conservative policy as E4-C-A0-c2:

\[
r_c(x)=0
\quad
\text{if}
\quad
G_c=\emptyset.
\]

This is strict and may heavily penalize empty classes:

\[
\log(0+\epsilon)\ll 0.
\]

This is acceptable for the first diagnostic experiment because it directly tests whether the current GPA global cache has enough class coverage to act as a final prediction branch. If it is too harsh, a later c3-b variant can introduce a coverage gate or neutral evidence for empty classes.

Important risk:

\[
e_{\mathrm{odds}}(c,x)
\]

can be sharper than c2 because both the positive class term and the one-vs-rest background term are class-dependent. Therefore a negative c3 result should not be interpreted as "GPA global has no information"; it may mean "GPA global needs coverage-aware gating before it can safely enter final logits."

## 7. Experiment Settings

Use E4-C-A0 as the base:

```text
E4_SCORE_NORM_MODE=running_zscore
E4_SCORE_NORM_MIN_COUNT=8
E4_SCORE_NORM_EPS=1e-6
E4_SCORE_NORM_CLIP=0
E4_TEXT_SCORE_WEIGHT=0.1
```

Use GPA global log-odds settings:

```text
E4_GPA_GLOBAL_EPS=1e-12
E4_GPA_GLOBAL_BETA=positive.beta
E4_GPA_GLOBAL_GAMMAS=0.05,0.1,0.25
```

Use smaller \(\gamma\) than c2 because one-vs-rest log-odds is more discriminative and can be sharper.

## 8. Output Requirements

The runner should output these formulas in one pass:

```text
original_formula
raw_gpa_global_logits
prior_corrected_gpa_global_gamma_0.1
prior_corrected_gpa_global_gamma_0.25
prior_corrected_gpa_global_gamma_0.5
one_vs_rest_gpa_global_odds_gamma_0.05
one_vs_rest_gpa_global_odds_gamma_0.1
one_vs_rest_gpa_global_odds_gamma_0.25
```

Keeping c2 formulas in the same runner makes the c2-vs-c3 comparison self-contained.

## 9. Implementation Files

Implement as new independent files:

```text
Point-Cache/runners/E4_distribution_guided_cache/model_e4_c_a0_c3_one_vs_rest_gpa_global_log_odds.py
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a0_c3_ulip_modelnetc_s2_one_vs_rest_gpa_global_log_odds.py
Point-Cache/scripts/E4_distribution_guided_cache/02_6_run_e4_c_a0_c3_ulip_modelnetc_s2_common.sh
Point-Cache/scripts/E4_distribution_guided_cache/02_6_ulip_modelnetc_s2_zs_global_local_e4_c_a0_c3_one_vs_rest_gpa_global_log_odds_manual_full.sh
```

Do not modify or overwrite E4-C-A0-c2 files while the c2 experiment is running.
