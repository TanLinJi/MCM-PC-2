# E4 Distribution-Guided Cache Optimization Worklist

> Last updated: 2026-06-08
>
> Scope: E4-A / E4-B / E4-C distribution-guided GPA-Cache experiments under `Point-Cache`.
>
> Purpose: record optimization opportunities, implementation status, result evidence, and whether each optimization produced measurable gain.

## 1. Current Distribution Assumption

The current E4-C implementation uses a **single-modal diagonal Gaussian approximation** for each class and each modality.

For class \(c\), the visual distribution is:

\[
p_v(x \mid c) \approx \mathcal{N}(x \mid \mu^v_c, \mathrm{diag}((\sigma^v_c)^2)).
\]

The text distribution is:

\[
p_t(x \mid c) \approx \mathcal{N}(x \mid \mu^t_c, \mathrm{diag}((\sigma^t_c)^2)).
\]

Where:

- \(x\): current test sample feature.
- \(c\): predicted class index.
- \(\mu^v_c\): class-wise mean of accepted-history visual features.
- \((\sigma^v_c)^2\): class-wise diagonal variance of accepted-history visual features.
- \(\mu^t_c\): class-wise mean of prompt-level text embeddings.
- \((\sigma^t_c)^2\): class-wise diagonal variance of prompt-level text embeddings.

This means each class is represented as one ellipsoidal cluster in feature space. It is **not** a multi-modal distribution.

## 2. Why Single Gaussian May Be Insufficient

One object category may contain multiple geometric or semantic modes. Examples:

- `chair`: armchair, office chair, dining chair, stool-like chair.
- `table`: round table, rectangular table, desk-like table.
- `lamp`: floor lamp, desk lamp, hanging lamp.
- `plant`: potted plant with different foliage shapes.

Under corruptions such as rotation, scale, dropout, and local/global additions, the same class can occupy several separated regions in embedding space. A single Gaussian may over-smooth these modes and produce a mean that does not represent any real subtype.

This can cause two problems:

1. A valid sample from a minority subtype may be assigned a low score because it is far from the single class mean.
2. The diagonal variance may become too broad after accumulating diverse samples, weakening the ability to reject outliers.

BayesMM also assumes Gaussian textual and geometric distributions, but it further uses Bayesian posterior updates and Bayesian model averaging. Our E4-C currently uses only a lightweight diagonal Gaussian score and manual text-visual weighting.

## 3. Current E4-C Score

For a distribution entry of class \(c\), current E4-C computes:

\[
s(x,c) =
-\frac{1}{d}
\sum_{i=1}^{d}
\frac{(x_i-\mu_{c,i})^2}{\sigma_{c,i}^2+\epsilon}.
\]

Where:

- \(d\): feature dimension.
- \(x_i\): the \(i\)-th feature dimension of the current test sample.
- \(\mu_{c,i}\): the \(i\)-th dimension of the class mean.
- \(\sigma_{c,i}^2\): the \(i\)-th diagonal variance of the class distribution.
- \(\epsilon\): numerical stabilizer.

For E4-C:

\[
s_{\text{joint}}(x,c)
=
s_v(x,c) + \lambda_t s_t(x,c).
\]

Where:

- \(s_v(x,c)\): accepted-history visual distribution score.
- \(s_t(x,c)\): fixed prompt-level text distribution score.
- \(\lambda_t\): text score weight, currently `E4_TEXT_SCORE_WEIGHT=0.1`.

This is a log-linear / product-of-experts style approximation, not full BayesMM Bayesian model averaging.

## 4. Optimization Worklist

| ID | Optimization | Motivation | Status | Current Evidence | Measured Gain |
|---|---|---|---|---|---|
| E4-C-O1 | Accepted-history visual distribution | E4-B current-cache snapshot is too narrow under `shot_capacity=3`; historical accepted samples preserve trusted diversity. | Done | E4-C avg `54.50`; E4-B add_global only `46.39`; E4-C exceeds E2 `54.00` and E3-V2-C `54.04`. | `+0.50` vs E2, `+0.46` vs E3-V2-C, `+1.28` vs E4-A. |
| E4-C-O2 | Text weight ablation | Current text contribution is unproven; `0.1` may help or hurt depending on score scale. | Not started | Current E4-C only ran `E4_TEXT_SCORE_WEIGHT=0.1`. This should be evaluated after score normalization is tested. | Unknown. |
| E4-C-O3 | Text weight sweep | Test whether `0.05`, `0.10`, `0.15` or `0` is best. | Not started | User proposed `0.05` and `0.15`; current report recommends adding these. | Unknown. |
| E4-C-O4 | Accepted source ablation | Need to know whether `EntropyCache accepted`, `GPACache accepted`, or their union gives the best visual distribution. | Not started | E4-C union improves average but loses E4-A's add_global strength, suggesting possible source contamination. | Unknown. |
| E4-C-O5 | GPA global logits branch | Current GPA-Cache controls only `GPA-local-cache`; GPA global features do not directly contribute to final logits. | Backlog / specified as E4-C-c1 | Current final formula uses EntropyCache global logits and GPA-local logits only. User requested a dual-output implementation that compares original logits and GPA-global-included logits in one pass. | Unknown. |
| E4-C-O6 | Dual-output evaluation in one pass | Avoid re-running full inference when only final-logit combination changes. Save original-formula results and GPA-global-included results separately while also writing both sections into `summary.csv`. | Backlog / specified as E4-C-c1 | User requested `summary.csv` to contain the original formula rows first and the GPA-global formula rows below them, with results separated in `results/`. | Unknown. |
| E4-C-O7 | Full Gaussian log-likelihood | Current score omits `log |Σ|` and constant terms. Including log determinant may penalize over-broad distributions. | Not started | Current score is simplified negative diagonal Mahalanobis distance. | Unknown. |
| E4-C-O8 | Full/shared covariance instead of diagonal covariance | Diagonal covariance ignores feature correlations. Shared covariance may better approximate BayesMM while controlling memory. | Not started | BayesMM discusses covariance and shared covariance; current E4-C uses diagonal variance only. | Unknown. |
| E4-C-O9 | Multi-modal visual distribution | One class may have multiple visual subtypes; single Gaussian can over-smooth class modes. If activated later, start with at most \(K=3\) modes per class. | Deferred | Current model is single-modal Gaussian; rotate remains weak and add_global regresses from E4-A. User confirmed this is not part of the immediate implementation. | Unknown. |
| E4-C-O10 | Bayesian text-visual weighting | Manual \(\lambda_t\) is heuristic; BayesMM uses Bayesian model averaging to adjust modality weights. | Not started | Current `joint_score = visual_score + lambda * text_score`. | Unknown. |
| E4-C-O11 | Distribution score as final logits | Current distribution score is only used for GPA replacement, not final classification. | Not started | E4-C changes cache admission only; final logits formula is unchanged. | Unknown. |
| E4-C-O12 | Rotation-specific distribution diagnostics | Rotate remains below E2/E3-V2-C. Need to inspect whether distribution scores reject valid rotated samples. | Not started | E4-C `rotate_2=60.98`, below E2 `62.07` and E3-V2-C `61.67`. | Unknown. |
| E4-C-O13 | Text/visual score normalization before fusion | Current visual and text scores have different numeric ranges; direct addition with a manual text weight may be scale-sensitive. Normalize or calibrate both scores before applying the fusion rule. | P0 / First priority | Current replacement events show text scores are often around `-3.x`, while visual scores vary more with accepted-history statistics. User confirmed this should be the first priority before ordinary text-weight sweeping. | Unknown. |

## 5. Confirmed Optimization Order

After the 2026-06-08 review, the confirmed order is:

1. **Direction A0 / P0: Text/visual score normalization before fusion.**
2. **Direction A1 / P1: Text weight ablation and sweep after normalization.**
3. **Direction C / P2: Full Gaussian log-likelihood.**
4. Keep accepted-source ablation, GPA-global logits, multi-modal distribution, covariance upgrades, and Bayesian weighting in the backlog until the first two directions are evaluated.
5. Shared covariance, multi-modal distribution with \(K=3\) modes per class, and automatic Bayesian modality weighting are explicitly deferred for now.

Rationale:

- Direction A0 is the lowest-cost way to remove the score-scale mismatch between text and visual scores.
- Direction A1 then tests whether the text distribution contributes positive signal after scale calibration.
- Direction C directly improves the current probability score while keeping the E4-C architecture stable.
- More complex changes such as Gaussian mixture, shared covariance, or Bayesian weighting should wait until we know whether the simple score and weight choices are already sufficient.
- Score normalization is now the first priority and should be tested before ordinary text-weight sweeping.

## 6. Recommended Immediate Experiments

### P0: Text/Visual Score Normalization

Before fusing visual and text scores, calibrate their numeric scales:

\[
\tilde{s}_v(x,c)=\mathrm{Norm}_v(s_v(x,c)),
\]

\[
\tilde{s}_t(x,c)=\mathrm{Norm}_t(s_t(x,c)).
\]

Then compute:

\[
s_{\text{joint}}(x,c)
=
\tilde{s}_v(x,c)
+
\lambda_t \tilde{s}_t(x,c).
\]

Recommended first implementation:

\[
\tilde{s}
=
\frac{s-\mu_s}{\sigma_s+\epsilon}.
\]

Where:

- \(s\): raw visual or text distribution score.
- \(\mu_s\): running or candidate-set mean of that score type.
- \(\sigma_s\): running or candidate-set standard deviation of that score type.
- \(\epsilon\): numerical stabilizer.

Interpretation:

- If normalization improves accuracy, previous text-weight sensitivity was partly caused by score-scale mismatch.
- If normalization hurts, the raw score magnitude may already carry useful reliability information.
- If normalization plus \(\lambda_t=0\) is best, text distribution should not be claimed as a positive contributor.

### P1: Text Weight Ablation After Normalization

Run E4-C with:

\[
\lambda_t \in \{0, 0.05, 0.10, 0.15\}.
\]

Interpretation:

- If \(\lambda_t=0\) is best, current gains mainly come from accepted-history visual distribution; text distribution should not be claimed as a positive contributor.
- If \(0.05\) or \(0.10\) is best, text helps as a weak semantic regularizer.
- If \(0.15\) is worse, text score likely overwhelms visual distribution or has mismatched scale.

### P2: Full Gaussian Log-Likelihood

Current E4-C score uses only the diagonal Mahalanobis term:

\[
s(x,c)=
-\frac{1}{d}
\sum_i
\frac{(x_i-\mu_{c,i})^2}{\sigma_{c,i}^2+\epsilon}.
\]

The full diagonal Gaussian log-likelihood adds the variance-volume penalty:

\[
\log p(x \mid c)
=
-\frac{1}{2}
\sum_i
\left[
\frac{(x_i-\mu_{c,i})^2}{\sigma_{c,i}^2+\epsilon}
+ \log(\sigma_{c,i}^2+\epsilon)
\right]
+ \mathrm{const}.
\]

Expected effect:

- Penalizes overly broad distributions.
- May reduce false acceptance after accepted-history accumulation.
- Keeps the distribution form simple and still compatible with current E4-C.

### Backlog: Accepted Source Ablation

Compare:

1. `visual_dist = history(GPACache accepted only)`
2. `visual_dist = history(EntropyCache accepted only)`
3. `visual_dist = history(GPACache accepted ∪ EntropyCache accepted)`

Interpretation:

- If GPA-only wins, EntropyCache accepted samples may introduce noisy or over-broad visual history.
- If Entropy-only wins, the GPA replacement gate may be too conservative or biased.
- If union wins, E4-C's current design is supported.

### Backlog: Dual Final-Logit Output

Codename:

`E4-C-c1`

Purpose:

In one inference pass, report two accuracies:

\[
\mathrm{logits}^{(0)}
=
\mathrm{logits}_{zs}
+\mathrm{logits}_{entropy}
+\mathrm{logits}_{gpa\_local}
-\mathrm{logits}_{neg}.
\]

\[
\mathrm{logits}^{(1)}
=
\mathrm{logits}_{zs}
+\mathrm{logits}_{entropy}
+\mathrm{logits}_{gpa\_global}
+\mathrm{logits}_{gpa\_local}
-\mathrm{logits}_{neg}.
\]

This tests whether GPA global features should participate directly in final prediction, without changing the cache-building process.

Implementation requirements:

1. Do not run the full test stream twice only to compare final-logit formulas.
2. During each sample step, compute both final logits after the same cache update:

\[
\mathrm{logits}^{(0)}
=
\mathrm{logits}_{zs}
+
\mathrm{logits}_{entropy}
+
\mathrm{logits}_{gpa\_local}
-
\mathrm{logits}_{neg},
\]

\[
\mathrm{logits}^{(1)}
=
\mathrm{logits}_{zs}
+
\mathrm{logits}_{entropy}
+
\mathrm{logits}_{gpa\_global}
+
\mathrm{logits}_{gpa\_local}
-
\mathrm{logits}_{neg}.
\]

3. Maintain two accuracy streams:

- `acc_original_formula`: accuracy of \(\mathrm{logits}^{(0)}\).
- `acc_with_gpa_global`: accuracy of \(\mathrm{logits}^{(1)}\).

4. Save two separated result views under the same experiment root:

```text
results/E4_distribution_guided_cache/<exp_id>/
  original_formula/
    summary.csv
    logs/
    gpa_stats/
  with_gpa_global/
    summary.csv
    logs/
    gpa_stats/
  summary.csv
```

5. The top-level `summary.csv` must contain both result blocks in one file:

- First block: original formula rows.
- Second block: GPA-global-included formula rows.

To keep the CSV machine-readable, use the same header and add a new column:

```text
logit_formula
```

Recommended values:

```text
original_formula
with_gpa_global
```

The top-level row order should be:

```text
header
original_formula, add_global_2, ...
original_formula, add_local_2, ...
...
with_gpa_global, add_global_2, ...
with_gpa_global, add_local_2, ...
...
```

6. The per-formula `summary.csv` files should contain only their corresponding rows. This allows quick comparison in one top-level file while still keeping formula-specific results separated in `results/`.

7. The printed terminal output should clearly show both final accuracies:

```text
DONE: add_global_2, acc_original_formula=..., acc_with_gpa_global=...
```

8. The method names should make the formula explicit:

```text
method=E4-C-c1 original formula
method=E4-C-c1 with GPA global logits
```

## 7. Single Gaussian vs Multi-Modal Distribution

Current E4-C uses one Gaussian per class:

\[
p(x \mid c)=\mathcal{N}(x \mid \mu_c, \Sigma_c).
\]

A more expressive alternative is a Gaussian mixture:

\[
p(x \mid c)=
\sum_{m=1}^{M_c}
\pi_{c,m}
\mathcal{N}(x \mid \mu_{c,m}, \Sigma_{c,m}).
\]

Where:

- \(M_c\): number of modes for class \(c\).
- \(\pi_{c,m}\): mixture weight of mode \(m\).
- \(\mu_{c,m}\): mean of mode \(m\).
- \(\Sigma_{c,m}\): covariance of mode \(m\).

Potential benefits:

- Better represents multiple object subtypes.
- Reduces over-smoothing caused by one broad class distribution.
- May improve rotate/scale/dropout cases where features form separated subclusters.

Risks:

- More parameters and memory.
- Online mode assignment is noisy under pseudo-label errors.
- Requires a stable rule for creating, merging, and deleting modes.
- Harder to keep the method training-free and simple.

Suggested minimal version:

1. Start with a fixed small maximum mode count, e.g. \(M_c \le 2\) or \(M_c \le 3\).
2. Use online clustering within accepted-history samples.
3. Score a sample by the best mode or log-sum-exp over modes.
4. Only create a new mode when a sample is low entropy but far from all existing modes.

## 8. Status Legend

- `Done`: implemented and evaluated.
- `In progress`: implementation or experiment is running.
- `Not started`: concept identified but no code/result yet.
- `Blocked`: requires a decision, dependency, or compute resource.
