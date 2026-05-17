# Stage 4: E4-CANC-DIAG

## Experiment Name

**E4-CANC-DIAG-v2: Conflict Signal Validation**

Chinese name:

**实验 E4-CANC-DIAG-v2：全局-局部冲突诊断实验**

---

## Goal

Before implementing a new negative cache, verify whether global-local conflict can actually identify wrong global pseudo-labels.

This stage does not change:

- final logits
- positive cache
- negative cache
- cache update rule
- prediction behavior

It only records diagnostic statistics.

---

## Core Definitions

Global prediction:

\[
\hat y_g=\arg\max_k p_g(k\mid x)
\]

Mean-local disagreement:

\[
D_{gl}^{mean}(x)
=
\left[
\max_{k\ne \hat y_g}p_l^{mean}(k\mid x)
-
p_l^{mean}(\hat y_g\mid x)
\right]_+
\]

Max-part disagreement:

\[
D_{gl}^{max}(x)
=
\max_r
\left[
\max_{k\ne \hat y_g}p_{l,r}(k\mid x)
-
p_{l,r}(\hat y_g\mid x)
\right]_+
\]

Important:

\[
D_{gl}(x)>\tau_d
\not\Rightarrow
k_l^\star=y
\]

Therefore, local alternative class should not be used as a positive pseudo-label.

---

## Main Diagnostic Result

Average over 7 ModelNet-C corruptions:

| Metric | Value |
|---|---:|
| Hierarchical accuracy | 62.81 |
| Global error rate | 42.06 |
| Mean-conflict wrong precision | 77.79 |
| Max-conflict wrong precision | 75.59 |
| Safe-mean candidate wrong precision | 79.59 |
| Safe-max candidate wrong precision | 79.54 |
| Mean-conflict rate | 7.85 |
| Max-conflict rate | 11.01 |
| Safe candidate rate | about 24.2 |
| Local alternative correct rate | about 20.64 |

---

## Per-Corruption Diagnostic Result

| Corruption | Global Error | Mean Conflict Wrong | Improvement |
|---|---:|---:|---:|
| add_global_2 | 34.48 | 81.86 | +47.38 |
| dropout_global_2 | 31.08 | 72.66 | +41.58 |
| rotate_2 | 29.34 | 70.10 | +40.77 |
| jitter_2 | 78.89 | 89.41 | +10.52 |
| add_local_2 | 45.54 | 80.09 | +34.54 |
| dropout_local_2 | 42.02 | 85.82 | +43.80 |
| scale_2 | 33.10 | 64.58 | +31.48 |

---

## Interpretation

Global-local conflict substantially increases the probability that the global pseudo-label is wrong.

\[
P(\hat y_g\ne y)
\approx 42.06\%
\]

\[
P(\hat y_g\ne y\mid I_D^{mean})
\approx 77.79\%
\]

Therefore:

\[
D_{gl}
\Rightarrow
\text{global pseudo-label likely wrong}
\]

But local alternative correctness is low:

\[
P(k_l^\star=y\mid I_D)
\approx 20.64\%
\]

Therefore:

\[
D_{gl}
\not\Rightarrow
k_l^\star \text{ is correct}
\]

---

## Stage 4 Conclusion

E4-CANC-DIAG-v2 is a successful diagnostic experiment.

It supports the following design principle:

> Use global-local conflict for negative suppression of the global predicted class, not for positive correction toward the local alternative class.

---

## Next Method

**E4-CANC-v0: Conservative Conflict-Aware Negative Cache**

Recommended rule:

\[
I_{neg}^{E4}(x)=I_H(x)\lor(I_M(x)\land I_D(x))
\]

where:

\[
I_H(x)=\mathbf{1}[\tau_l < H_g(x)<\tau_u]
\]

\[
I_M(x)=
\mathbf{1}[M_g(x)<\tau_m]\cdot
\mathbf{1}[p_g^{(1)}(x)>\tau_p]
\]

\[
I_D(x)=\mathbf{1}[D_{gl}^{mean}(x)>\tau_d]
\]

First version recommendation:

\[
\tau_d = 5\times 10^{-5}
\]

Use mean-local conflict first.


---

## E4-CANC-v0: Conservative Conflict-Aware Negative Cache

### Rule

\[
I_{neg}^{E4}(x)=I_H(x)\lor(I_M(x)\land I_D(x))
\]

### Result

| Corruption | Baseline | E4-CANC-v0 | Delta |
|---|---:|---:|---:|
| add_global_2 | 68.15 | 68.19 | +0.04 |
| add_local_2 | 61.30 | 61.43 | +0.13 |
| dropout_global_2 | 73.22 | 73.14 | -0.08 |
| dropout_local_2 | 63.65 | 63.57 | -0.08 |
| rotate_2 | 73.30 | 73.38 | +0.08 |
| scale_2 | 70.46 | 70.54 | +0.08 |
| jitter_2 | 29.58 | 29.58 | 0.00 |
| **Average** | **62.81** | **62.83** | **+0.02** |

### Observation

E4-CANC-v0 is safe but under-active. It slightly improves the average accuracy, but the actual newly added negative samples are only about 0.78% on average.

### Next

E4-CANC-v1 should relax the candidate rule from:

\[
I_H\lor(I_M\land I_D)
\]

to:

\[
I_H\lor(I_D\land p_g^{(1)}>\tau_p)
\]

