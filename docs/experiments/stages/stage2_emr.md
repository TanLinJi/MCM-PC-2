# Stage 2: E2-EMR

## Experiment Name

**E2-EMR: Entropy-Margin Reliability Cache Admission**

Chinese name:

**实验 E2-EMR：熵-间隔可靠性缓存准入实验**

---

## Goal

Test whether positive cache admission can be improved by replacing entropy-only ranking with an entropy-margin reliability score.

Original Point-Cache positive cache admission mainly uses entropy:

\[
H(x) \downarrow \Rightarrow x \text{ is more reliable}
\]

E2-EMR changes the positive cache ranking criterion to:

\[
R(x) = -H(x) + \lambda_m M(x)
\]

where:

\[
M(x)=p_{(1)}(x)-p_{(2)}(x)
\]

---

## Meaning

The intuition is:

- Low entropy means the model is confident.
- Large top-1/top-2 margin means the model is less confused between the best and second-best classes.
- A good positive-cache sample should satisfy both.

---

## Results

| Corruption | Baseline | E2-EMR | Delta |
|---|---:|---:|---:|
| add_global_2 | 68.15 | 68.68 | +0.53 |
| add_local_2 | 61.30 | 60.25 | -1.05 |
| dropout_global_2 | 73.22 | 73.34 | +0.12 |
| dropout_local_2 | 63.65 | 63.01 | -0.64 |
| rotate_2 | 73.30 | 72.85 | -0.45 |
| scale_2 | 70.46 | 69.85 | -0.61 |
| jitter_2 | 29.58 | 29.54 | -0.04 |
| **Average** | **62.81** | **62.50** | **-0.31** |

---

## Conclusion

E2-EMR improves global corruption cases such as `add_global_2`, but it does not improve the full corruption average.

This shows:

> Entropy-margin reliability is useful but insufficient for robust 3D test-time cache admission.

The result motivates the need to consider 3D-specific structural signals.

---

## Paper Note

**Section: Ablation / Motivation**

A simple entropy-margin reliability score improves some global corruption cases but fails to improve the full ModelNet-C average. This suggests that class-level confidence alone is insufficient for reliable 3D positive cache admission.

