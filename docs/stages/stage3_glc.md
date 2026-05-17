# Stage 3: E3-GLC

## Experiment Family

**E3-GLC: Global-Local Consistency for Positive Cache Admission**

---

## Motivation

E2-EMR only considers global class-level confidence:

\[
R(x) = -H(x) + \lambda_m M(x)
\]

It does not consider whether local point-cloud parts support the global prediction.

Stage 3 tests whether global-local consistency can improve positive cache admission.

---

## E3-GLC-v0

### Name

**E3-GLC-v0: Naive Global-Local Consistency Reliability**

### Formula

\[
R(x) =
-H(x)
+
\lambda_m M(x)
+
\lambda_c C_{\text{global-local}}(x)
\]

where:

\[
C_{\text{global-local}}(x)=p_{\text{local}}(\hat y_{\text{global}})
\]

### Result

| Corruption | Accuracy |
|---|---:|
| add_global_2 | 68.31 |
| add_local_2 | 60.86 |
| dropout_global_2 | 73.34 |
| dropout_local_2 | 62.76 |
| rotate_2 | 73.22 |
| scale_2 | 69.69 |
| jitter_2 | 29.29 |
| **Average** | **62.50** |

### Conclusion

Naively adding local support into the positive reliability score is not stable.

---

## E3-GLC-v1

### Name

**E3-GLC-v1: Non-expansive Global-Local Consistency Gate**

### Formula

\[
S_g(x)=1-H_g(x)+\lambda_m M_g(x)
\]

\[
C_{glm}(x)=
\left[
p_l(\hat y_g)
-
\max_{k\ne \hat y_g}p_l(k)
\right]_+
\]

\[
G_{gl}(x)=\epsilon+(1-\epsilon)C_{glm}(x)
\]

\[
R(x)=S_g(x)\cdot G_{gl}(x)
\]

This fixes the mathematical risk of letting local consistency over-amplify unreliable global predictions:

\[
0\le R(x)\le S_g(x)
\]

### Result

| Corruption | Accuracy |
|---|---:|
| add_global_2 | 68.76 |
| add_local_2 | 60.45 |
| dropout_global_2 | 73.26 |
| dropout_local_2 | 63.01 |
| rotate_2 | 73.06 |
| scale_2 | 69.41 |
| jitter_2 | 29.58 |
| **Average** | **62.50** |

---

## Stage 3 Conclusion

E3-GLC-v1 is mathematically cleaner than v0, but it still does not improve the full corruption average.

This suggests:

> Global-local consistency should not be used to promote positive cache samples.

Instead, it should be used to detect unreliable global pseudo-labels and support negative / boundary suppression.

---

## Transition to E4

Stage 3 result motivates:

**E4-CANC: Confusion-Aware Negative Cache**

Core change:

\[
\text{GLC for positive promotion}
\quad\rightarrow\quad
\text{GLC for conflict detection}
\]

