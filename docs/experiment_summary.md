# MCM-PC Experiment Summary

## Project

**MCM-PC: Multi-Cache Matrix for 3D Point Cloud Test-Time Adaptation**

Current base code: `Point-Cache`  
Main branch: `mcm-pc-reliability-v1`  
Backbone used in current experiments: `ULIP-2`  
Dataset: `ModelNet-C`  
Corruption severity: `level 2`

---

## Global Summary

| Stage | Experiment | Main Change | Average Accuracy | Status |
|---|---|---|---:|---|
| Stage 1 | Baseline Reproduction | Original Point-Cache hierarchical cache | 62.81 | Reproduced |
| Stage 2 | E2-EMR | Entropy-margin reliability for positive cache admission | 62.50 | Not better than baseline |
| Stage 3 | E3-GLC-v0 | Additive global-local consistency for positive cache admission | 62.50 | Not stable |
| Stage 3 | E3-GLC-v1 | Non-expansive global-local gate for positive cache admission | 62.50 | Not better than baseline |
| Stage 4 | E4-CANC-DIAG-v2 | Diagnose whether global-local conflict detects wrong pseudo-labels | 62.81 | Positive diagnostic result |

---

## Baseline Hierarchical Results

| Corruption | Accuracy |
|---|---:|
| add_global_2 | 68.15 |
| add_local_2 | 61.30 |
| dropout_global_2 | 73.22 |
| dropout_local_2 | 63.65 |
| rotate_2 | 73.30 |
| scale_2 | 70.46 |
| jitter_2 | 29.58 |
| **Average** | **62.81** |

---

## Main Method Comparison

| Corruption | Baseline | E2-EMR | E3-GLC-v0 | E3-GLC-v1 |
|---|---:|---:|---:|---:|
| add_global_2 | 68.15 | 68.68 | 68.31 | 68.76 |
| add_local_2 | 61.30 | 60.25 | 60.86 | 60.45 |
| dropout_global_2 | 73.22 | 73.34 | 73.34 | 73.26 |
| dropout_local_2 | 63.65 | 63.01 | 62.76 | 63.01 |
| rotate_2 | 73.30 | 72.85 | 73.22 | 73.06 |
| scale_2 | 70.46 | 69.85 | 69.69 | 69.41 |
| jitter_2 | 29.58 | 29.54 | 29.29 | 29.58 |
| **Average** | **62.81** | **62.50** | **62.50** | **62.50** |

---

## Current Key Conclusion

The E2 and E3 positive-cache-admission variants do not outperform the original Point-Cache hierarchical baseline on the full ModelNet-C corruption set.

However, E4-CANC-DIAG-v2 shows that global-local conflict is a strong diagnostic signal for wrong global pseudo-labels.

The next method should therefore use global-local conflict for **negative / boundary suppression**, not for positive cache promotion.

---

## Jitter Observation

`jitter_2` remains consistently low across all methods.

| Method | jitter_2 |
|---|---:|
| Baseline | 29.58 |
| E2-EMR | 29.54 |
| E3-GLC-v0 | 29.29 |
| E3-GLC-v1 | 29.58 |

Interpretation:

- Jitter perturbs individual point coordinates.
- It destroys local geometric structure more directly than rotate or scale.
- Hierarchical local cache relies heavily on stable local part features.
- Therefore, jitter makes both global pseudo-labels and local part evidence unreliable.
- E4 should focus on suppressing wrong global pseudo-labels under jitter rather than using local alternative classes as positive labels.

