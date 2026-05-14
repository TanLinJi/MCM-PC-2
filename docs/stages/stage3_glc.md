# Stage 3: Global-Local Consistency Experiments

## Experiment Names

| ID | Name | Description |
|---|---|---|
| E3-GLC-v0 | Naive Global-Local Consistency Reliability | Add local support directly into positive cache reliability score |
| E3-GLC-v1 | Non-expansive Global-Local Consistency Gate | Use local consistency as a non-expansive gate for positive cache admission |

## Goal

Test whether global-local consistency can improve positive cache admission in Point-Cache hierarchical inference.

## Results

| Method | Avg. Accuracy | Delta vs Baseline |
|---|---:|---:|
| Baseline Hierarchical | 62.81 | - |
| E2-EMR | 62.50 | -0.31 |
| E3-GLC-v0 | 62.50 | -0.31 |
| E3-GLC-v1 | 62.50 | -0.30 |

## Key Observation

E3-GLC-v1 fixes the mathematical risk of additive local consistency by using a non-expansive gate. However, it still does not improve the full-corruption average accuracy.

## Conclusion

Global-local consistency is not stable when directly used to promote positive cache admission.  
It should instead be used to detect global-local conflict samples and guide negative or boundary cache construction.

## Next Stage

E4-CANC: Confusion-Aware Negative Cache.
