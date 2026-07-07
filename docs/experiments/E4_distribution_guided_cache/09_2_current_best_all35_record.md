# 09-2 Current Best All35 Record

Recorded on 2026-07-07.

## Scope

Strict ULIP + ModelNet-C all35 under the E4 distribution-guided cache / explicit final-score line.

This record excludes ULIP2, OpenShape, and Uni3D results.

## Best Result

| metric | value |
|---|---:|
| all35 avg | **53.7138** |
| S0 avg | 59.9792 |
| S1 avg | 56.8187 |
| S2 avg | 55.1459 |
| S3 avg | 51.0651 |
| S4 avg | 45.5603 |

## Best Configuration

```text
capacity = (entropy_cap, gpa_cap, local_cap, neg_cap) = (3, 3, 3, 6)
local_centers = 3
weight_name = s2_best_dense
alpha_g = 4.4
alpha_l = 3.9
alpha_n = 0.19
final_score = y_zs + alpha_g * y_g + alpha_l * y_l - alpha_n * y_n
final_score = y_zs + 4.4 * y_g + 3.9 * y_l - 0.19 * y_n
text_weight = 0.15
score_norm = running_zscore
prompt_source = manual_full
text_distribution_prompt_source = manualfull_llm_dynamic_init
```

## Result Location

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_e3_g3_l3_n6_priority_cap678_w6_ulip_modelnetc_all35_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

Key files:

```text
summary.csv
final_score_weight_summary.csv
logs/
```

## Comparison

| baseline | all35 | delta from best |
|---|---:|---:|
| previous tuned all35 best | 53.6872 | +0.0266 |
| capacity-only `(3,3,3,8)` | 53.5623 | +0.1515 |
| capacity-only `(3,3,3,7)` | 53.5414 | +0.1724 |
| capacity-only `(3,3,3,6)` | 53.4863 | +0.2275 |

## Notes

- The best capacity remains `(3,3,3,6)` after final-score tuning.
- Increasing `neg_cap` to 7 or 8 improves capacity-only all35, but does not beat `(3,3,3,6)` once final-score weights are tuned.
- The main improvement over the previous tuned setting comes from `add_global`; the main small tradeoff is on `jitter`.
