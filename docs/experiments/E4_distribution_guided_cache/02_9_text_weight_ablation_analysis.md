# E4-C-A0+E1 Text Distribution Weight Ablation

Date: 2026-06-10

## Experiment Scope

This ablation evaluates the text distribution score weight in the E4-C-A0+E1 text-distribution-only setting.

The base setting is unchanged:

- Backbone: ULIP
- Dataset: ModelNet-C
- Severity: 2
- Corruptions: add_global, add_local, dropout_global, dropout_local, rotate, scale, jitter
- Final classifier and final logits: manual_full, unchanged from E4-C-A0
- E1 cached LLM descriptions: used only to build the text distribution for GPA replacement
- Score normalization: running_zscore

## Weight Mapping

```text
02_9_1 -> E4_TEXT_SCORE_WEIGHT=0.05
02_8   -> E4_TEXT_SCORE_WEIGHT=0.10
02_9_2 -> E4_TEXT_SCORE_WEIGHT=0.15
02_9_3 -> E4_TEXT_SCORE_WEIGHT=0.20
02_9_4 -> E4_TEXT_SCORE_WEIGHT=0.25
```

`02_9_0 -> 0.00` exists as a planned control script, but it has not been included in the completed-result comparison here.

## Results

Exact averages below are computed from `gpa_stats/*_gpa_stats.json` `final_acc`, not from rounded `summary.csv` values.

| Experiment | Text weight | Avg acc |
|---|---:|---:|
| 02_9_5 | 0.14 | 54.5771 |
| 02_9_1 | 0.05 | 54.1514 |
| 02_8 | 0.10 | 54.6712 |
| 02_9_2 | 0.15 | 54.7060 |
| 02_9_6 | 0.16 | 54.6243 |
| 02_9_3 | 0.20 | 54.5265 |
| 02_9_4 | 0.25 | 54.7002 |

## Selected Setting

`02_9_2` is fixed as the current E4-C-A0+E1 text-distribution default:

```text
E4_TEXT_SCORE_WEIGHT=0.15
avg_acc=54.70595045
```

It is the highest complete severity-2 result currently recorded across E1/E2/E3/E4 summaries.

`02_9_4` with `E4_TEXT_SCORE_WEIGHT=0.25` is nearly tied, but remains lower by `0.0058` average points. It improves add_global, dropout_global, dropout_local, and rotate, but loses more on add_local and scale. Therefore, `0.15` is the safer default for the main result, while `0.25` should be reported as a near-tie supplementary ablation.

## Interpretation

The useful text-distribution weight range is around `0.15` to `0.25`. Increasing the text score beyond `0.15` is not monotonically beneficial because GPA replacement is a hard trajectory-dependent decision:

```text
replace if joint_score(new) > joint_score(old)
joint_score = normalized_visual_score + E4_TEXT_SCORE_WEIGHT * normalized_text_score
```

A larger text weight can help semantic corruptions or dropout-like changes, but it can also over-constrain local geometric corruptions. The current evidence supports using text distribution as a moderate prior, not as a dominant replacement signal.

## Refinement Sweep

We further refined the neighborhood around `0.15` with:

- `0.14` -> `02_9_5`
- `0.16` -> `02_9_6`
- `0.145` -> `02_9_7` planned
- `0.155` -> `02_9_8` planned

The refinement record is maintained in:

`docs/experiments/E4_distribution_guided_cache/02_9_7_02_9_8_text_weight_refinement.md`
