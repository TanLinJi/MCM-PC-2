# E4-C-A0+E1 Text Distribution Weight Refinement

Date: 2026-06-29

## Experiment Scope

This refinement continues the E4-C-A0+E1 text-distribution-only ablation around the current best region near `0.15`.

The fixed base setting remains unchanged:

- Backbone: ULIP
- Dataset: ModelNet-C
- Severity: 2
- Corruptions: add_global, add_local, dropout_global, dropout_local, rotate, scale, jitter
- Final classifier and final logits: manual_full, unchanged from E4-C-A0
- E1 cached LLM descriptions: used only to build the text distribution for GPA replacement
- Score normalization: running_zscore

## Recorded Results So Far

Exact averages below are computed from `gpa_stats/*_gpa_stats.json` `final_acc`, not from rounded `summary.csv` values.

| Experiment | Text weight | Avg acc |
|---|---:|---:|
| 02_9_5 | 0.14 | 54.5771 |
| 02_9_7 | 0.145 | 54.6600 |
| 02_9_9 | 0.148 | 54.6770 |
| 02_9_10 | 0.149 | 54.6770 |
| 02_9_2 | 0.15 | 54.7060 |
| 02_9_8 | 0.155 | 54.6186 |
| 02_9_6 | 0.16 | 54.6243 |
| 02_9_3 | 0.20 | 54.5265 |
| 02_9_4 | 0.25 | 54.7002 |

Note: the historical `02_9_2` anchor remains `54.7060`. A later rerun of `02_9_2` produced `54.6770`, exactly matching `0.148` and `0.149`.

## Interpretation

The best point in the current sweep remains `0.15`.

`0.148`, `0.149`, and the later rerun of `0.15` are exactly identical at the per-corruption `final_acc` level. This indicates that the hard GPA replacement decisions are unchanged throughout this very narrow weight interval.

Verification note: this is not a script parameter-passing error. The rerun logs and `gpa_stats` record distinct `e4_text_score_weight` values for `0.148`, `0.149`, and `0.15`; replacement-event `joint_score` values also differ across these weights. However, the accept/reject decision sequence is identical for all 4936 recorded replacement events in each of the seven severity-2 corruptions, so the final cache trajectory and accuracy remain identical.

`0.145` improves `add_global` substantially versus `0.15`, but the gain does not carry through to the mean because `rotate` and several other corruptions remain better at `0.15`.

`0.14` and `0.16` both stay close, which suggests the useful region is narrow and centered near `0.15`, but neither beats the existing default.

`0.25` remains the strongest setting on `add_global` and `rotate`, but its overall mean is still slightly below `0.15`.

## Notes

The following results are now part of the ablation record and should be preserved for later summary tables:

- 0.14
- 0.145
- 0.148
- 0.149
- 0.15
- 0.155
- 0.16
- 0.20
- 0.25

Future runs in this refinement line should keep the same base runner, prompt cache, score normalization, and final classifier settings, changing only `E4_TEXT_SCORE_WEIGHT`.
