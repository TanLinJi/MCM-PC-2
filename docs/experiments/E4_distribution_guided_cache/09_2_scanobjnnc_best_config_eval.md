# 09-2 ScanObjNN-C Best-Config Evaluation

Recorded on 2026-07-07.

## Purpose

把当前 ULIP + ModelNet-C all35 的最优设置迁移到 ScanObjNN-C all35，先验证跨数据集是否仍然有效。

## Configuration

```text
dataset = sonn_c
sonn_variant = hardest
capacity = (entropy_cap, gpa_cap, local_cap, neg_cap) = (3, 3, 3, 6)
local_centers = 3
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

The configuration comes from the current strict ULIP + ModelNet-C all35 best record:

```text
all35 avg = 53.7138
capacity = (3, 3, 3, 6)
weight_name = s2_best_dense
weights = (4.4, 3.9, 0.19)
```

## Code

```text
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a0_e1_explicit_final_score_ulip_scanobjnnc_hardest_all35.py
Point-Cache/scripts/E4_distribution_guided_cache/09_run_e4_c_a0_e1_explicit_final_score_scanobjnnc_all35_common.sh
Point-Cache/scripts/E4_distribution_guided_cache/09_2_ulip_scanobjnnc_hardest_all35_explicit_final_score_best.sh
```

## Command

From `/root/autodl-tmp/MCM-PC-2/Point-Cache`:

```bash
bash scripts/E4_distribution_guided_cache/09_2_ulip_scanobjnnc_hardest_all35_explicit_final_score_best.sh 0
```

Change the final argument to select the physical GPU.

## Result Location

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_e3_g3_l3_n6_modelnetc_best_w4p4_3p9_0p19_ulip_scanobjnnc_hardest_all35_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

Key files:

```text
summary.csv
final_score_weight_summary.csv
logs/
```

## Results

Finished on 2026-07-07.

| method | all35 | S0 | S1 | S2 | S3 | S4 |
|---|---:|---:|---:|---:|---:|---:|
| ULIP zero-shot | 23.6457 | 26.8614 | 25.1029 | 23.9129 | 22.1129 | 20.2386 |
| ULIP + global cache | 26.6006 | 30.1957 | 28.2086 | 26.8400 | 24.7400 | 23.0186 |
| ULIP + original global/local cache | 27.4126 | 31.4229 | 28.9257 | 27.9386 | 25.4943 | 23.2814 |
| 09-2 explicit final-score, transferred best | **29.6649** | **34.2173** | **30.6434** | **30.5344** | **27.7238** | **25.2057** |

Compared with the original ULIP + global/local cache baseline:

```text
all35 delta = +2.2523
S0 delta = +2.7944
S1 delta = +1.7177
S2 delta = +2.5958
S3 delta = +2.2295
S4 delta = +1.9243
```

Corruption-wise average deltas against the original ULIP + global/local cache baseline:

| corruption | baseline | transferred best | delta |
|---|---:|---:|---:|
| add_global | 23.6640 | 27.0260 | +3.3620 |
| add_local | 22.0420 | 23.0480 | +1.0060 |
| dropout_global | 33.2580 | 35.8780 | +2.6200 |
| dropout_local | 28.4260 | 31.2840 | +2.8580 |
| rotate | 30.3280 | 32.7160 | +2.3880 |
| scale | 28.5280 | 30.3960 | +1.8680 |
| jitter | 25.6420 | 27.3140 | +1.6720 |

Diagnostic transition summary from zero-shot to final prediction:

```text
total samples across all35 = 100870
wrong -> right = 10709
right -> wrong = 4644
net corrected samples = +6065
net accuracy gain = +6.01 pp
```

Stage-wise net gains:

```text
zero-shot -> entropy stage = +1.65 pp
entropy -> local stage = +2.07 pp
local -> negative stage = +2.29 pp
```

## Analysis

The transferred ModelNet-C best setting is effective on ScanObjNN-C hardest all35. It improves over the original ULIP + global/local cache baseline on 34/35 corruption-severity pairs; the only negative cell is `add_local_4`, with a small drop of `-0.34`.

The largest average gains are on `add_global`, `dropout_local`, and `dropout_global`. The smaller gains are on `add_local`, `jitter`, and `scale`, which suggests that ScanObjNN-C may still benefit from a small dataset-specific sweep around local score and negative score weights.

The diagnostic transition result is healthy: all three stages have positive net gains, and the negative-cache stage contributes the largest single-stage net gain in this run. This supports keeping the negative cache in the ScanObjNN-C line, but this run alone does not prove that `(3, 3, 3, 6)` is ScanObjNN-C optimal.

## Notes

- The runner is new and does not modify the existing ModelNet-C runner.
- The script intentionally reuses the existing ScanObjNN-C E1 prompt cache and stops if it is missing or incomplete.
- `COMBINATIONS` and `FINAL_SCORE_WEIGHTS` are kept as arrays in the queue script, so later ScanObjNN-C capacity or score-weight sweeps can be added by appending lines.
