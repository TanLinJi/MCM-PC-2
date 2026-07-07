# 09-2 ScanObjNN-C vs Point-Cache Paper Table 7

Recorded on 2026-07-07.

## Source

Paper PDF:

```text
/root/autodl-tmp/MCM-PC-2/docs/references/Sun 等 - 2025 - Point-Cache Test-time Dynamic and Hierarchical Cache for Robust and Generalizable Point Cloud Analy.pdf
```

The paper values are from Table 7, S-PB T50-RS-C, i.e. the hardest split of ScanObjectNN-C.

Important correction: Table 7 is the correct split for our `sonn_variant=hardest` setting, but it is not an all35 table. Although its caption does not explicitly say "severity level 2" as Table 5 and Table 6 do, the numbers align with the local `hardest` severity-2 baseline much more closely than with the all35 average. Therefore, paper Table 7 should be treated as the paper-level comparison target for `hardest, severity=2`, while our current run is `hardest, all35`.

Current result:

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_e3_g3_l3_n6_modelnetc_best_w4p4_3p9_0p19_ulip_scanobjnnc_hardest_all35_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

Current configuration:

```text
capacity = (3, 3, 3, 6)
weights = (alpha_g, alpha_l, alpha_n) = (4.4, 3.9, 0.19)
final_score = y_zs + 4.4 * y_g + 3.9 * y_l - 0.19 * y_n
```

## Paper Table 7 Granularity Comparison

This table compares the current severity-2 rows against the paper Table 7 values.

| Corruption | Paper ULIP ZS | Paper PointCache-G | Paper PointCache-H | Local original PC-H S2 | Current S2 | Current S2 - Paper H | Current S2 - Local PC-H S2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Add Global | 19.26 | 22.87 | 23.46 | 23.87 | 27.45 | +3.99 | +3.58 |
| Add Local | 18.39 | 20.85 | 22.69 | 22.73 | 23.98 | +1.29 | +1.25 |
| Drop Global | 30.99 | 33.31 | 34.70 | 34.25 | 38.48 | +3.78 | +4.23 |
| Drop Local | 23.91 | 27.90 | 31.75 | 30.26 | 33.41 | +1.66 | +3.15 |
| Rotate | 27.48 | 30.85 | 33.00 | 32.55 | 35.29 | +2.29 | +2.74 |
| Scale | 26.34 | 28.63 | 28.28 | 28.38 | 29.63 | +1.35 | +1.25 |
| Jitter | 21.44 | 24.53 | 25.05 | 23.53 | 25.50 | +0.45 | +1.97 |
| Avg | 23.97 | 26.99 | 28.42 | 27.94 | 30.53 | +2.11 | +1.66 |

## Current All35 Result

The current experiment is all35: 7 corruption types x 5 severities. The paper does not provide corresponding all35 cells. The table below keeps the paper PointCache-H value as the severity-2 reference, so these deltas are diagnostic rather than a strict same-granularity comparison.

```text
current accuracy (current accuracy - paper Table 7 PointCache-H value)
```

| Corruption | Paper PointCache-H avg | S0 | S1 | S2 | S3 | S4 | Current avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Add Global | 23.46 | 29.53 (+6.07) | 27.90 (+4.44) | 27.45 (+3.99) | 26.72 (+3.26) | 23.53 (+0.07) | 27.03 |
| Add Local | 22.69 | 28.28 (+5.59) | 22.52 (-0.17) | 23.98 (+1.29) | 21.13 (-1.56) | 19.33 (-3.36) | 23.05 |
| Drop Global | 34.70 | 39.10 (+4.40) | 36.78 (+2.08) | 38.48 (+3.78) | 36.02 (+1.32) | 29.01 (-5.69) | 35.88 |
| Drop Local | 31.75 | 36.26 (+4.51) | 33.14 (+1.39) | 33.41 (+1.66) | 27.00 (-4.75) | 26.61 (-5.14) | 31.28 |
| Rotate | 33.00 | 36.75 (+3.75) | 33.87 (+0.87) | 35.29 (+2.29) | 29.91 (-3.09) | 27.76 (-5.24) | 32.72 |
| Scale | 28.28 | 32.41 (+4.13) | 30.57 (+2.29) | 29.63 (+1.35) | 28.97 (+0.69) | 30.40 (+2.12) | 30.40 |
| Jitter | 25.05 | 37.20 (+12.15) | 29.74 (+4.69) | 25.50 (+0.45) | 24.32 (-0.73) | 19.81 (-5.24) | 27.31 |

## All35 Current Result vs Local Original PointCache-H

This is the exact all35 cell-to-cell comparison against the local original Point-Cache hierarchical baseline. This is the correct table for all35 claims.

```text
local original PC-H -> current (delta)
```

| Corruption | S0 | S1 | S2 | S3 | S4 | Mean Delta |
|---|---:|---:|---:|---:|---:|---:|
| Add Global | 27.48 -> 29.53 (+2.05) | 23.53 -> 27.90 (+4.37) | 23.87 -> 27.45 (+3.58) | 22.73 -> 26.72 (+3.99) | 20.71 -> 23.53 (+2.82) | +3.36 |
| Add Local | 25.54 -> 28.28 (+2.74) | 22.21 -> 22.52 (+0.31) | 22.73 -> 23.98 (+1.25) | 20.06 -> 21.13 (+1.07) | 19.67 -> 19.33 (-0.34) | +1.01 |
| Drop Global | 35.29 -> 39.10 (+3.81) | 35.88 -> 36.78 (+0.90) | 34.25 -> 38.48 (+4.23) | 32.76 -> 36.02 (+3.26) | 28.11 -> 29.01 (+0.90) | +2.62 |
| Drop Local | 32.76 -> 36.26 (+3.50) | 30.71 -> 33.14 (+2.43) | 30.26 -> 33.41 (+3.15) | 26.16 -> 27.00 (+0.84) | 22.24 -> 26.61 (+4.37) | +2.86 |
| Rotate | 34.32 -> 36.75 (+2.43) | 33.66 -> 33.87 (+0.21) | 32.55 -> 35.29 (+2.74) | 26.16 -> 29.91 (+3.75) | 24.95 -> 27.76 (+2.81) | +2.39 |
| Scale | 31.02 -> 32.41 (+1.39) | 28.35 -> 30.57 (+2.22) | 28.38 -> 29.63 (+1.25) | 26.75 -> 28.97 (+2.22) | 28.14 -> 30.40 (+2.26) | +1.87 |
| Jitter | 33.55 -> 37.20 (+3.65) | 28.14 -> 29.74 (+1.60) | 23.53 -> 25.50 (+1.97) | 23.84 -> 24.32 (+0.48) | 19.15 -> 19.81 (+0.66) | +1.67 |

## Notes

- Table 7 is the correct paper table for the `hardest` split, but it should not be called an all35 paper table.
- For the same paper-level severity-2 comparison, the current method is above paper ULIP + hierarchical PointCache by `+2.11` points.
- For the actual all35 comparison, use the local original PointCache-H baseline: the current method is above it by `+2.25` all35 points.
- Against the local original PointCache-H all35 cells, the current method improves 34/35 cells; only `Add Local S4` drops by `-0.34`.
