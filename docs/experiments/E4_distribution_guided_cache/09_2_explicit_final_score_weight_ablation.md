# 09-2 Explicit Final-Score Weight Ablation

## Current Best Record

Recorded on 2026-07-07.

Current best strict ULIP + ModelNet-C all35 result:

```text
all35 avg = 53.7138
capacity = (entropy_cap, gpa_cap, local_cap, neg_cap) = (3, 3, 3, 6)
local_centers = 3
final score = y_zs + 4.4 * y_g + 3.9 * y_l - 0.19 * y_n
weight_name = s2_best_dense
```

Severity-wise result:

| S0 | S1 | S2 | S3 | S4 | all35 |
|---:|---:|---:|---:|---:|---:|
| 59.9792 | 56.8187 | 55.1459 | 51.0651 | 45.5603 | **53.7138** |

Result directory:

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_e3_g3_l3_n6_priority_cap678_w6_ulip_modelnetc_all35_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

Key row is in:

```text
final_score_weight_summary.csv
```

## Goal

在 09-1 缓存容量消融的基础上，固定缓存容量为 `(entropy_cap, gpa_cap, local_cap, neg_cap) = (3, 3, 3, 6)`，只调最终得分公式中的权重：

```text
y = y_zs + alpha_g * y_g + alpha_l * y_l - alpha_n * y_n
```

这里 `y_g / y_l / y_n` 均是不含前置权重的原始缓存得分项。09-2 新代码已将权重从 `compute_cache_score` 和 `compute_local_cache_score` 中移出，只在最终公式处统一加权。

## Cache-Capacity Selection

ModelNet-C severity=2, 7 corruptions:

| setting | avg | add_global | add_local | dropout_global | dropout_local | rotate | scale | jitter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `(3,3,3,2)` default | 54.6786 | 48.06 | 50.85 | 59.12 | 57.46 | 60.94 | 55.83 | 50.49 |
| `(3,3,3,5)` | 54.8514 | 49.39 | 50.81 | 59.04 | 57.09 | 61.06 | 56.00 | 50.57 |
| `(3,3,3,6)` | **54.9843** | 49.92 | 50.89 | 59.04 | 57.09 | 61.18 | 56.08 | 50.69 |
| `(3,3,3,7)` | 54.9314 | 50.00 | 50.81 | 58.83 | 56.97 | 61.10 | 56.16 | 50.65 |
| `(3,3,3,8)` | 54.9600 | 50.20 | 50.73 | 58.91 | 56.85 | 61.14 | 56.08 | 50.81 |
| `(2,3,3,5)` | 54.9100 | 50.41 | 50.41 | 59.16 | 56.89 | 60.66 | 55.75 | 51.09 |

选择 `(3,3,3,6)` 的原因：

- S2 平均值当前最高：54.9843。
- 相比 `(3,3,3,5)`，7 个扰动整体提升 `+0.1329`，主要来自 `add_global +0.53`，同时 `add_local / rotate / scale / jitter` 均有小幅提升。
- 相比 `(2,3,3,5)`，平均值 `+0.0743`，并且 `add_local / dropout_local / rotate / scale` 更稳；虽然 `add_global / jitter` 略低，但整体更均衡。

All35 / clean 补充结果：

| setting | S2 avg | all35 avg | clean |
|---|---:|---:|---:|
| `(2,3,3,5)` | 54.9100 | 53.0857 | 63.21 |
| `(3,3,3,5)` | 54.8514 | 53.4063 | 64.14 |
| `(3,3,3,6)` | 54.9843 | 53.4863 | 64.06 |
| `(3,3,3,7)` | 54.9314 | 53.5414 | 64.10 |
| `(3,3,3,8)` | 54.9600 | 53.5623 | 63.94 |

## Weight Grid

第一轮脚本使用 100 组权重：

```text
alpha_g in {1.6, 1.8, 2.0, 2.2, 2.4}
alpha_l in {1.6, 1.8, 2.0, 2.2}
alpha_n in {0.10, 0.117, 0.13, 0.15, 0.18}
```

注意：第一轮网格没有覆盖 09-1 实际运行时的正缓存权重。日志显示 09-1 / 09-2 的 runtime config 中：

```text
positive.alpha = 4.0
negative.alpha = 0.117
```

因此真正的原始配置应是：

```text
alpha_g = 4.0
alpha_l = 4.0
alpha_n = 0.117
```

## First Sweep Result

Result directory:

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_e3_g3_l3_n6_ag5_al4_an5_grid_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

Top-10 within the first scanned grid:

| rank | alpha_g | alpha_l | alpha_n | avg | add_global | add_local | dropout_global | dropout_local | rotate | scale | jitter |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.8 | 2.2 | 0.18 | 54.8449 | 52.43 | 50.85 | 58.27 | 55.27 | 60.70 | 55.71 | 50.69 |
| 2 | 2.0 | 2.2 | 0.18 | 54.8217 | 52.59 | 50.73 | 58.18 | 55.31 | 60.70 | 55.63 | 50.61 |
| 3 | 2.2 | 2.2 | 0.18 | 54.8101 | 52.51 | 50.69 | 58.10 | 55.43 | 60.66 | 55.67 | 50.61 |
| 4 | 2.0 | 2.2 | 0.15 | 54.7928 | 52.23 | 50.41 | 58.31 | 55.55 | 60.70 | 55.67 | 50.69 |
| 5 | 2.2 | 2.0 | 0.18 | 54.7928 | 52.47 | 50.73 | 57.98 | 55.35 | 60.66 | 55.55 | 50.81 |
| 6 | 1.6 | 2.2 | 0.18 | 54.7870 | 52.47 | 50.73 | 58.27 | 55.06 | 60.74 | 55.63 | 50.61 |
| 7 | 2.4 | 2.0 | 0.18 | 54.7812 | 52.39 | 50.61 | 57.98 | 55.39 | 60.70 | 55.63 | 50.77 |
| 8 | 2.4 | 2.2 | 0.15 | 54.7812 | 52.07 | 50.53 | 58.27 | 55.59 | 60.70 | 55.71 | 50.61 |
| 9 | 1.8 | 2.0 | 0.18 | 54.7754 | 52.43 | 50.89 | 57.98 | 55.19 | 60.62 | 55.63 | 50.69 |
| 10 | 1.8 | 2.2 | 0.15 | 54.7754 | 52.11 | 50.53 | 58.27 | 55.43 | 60.62 | 55.71 | 50.77 |
 
Interpretation:

- 第一轮 top-1 是 `(alpha_g, alpha_l, alpha_n) = (1.8, 2.2, 0.18)`，S2 avg 为 54.8449。
- 但该结果低于 09-1 同容量真实基线 `(3,3,3,6)` 的 54.9843，因此不能作为最终改进结论。
- 第一轮网格整体显示 `alpha_n` 增大有收益：按 `alpha_n` 分组，平均值从 `0.10 -> 0.18` 单调上升。
- 第一轮网格也显示 `alpha_l` 越大越好，且 `alpha_g` 的均值在上边界 2.4 仍最高。这进一步说明应该把第二轮网格移动到真实正缓存权重 4.0 附近。

## Second Sweep Grid

第二轮脚本已改为围绕真实基线 `(4.0, 4.0, 0.117)` 搜索：

```text
alpha_g in {3.2, 3.6, 4.0, 4.4, 4.8}
alpha_l in {3.2, 3.6, 4.0, 4.4, 4.8}
alpha_n in {0.117, 0.15, 0.18, 0.22, 0.26}
```

共 125 组，包含真实原始配置：

```text
alpha_g = 4.0
alpha_l = 4.0
alpha_n = 0.117
```

## Second Sweep Result

Result directory:

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_e3_g3_l3_n6_ag5_al5_an5_center4_grid_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

The explicit-score implementation reproduces the 09-1 `(3,3,3,6)` baseline:

| setting | alpha_g | alpha_l | alpha_n | avg | add_global | add_local | dropout_global | dropout_local | rotate | scale | jitter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 09-1 baseline | 4.0 | 4.0 | 0.117 | 54.9843 | 49.92 | 50.89 | 59.04 | 57.09 | 61.18 | 56.08 | 50.69 |
| 09-2 reproduced baseline | 4.0 | 4.0 | 0.117 | 54.9838 | 49.92 | 50.89 | 59.04 | 57.09 | 61.18 | 56.08 | 50.69 |

Top-10 within the second grid:

| rank | alpha_g | alpha_l | alpha_n | avg | add_global | add_local | dropout_global | dropout_local | rotate | scale | jitter |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.4 | 4.0 | 0.18 | **55.1227** | 50.93 | 51.01 | 58.83 | 56.81 | 61.14 | 56.32 | 50.81 |
| 2 | 4.8 | 3.6 | 0.18 | 55.0938 | 51.42 | 50.89 | 58.67 | 56.77 | 61.14 | 56.12 | 50.65 |
| 3 | 3.6 | 4.0 | 0.22 | 55.0880 | 51.54 | 50.85 | 58.51 | 56.77 | 61.02 | 56.28 | 50.65 |
| 4 | 4.0 | 4.0 | 0.22 | 55.0880 | 51.58 | 50.77 | 58.47 | 56.77 | 61.14 | 56.32 | 50.57 |
| 5 | 3.2 | 3.2 | 0.22 | 55.0822 | 52.15 | 50.77 | 58.43 | 56.36 | 61.10 | 56.08 | 50.69 |
| 6 | 4.0 | 4.0 | 0.18 | 55.0764 | 50.81 | 50.97 | 58.75 | 56.77 | 61.18 | 56.20 | 50.85 |
| 7 | 4.8 | 3.2 | 0.15 | 55.0764 | 51.13 | 51.01 | 58.75 | 56.77 | 61.10 | 55.92 | 50.85 |
| 8 | 3.6 | 3.6 | 0.18 | 55.0706 | 51.26 | 50.93 | 58.75 | 56.48 | 61.10 | 56.24 | 50.73 |
| 9 | 4.8 | 3.2 | 0.18 | 55.0706 | 51.54 | 50.85 | 58.47 | 56.73 | 61.26 | 55.96 | 50.69 |
| 10 | 3.2 | 3.6 | 0.18 | 55.0648 | 51.13 | 51.01 | 58.83 | 56.44 | 61.14 | 56.16 | 50.73 |

Best setting:

```text
alpha_g = 4.4
alpha_l = 4.0
alpha_n = 0.18
avg = 55.1227
```

Compared with the reproduced baseline `(4.0,4.0,0.117)`:

```text
avg +0.1389
add_global +1.01
add_local +0.12
dropout_global -0.21
dropout_local -0.28
rotate -0.04
scale +0.24
jitter +0.12
```

Grouped trend:

```text
alpha_n mean:
0.117 -> 54.8947
0.15  -> 54.9627
0.18  -> 55.0183
0.22  -> 54.9958
0.26  -> 54.9000

alpha_g mean:
3.2 -> 54.9241
3.6 -> 54.9447
4.0 -> 54.9613
4.4 -> 54.9731
4.8 -> 54.9683

alpha_l mean:
3.2 -> 54.9803
3.6 -> 55.0044
4.0 -> 54.9928
4.4 -> 54.9254
4.8 -> 54.8685
```

Interpretation:

- `alpha_n` 的最佳区域在 `0.18 ~ 0.22`，继续增大到 `0.26` 会下降。
- `alpha_g` 的均值峰值在 `4.4` 附近，说明全局正缓存略强于原始 `4.0` 有收益。
- `alpha_l` 的均值峰值在 `3.6 ~ 4.0`，继续增大到 `4.4/4.8` 会下降。
- Top-1 的收益主要来自 `add_global` 和 `scale`，代价主要在 `dropout_global/dropout_local`。

## Third Sweep Result

The same `ag5_al5_an5_center4_grid` directory was later updated with a denser local grid:

```text
alpha_g in {4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7}
alpha_l in {3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3}
alpha_n in {0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22}
```

This grid has 490 settings. It does not include the original baseline `(4.0,4.0,0.117)`, so comparisons use the reproduced baseline from the second sweep.

Top-10:

| rank | alpha_g | alpha_l | alpha_n | avg | add_global | add_local | dropout_global | dropout_local | rotate | scale | jitter |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.4 | 3.9 | 0.19 | **55.1459** | 51.26 | 50.89 | 58.75 | 56.85 | 61.22 | 56.32 | 50.73 |
| 2 | 4.3 | 3.9 | 0.19 | 55.1343 | 51.26 | 50.89 | 58.79 | 56.77 | 61.22 | 56.28 | 50.73 |
| 3 | 4.3 | 4.0 | 0.19 | 55.1285 | 51.13 | 50.89 | 58.75 | 56.89 | 61.22 | 56.28 | 50.73 |
| 4 | 4.2 | 3.9 | 0.19 | 55.1227 | 51.22 | 50.85 | 58.83 | 56.73 | 61.22 | 56.20 | 50.81 |
| 5 | 4.4 | 4.0 | 0.18 | 55.1227 | 50.93 | 51.01 | 58.83 | 56.81 | 61.14 | 56.32 | 50.81 |
| 6 | 4.5 | 3.9 | 0.19 | 55.1227 | 51.30 | 50.89 | 58.63 | 56.89 | 61.10 | 56.36 | 50.69 |
| 7 | 4.1 | 3.9 | 0.19 | 55.1169 | 51.22 | 50.89 | 58.79 | 56.73 | 61.22 | 56.16 | 50.81 |
| 8 | 4.1 | 4.0 | 0.19 | 55.1169 | 51.09 | 50.89 | 58.79 | 56.77 | 61.26 | 56.20 | 50.81 |
| 9 | 4.1 | 3.7 | 0.17 | 55.1111 | 50.89 | 50.97 | 58.83 | 56.81 | 61.22 | 56.24 | 50.81 |
| 10 | 4.2 | 4.0 | 0.18 | 55.1111 | 50.93 | 51.05 | 58.79 | 56.77 | 61.22 | 56.20 | 50.81 |

Best setting:

```text
alpha_g = 4.4
alpha_l = 3.9
alpha_n = 0.19
avg = 55.1459
```

Compared with the reproduced baseline `(4.0,4.0,0.117)`:

```text
avg +0.1616
add_global +1.34
add_local +0.00
dropout_global -0.29
dropout_local -0.24
rotate +0.04
scale +0.24
jitter +0.04
```

Compared with the previous best `(4.4,4.0,0.18)`:

```text
avg +0.0232
add_global +0.33
add_local -0.12
dropout_global -0.08
dropout_local +0.04
rotate +0.08
scale +0.00
jitter -0.08
```

Grouped trend in the third sweep:

```text
alpha_n mean:
0.16 -> 55.0316
0.17 -> 55.0425
0.18 -> 55.0566
0.19 -> 55.0546
0.20 -> 55.0381
0.21 -> 55.0237
0.22 -> 55.0215

alpha_g mean:
4.1 -> 55.0437
4.2 -> 55.0468
4.3 -> 55.0410
4.4 -> 55.0403
4.5 -> 55.0357
4.6 -> 55.0330
4.7 -> 55.0281

alpha_l mean:
3.4 -> 55.0313
3.5 -> 55.0247
3.6 -> 55.0333
3.7 -> 55.0384
3.8 -> 55.0483
3.9 -> 55.0600
4.0 -> 55.0601
4.1 -> 55.0528
4.2 -> 55.0188
4.3 -> 55.0163
```

Interpretation:

- The best region is a plateau around `alpha_g=4.2~4.4`, `alpha_l=3.9~4.0`, `alpha_n=0.18~0.19`.
- The exact top-1 `(4.4,3.9,0.19)` improves only `+0.0232` over `(4.4,4.0,0.18)`, so the difference is small.
- The main consistent gain over baseline is still `add_global`; the main cost remains `dropout_global/dropout_local`.

## Fine Scan Around `(4.4, 4.0, 0.18)`

Script:

```text
Point-Cache/scripts/E4_distribution_guided_cache/09_2_ulip_modelnetc_s2_explicit_final_score_fine_scan.sh
```

The user's requested full Cartesian grid would contain:

```text
alpha_g: 4.4 +/- 1.00, step 0.01 -> 201 values
alpha_l: 4.0 +/- 1.00, step 0.01 -> 201 values
alpha_n: 0.18 +/- 0.100, step 0.001 -> 201 values
full Cartesian = 201^3 = 8,120,601 settings
```

This is too large for the current per-sample Python loop and CSV output. The fine-scan script therefore uses:

```text
1. Full alpha_g axis scan:
   alpha_g = 3.40..5.40, step 0.01
   alpha_l = 4.00, alpha_n = 0.180

2. Full alpha_l axis scan:
   alpha_l = 3.00..5.00, step 0.01
   alpha_g = 4.40, alpha_n = 0.180

3. Full alpha_n axis scan:
   alpha_n = 0.080..0.280, step 0.001
   alpha_g = 4.40, alpha_l = 4.00

4. Local 3D interaction box:
   alpha_g = 4.35..4.45, step 0.01
   alpha_l = 3.95..4.05, step 0.01
   alpha_n = 0.175..0.185, step 0.001
```

After de-duplication this produces 1901 settings. It contains values such as:

```text
4.40,4.00,0.180
4.40,4.00,0.181
4.40,4.00,0.182
```

Command:

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/09_2_ulip_modelnetc_s2_explicit_final_score_fine_scan.sh 0
```

Output:

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_e3_g3_l3_n6_fine_axis_pm1_pm0p1_local_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

Fine scan completed with 1901 settings.

Top-10:

| rank | alpha_g | alpha_l | alpha_n | avg | add_global | add_local | dropout_global | dropout_local | rotate | scale | jitter |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.35 | 3.96 | 0.182 | **55.1459** | 51.09 | 50.97 | 58.83 | 56.85 | 61.22 | 56.32 | 50.73 |
| 2 | 4.40 | 4.00 | 0.182 | 55.1401 | 51.01 | 51.05 | 58.83 | 56.85 | 61.14 | 56.32 | 50.77 |
| 3 | 4.35 | 3.97 | 0.182 | 55.1401 | 51.05 | 50.97 | 58.83 | 56.81 | 61.22 | 56.32 | 50.77 |
| 4 | 4.35 | 3.97 | 0.184 | 55.1401 | 51.13 | 50.93 | 58.79 | 56.85 | 61.18 | 56.32 | 50.77 |
| 5 | 4.35 | 3.98 | 0.185 | 55.1401 | 51.13 | 50.89 | 58.75 | 56.93 | 61.18 | 56.32 | 50.77 |
| 6 | 4.35 | 3.99 | 0.184 | 55.1401 | 51.13 | 50.97 | 58.79 | 56.85 | 61.14 | 56.32 | 50.77 |
| 7 | 4.35 | 3.99 | 0.185 | 55.1401 | 51.13 | 50.93 | 58.79 | 56.89 | 61.14 | 56.32 | 50.77 |
| 8 | 4.39 | 3.98 | 0.182 | 55.1401 | 51.01 | 51.05 | 58.79 | 56.85 | 61.18 | 56.32 | 50.77 |
| 9 | 4.40 | 3.99 | 0.184 | 55.1401 | 51.09 | 50.97 | 58.79 | 56.85 | 61.18 | 56.32 | 50.77 |
| 10 | 4.40 | 4.03 | 0.182 | 55.1401 | 50.97 | 51.05 | 58.83 | 56.89 | 61.14 | 56.28 | 50.81 |

Compared with reproduced baseline `(4.0,4.0,0.117)`:

```text
best fine scan avg +0.1621
add_global +1.17
add_local +0.08
dropout_global -0.21
dropout_local -0.24
rotate +0.04
scale +0.24
jitter +0.04
```

Compared with previous dense-grid best `(4.4,3.9,0.19)`:

```text
avg +0.0000
add_global -0.17
add_local +0.08
dropout_global +0.08
dropout_local +0.00
rotate +0.00
scale +0.00
jitter +0.00
```

Interpretation:

- Fine scan did not improve beyond the previous best S2 avg `55.1459`.
- It found an equal-best point closer to the center: `(4.35,3.96,0.182)`.
- The local optimum is a plateau rather than a sharp point. Many settings around `alpha_g=4.35~4.40`, `alpha_l=3.96~4.03`, `alpha_n=0.182~0.185` are within `0.006` avg.
- For a clean paper setting, `(4.4,4.0,0.18)` remains a strong rounded choice with avg `55.1227`; for the strict S2 best, use `(4.35,3.96,0.182)` or the previous `(4.4,3.9,0.19)`.

## Code

- Model: `Point-Cache/runners/E4_distribution_guided_cache/model_e4_c_a0_e1_explicit_final_score.py`
- Runner: `Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a0_e1_explicit_final_score_ulip_modelnetc_s2.py`
- Script: `Point-Cache/scripts/E4_distribution_guided_cache/09_2_ulip_modelnetc_s2_explicit_final_score_ablation.sh`

## Command

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/09_2_ulip_modelnetc_s2_explicit_final_score_ablation.sh 0
```

## Output

Result directory:

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_e3_g3_l3_n6_ag5_al5_an5_center4_grid_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

Key files:

```text
summary.csv
final_score_weight_summary.csv
```

`summary.csv` 记录逐扰动、逐权重结果；`final_score_weight_summary.csv` 记录每组权重在 7 个扰动上的平均结果。

## All35 Validation Plan

下一步在完整 ModelNet-C 上验证当前暂定的 rounded best：

```text
cache capacity = (entropy_cap, gpa_cap, local_cap, neg_cap) = (3, 3, 3, 6)
final score = (alpha_g, alpha_l, alpha_n) = (4.4, 4.0, 0.18)
local_centers = 3
```

新增代码保持 09-2 显式得分公式不变：

```text
y = y_zs + alpha_g * y_g + alpha_l * y_l - alpha_n * y_n
```

脚本支持两层队列：

- `COMBINATIONS`：缓存容量组合，例如 `"3,3,3,6"`。
- `FINAL_SCORE_WEIGHTS`：最终得分权重组合，例如 `"4.4,4.0,0.18"`。

默认只跑当前暂定设置；如果要同时统计多组权重，可在 `FINAL_SCORE_WEIGHTS` 里继续加行。一次 all35 推理内会复用同一轮缓存更新，只在最终得分处统计多组 alpha。

All35 code:

- Runner: `Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a0_e1_explicit_final_score_ulip_modelnetc_all35.py`
- Common script: `Point-Cache/scripts/E4_distribution_guided_cache/09_run_e4_c_a0_e1_explicit_final_score_all35_common.sh`
- Queue script: `Point-Cache/scripts/E4_distribution_guided_cache/09_2_ulip_modelnetc_all35_explicit_final_score_ablation.sh`

Command:

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/09_2_ulip_modelnetc_all35_explicit_final_score_ablation.sh 0
```

Default output directory:

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_e3_g3_l3_n6_ag4p4_al4p0_an0p18_ulip_modelnetc_all35_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

Key files:

```text
summary.csv
final_score_weight_summary.csv
logs/
```

`summary.csv` 记录 35 个 `cor_type` 上逐权重结果；`final_score_weight_summary.csv` 记录每组权重的 all35 平均、每个 severity 平均，以及 35 个具体扰动结果。

## All35 Validation Result

Result directory:

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_e3_g3_l3_n6_ag4p4_al4p0_an0p18_ulip_modelnetc_all35_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

The run completed all 35 ModelNet-C corruptions with:

```text
cache capacity = (3, 3, 3, 6)
final score = (alpha_g, alpha_l, alpha_n) = (4.4, 4.0, 0.18)
all35 avg = 53.6872
```

Comparison with 09-1 capacity baselines:

| setting | final-score weights | all35 avg | 09-2 minus setting |
|---|---:|---:|---:|
| 09-1 `(3,3,3,6)` | `(4.0,4.0,0.117)` | 53.4863 | +0.2009 |
| 09-1 `(3,3,3,7)` | `(4.0,4.0,0.117)` | 53.5414 | +0.1458 |
| 09-1 `(3,3,3,8)` | `(4.0,4.0,0.117)` | 53.5623 | +0.1249 |
| 09-2 `(3,3,3,6)` | `(4.4,4.0,0.18)` | **53.6872** | - |

Severity-wise comparison with 09-1 `(3,3,3,6)`:

| setting | S0 | S1 | S2 | S3 | S4 | all35 |
|---|---:|---:|---:|---:|---:|---:|
| 09-1 `(3,3,3,6)` | 59.89 | 56.69 | 54.98 | 50.76 | 45.11 | 53.49 |
| 09-2 `(4.4,4.0,0.18)` | **59.97** | **56.78** | **55.12** | **51.05** | **45.51** | **53.69** |
| delta | +0.08 | +0.09 | +0.14 | +0.29 | +0.40 | +0.20 |

Corruption-wise comparison with 09-1 `(3,3,3,6)`:

| corruption | 09-1 avg | 09-2 avg | delta |
|---|---:|---:|---:|
| add_global | 49.42 | **50.53** | +1.11 |
| add_local | 50.77 | **50.92** | +0.15 |
| dropout_global | **58.09** | 58.01 | -0.08 |
| dropout_local | 55.44 | **55.48** | +0.04 |
| rotate | 58.24 | **58.31** | +0.07 |
| scale | 56.09 | **56.16** | +0.07 |
| jitter | 46.35 | **46.40** | +0.05 |

Interpretation:

- The S2 gain found during weight tuning transfers to all35: S2 improves from 54.98 to 55.12.
- The all35 gain is stronger on harder severities: S3 +0.29 and S4 +0.40, so the tuned final score is not merely overfitting the S2 validation point.
- The main source of improvement is still `add_global`, especially higher severity levels.
- The main tradeoff is `dropout_global`, which drops slightly on average.
- This all35 result also exceeds the previous best capacity-only all35 run `(3,3,3,8)` by +0.1249, so `(3,3,3,6)` plus tuned final-score weights is currently the stronger setting.

## All35 Priority Search Queue

Goal: push the strict ULIP + ModelNet-C all35 result beyond the current best `53.6872`.

Rationale:

- Capacity-only all35 prefers larger negative cache capacity: `(3,3,3,8)` gives `53.5623`, better than `(3,3,3,6)` at `53.4863`.
- S2 final-score tuning found a stable plateau around `alpha_g=4.35~4.40`, `alpha_l=3.90~4.03`, `alpha_n=0.18~0.19`.
- The first all35 validation showed `(4.4,4.0,0.18)` transfers from S2 to all35, especially on S3/S4.

Queue script:

```text
Point-Cache/scripts/E4_distribution_guided_cache/09_2_ulip_modelnetc_all35_explicit_final_score_ablation.sh
```

Capacity combinations:

```text
(3,3,3,8)
(3,3,3,7)
(3,3,3,6)
```

Final-score weights:

```text
rounded_all35 = (4.4, 4.0, 0.18)
s2_best_fine = (4.35, 3.96, 0.182)
s2_best_dense = (4.4, 3.9, 0.19)
near_1 = (4.4, 4.0, 0.182)
near_2 = (4.35, 3.97, 0.184)
near_3 = (4.35, 3.98, 0.185)
```

Command:

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/09_2_ulip_modelnetc_all35_explicit_final_score_ablation.sh 0
```

Expected output directories:

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_e3_g3_l3_n8_priority_cap678_w6_ulip_modelnetc_all35_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_e3_g3_l3_n7_priority_cap678_w6_ulip_modelnetc_all35_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/09_2_e3_g3_l3_n6_priority_cap678_w6_ulip_modelnetc_all35_zs_global_local_e4_c_a0_e1_explicit_final_score_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

Key file for comparison:

```text
final_score_weight_summary.csv
```

## All35 Priority Search Result

The `priority_cap678_w6` queue completed:

```text
capacity settings: (3,3,3,8), (3,3,3,7), (3,3,3,6)
weights per setting: 6
rows per setting: 35 corruptions * 6 weights = 210
status: all done
```

Top results:

| rank | capacity | weight name | `(alpha_g, alpha_l, alpha_n)` | all35 | S0 | S1 | S2 | S3 | S4 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `(3,3,3,6)` | `s2_best_dense` | `(4.4,3.9,0.19)` | **53.7138** | 59.9792 | 56.8187 | 55.1459 | 51.0651 | 45.5603 |
| 2 | `(3,3,3,6)` | `near_1` | `(4.4,4.0,0.182)` | 53.6918 | 59.9676 | 56.7898 | 55.1401 | 51.0361 | 45.5256 |
| 3 | `(3,3,3,6)` | `near_3` | `(4.35,3.98,0.185)` | 53.6895 | 59.9618 | 56.8071 | 55.1401 | 51.0361 | 45.5024 |
| 4 | `(3,3,3,6)` | `near_2` | `(4.35,3.97,0.184)` | 53.6872 | 59.9502 | 56.7956 | 55.1401 | 51.0477 | 45.5024 |
| 5 | `(3,3,3,6)` | `rounded_all35` | `(4.4,4.0,0.18)` | 53.6849 | 59.9676 | 56.7782 | 55.1227 | 51.0419 | 45.5140 |

Best per capacity:

| capacity | best weight | all35 |
|---|---|---:|
| `(3,3,3,8)` | `s2_best_dense` `(4.4,3.9,0.19)` | 53.6687 |
| `(3,3,3,7)` | `near_1` `(4.4,4.0,0.182)` | 53.6814 |
| `(3,3,3,6)` | `s2_best_dense` `(4.4,3.9,0.19)` | **53.7138** |

Comparison:

| baseline | all35 | delta from new best |
|---|---:|---:|
| previous recorded best `(3,3,3,6)` rounded/tuned setting | 53.6872 | +0.0266 |
| capacity-only `(3,3,3,8)` | 53.5623 | +0.1515 |
| capacity-only `(3,3,3,7)` | 53.5414 | +0.1724 |
| capacity-only `(3,3,3,6)` | 53.4863 | +0.2275 |

Interpretation:

- The best all35 setting is now `(entropy_cap,gpa_cap,local_cap,neg_cap)=(3,3,3,6)` with `(alpha_g,alpha_l,alpha_n)=(4.4,3.9,0.19)`.
- Increasing `neg_cap` to 7 or 8 improved capacity-only all35, but after final-score tuning it did not beat `neg_cap=6`.
- The new best keeps the same S2 score as the previous S2 top plateau (`55.1459`) while improving S0/S1/S4 slightly, so the all35 gain is not coming from S2 alone.
- Relative to the previous tuned setting, the main gain comes from `add_global` (+0.17 average), with a small tradeoff on `jitter` (-0.04 average).
