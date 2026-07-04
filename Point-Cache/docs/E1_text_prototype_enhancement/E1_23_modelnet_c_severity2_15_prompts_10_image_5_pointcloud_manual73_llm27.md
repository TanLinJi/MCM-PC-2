# E1_23 ModelNet-C Severity2 15 Prompts 10 Image 5 Pointcloud Manual73 LLM27

Updated: 2026-06-17

## Basic Setting

| Item | Value |
|---|---|
| Experiment ID | E1_23 severity2 diagnostic |
| Result `exp_id` | `E1_23_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual73_llm27` |
| Purpose | Test whether changing fusion weight from `0.75:0.25` to `0.73:0.27` improves ModelNet-C severity=2 on the same prompt composition as E1_20 |
| Dataset and scope | ModelNet-C severity=2, 7 evaluations = 7 corruption types x 1 severity |
| Backbone | ULIP |
| Task setting | zero-shot |
| Text prototype method | `manual_full + LLM` weighted fusion |
| LLM prompt | 15 prompts/class = 10 image + 5 pointcloud |
| Fusion weights | `manual_full = 0.73`, `LLM = 0.27` |
| Prompt JSON | `Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json` |
| Default result directory | `Point-Cache/results/E1_text_prototype_enhancement/E1_23_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual73_llm27/` |

## Relation to Previous Experiments

This experiment keeps the same prompt composition as E1_20:

```text
15 prompts = 10 image + 5 pointcloud
```

The only intentional change is the fusion weight:

```text
E1_20: 0.75 : 0.25
E1_23: 0.73 : 0.27
```

So E1_23 is a pure severity2 weight ablation on the E1_20 prompt set.

## Execution Command

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E1_text_prototype_enhancement/prompt_composition_ablation/E1_23_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual73_llm27.sh 0
```

## Output Location

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_23_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual73_llm27/
Point-Cache/results/E1_text_prototype_enhancement/E1_23_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual73_llm27/summary.csv
Point-Cache/results/E1_text_prototype_enhancement/E1_23_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual73_llm27/logs/
```

## Completion Check

| Check | Result |
|---|---|
| `summary.csv` rows | 7 |
| `status` | all `done` |
| severity | all `2` |
| corruption coverage | 7 types |
| average accuracy | 49.91 |

## Comparison Set

| Compared experiment | Path | Meaning |
|---|---|---|
| E1_20 severity2 slice | `Point-Cache/results/E1_text_prototype_enhancement/E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25/summary.csv` | Same prompt composition, different weight; use severity=2 rows only |
| E1_22 severity2 diagnostic | `Point-Cache/results/E1_text_prototype_enhancement/E1_22_modelnet_c_severity2_15_prompts_12_image_3_pointcloud_manual75_llm25/summary.csv` | 15 prompts = 12 image + 3 pointcloud |
| E1_21 severity2 diagnostic | `Point-Cache/results/E1_text_prototype_enhancement/E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25/summary.csv` | 15 prompts = 5 image + 10 pointcloud |
| E1_13 severity2 diagnostic | `Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_severity2_manual75_llm25/summary.csv` | 10 prompts = 4 image + 4 pointcloud + 2 bridge |
| E0 baseline severity2 slice | `Point-Cache/results/E0_baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv` | ULIP zero-shot baseline |

## Overall Result

| Setting | severity=2 average | Delta vs E1_23 |
|---|---:|---:|
| E1_23 `0.73:0.27`, 15 prompts = 10 image + 5 pointcloud | 49.91 | 0.00 |
| E1_20 severity2 slice `0.75:0.25`, same prompt composition | 49.79 | +0.12 |
| E1_22 `0.75:0.25`, 15 prompts = 12 image + 3 pointcloud | 49.19 | +0.72 |
| E1_21 `0.75:0.25`, 15 prompts = 5 image + 10 pointcloud | 48.55 | +1.36 |
| E1_13 severity2 diagnostic `0.75:0.25`, 10 prompts | 48.90 | +1.01 |
| E0 baseline | 47.68 | +2.23 |

## Per-Corruption Comparison

| corruption | E1_23 | E1_20 s2 | diff |
|---|---:|---:|---:|
| add_global | 37.60 | 37.48 | +0.12 |
| add_local | 45.10 | 45.02 | +0.08 |
| dropout_global | 56.97 | 56.85 | +0.12 |
| dropout_local | 54.58 | 54.29 | +0.29 |
| rotate | 56.08 | 55.96 | +0.12 |
| scale | 53.44 | 53.40 | +0.04 |
| jitter | 45.62 | 45.54 | +0.08 |

## Analysis

1. E1_23 is a clean weight-only ablation against the severity2 slice of E1_20.
2. The new `0.73:0.27` setting is slightly better than `0.75:0.25` on severity2, but the gain is small: only `+0.12` average.
3. The improvement is consistent across all 7 corruption types, which makes the result more believable than a one-off spike.
4. Compared with the other severity2 prompt compositions, E1_23 is now the best tested severity2 setting so far.
5. The margin over E1_20 is still tiny, so this is a mild preference, not a dramatic shift.

## Current Conclusion

E1_23 suggests that `0.73:0.27` may be a slightly better severity2 fusion weight than `0.75:0.25` when the prompt composition stays fixed at `10 image + 5 pointcloud`. The effect is small, so it should be treated as a fine-tuning result rather than a new regime.
