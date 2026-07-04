# E1_24 ModelNet-C Severity2 15 Prompts 10 Image 5 Pointcloud Manual74 LLM26

Updated: 2026-06-17

## Basic Setting

| Item | Value |
|---|---|
| Experiment ID | E1_24 severity2 diagnostic |
| Result `exp_id` | `E1_24_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual74_llm26` |
| Purpose | Test whether changing fusion weight from `0.73:0.27` to `0.74:0.26` improves ModelNet-C severity=2 on the same prompt composition as E1_23 |
| Dataset and scope | ModelNet-C severity=2, 7 evaluations = 7 corruption types x 1 severity |
| Backbone | ULIP |
| Task setting | zero-shot |
| Text prototype method | `manual_full + LLM` weighted fusion |
| LLM prompt | 15 prompts/class = 10 image + 5 pointcloud |
| Fusion weights | `manual_full = 0.74`, `LLM = 0.26` |
| Prompt JSON | `Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json` |
| Default result directory | `Point-Cache/results/E1_text_prototype_enhancement/E1_24_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual74_llm26/` |

## Relation to Previous Experiments

This experiment keeps the same prompt composition as E1_23 and E1_20:

```text
15 prompts = 10 image + 5 pointcloud
```

The only intentional change is the fusion weight:

```text
E1_23: 0.73 : 0.27
E1_24: 0.74 : 0.26
E1_20 severity2 slice: 0.75 : 0.25
```

So E1_24 is a pure severity2 weight ablation on the same prompt set.

## Execution Command

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E1_text_prototype_enhancement/prompt_composition_ablation/E1_24_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual74_llm26.sh 0
```

## Output Location

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_24_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual74_llm26/
Point-Cache/results/E1_text_prototype_enhancement/E1_24_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual74_llm26/summary.csv
Point-Cache/results/E1_text_prototype_enhancement/E1_24_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual74_llm26/logs/
```

## Completion Check

| Check | Result |
|---|---|
| `summary.csv` rows | 7 |
| `status` | all `done` |
| severity | all `2` |
| corruption coverage | 7 types |
| average accuracy | 49.84 |

## Comparison Set

| Compared experiment | Path | Meaning |
|---|---|---|
| E1_23 severity2 diagnostic | `Point-Cache/results/E1_text_prototype_enhancement/E1_23_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual73_llm27/summary.csv` | Same prompt composition, weight `0.73:0.27` |
| E1_20 severity2 slice | `Point-Cache/results/E1_text_prototype_enhancement/E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25/summary.csv` | Same prompt composition, weight `0.75:0.25` |
| E1_22 severity2 diagnostic | `Point-Cache/results/E1_text_prototype_enhancement/E1_22_modelnet_c_severity2_15_prompts_12_image_3_pointcloud_manual75_llm25/summary.csv` | 15 prompts = 12 image + 3 pointcloud |
| E1_21 severity2 diagnostic | `Point-Cache/results/E1_text_prototype_enhancement/E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25/summary.csv` | 15 prompts = 5 image + 10 pointcloud |
| E1_13 severity2 diagnostic | `Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_severity2_manual75_llm25/summary.csv` | 10 prompts = 4 image + 4 pointcloud + 2 bridge |
| E0 baseline severity2 slice | `Point-Cache/results/E0_baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv` | ULIP zero-shot baseline |

## Overall Result

| Setting | severity=2 average | Delta vs E1_24 |
|---|---:|---:|
| E1_24 `0.74:0.26`, 15 prompts = 10 image + 5 pointcloud | 49.84 | 0.00 |
| E1_23 `0.73:0.27`, same prompt composition | 49.91 | -0.07 |
| E1_20 severity2 slice `0.75:0.25`, same prompt composition | 49.79 | +0.05 |
| E1_22 `0.75:0.25`, 15 prompts = 12 image + 3 pointcloud | 49.19 | +0.65 |
| E1_21 `0.75:0.25`, 15 prompts = 5 image + 10 pointcloud | 48.55 | +1.29 |
| E1_13 severity2 diagnostic `0.75:0.25`, 10 prompts | 48.90 | +0.94 |
| E0 baseline | 47.68 | +2.16 |

## Per-Corruption Comparison

| corruption | E1_24 | E1_23 | diff |
|---|---:|---:|---:|
| add_global | 37.40 | 37.60 | -0.20 |
| add_local | 45.10 | 45.10 | +0.00 |
| dropout_global | 56.81 | 56.97 | -0.16 |
| dropout_local | 54.62 | 54.58 | +0.04 |
| rotate | 56.00 | 56.08 | -0.08 |
| scale | 53.40 | 53.44 | -0.04 |
| jitter | 45.58 | 45.62 | -0.04 |

## Analysis

1. E1_24 is a clean weight-only ablation against E1_23.
2. The new `0.74:0.26` setting does not beat `0.73:0.27`; it is lower by `0.07` average.
3. Compared with E1_20 severity2 slice, E1_24 is slightly better by `+0.05`, but the margin is very small.
4. The best tested weight among `0.73:0.27`, `0.74:0.26`, and `0.75:0.25` remains `0.73:0.27`.
5. All differences are tiny, so this is a fine-grained local optimum rather than a strong regime change.

## Current Conclusion

E1_24 suggests that `0.74:0.26` is not better than `0.73:0.27` on ModelNet-C severity=2 with `15 prompts = 10 image + 5 pointcloud`. The weight optimum in this narrow S2 sweep is still `0.73:0.27`.
