# E1_36 ModelNet-C Full 15 Prompts 10 Image 5 Pointcloud Manual60 LLM40

更新日期：2026-06-17

## 基本实验设置

| 项目 | 设置 |
|---|---|
| 实验编号 | E1_36 |
| Result `exp_id` | `E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40` |
| 实验目的 | 将当前采用的 E1_33 设置从 severity=2 扩展到完整 ModelNet-C，验证 all35 是否仍优于 E1_20 |
| 数据集与评估范围 | ModelNet-C full，35 evaluations = 7 corruption types x 5 severities |
| Backbone | ULIP |
| 任务设置 | zero-shot |
| 文本原型方法 | `manual_full + LLM` weighted fusion |
| LLM prompt | 15 prompts/class = 10 image + 5 pointcloud |
| 融合权重 | `manual_full = 0.60`, `LLM = 0.40` |
| Prompt JSON | `Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json` |
| 默认结果目录 | `Point-Cache/results/E1_text_prototype_enhancement/E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40/` |

## 与 E1_33 的关系

E1_36 完全继承 E1_33 的方法设置，只改变评估范围：

| 项目 | E1_33 | E1_36 |
|---|---|---|
| 权重 | `manual60_llm40` | `manual60_llm40` |
| prompt | 15 prompts = 10 image + 5 pointcloud | 15 prompts = 10 image + 5 pointcloud |
| 数据范围 | ModelNet-C severity=2 | ModelNet-C full |
| evaluations | 7 | 35 |

## 执行命令

默认模式不会重新调用 API，只读取已有 prompt JSON。

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E1_text_prototype_enhancement/modelnet_c_full_validation/E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40.sh 0
```

输出位置：

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40/
Point-Cache/results/E1_text_prototype_enhancement/E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40/summary.csv
Point-Cache/results/E1_text_prototype_enhancement/E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40/logs/
```

## 备用：重新生成 prompt 后运行

只有显式设置 `E1_USE_DEFAULT_PROMPTS=0` 时才会调用 API。重新生成的 prompt 会保存到 `Point-Cache/llm/generated/` 下的新目录。

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_USE_DEFAULT_PROMPTS=0
export E1_GENERATED_PROMPT_TAG="trial01"
bash scripts/E1_text_prototype_enhancement/modelnet_c_full_validation/E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40.sh 0
```

重新生成 prompt 的默认保存位置：

```text
Point-Cache/llm/generated/E1_36_15_prompts_10_image_5_pointcloud_trial01/
```

重新生成模式的默认结果目录会带上 tag：

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40_regenerated_trial01/
```

## 完成检查

| 检查项 | 结果 |
|---|---|
| `summary.csv` 数据行数 | 35 |
| `status` | 全部 `done` |
| corruption 覆盖 | 7 类 |
| severity 覆盖 | 0, 1, 2, 3, 4 |
| full average | 49.36 |

## 总体结果

| 设置 | ModelNet-C full average | 与 E1_36 差值 |
|---|---:|---:|
| E1_36 `15 prompts, 10 image + 5 pointcloud, manual60_llm40` | 49.36 | 0.00 |
| E1_20 `15 prompts, 10 image + 5 pointcloud, manual75_llm25` | 48.81 | +0.55 |
| E1_13 `10 prompts, manual75_llm25` | 48.07 | +1.29 |
| E1_10 `10 prompts, manual90_llm10` | 47.60 | +1.76 |
| E0 baseline | 46.85 | +2.51 |

## 按 Severity 对比

| severity | E1_36 | E1_20 | diff | E1_13 | diff | E0 | diff |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 56.02 | 55.32 | +0.70 | 54.85 | +1.17 | 53.40 | +2.62 |
| 1 | 53.58 | 53.06 | +0.52 | 52.32 | +1.25 | 50.94 | +2.64 |
| 2 | 50.41 | 49.79 | +0.61 | 48.90 | +1.50 | 47.68 | +2.73 |
| 3 | 46.28 | 45.76 | +0.52 | 45.07 | +1.22 | 43.85 | +2.44 |
| 4 | 40.51 | 40.14 | +0.37 | 39.21 | +1.30 | 38.39 | +2.12 |

## 按 Corruption 对比

| corruption | E1_36 | E1_20 | diff | E1_13 | diff | E0 | diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| add_global | 38.71 | 37.99 | +0.72 | 36.25 | +2.46 | 34.89 | +3.83 |
| add_local | 45.45 | 45.88 | -0.44 | 44.96 | +0.49 | 44.57 | +0.88 |
| dropout_global | 55.84 | 54.87 | +0.97 | 54.71 | +1.14 | 52.92 | +2.92 |
| dropout_local | 53.22 | 52.65 | +0.57 | 51.27 | +1.95 | 50.22 | +3.00 |
| jitter | 43.16 | 43.05 | +0.11 | 42.01 | +1.15 | 41.66 | +1.50 |
| rotate | 54.62 | 53.97 | +0.65 | 54.15 | +0.47 | 52.69 | +1.93 |
| scale | 54.52 | 53.29 | +1.23 | 53.15 | +1.37 | 51.00 | +3.52 |

## All35 逐项胜负

| 对比对象 | E1_36 wins | ties | losses |
|---|---:|---:|---:|
| E1_20 | 27 | 0 | 8 |
| E1_13 | 34 | 0 | 1 |
| E0 baseline | 35 | 0 | 0 |

E1_36 相对 E1_20 的 8 个下降项：

| item | E1_36 | E1_20 | diff |
|---|---:|---:|---:|
| add_local_0 | 51.58 | 52.47 | -0.89 |
| jitter_4 | 23.95 | 24.84 | -0.89 |
| jitter_3 | 34.36 | 34.93 | -0.57 |
| add_local_1 | 48.10 | 48.58 | -0.48 |
| add_local_3 | 42.63 | 43.03 | -0.40 |
| add_local_2 | 44.77 | 45.02 | -0.25 |
| add_local_4 | 40.15 | 40.32 | -0.17 |
| jitter_2 | 45.42 | 45.54 | -0.12 |

E1_36 相对 E1_20 的最大提升项：

| item | E1_36 | E1_20 | diff |
|---|---:|---:|---:|
| scale_3 | 54.25 | 52.67 | +1.58 |
| scale_2 | 54.86 | 53.40 | +1.46 |
| scale_4 | 51.70 | 50.36 | +1.34 |
| add_global_3 | 33.95 | 32.70 | +1.25 |
| rotate_0 | 59.24 | 58.02 | +1.22 |
| dropout_global_0 | 59.08 | 57.86 | +1.22 |
| jitter_1 | 54.82 | 53.65 | +1.17 |
| scale_0 | 56.73 | 55.59 | +1.14 |

## 分析记录

1. E1_36 的 full average 为 49.36，比 E1_20 高 +0.55，比 E0 baseline 高 +2.51。S2 上观察到的高 LLM 权重收益可以推广到完整 ModelNet-C。
2. 按 severity 看，E1_36 在 0 到 4 的所有 severity 上都高于 E1_20，说明提升不是只来自 severity=2。
3. 按 corruption 看，E1_36 在 7 类 corruption 中有 6 类高于 E1_20，唯一低于 E1_20 的是 `add_local`，差值 -0.44。
4. `scale` 是最大收益来源，相对 E1_20 提升 +1.23；`dropout_global`、`add_global`、`rotate` 也有明显提升。
5. `jitter` full 平均仍略高于 E1_20 (+0.11)，但高 severity 的 `jitter_3` 和 `jitter_4` 低于 E1_20，说明高 LLM 权重对强 jitter 的稳定性有代价。
6. 相对 E1_20，E1_36 在 35 个评估项中 27 项更高、8 项更低；下降项集中在 `add_local` 全部 severity 和 `jitter_2/3/4`。
7. 相对 E0 baseline，E1_36 在 35 个评估项全部更高，说明该设置没有破坏 baseline 稳定性。

## 当前结论

E1_36 确认 E1_33 设置可以进入当前 E1 的正式候选：`manual60_llm40 + 15 prompts, 10 image + 5 pointcloud` 在完整 ModelNet-C 上显著优于 E1_20 和 E0 baseline。主要风险是 `add_local` 和高 severity `jitter`，后续跨数据集验证时需要重点观察这两类扰动是否继续成为弱项。
