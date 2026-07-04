# E1_20 ModelNet-C Full 15 Prompts 10 Image 5 Pointcloud Manual75 LLM25

更新日期：2026-06-17

## 基本实验设置

| 项目 | 设置 |
|---|---|
| 实验编号 | E1_20 |
| Result `exp_id` | `E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25` |
| 实验目的 | 在当前更优权重 `manual75_llm25` 下，测试 15 条 LLM 描述、10 image + 5 pointcloud 的 prompt 数量与组成是否优于 E1_13 |
| 数据集与评估范围 | ModelNet-C full，35 evaluations = 7 corruption types x 5 severities |
| Backbone | ULIP |
| 任务设置 | zero-shot |
| 文本原型方法 | `manual_full + LLM` weighted fusion |
| LLM prompt | 15 prompts/class = 10 image + 5 pointcloud |
| 融合权重 | `manual_full = 0.75`, `LLM = 0.25` |
| 默认 Prompt JSON | `Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json` |
| 默认结果目录 | `Point-Cache/results/E1_text_prototype_enhancement/E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25/` |

## 与当前最佳设置的关系

当前 E1_13 使用的 prompt 文件是：

```text
Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_10_prompts_4_image_4_pointcloud_2_bridge.json
```

E1_20 将其替换为：

```text
Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json
```

除了 prompt 数量与组成之外，其他关键设置保持 E1_13 一致：ModelNet-C full、ULIP zero-shot、`manual75_llm25`。

## 执行命令：使用默认 Prompt JSON

默认模式不会调用 API，只读取已经存在的 E1_02 prompt JSON：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="conda run -n mcmpc python"
bash scripts/E1_text_prototype_enhancement/prompt_composition_ablation/E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25.sh 0
```

输出位置：

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25/
Point-Cache/results/E1_text_prototype_enhancement/E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25/summary.csv
Point-Cache/results/E1_text_prototype_enhancement/E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25/logs/
```

## 执行命令：调用 API 重新生成 Prompt

只有显式设置 `E1_USE_DEFAULT_PROMPTS=0` 时才会调用 API。重新生成的 prompt 会保存到 `Point-Cache/llm/` 下的新子目录。

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="conda run -n mcmpc python"
export E1_USE_DEFAULT_PROMPTS=0
export E1_GENERATED_PROMPT_TAG="trial01"
bash scripts/E1_text_prototype_enhancement/prompt_composition_ablation/E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25.sh 0
```

重新生成 prompt 的默认保存位置：

```text
Point-Cache/llm/generated/E1_20_15_prompts_10_image_5_pointcloud_trial01/
Point-Cache/llm/generated/E1_20_15_prompts_10_image_5_pointcloud_trial01/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json
```

重新生成模式的默认结果目录会带上 tag，避免覆盖默认 prompt 实验：

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25_regenerated_trial01/
```

可选开关：

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `E1_USE_DEFAULT_PROMPTS` | `1` | `1` 使用已有 prompt JSON；`0` 调用 API 重新生成 |
| `E1_GENERATED_PROMPT_TAG` | 当前时间戳 | 重新生成 prompt 的目录和结果目录 tag |
| `E1_GENERATED_PROMPT_DIR` | `llm/generated/E1_20_15_prompts_10_image_5_pointcloud_${tag}` | 自定义重新生成 prompt 保存目录 |
| `E1_GENERATED_PROMPT_FILE` | 默认 E1_02 文件名 | 自定义重新生成 prompt 文件名 |
| `E1_EXP_ID` | 自动生成 | 自定义结果目录名 |
| `E1_PROMPT_CACHE_DIR` | `llm` | 默认模式下自定义 prompt 目录 |
| `E1_PROMPT_CACHE_FILE` | 默认 E1_02 文件名 | 默认模式下自定义 prompt 文件名 |

## 完成检查

| 检查项 | 结果 |
|---|---|
| `summary.csv` 数据行数 | 35 |
| `status` | 全部 `done` |
| corruption 覆盖 | 7 类 |
| severity 覆盖 | 0, 1, 2, 3, 4 |
| full average | 48.81 |
| prompt 来源 | 默认 E1_02 prompt JSON，未调用 API 重新生成 |

## 对比对象

| 对比对象 | 路径 | 说明 |
|---|---|---|
| E1_13 `manual75_llm25` | `Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_full_manual75_llm25/summary.csv` | 当前更优权重，10 prompts = 4 image + 4 pointcloud + 2 bridge |
| E1_10 `manual90_llm10` | `Point-Cache/results/E1_text_prototype_enhancement/E1_10_modelnet_c_full_w90_10/summary.csv` | 10 prompts = 4 image + 4 pointcloud + 2 bridge，权重为 90:10 |
| E0 baseline | `Point-Cache/results/E0_baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv` | ULIP zero-shot baseline |

## 总体结果

| 设置 | ModelNet-C full average | 与 E1_20 差值 |
|---|---:|---:|
| E1_20 `15 prompts, 10 image + 5 pointcloud, manual75_llm25` | 48.81 | 0.00 |
| E1_13 `10 prompts, 4 image + 4 pointcloud + 2 bridge, manual75_llm25` | 48.07 | +0.74 |
| E1_10 `10 prompts, 4 image + 4 pointcloud + 2 bridge, manual90_llm10` | 47.60 | +1.21 |
| E0 baseline | 46.85 | +1.96 |

## 按 Severity 对比

| severity | E1_20 | E1_13 | diff | E1_10 | diff | E0 | diff |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 55.32 | 54.85 | +0.47 | 54.38 | +0.94 | 53.40 | +1.92 |
| 1 | 53.06 | 52.32 | +0.73 | 51.79 | +1.26 | 50.94 | +2.12 |
| 2 | 49.79 | 48.90 | +0.89 | 48.41 | +1.38 | 47.68 | +2.11 |
| 3 | 45.76 | 45.07 | +0.69 | 44.54 | +1.22 | 43.85 | +1.92 |
| 4 | 40.14 | 39.21 | +0.93 | 38.88 | +1.26 | 38.39 | +1.75 |

## 按 Corruption 对比

| corruption | E1_20 | E1_13 | diff | E1_10 | diff | E0 | diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| add_global | 37.99 | 36.25 | +1.74 | 35.69 | +2.30 | 34.89 | +3.11 |
| add_local | 45.88 | 44.96 | +0.92 | 45.13 | +0.76 | 44.57 | +1.32 |
| dropout_global | 54.87 | 54.71 | +0.16 | 53.79 | +1.08 | 52.92 | +1.95 |
| dropout_local | 52.65 | 51.27 | +1.38 | 51.17 | +1.48 | 50.22 | +2.43 |
| jitter | 43.05 | 42.01 | +1.04 | 42.05 | +1.00 | 41.66 | +1.39 |
| rotate | 53.97 | 54.15 | -0.18 | 53.47 | +0.50 | 52.69 | +1.28 |
| scale | 53.29 | 53.15 | +0.14 | 51.90 | +1.38 | 51.00 | +2.28 |

## 分析记录

与 E1_13 对比：

1. E1_20 和 E1_13 使用同一权重 `manual75_llm25`、同一 ModelNet-C full 评估范围、同一 ULIP zero-shot 设置；主要差异是 prompt 数量与组成。
2. E1_20 使用 15 prompts = 10 image + 5 pointcloud；E1_13 使用 10 prompts = 4 image + 4 pointcloud + 2 bridge。
3. E1_20 full average 为 48.81，高于 E1_13 的 48.07，提升 +0.74。
4. E1_20 在 5 个 severity 平均上全部高于 E1_13，提升最大的是 severity 4 (+0.93) 和 severity 2 (+0.89)。
5. E1_20 在 7 类 corruption 平均中有 6 类高于 E1_13，仅 `rotate` 低 -0.18。
6. 提升最明显的是 `add_global` (+1.74)、`dropout_local` (+1.38)、`jitter` (+1.04)、`add_local` (+0.92)。
7. 逐项看，E1_20 在 35 个评估项中有 27 项高于 E1_13，8 项低于 E1_13。

与 E1_10 对比：

1. E1_20 相比 E1_10 同时改变了权重和 prompt 设置，因此不能把 +1.21 完全归因到 prompt。
2. E1_20 在 35 个评估项中 35 项全部高于 E1_10。
3. 这说明当前的 `manual75_llm25 + 15 prompts, 10 image + 5 pointcloud` 明显优于早期 `manual90_llm10 + 10 prompts`。

与 E0 baseline 对比：

1. E1_20 full average 为 48.81，高于 E0 baseline 的 46.85，整体提升 +1.96。
2. E1_20 在所有 severity 平均上都高于 E0，提升范围为 +1.75 到 +2.12。
3. E1_20 在所有 corruption 平均上都高于 E0。
4. 逐项看，E1_20 在 35 个评估项中 35 项全部高于 E0。
5. 相比 E1_13，E1_20 修复了 `jitter` 原本偏弱的问题：`jitter` 相对 E1_13 提升 +1.04，相对 E0 提升 +1.39。

## 当前结论

在当前 ModelNet-C full 结果中，E1_20 是目前最强的 E1 设置：`manual75_llm25 + 15 prompts, 10 image + 5 pointcloud` 的 full average 达到 48.81，比 E1_13 高 +0.74，比 E0 baseline 高 +1.96。该结果说明，在当前权重下，增加到每类 15 条描述并偏向 image-style 描述是有效的；下一步应测试 E1_21，也就是 15 prompts = 5 image + 10 pointcloud，以判断收益来自“15 条数量”还是“10 image + 5 pointcloud 的比例”。
