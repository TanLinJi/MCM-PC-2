# E1_21 ModelNet-C Severity2 15 Prompts 5 Image 10 Pointcloud Manual75 LLM25

更新日期：2026-06-17

## 基本实验设置

| 项目 | 设置 |
|---|---|
| 实验编号 | E1_21 severity2 diagnostic |
| Result `exp_id` | `E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25` |
| 实验目的 | 先在 ModelNet-C severity=2 上诊断 15 条 LLM 描述、5 image + 10 pointcloud 是否值得跑 full |
| 数据集与评估范围 | ModelNet-C severity=2，7 evaluations = 7 corruption types x 1 severity |
| Backbone | ULIP |
| 任务设置 | zero-shot |
| 文本原型方法 | `manual_full + LLM` weighted fusion |
| LLM prompt | 15 prompts/class = 5 image + 10 pointcloud |
| 融合权重 | `manual_full = 0.75`, `LLM = 0.25` |
| 默认 Prompt JSON | `Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_5_image_10_pointcloud.json` |
| 默认结果目录 | `Point-Cache/results/E1_text_prototype_enhancement/E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25/` |

## 与 E1_20 的关系

E1_20 使用：

```text
15 prompts = 10 image + 5 pointcloud
```

本诊断实验使用：

```text
15 prompts = 5 image + 10 pointcloud
```

两者都固定 `manual75_llm25`。先只跑 severity=2，用来判断 pointcloud-style 占比更高是否值得进入完整 ModelNet-C full。

## 执行命令：使用默认 Prompt JSON

默认模式不会调用 API，只读取已经存在的 E1_03 prompt JSON：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="conda run -n mcmpc python"
bash scripts/E1_text_prototype_enhancement/prompt_composition_ablation/E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25.sh 0
```

输出位置：

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25/
Point-Cache/results/E1_text_prototype_enhancement/E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25/summary.csv
Point-Cache/results/E1_text_prototype_enhancement/E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25/logs/
```

## 执行命令：调用 API 重新生成 Prompt

只有显式设置 `E1_USE_DEFAULT_PROMPTS=0` 时才会调用 API。重新生成的 prompt 会保存到 `Point-Cache/llm/` 下的新子目录。

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="conda run -n mcmpc python"
export E1_USE_DEFAULT_PROMPTS=0
export E1_GENERATED_PROMPT_TAG="trial01"
bash scripts/E1_text_prototype_enhancement/prompt_composition_ablation/E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25.sh 0
```

重新生成 prompt 的默认保存位置：

```text
Point-Cache/llm/generated/E1_21_15_prompts_5_image_10_pointcloud_trial01/
Point-Cache/llm/generated/E1_21_15_prompts_5_image_10_pointcloud_trial01/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_5_image_10_pointcloud.json
```

重新生成模式的默认结果目录会带上 tag：

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25_regenerated_trial01/
```

## 完成检查

| 检查项 | 结果 |
|---|---|
| `summary.csv` 数据行数 | 7 |
| `status` | 全部 `done` |
| severity | 全部为 `2` |
| corruption 覆盖 | 7 类 |
| average | 48.55 |
| prompt 来源 | 默认 E1_03 prompt JSON，未调用 API 重新生成 |

## 对比对象

| 对比对象 | 路径 | 说明 |
|---|---|---|
| E1_20 severity=2 子集 | `Point-Cache/results/E1_text_prototype_enhancement/E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25/summary.csv` | 15 prompts = 10 image + 5 pointcloud |
| E1_13 severity=2 子集 | `Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_full_manual75_llm25/summary.csv` | 10 prompts = 4 image + 4 pointcloud + 2 bridge |
| E1_10 severity=2 子集 | `Point-Cache/results/E1_text_prototype_enhancement/E1_10_modelnet_c_full_w90_10/summary.csv` | 10 prompts = 4 image + 4 pointcloud + 2 bridge, 90:10 |
| E0 baseline severity=2 子集 | `Point-Cache/results/E0_baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv` | ULIP zero-shot baseline |

## 总体结果

| 设置 | severity=2 average | 与 E1_21 差值 |
|---|---:|---:|
| E1_21 `15 prompts, 5 image + 10 pointcloud, manual75_llm25` | 48.55 | 0.00 |
| E1_20 `15 prompts, 10 image + 5 pointcloud, manual75_llm25` | 49.79 | -1.24 |
| E1_13 `10 prompts, 4 image + 4 pointcloud + 2 bridge, manual75_llm25` | 48.90 | -0.35 |
| E1_10 `10 prompts, 4 image + 4 pointcloud + 2 bridge, manual90_llm10` | 48.41 | +0.14 |
| E0 baseline | 47.68 | +0.87 |

## 逐 Corruption 对比

| corruption | E1_21 | E1_20 | diff | E1_13 | diff | E0 | diff |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 33.47 | 37.48 | -4.01 | 35.33 | -1.86 | 34.00 | -0.53 |
| add_local | 44.00 | 45.02 | -1.02 | 44.49 | -0.49 | 43.92 | +0.08 |
| dropout_global | 57.25 | 56.85 | +0.40 | 57.13 | +0.12 | 54.70 | +2.55 |
| dropout_local | 52.03 | 54.29 | -2.26 | 51.82 | +0.21 | 50.57 | +1.46 |
| jitter | 44.45 | 45.54 | -1.09 | 44.12 | +0.33 | 44.49 | -0.04 |
| rotate | 56.32 | 55.96 | +0.36 | 56.16 | +0.16 | 55.19 | +1.13 |
| scale | 52.35 | 53.40 | -1.05 | 53.28 | -0.93 | 50.89 | +1.46 |

## 分析记录

与 E1_20 对比：

1. E1_21 与 E1_20 都使用 15 prompts 和 `manual75_llm25`，主要差异是 prompt 比例：E1_21 为 5 image + 10 pointcloud，E1_20 为 10 image + 5 pointcloud。
2. E1_21 severity=2 average 为 48.55，低于 E1_20 的 49.79，差值 -1.24。
3. E1_21 在 7 类 corruption 中只有 `dropout_global` (+0.40) 和 `rotate` (+0.36) 高于 E1_20，其余 5 类低于 E1_20。
4. 下降最明显的是 `add_global` (-4.01)、`dropout_local` (-2.26)、`jitter` (-1.09)、`scale` (-1.05)、`add_local` (-1.02)。
5. 这说明在 severity=2 诊断范围内，pointcloud-style 占比更高的 5 image + 10 pointcloud 明显弱于 image-style 占比更高的 10 image + 5 pointcloud。

与 E1_13 对比：

1. E1_21 平均 48.55，低于 E1_13 severity=2 子集的 48.90，差值 -0.35。
2. E1_21 在 `dropout_global`、`dropout_local`、`jitter`、`rotate` 上高于 E1_13，但在 `add_global`、`add_local`、`scale` 上低于 E1_13。
3. E1_21 在 `jitter` 上比 E1_13 高 +0.33，但仍低于 E1_20 的 `jitter` 结果。

与 E0 baseline 对比：

1. E1_21 平均 48.55，高于 E0 severity=2 baseline 的 47.68，差值 +0.87。
2. E1_21 在 7 类 corruption 中有 5 类高于 E0，低于 E0 的是 `add_global` (-0.53) 和 `jitter` (-0.04)。
3. 这说明 E1_21 仍然优于 baseline，但不如 E1_20 稳定。

## 当前结论

E1_21 severity2 diagnostic 不支持直接进入 full 实验：`15 prompts, 5 image + 10 pointcloud` 在 severity=2 上明显低于 E1_20 的 `15 prompts, 10 image + 5 pointcloud`，也低于 E1_13 的 10 prompt 设置。当前应保留 E1_20 作为 prompt 数量与组成消融中的更优候选；E1_21 full 可以暂缓，除非后续需要系统性补全消融矩阵。
