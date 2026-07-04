# E1_22 ModelNet-C Severity2 15 Prompts 12 Image 3 Pointcloud Manual75 LLM25

更新日期：2026-06-17

## 基本实验设置

| 项目 | 设置 |
|---|---|
| 实验编号 | E1_22 severity2 diagnostic |
| Result `exp_id` | `E1_22_modelnet_c_severity2_15_prompts_12_image_3_pointcloud_manual75_llm25` |
| 实验目的 | 在 ModelNet-C severity=2 上诊断 15 条 LLM 描述、12 image + 3 pointcloud 是否进一步优于 E1_20 |
| 数据集与评估范围 | ModelNet-C severity=2，7 evaluations = 7 corruption types x 1 severity |
| Backbone | ULIP |
| 任务设置 | zero-shot |
| 文本原型方法 | `manual_full + LLM` weighted fusion |
| LLM prompt | 15 prompts/class = 12 image + 3 pointcloud |
| 融合权重 | `manual_full = 0.75`, `LLM = 0.25` |
| Prompt JSON | `Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_12_image_3_pointcloud.json` |
| 默认结果目录 | `Point-Cache/results/E1_text_prototype_enhancement/E1_22_modelnet_c_severity2_15_prompts_12_image_3_pointcloud_manual75_llm25/` |

## 与已有 Prompt 比例的关系

当前 prompt 比例诊断链：

| 实验 | Prompt 设置 | 范围 |
|---|---|---|
| E1_20 | 15 prompts = 10 image + 5 pointcloud | ModelNet-C full |
| E1_21 severity2 diagnostic | 15 prompts = 5 image + 10 pointcloud | ModelNet-C severity=2 |
| E1_22 severity2 diagnostic | 15 prompts = 12 image + 3 pointcloud | ModelNet-C severity=2 |

E1_22 用来判断 image-style 描述继续增加到 12/15 是否比 E1_20 的 10/15 更好。

## 第一步：生成 Prompt JSON

当前 `12 image + 3 pointcloud` 是新增比例，需要先调用 API 生成 prompt JSON。

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="python"
bash scripts/E1_text_prototype_enhancement/prompt_generation/E1_04_15_prompts_12_image_3_pointcloud.sh
```

输出保存位置：

```text
Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_12_image_3_pointcloud.json
```

完成后必须检查：

1. `class_names` 为 40；
2. `completed_class_names` 为 40；
3. 每类 prompt 数量为 15；
4. `failed_classes` 为空；
5. `llm_prompt_mode` 为 `image12_pointcloud3`。

## 第二步：运行 Severity2 诊断

默认模式不会调用 API，只读取上一步生成的 prompt JSON：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="python"
bash scripts/E1_text_prototype_enhancement/prompt_composition_ablation/E1_22_modelnet_c_severity2_15_prompts_12_image_3_pointcloud_manual75_llm25.sh 0
```

输出位置：

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_22_modelnet_c_severity2_15_prompts_12_image_3_pointcloud_manual75_llm25/
Point-Cache/results/E1_text_prototype_enhancement/E1_22_modelnet_c_severity2_15_prompts_12_image_3_pointcloud_manual75_llm25/summary.csv
Point-Cache/results/E1_text_prototype_enhancement/E1_22_modelnet_c_severity2_15_prompts_12_image_3_pointcloud_manual75_llm25/logs/
```

## 备用：一条命令中重新生成并运行

只有显式设置 `E1_USE_DEFAULT_PROMPTS=0` 时才会调用 API。重新生成的 prompt 会保存到 `Point-Cache/llm/` 下的新子目录。

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="python"
export E1_USE_DEFAULT_PROMPTS=0
export E1_GENERATED_PROMPT_TAG="trial01"
bash scripts/E1_text_prototype_enhancement/prompt_composition_ablation/E1_22_modelnet_c_severity2_15_prompts_12_image_3_pointcloud_manual75_llm25.sh 0
```

重新生成 prompt 的默认保存位置：

```text
Point-Cache/llm/generated/E1_22_15_prompts_12_image_3_pointcloud_trial01/
Point-Cache/llm/generated/E1_22_15_prompts_12_image_3_pointcloud_trial01/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_12_image_3_pointcloud.json
```

## 完成检查

| 检查项 | 结果 |
|---|---|
| `summary.csv` 数据行数 | 7 |
| `status` | 全部 `done` |
| severity | 全部为 `2` |
| corruption 覆盖 | 7 类 |
| average | 49.19 |
| prompt 来源 | E1_04 prompt JSON，未在运行阶段重新生成 |

## 对比对象

| 对比对象 | 路径 | 说明 |
|---|---|---|
| E1_20 severity=2 子集 | `Point-Cache/results/E1_text_prototype_enhancement/E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25/summary.csv` | 15 prompts = 10 image + 5 pointcloud |
| E1_21 severity2 diagnostic | `Point-Cache/results/E1_text_prototype_enhancement/E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25/summary.csv` | 15 prompts = 5 image + 10 pointcloud |
| E1_13 severity=2 子集 | `Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_full_manual75_llm25/summary.csv` | 10 prompts = 4 image + 4 pointcloud + 2 bridge |
| E1_10 severity=2 子集 | `Point-Cache/results/E1_text_prototype_enhancement/E1_10_modelnet_c_full_w90_10/summary.csv` | 10 prompts = 4 image + 4 pointcloud + 2 bridge, 90:10 |
| E0 baseline severity=2 子集 | `Point-Cache/results/E0_baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv` | ULIP zero-shot baseline |

## 总体结果

| 设置 | severity=2 average | 与 E1_22 差值 |
|---|---:|---:|
| E1_22 `15 prompts, 12 image + 3 pointcloud, manual75_llm25` | 49.19 | 0.00 |
| E1_20 `15 prompts, 10 image + 5 pointcloud, manual75_llm25` | 49.79 | -0.60 |
| E1_21 `15 prompts, 5 image + 10 pointcloud, manual75_llm25` | 48.55 | +0.64 |
| E1_13 `10 prompts, 4 image + 4 pointcloud + 2 bridge, manual75_llm25` | 48.90 | +0.29 |
| E1_10 `10 prompts, 4 image + 4 pointcloud + 2 bridge, manual90_llm10` | 48.41 | +0.78 |
| E0 baseline | 47.68 | +1.51 |

## 逐 Corruption 对比

| corruption | E1_22 | E1_20 | diff | E1_21 | diff | E1_13 | diff | E0 | diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| add_global | 34.04 | 37.48 | -3.44 | 33.47 | +0.57 | 35.33 | -1.29 | 34.00 | +0.04 |
| add_local | 44.53 | 45.02 | -0.49 | 44.00 | +0.53 | 44.49 | +0.04 | 43.92 | +0.61 |
| dropout_global | 56.93 | 56.85 | +0.08 | 57.25 | -0.32 | 57.13 | -0.20 | 54.70 | +2.23 |
| dropout_local | 54.05 | 54.29 | -0.24 | 52.03 | +2.02 | 51.82 | +2.23 | 50.57 | +3.48 |
| jitter | 45.06 | 45.54 | -0.48 | 44.45 | +0.61 | 44.12 | +0.94 | 44.49 | +0.57 |
| rotate | 56.52 | 55.96 | +0.56 | 56.32 | +0.20 | 56.16 | +0.36 | 55.19 | +1.33 |
| scale | 53.20 | 53.40 | -0.20 | 52.35 | +0.85 | 53.28 | -0.08 | 50.89 | +2.31 |

## 分析记录

与 E1_20 对比：

1. E1_22 和 E1_20 都是 15 prompts、`manual75_llm25`、ModelNet-C severity=2，主要差异是 prompt 比例：E1_22 为 12 image + 3 pointcloud，E1_20 为 10 image + 5 pointcloud。
2. E1_22 平均 49.19，低于 E1_20 的 49.79，差值 -0.60。
3. E1_22 在 `dropout_global` (+0.08) 和 `rotate` (+0.56) 上高于 E1_20，但在其余 5 类 corruption 上低于 E1_20。
4. 最大损失来自 `add_global` (-3.44)，这是 E1_22 没超过 E1_20 的主要原因。
5. 因此，继续把 image-style 比例从 10/15 提高到 12/15 并没有带来整体收益。

与 E1_21 对比：

1. E1_22 平均 49.19，高于 E1_21 的 48.55，差值 +0.64。
2. E1_22 在 7 类 corruption 中有 6 类高于 E1_21，仅 `dropout_global` 低 -0.32。
3. 这进一步说明，当前设置下 image-style 描述比例高于 pointcloud-style 描述比例更好。

与 E1_13 对比：

1. E1_22 平均 49.19，高于 E1_13 severity=2 子集的 48.90，差值 +0.29。
2. E1_22 在 `dropout_local` (+2.23)、`jitter` (+0.94)、`rotate` (+0.36)、`add_local` (+0.04) 上高于 E1_13。
3. E1_22 在 `add_global` (-1.29)、`dropout_global` (-0.20)、`scale` (-0.08) 上低于 E1_13。

与 E0 baseline 对比：

1. E1_22 平均 49.19，高于 E0 severity=2 baseline 的 47.68，差值 +1.51。
2. E1_22 在 7 类 corruption 上全部高于 E0。
3. 最大提升来自 `dropout_local` (+3.48)、`scale` (+2.31)、`dropout_global` (+2.23)、`rotate` (+1.33)。

## 当前结论

E1_22 证明 `12 image + 3 pointcloud` 明显优于 `5 image + 10 pointcloud`，也优于 E1_13 的 10 prompt 设置；但它仍低于 E1_20 的 `10 image + 5 pointcloud`。当前最优 prompt 比例仍是 E1_20：15 prompts = 10 image + 5 pointcloud。E1_22 不建议优先进入 full，除非后续需要系统性补全 prompt 比例消融矩阵。
