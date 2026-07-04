# E1_40 ModelNet Clean 15 Prompts 10 Image 5 Pointcloud Manual60 LLM40

更新日期：2026-06-18

## 基本实验设置

| 项目 | 设置 |
|---|---|
| 实验编号 | E1_40 |
| Result `exp_id` | `E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40` |
| 实验目的 | 验证 E1_36 当前候选配置在干净 ModelNet 上是否保持或提升 E0 baseline |
| 数据集与评估范围 | clean ModelNet，单次评估 |
| 数据口径 | 使用仓库内 clean ModelNet 文件：`Point-Cache/data/modelnet_c/clean.h5` |
| Loader dataset | `modelnet_c` |
| `cor_type` | `clean` |
| Backbone | ULIP |
| 任务设置 | zero-shot |
| 文本原型方法 | `manual_full + LLM` weighted fusion |
| LLM prompt | 15 prompts/class = 10 image + 5 pointcloud |
| LLM prompt mode | `image10_pointcloud5` |
| 融合权重 | `manual_full = 0.60`, `LLM = 0.40` |
| Prompt JSON | `Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json` |
| 默认结果目录 | `Point-Cache/results/E1_text_prototype_enhancement/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40/` |

## 与 E1_36 的关系

E1_40 完全继承 E1_36 的方法设置，只改变评估数据：

| 项目 | E1_36 | E1_40 |
|---|---|---|
| 权重 | `manual60_llm40` | `manual60_llm40` |
| prompt | 15 prompts = 10 image + 5 pointcloud | 15 prompts = 10 image + 5 pointcloud |
| 数据范围 | ModelNet-C full，35 evaluations | clean ModelNet，1 evaluation |
| 数据文件 | `data/modelnet_c/{corruption}_{severity}.h5` | `data/modelnet_c/clean.h5` |

## 与 E0 Baseline 的对照

E0 clean ModelNet baseline 使用同一数据口径：

```text
Point-Cache/scripts/E0_baseline/01_1_ulip_modelnet_clean_zs_single_gpu.sh
Point-Cache/results/E0_baseline/01_1_ulip_modelnet_clean_zs/summary.csv
```

已知 E0 baseline：

| 实验 | 设置 | clean ModelNet accuracy |
|---|---|---:|
| E0 `01_1_ulip_modelnet_clean_zs` | ULIP zero-shot, original manual prompt | 56.77 |
| E1_40 | `manual60_llm40`, 15 prompts = 10 image + 5 pointcloud | 59.24 |

## 执行命令

默认模式不会重新调用 API，只读取已有 prompt JSON。

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E1_text_prototype_enhancement/modelnet_clean_validation/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40.sh 0
```

输出位置：

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40/
Point-Cache/results/E1_text_prototype_enhancement/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40/summary.csv
Point-Cache/results/E1_text_prototype_enhancement/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40/logs/
```

## 备用：重新生成 prompt 后运行

只有显式设置 `E1_USE_DEFAULT_PROMPTS=0` 时才会调用 API。重新生成的 prompt 会保存到 `Point-Cache/llm/generated/` 下的新目录。

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_USE_DEFAULT_PROMPTS=0
export E1_GENERATED_PROMPT_TAG="trial01"
bash scripts/E1_text_prototype_enhancement/modelnet_clean_validation/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40.sh 0
```

重新生成 prompt 的默认保存位置：

```text
Point-Cache/llm/generated/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_trial01/
```

重新生成模式的默认结果目录会带上 tag：

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40_regenerated_trial01/
```

## 完成检查

| 检查项 | 结果 |
|---|---|
| `summary.csv` 数据行数 | 1 |
| `status` | `done` |
| `corruption` | `clean` |
| `severity` | `-` |
| `cor_type` | `clean` |
| `file` | `data/modelnet_c/clean.h5` |
| clean ModelNet accuracy | 59.24 |

## 结果记录

结果文件：

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40/summary.csv
Point-Cache/results/E1_text_prototype_enhancement/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40/logs/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40_clean_20260618_002118.log
```

| 实验 | clean ModelNet accuracy | 相对 E0 baseline |
|---|---:|---:|
| E1_40 | 59.24 | +2.47 |
| E0 baseline | 56.77 | 0.00 |

## 对比分析

1. E1_40 在 clean ModelNet 上达到 59.24，比 E0 clean baseline 的 56.77 高 +2.47，说明 E1_36 候选配置没有破坏干净 ModelNet 上的原始识别能力。
2. 该提升幅度与 E1_36 在完整 ModelNet-C 上相对 E0 baseline 的 +2.51 非常接近，说明 `manual60_llm40 + 15 prompts, 10 image + 5 pointcloud` 的收益不只来自 corrupted setting，也能延续到 clean setting。
3. E1_40 与 E0 baseline 使用同一数据文件 `data/modelnet_c/clean.h5`，因此本次差异主要来自文本原型构造：E0 使用原始 manual prompt，E1_40 使用 `manual_full + LLM` 加权融合。
4. 从当前 ModelNet-C full 与 clean ModelNet 两个结果看，E1_36 候选配置具有继续进入 ScanObjectNN 与 ScanObjectNN-C 验证的价值。
5. E1_40 只覆盖 clean ModelNet，不能替代真实扫描 clean 数据和真实扫描 corruption 数据。下一步仍需在 ScanObjectNN、ScanObjectNN-C 上检查该配置是否稳定。
