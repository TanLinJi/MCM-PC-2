# E1 Result Summary

更新日期：2026-06-18

## 说明

E1 正式结果只记录新版 E1 中 `manual_full + LLM` 的实验结果。E0 baseline 只作为外部对照引用。

本文件只做结果索引和总览，不写长篇分析。每个具体实验的基本设置、执行命令、结果路径、检查项和详细分析写入独立实验文档。

## 当前正式候选配置

当前 E1 正式候选配置固定为 E1_36：

```text
manual_full + LLM weighted fusion
manual_full = 0.60
LLM = 0.40
15 prompts/class = 10 image + 5 pointcloud
```

配置说明见 `current_candidate_config.md`。该配置用于后续 ModelNet、ScanObjectNN 和 ScanObjectNN-C 验证；最终配置需等待四数据集验证完成后再冻结。

## 实验文档索引

| 实验 | 范围 | 设置 | 状态 | 详细文档 |
|---|---|---|---|---|
| E1_10 | ModelNet-C full | `manual90_llm10`, 10 prompts/class = 4 image + 4 pointcloud + 2 bridge | done | `E1_10_modelnet_c_full_manual90_llm10.md` |
| E1_13 | ModelNet-C full | `manual75_llm25`, 10 prompts/class = 4 image + 4 pointcloud + 2 bridge | done | `E1_13_modelnet_c_full_manual75_llm25.md` |
| E1_13 severity2 diagnostic | ModelNet-C severity=2 | `manual75_llm25`, 10 prompts/class = 4 image + 4 pointcloud + 2 bridge | done | `E1_13_modelnet_c_severity2_manual75_llm25.md` |
| E1_20 | ModelNet-C full | `manual75_llm25`, 15 prompts/class = 10 image + 5 pointcloud | done | `E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25.md` |
| E1_21 severity2 diagnostic | ModelNet-C severity=2 | `manual75_llm25`, 15 prompts/class = 5 image + 10 pointcloud | done | `E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25.md` |
| E1_22 severity2 diagnostic | ModelNet-C severity=2 | `manual75_llm25`, 15 prompts/class = 12 image + 3 pointcloud | done | `E1_22_modelnet_c_severity2_15_prompts_12_image_3_pointcloud_manual75_llm25.md` |
| E1_23 severity2 diagnostic | ModelNet-C severity=2 | `manual73_llm27`, 15 prompts/class = 10 image + 5 pointcloud | done | `E1_23_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual73_llm27.md` |
| E1_24 severity2 diagnostic | ModelNet-C severity=2 | `manual74_llm26`, 15 prompts/class = 10 image + 5 pointcloud | done | `E1_24_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual74_llm26.md` |
| E1_33 severity2 diagnostic | ModelNet-C severity=2 | `manual60_llm40`, 15 prompts/class = 10 image + 5 pointcloud | adopted for full validation | `E1_33_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual60_llm40.md` |
| E1_36 | ModelNet-C full | `manual60_llm40`, 15 prompts/class = 10 image + 5 pointcloud | done | `E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40.md` |
| E1_40 | clean ModelNet | `manual60_llm40`, 15 prompts/class = 10 image + 5 pointcloud | done | `E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40.md` |

## ModelNet-C Full Index

ModelNet-C full 指：

```text
7 corruption types x 5 severities = 35 evaluations
```

| 实验 | 平均准确率 | E0 对照 | 差值 | 结果目录 |
|---|---:|---:|---:|---|
| E1_10 | 47.60 | 46.85 | +0.75 | `Point-Cache/results/E1_text_prototype_enhancement/E1_10_modelnet_c_full_w90_10/` |
| E1_13 | 48.07 | 46.85 | +1.22 | `Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_full_manual75_llm25/` |
| E1_20 | 48.81 | 46.85 | +1.96 | `Point-Cache/results/E1_text_prototype_enhancement/E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25/` |
| E1_36 | 49.36 | 46.85 | +2.51 | `Point-Cache/results/E1_text_prototype_enhancement/E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40/` |

## ModelNet Clean Index

clean ModelNet 使用 `Point-Cache/data/modelnet_c/clean.h5`，与 E0 clean ModelNet baseline 保持同一数据口径。

| 实验 | 准确率 | E0 对照 | 差值 | 结果目录 |
|---|---:|---:|---:|---|
| E1_40 | 59.24 | 56.77 | +2.47 | `Point-Cache/results/E1_text_prototype_enhancement/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40/` |

## Diagnostic Index

| 实验 | 范围 | 平均准确率 | 对照 | 差值 | 当前用途 |
|---|---|---:|---:|---:|---|
| E1_13 severity2 diagnostic | ModelNet-C severity=2 | 48.90 | E1_10 severity=2: 48.41 | +0.49 | 隔离同一新版 prompt 下 `manual75_llm25` 与 `manual90_llm10` 的 severity=2 差异 |
| E1_21 severity2 diagnostic | ModelNet-C severity=2 | 48.55 | E1_20 severity=2: 49.79 | -1.24 | 诊断 `15 prompts, 5 image + 10 pointcloud` 是否值得进入 full |
| E1_22 severity2 diagnostic | ModelNet-C severity=2 | 49.19 | E1_20 severity=2: 49.79 | -0.60 | 诊断 `15 prompts, 12 image + 3 pointcloud` 是否优于 E1_20 |
| E1_23 severity2 diagnostic | ModelNet-C severity=2 | 49.91 | E1_20 severity=2: 49.79 | +0.12 | 诊断同一 prompt 组成下 `manual73_llm27` 是否优于 `manual75_llm25` |
| E1_24 severity2 diagnostic | ModelNet-C severity=2 | 49.84 | E1_23 severity=2: 49.91 | -0.07 | 诊断同一 prompt 组成下 `manual74_llm26` 是否优于 `manual73_llm27` |
| E1_33 severity2 diagnostic | ModelNet-C severity=2 | 50.41 | E1_20 severity=2: 49.79 | +0.61 | 当前采用候选；进入 ModelNet-C full 验证 |
