# E1 Current Candidate Configuration

更新日期：2026-06-18

## 当前状态

E1 当前正式候选配置固定为 E1_36：

```text
manual_full + LLM weighted fusion
manual_full = 0.60
LLM = 0.40
15 prompts/class = 10 image + 5 pointcloud
```

该配置不是最终跨数据集结论，而是后续 ModelNet、ScanObjectNN、ScanObjectNN-C 验证的默认候选配置。最终配置需要等四数据集验证完成后再冻结。

## 配置明细

| 项目 | 固定值 |
|---|---|
| 配置来源 | E1_36 |
| 选择依据 | ModelNet-C full 结果优于 E1_20、E1_13 和 E0 baseline |
| Backbone | ULIP |
| 任务设置 | zero-shot |
| 文本原型方法 | `manual_full + LLM` weighted fusion |
| 手工模板分支 | `manual_full` |
| LLM prompt 数量 | 15 prompts/class |
| LLM prompt 组成 | 10 image-style + 5 pointcloud-style |
| LLM prompt mode | `image10_pointcloud5` |
| Prompt JSON | `Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json` |
| 融合权重 | `manual_full = 0.60`, `LLM = 0.40` |
| ModelNet-C full 实验 | E1_36 |
| E1_36 结果目录 | `Point-Cache/results/E1_text_prototype_enhancement/E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40/` |
| E1_36 详细文档 | `E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40.md` |

## 固定原因

E1_36 使用 E1_33 的方法设置，并在完整 ModelNet-C 上验证：

| 实验 | 设置 | ModelNet-C full average |
|---|---|---:|
| E1_36 | 15 prompts, `manual60_llm40` | 49.36 |
| E1_20 | 15 prompts, `manual75_llm25` | 48.81 |
| E1_13 | 10 prompts, `manual75_llm25` | 48.07 |
| E1_10 | 10 prompts, `manual90_llm10` | 47.60 |
| E0 baseline | ULIP zero-shot | 46.85 |

E1_36 相比 E1_20 提升 +0.55，相比 E0 baseline 提升 +2.51。按 severity 看，E1_36 在 0 到 4 的所有 severity 上均优于 E1_20，说明 E1_33 在 severity=2 上的收益可以推广到完整 ModelNet-C。

clean ModelNet 验证结果：

| 实验 | 设置 | clean ModelNet accuracy |
|---|---|---:|
| E1_40 | 15 prompts, `manual60_llm40` | 59.24 |
| E0 baseline | ULIP zero-shot | 56.77 |

E1_40 相比 E0 clean ModelNet baseline 提升 +2.47，说明当前候选配置在 clean ModelNet 上没有牺牲原始识别能力。

## 与 E1_34 的关系

E1_34 在 ModelNet-C severity=2 上的平均值略高于 E1_33：

| 实验 | 权重 | severity=2 average |
|---|---|---:|
| E1_34 | `manual55_llm45` | 50.42 |
| E1_33 | `manual60_llm40` | 50.41 |
| E1_35 | `manual57.5_llm42.5` | 50.40 |

但 E1_34 相比 E1_33 的平均优势只有 +0.01，并且在 `add_local`、`dropout_local`、`jitter` 上出现更明显的 trade-off。因此当前采用更稳妥、已完成 full 验证的 E1_36，而不是只在 severity=2 略高的 E1_34。

## 已知风险

E1_36 相对 E1_20 的下降项集中在：

| 类型 | 说明 |
|---|---|
| `add_local` | 5 个 severity 均低于 E1_20，corruption 平均 -0.44 |
| high-severity `jitter` | `jitter_2/3/4` 低于 E1_20，尤其 `jitter_4` 为 -0.89 |

后续跨数据集验证必须重点观察局部扰动和强 jitter 扰动下是否继续存在类似弱点。

## 后续实验默认设置

后续 E1_40、E1_50、E1_60 系列验证应默认使用：

```text
prompt_source = manualfull_llm_dynamic_init
prompt_cache_file = modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json
llm_prompt_mode = image10_pointcloud5
dynamic_prompt_count = 15
prompt_static_weight = 0.60
prompt_dynamic_weight = 0.40
```

如果后续数据集结果显示该配置在 clean 或真实扫描数据上不稳定，再回到 E1_34/E1_35 或新的权重进行补充验证。
