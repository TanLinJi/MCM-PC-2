# E1_33 ModelNet-C Severity2 15 Prompts 10 Image 5 Pointcloud Manual60 LLM40

更新日期：2026-06-17

## 基本实验设置

| 项目 | 设置 |
|---|---|
| 实验编号 | E1_33 severity2 diagnostic |
| Result `exp_id` | `E1_33_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual60_llm40` |
| 实验目的 | 在 ModelNet-C severity=2 上验证更高 LLM 权重是否优于此前 `manual75_llm25` 与 `manual68_llm32` |
| 数据集与评估范围 | ModelNet-C severity=2，7 evaluations = 7 corruption types x 1 severity |
| Backbone | ULIP |
| 任务设置 | zero-shot |
| 文本原型方法 | `manual_full + LLM` weighted fusion |
| LLM prompt | 15 prompts/class = 10 image + 5 pointcloud |
| 融合权重 | `manual_full = 0.60`, `LLM = 0.40` |
| Prompt JSON | `Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json` |
| 结果目录 | `Point-Cache/results/E1_text_prototype_enhancement/E1_33_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual60_llm40/` |

## 当前采用状态

E1_33 是当前 E1 的临时采用版本，用于进入完整 ModelNet-C full 验证。虽然 E1_34 的 severity=2 平均值略高于 E1_33，但 E1_34 相对 E1_33 主要通过提高 `add_global` 和 `dropout_global` 获益，同时明显损失 `add_local`、`dropout_local` 和 `jitter`。因此当前先采用更稳妥的 E1_33：`manual60_llm40`。

## 执行命令

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E1_text_prototype_enhancement/prompt_composition_ablation/E1_33_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual60_llm40.sh 0
```

输出位置：

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_33_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual60_llm40/
Point-Cache/results/E1_text_prototype_enhancement/E1_33_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual60_llm40/summary.csv
Point-Cache/results/E1_text_prototype_enhancement/E1_33_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual60_llm40/logs/
```

## 完成检查

| 检查项 | 结果 |
|---|---|
| `summary.csv` 数据行数 | 7 |
| `status` | 全部 `done` |
| severity | 全部为 `2` |
| corruption 覆盖 | 7 类 |
| average | 50.41 |

## 逐 Corruption 结果

| corruption | E1_33 |
|---|---:|
| add_global | 37.93 |
| add_local | 44.77 |
| dropout_global | 57.78 |
| dropout_local | 55.11 |
| rotate | 56.97 |
| scale | 54.86 |
| jitter | 45.42 |

## 关键对比

| 设置 | severity=2 average | 与 E1_33 差值 |
|---|---:|---:|
| E1_34 `manual55_llm45` | 50.42 | -0.01 |
| E1_33 `manual60_llm40` | 50.41 | 0.00 |
| E1_35 `manual57.5_llm42.5` | 50.40 | +0.01 |
| E1_32 `manual68_llm32` | 50.14 | +0.26 |
| E1_20 `manual75_llm25` | 49.79 | +0.61 |
| E0 baseline | 47.68 | +2.73 |

## 分析记录

1. 从 `manual75_llm25` 到 `manual60_llm40`，severity=2 平均从 49.79 提升到 50.41，提升 +0.61。
2. E1_33 明显优于低 LLM 权重区域，说明每类 15 条 prompt 下，LLM 描述权重应显著高于 25%。
3. E1_34 平均值只比 E1_33 高 +0.01，但扰动间 trade-off 更明显：`add_local`、`dropout_local` 和 `jitter` 更弱。
4. E1_35 作为 40% 和 45% 的中点没有超过 E1_33/E1_34，说明当前高权重区已经接近平台。
5. 因此当前采用 E1_33 作为更稳妥的 full 验证候选，而不是单看平均值略高的 E1_34。

## 下一步

使用 E1_33 的设置在完整 ModelNet-C 上运行 E1_36：

```text
manual_full = 0.60
LLM = 0.40
15 prompts/class = 10 image + 5 pointcloud
ModelNet-C full = 35 evaluations
```
