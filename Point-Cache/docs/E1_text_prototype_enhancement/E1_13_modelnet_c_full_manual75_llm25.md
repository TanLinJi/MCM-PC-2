# E1_13 ModelNet-C Full Manual75 LLM25

更新日期：2026-06-17

## 基本实验设置

| 项目 | 设置 |
|---|---|
| 实验编号 | E1_13 |
| Result `exp_id` | `E1_13_modelnet_c_full_manual75_llm25` |
| 实验目的 | 验证 `manual_full:LLM = 75:25` 在完整 ModelNet-C 上是否优于 E1_10 `manual90_llm10` 和 E0 baseline |
| 数据集与评估范围 | ModelNet-C full，35 evaluations = 7 corruption types x 5 severities |
| Backbone | ULIP |
| 任务设置 | zero-shot |
| 文本原型方法 | `manual_full + LLM` weighted fusion |
| LLM prompt | 10 prompts/class = 4 image + 4 pointcloud + 2 bridge |
| 融合权重 | `manual_full = 0.75`, `LLM = 0.25` |
| Prompt JSON | `Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_10_prompts_4_image_4_pointcloud_2_bridge.json` |
| 结果目录 | `Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_full_manual75_llm25/` |

## 与 Severity2 诊断实验的关系

已完成的 `E1_13 severity2 diagnostic` 使用同一权重和同一 prompt JSON，但只跑 ModelNet-C severity=2。

本实验是正式 full 版本，需要跑完整 ModelNet-C：

```text
7 corruption types x 5 severities = 35 evaluations
```

## 执行命令

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="conda run -n mcmpc python"
bash scripts/E1_text_prototype_enhancement/fusion_weight_ablation/E1_13_modelnet_c_full_manual75_llm25.sh 0
```

## 输出位置

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_full_manual75_llm25/
Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_full_manual75_llm25/summary.csv
Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_full_manual75_llm25/logs/
```

## 完成检查

| 检查项 | 结果 |
|---|---|
| `summary.csv` 数据行数 | 35 |
| `status` | 全部 `done` |
| corruption 覆盖 | 7 类 |
| severity 覆盖 | 0, 1, 2, 3, 4 |
| full average | 48.07 |

## 对比对象

| 对比对象 | 路径 | 说明 |
|---|---|---|
| E1_10 `manual90_llm10` | `Point-Cache/results/E1_text_prototype_enhancement/E1_10_modelnet_c_full_w90_10/summary.csv` | 同一新版 prompt，权重为 90:10 |
| E0 baseline | `Point-Cache/results/E0_baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv` | ULIP zero-shot baseline |
| E1_13 severity2 diagnostic | `Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_severity2_manual75_llm25/summary.csv` | 同权重同 prompt，只跑 severity=2 |
| 旧 `0.75:0.25` | `Point-Cache/results/E1_text_prototype_enhancement/01_3_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w075_025/summary.csv` | 旧 prompt，只跑 severity=2 |

## 总体结果

| 设置 | ModelNet-C full average | 与 E1_13 差值 |
|---|---:|---:|
| E1_13 `manual75_llm25` | 48.07 | 0.00 |
| E1_10 `manual90_llm10` | 47.60 | +0.47 |
| E0 baseline | 46.85 | +1.22 |

## 按 Severity 对比

| severity | E1_13 | E1_10 | diff | E0 | diff |
|---:|---:|---:|---:|---:|---:|
| 0 | 54.85 | 54.38 | +0.47 | 53.40 | +1.44 |
| 1 | 52.32 | 51.79 | +0.53 | 50.94 | +1.39 |
| 2 | 48.90 | 48.41 | +0.49 | 47.68 | +1.22 |
| 3 | 45.07 | 44.54 | +0.53 | 43.85 | +1.22 |
| 4 | 39.21 | 38.88 | +0.33 | 38.39 | +0.82 |

## 按 Corruption 对比

| corruption | E1_13 | E1_10 | diff | E0 | diff |
|---|---:|---:|---:|---:|---:|
| add_global | 36.25 | 35.69 | +0.56 | 34.89 | +1.36 |
| add_local | 44.96 | 45.13 | -0.17 | 44.57 | +0.39 |
| dropout_global | 54.71 | 53.79 | +0.91 | 52.92 | +1.78 |
| dropout_local | 51.27 | 51.17 | +0.10 | 50.22 | +1.05 |
| jitter | 42.01 | 42.05 | -0.04 | 41.66 | +0.35 |
| rotate | 54.15 | 53.47 | +0.68 | 52.69 | +1.46 |
| scale | 53.15 | 51.90 | +1.24 | 51.00 | +2.14 |

## Severity2 一致性检查

| 设置 | severity=2 average | 说明 |
|---|---:|---|
| E1_13 full 中的 severity=2 子集 | 48.90 | 从 35 项 full 结果中筛选 |
| E1_13 severity2 diagnostic | 48.90 | 单独跑 severity=2 |
| 差值 | 0.00 | 完全一致 |
| 旧 `0.75:0.25` | 48.88 | 旧 prompt，只能参考 |
| E1_13 full severity=2 - 旧 `0.75:0.25` | +0.02 | 基本持平 |

## 分析记录

与 E1_10 `manual90_llm10` 对比：

1. E1_13 与 E1_10 使用同一新版 prompt JSON、同一 ModelNet-C full 范围、同一 ULIP zero-shot 设置；主要差异是权重从 90:10 改为 75:25。
2. E1_13 full average 为 48.07，高于 E1_10 的 47.60，整体提升 +0.47。
3. E1_13 在 5 个 severity 上均高于 E1_10，说明 75:25 的收益不是只来自某一个扰动等级。
4. 逐 corruption 看，E1_13 在 `scale` (+1.24)、`dropout_global` (+0.91)、`rotate` (+0.68)、`add_global` (+0.56)、`dropout_local` (+0.10) 上优于 E1_10。
5. E1_13 在 `add_local` (-0.17) 和 `jitter` (-0.04) 上低于 E1_10，其中 `jitter` 的下降很小但延续了 severity=2 诊断里的负例趋势。
6. 逐项看，E1_13 在 35 个评估项中有 26 项高于 E1_10，9 项低于 E1_10。

与 E0 baseline 对比：

1. E1_13 full average 为 48.07，高于 E0 baseline 的 46.85，整体提升 +1.22。
2. E1_13 在所有 severity 平均上都高于 E0，最高提升出现在 severity 0 (+1.44) 和 severity 1 (+1.39)，最高扰动 severity 4 仍有 +0.82。
3. E1_13 在 7 类 corruption 平均上全部高于 E0。
4. 提升最大的 corruption 是 `scale` (+2.14)、`dropout_global` (+1.78)、`rotate` (+1.46)、`add_global` (+1.36)。
5. 逐项看，E1_13 在 35 个评估项中有 33 项高于 E0，仅 `jitter_2` 和 `jitter_4` 低于 E0。

与旧 `0.75:0.25` 对比：

1. 旧 `0.75:0.25` 只跑 severity=2，不能和 E1_13 full average 直接比较。
2. 在相同 severity=2 范围下，E1_13 full 子集平均为 48.90，旧结果为 48.88，差值 +0.02。
3. 因为旧结果使用旧 prompt JSON，而 E1_13 使用新版 prompt JSON，所以这只能说明新版 prompt 在同权重 severity=2 上基本保持旧结果水平，不能把 +0.02 解释为稳定收益。

## 当前结论

`manual75_llm25` 是目前比 `manual90_llm10` 更强的 ModelNet-C full 设置：它在 full average、所有 severity 平均、以及多数 corruption 平均上都优于 E1_10，同时相对 E0 baseline 有 +1.22 的整体提升。需要注意的风险点是 `jitter` 和 `add_local`：如果后续权重继续增大 LLM 比例，应重点观察这两类扰动是否进一步下降。
