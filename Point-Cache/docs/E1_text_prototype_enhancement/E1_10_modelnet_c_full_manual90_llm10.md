# E1_10 ModelNet-C Full Manual90 LLM10

更新日期：2026-06-17

## 基本实验设置

| 项目 | 设置 |
|---|---|
| 实验编号 | E1_10 |
| Canonical 设置名 | `manual90_llm10` |
| 实验目的 | 验证 `manual_full:LLM = 90:10` 在完整 ModelNet-C 上是否优于 E0 `manual_full` baseline |
| 数据集与评估范围 | ModelNet-C full，35 evaluations = 7 corruption types x 5 severities |
| Backbone | ULIP |
| 任务设置 | zero-shot |
| 文本原型方法 | `manual_full + LLM` weighted fusion |
| LLM prompt | 10 prompts/class = 4 image + 4 pointcloud + 2 bridge |
| 融合权重 | `manual_full = 0.90`, `LLM = 0.10` |
| Prompt JSON | `Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_10_prompts_4_image_4_pointcloud_2_bridge.json` |
| 结果目录 | `Point-Cache/results/E1_text_prototype_enhancement/E1_10_modelnet_c_full_w90_10/` |

注：该结果生成时结果目录仍使用旧缩写 `w90_10`。正式设置名和脚本名已经改为 `manual90_llm10`。

## 执行命令

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="conda run -n mcmpc python"
bash scripts/E1_text_prototype_enhancement/fusion_weight_ablation/E1_10_modelnet_c_full_manual90_llm10.sh 0
```

## 输出位置

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_10_modelnet_c_full_w90_10/
Point-Cache/results/E1_text_prototype_enhancement/E1_10_modelnet_c_full_w90_10/summary.csv
Point-Cache/results/E1_text_prototype_enhancement/E1_10_modelnet_c_full_w90_10/logs/
```

## 完成检查

| 检查项 | 结果 |
|---|---|
| `summary.csv` 行数 | 35 |
| `status` | 全部 `done` |
| corruption 覆盖 | 7 类 |
| severity 覆盖 | 0, 1, 2, 3, 4 |

## 总体结果

E0 baseline 文件：

```text
Point-Cache/results/E0_baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv
```

| 指标 | 数值 |
|---|---:|
| E1_10 full average | 47.60 |
| E0 ULIP zero-shot full average | 46.85 |
| E1_10 - E0 | +0.75 |
| E1_10 severity=2 subset average | 48.41 |

## 按 Severity 对比 E0

| severity | E1_10 | E0 | diff |
|---:|---:|---:|---:|
| 0 | 54.38 | 53.40 | +0.97 |
| 1 | 51.79 | 50.94 | +0.86 |
| 2 | 48.41 | 47.68 | +0.73 |
| 3 | 44.54 | 43.85 | +0.70 |
| 4 | 38.88 | 38.39 | +0.49 |

分析记录：

1. E1_10 在 5 个 severity 上均高于 E0 `manual_full` baseline。
2. 提升幅度随扰动等级加重而逐步变小：severity 0 为 +0.97，severity 4 为 +0.49。
3. 当前 `manual90_llm10` 对轻中度扰动帮助更明显，但最高扰动强度下仍保留正增益。

## 按 Corruption 对比 E0

| corruption | E1_10 | E0 | diff |
|---|---:|---:|---:|
| add_global | 35.69 | 34.89 | +0.80 |
| add_local | 45.13 | 44.57 | +0.56 |
| dropout_global | 53.79 | 52.92 | +0.87 |
| dropout_local | 51.17 | 50.22 | +0.95 |
| jitter | 42.05 | 41.66 | +0.39 |
| rotate | 53.47 | 52.69 | +0.78 |
| scale | 51.90 | 51.00 | +0.90 |

分析记录：

1. E1_10 在 7 类 corruption 上均高于 E0 `manual_full` baseline。
2. 提升较大的类型是 `dropout_local` (+0.95)、`scale` (+0.90)、`dropout_global` (+0.87)。
3. 提升最小的是 `jitter` (+0.39)，说明当前 10% LLM 描述对随机抖动类扰动的帮助较弱。
4. `add_global` (+0.80) 和 `add_local` (+0.56) 都是正增益，但局部加点的提升幅度更小。

## 与旧 0.75:0.25 的关系

旧 `0.75:0.25` 结果不是新版 E1 full 实验，不能直接与 E1_10 full average 比较。关键差异：

1. 旧结果只跑 ModelNet-C `severity=2`，共 7 行；E1_10 跑完整 ModelNet-C，共 35 行。
2. 旧结果使用旧 prompt cache：`results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json`。
3. E1_10 使用新版 prompt JSON：`Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_10_prompts_4_image_4_pointcloud_2_bridge.json`。
4. 两个 prompt JSON 内容不相同，40 个类别的 prompt 文本均发生变化。
5. 因此旧 `0.75:0.25` 与 E1_10 不是“只有权重不同”。

仅做 severity=2 子集的参考对比：

| 设置 | severity=2 average |
|---|---:|
| 旧 `0.75:0.25` | 48.88 |
| E1_10 `manual90_llm10` | 48.41 |
| 差值 | -0.47 |

该差值只能作为参考，因为 prompt 文本和实验范围均不同。
